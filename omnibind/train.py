"""Training loop for OmniBind models.

Supports both single-GPU and distributed training (Horovod).
Horovod is optional: if not installed, falls back to single-GPU training.
"""

import math
import os
import gc
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm import trange
from lifelines.utils import concordance_index
from sklearn.metrics import (
    auc, mean_squared_error, precision_recall_curve,
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
)

from omnibind.data_utils import CPIDataset, collate_fn
from omnibind.model import build_model
from omnibind.utils import makedirs, param_count, param_count_all

# Optional Horovod support
try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def _hvd_initialized() -> bool:
    """Check if Horovod is available and initialized."""
    if not HAS_HOROVOD:
        return False
    try:
        hvd.size()
        return True
    except ValueError:
        return False


def _is_distributed() -> bool:
    return _hvd_initialized() and hvd.size() > 1


def _rank() -> int:
    return hvd.rank() if _hvd_initialized() else 0


def _local_rank() -> int:
    return hvd.local_rank() if _hvd_initialized() else 0


def _world_size() -> int:
    return hvd.size() if _hvd_initialized() else 1


def rmse(targets: List[float], preds: List[float]) -> float:
    """Compute root mean squared error."""
    return math.sqrt(mean_squared_error(targets, preds))


LABEL_NAMES = ['Ki', 'Kd', 'IC50', 'EC50']


class Trainer:
    """Training and evaluation handler."""

    def __init__(self, model: nn.Module, optimizer, cfg: DictConfig):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.loss_func = nn.MSELoss(reduction='mean')

    def train(self, dataset: CPIDataset, epoch: int) -> tuple:
        """Run one training epoch.

        Returns:
            Tuple of (loss_total, loss_ki, loss_kd, loss_ic50, loss_ec50).
        """
        self.model.train()
        torch.cuda.empty_cache()

        kwargs = {'num_workers': self.cfg.training.num_workers}
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        if _is_distributed():
            import torch.utils.data.distributed as ddp
            train_sampler = ddp.DistributedSampler(
                dataset, num_replicas=_world_size(), rank=_rank())
            train_sampler.set_epoch(epoch)
        else:
            train_sampler = None

        dataloader = DataLoader(
            dataset, batch_size=self.cfg.training.batch_size_train,
            sampler=train_sampler, shuffle=(train_sampler is None),
            collate_fn=collate_fn, pin_memory=self.cfg.training.pin_memory,
            drop_last=self.cfg.training.drop_last, **kwargs,
        )

        loss_total = 0
        loss_ki_total = 0
        loss_kd_total = 0
        loss_ic50_total = 0
        loss_ec50_total = 0
        rmse_count = 0
        ki_count = 0
        kd_count = 0
        ic50_count = 0
        ec50_count = 0

        scaler = torch.cuda.amp.GradScaler()

        for i, (compounds, adjs, aas, sas, kis, kds, ic50s, ec50s, atom_num, aa_num, sa_num) in enumerate(dataloader):
            compounds = compounds.to(self.cfg.training.device)
            adjs = adjs.to(self.cfg.training.device)
            aas = aas.to(self.cfg.training.device)
            sas = sas.to(self.cfg.training.device)

            # Masks for missing labels (-1)
            kis_mask = (kis != -1)
            kds_mask = (kds != -1)
            ic50s_mask = (ic50s != -1)
            ec50s_mask = (ec50s != -1)

            kis = kis.to(self.cfg.training.device)
            kds = kds.to(self.cfg.training.device)
            ic50s = ic50s.to(self.cfg.training.device)
            ec50s = ec50s.to(self.cfg.training.device)

            with torch.cuda.amp.autocast():
                predicted_kis, predicted_kds, predicted_ic50s, predicted_ec50s = self.model(
                    compounds, adjs, aas, sas, atom_num, aa_num, sa_num)

                loss = torch.tensor(0.0).to(self.cfg.training.device)
                label_count = 0

                if kis_mask.any():
                    loss_kis = self.loss_func(predicted_kis[kis_mask], kis[kis_mask])
                    loss += loss_kis
                    label_count += 1
                    ki_count += 1
                    loss_kis_value = loss_kis.item()
                else:
                    loss_kis_value = 0

                if kds_mask.any():
                    loss_kds = self.loss_func(predicted_kds[kds_mask], kds[kds_mask])
                    loss += loss_kds
                    label_count += 1
                    kd_count += 1
                    loss_kds_value = loss_kds.item()
                else:
                    loss_kds_value = 0

                if ic50s_mask.any():
                    loss_ic50s = self.loss_func(predicted_ic50s[ic50s_mask], ic50s[ic50s_mask])
                    loss += loss_ic50s
                    label_count += 1
                    ic50_count += 1
                    loss_ic50s_value = loss_ic50s.item()
                else:
                    loss_ic50s_value = 0

                if ec50s_mask.any():
                    loss_ec50s = self.loss_func(predicted_ec50s[ec50s_mask], ec50s[ec50s_mask])
                    loss += loss_ec50s
                    label_count += 1
                    ec50_count += 1
                    loss_ec50s_value = loss_ec50s.item()
                else:
                    loss_ec50s_value = 0

                loss = loss / label_count
                loss_value = loss.item()

                # Gradient accumulation
                loss = loss / self.cfg.training.accumulate_grad_batches

            scaler.scale(loss).backward()
            del loss
            torch.cuda.memory_reserved(self.cfg.training.device)

            if (i + 1) % self.cfg.training.accumulate_grad_batches == 0:
                if _is_distributed():
                    self.optimizer.synchronize()
                    scaler.unscale_(self.optimizer)
                    with self.optimizer.skip_synchronize():
                        scaler.step(self.optimizer)
                else:
                    scaler.step(self.optimizer)
                scaler.update()

                for param in self.model.parameters():
                    param.grad = None

            rmse_count += 1
            loss_total += math.sqrt(loss_value)
            loss_ki_total += math.sqrt(loss_kis_value)
            loss_kd_total += math.sqrt(loss_kds_value)
            loss_ic50_total += math.sqrt(loss_ic50s_value)
            loss_ec50_total += math.sqrt(loss_ec50s_value)

        if rmse_count != 0:
            loss_total /= rmse_count
        if ki_count != 0:
            loss_ki_total /= ki_count
        if kd_count != 0:
            loss_kd_total /= kd_count
        if ic50_count != 0:
            loss_ic50_total /= ic50_count
        if ec50_count != 0:
            loss_ec50_total /= ec50_count

        if _is_distributed():
            loss_total = self._metric_average(loss_total, 'avg_loss_total')
            loss_ki_total = self._metric_average(loss_ki_total, 'avg_loss_ki')
            loss_kd_total = self._metric_average(loss_kd_total, 'avg_loss_kd')
            loss_ic50_total = self._metric_average(loss_ic50_total, 'avg_loss_ic50')
            loss_ec50_total = self._metric_average(loss_ec50_total, 'avg_loss_ec50')

        return loss_total, loss_ki_total, loss_kd_total, loss_ic50_total, loss_ec50_total

    def eval(self, dataset: CPIDataset) -> dict:
        """Evaluate model on a dataset.

        Returns:
            Nested dict: {all: {RMSE,...}, Ki: {...}, Kd: {...}, IC50: {...}, EC50: {...}}.
        """
        self.model.eval()

        kwargs = {'num_workers': self.cfg.training.num_workers}
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        if _is_distributed():
            import torch.utils.data.distributed as ddp
            valid_sampler = ddp.DistributedSampler(
                dataset, num_replicas=_world_size(), rank=_rank())
        else:
            valid_sampler = None

        dataloader = DataLoader(
            dataset, batch_size=self.cfg.training.batch_size_valid,
            sampler=valid_sampler, shuffle=False,
            pin_memory=self.cfg.training.pin_memory,
            collate_fn=collate_fn, **kwargs,
        )

        T_Kis, P_Kis = [], []
        T_Kds, P_Kds = [], []
        T_IC50s, P_IC50s = [], []
        T_EC50s, P_EC50s = [], []

        with torch.no_grad():
            for compounds, adjs, aas, sas, kis, kds, ic50s, ec50s, atom_num, aa_num, sa_num in dataloader:
                compounds = compounds.to(self.cfg.training.device)
                adjs = adjs.to(self.cfg.training.device)
                aas = aas.to(self.cfg.training.device)
                sas = sas.to(self.cfg.training.device)

                kis_mask = (kis != -1).numpy()
                kds_mask = (kds != -1).numpy()
                ic50s_mask = (ic50s != -1).numpy()
                ec50s_mask = (ec50s != -1).numpy()

                kis_np = kis.numpy()[kis_mask]
                kds_np = kds.numpy()[kds_mask]
                ic50s_np = ic50s.numpy()[ic50s_mask]
                ec50s_np = ec50s.numpy()[ec50s_mask]

                predicted_kis, predicted_kds, predicted_ic50s, predicted_ec50s = self.model(
                    compounds, adjs, aas, sas, atom_num, aa_num, sa_num)

                del compounds, adjs, aas, sas, atom_num, aa_num, sa_num
                torch.cuda.empty_cache()

                predicted_kis = predicted_kis.cpu().numpy()[kis_mask]
                predicted_kds = predicted_kds.cpu().numpy()[kds_mask]
                predicted_ic50s = predicted_ic50s.cpu().numpy()[ic50s_mask]
                predicted_ec50s = predicted_ec50s.cpu().numpy()[ec50s_mask]

                del kis_mask, kds_mask, ic50s_mask, ec50s_mask
                gc.collect()

                T_Kis.extend(kis_np)
                T_Kds.extend(kds_np)
                T_IC50s.extend(ic50s_np)
                T_EC50s.extend(ec50s_np)
                del kis_np, kds_np, ic50s_np, ec50s_np
                gc.collect()

                P_Kis.extend(predicted_kis)
                P_Kds.extend(predicted_kds)
                P_IC50s.extend(predicted_ic50s)
                P_EC50s.extend(predicted_ec50s)
                del predicted_kis, predicted_kds, predicted_ic50s, predicted_ec50s
                gc.collect()

        metrics = {'all': {}, 'Ki': {}, 'Kd': {}, 'IC50': {}, 'EC50': {}}

        for label_name, T, P in [('Ki', T_Kis, P_Kis), ('Kd', T_Kds, P_Kds),
                                  ('IC50', T_IC50s, P_IC50s), ('EC50', T_EC50s, P_EC50s)]:
            if len(T) > 0:
                metrics[label_name]['RMSE'] = rmse(T, P)
                metrics[label_name]['CINDEX'] = concordance_index(T, P)

                true_labels = np.array(T) >= self.cfg.dataset.threshold
                pred_labels = np.array(P) >= self.cfg.dataset.threshold

                metrics[label_name]['AUC'] = roc_auc_score(true_labels, P)
                tpr, fpr, _ = precision_recall_curve(true_labels, P)
                metrics[label_name]['PRC'] = auc(fpr, tpr)
                metrics[label_name]['ACC'] = accuracy_score(true_labels, pred_labels)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average='binary')
                metrics[label_name]['precision'] = precision
                metrics[label_name]['recall'] = recall
                metrics[label_name]['f1'] = f1
            else:
                metrics[label_name] = {k: float('nan') for k in
                                       ['RMSE', 'CINDEX', 'AUC', 'PRC', 'ACC', 'precision', 'recall', 'f1']}

        del T_Kis, P_Kis, T_Kds, P_Kds, T_IC50s, P_IC50s, T_EC50s, P_EC50s
        gc.collect()

        # Compute 'all' as mean of valid (non-nan) per-label metrics
        for metric_name in ['RMSE', 'CINDEX', 'AUC', 'PRC', 'ACC', 'precision', 'recall', 'f1']:
            valid_values = [metrics[lbl][metric_name] for lbl in LABEL_NAMES
                            if not math.isnan(metrics[lbl][metric_name])]
            metrics['all'][metric_name] = (sum(valid_values) / len(valid_values)
                                           if valid_values else float('nan'))

        if _is_distributed():
            for label_type in metrics:
                for metric_name in metrics[label_type]:
                    metrics[label_type][metric_name] = self._metric_average(
                        metrics[label_type][metric_name],
                        f'avg_{metric_name}_{label_type}')

        return metrics

    def _metric_average(self, val: float, name: str) -> float:
        """Average a metric across all Horovod workers."""
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()


class WarmupCosineLambda:
    """Warmup + cosine annealing learning rate lambda."""

    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float,
                 exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(
                    self.decay_scale, -epoch / self.warmup_steps)
            ratio = epoch / self.warmup_steps
        else:
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio


def init_scheduler(cfg: DictConfig, optimizer, num_steps_per_epoch: int = None):
    """Initialize learning rate scheduler from config.

    If scheduler.type is null/None, returns None (manual lr decay will be used).

    Args:
        cfg: Scheduler config section.
        optimizer: Optimizer instance.
        num_steps_per_epoch: Steps per epoch (for step-level schedulers).

    Returns:
        Scheduler instance or None.
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, CosineAnnealingWarmRestarts,
        ExponentialLR, ReduceLROnPlateau, StepLR,
    )
    from transformers import get_cosine_schedule_with_warmup

    if cfg.type is None:
        return None
    elif cfg.type == "step_lr":
        return StepLR(optimizer, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay_rate)
    elif cfg.type == "exponential_lr":
        return ExponentialLR(optimizer, gamma=cfg.lr_decay_rate)
    elif cfg.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_decay_rate,
            patience=cfg.patience, verbose=True, min_lr=cfg.min_lr)
    elif cfg.type == "cosine_annealing_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    elif cfg.type == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    elif cfg.type == "cosine_warmup":
        warmup_steps = cfg.max_epochs * cfg.warmup_steps_ratio
        cycle_steps = cfg.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, cfg.lr_decay_scale)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.type == "cosine":
        steps_per_epoch = num_steps_per_epoch or cfg.num_steps_per_epoch
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps_per_epoch * cfg.max_epochs * cfg.warmup_steps_ratio),
            num_training_steps=int(steps_per_epoch * cfg.max_epochs),
            num_cycles=cfg.num_cycles)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")


def init_optimizer(cfg: DictConfig, params):
    """Initialize optimizer from config.

    Args:
        cfg: Optimizer config section.
        params: Model parameters.

    Returns:
        Configured optimizer.
    """
    from torch.optim import SGD, Adam, AdamW, RAdam

    opt_type = cfg.type
    if opt_type == "adam":
        return Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.eps, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    elif opt_type == "adamw":
        return AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                     eps=cfg.eps, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    elif opt_type == "sgd":
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif opt_type == "radam":
        return RAdam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                     eps=cfg.eps, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def run_training(cfg: DictConfig, train_data: CPIDataset, validation_data: CPIDataset,
                 logger=None) -> dict:
    """Main training loop.

    Args:
        cfg: Full Hydra config.
        train_data: Training dataset.
        validation_data: Validation dataset.
        logger: Optional logger.

    Returns:
        Dict of last epoch's validation metrics (only on rank 0).
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    torch.backends.cudnn.deterministic = True

    # NOTE: hvd.init() should be called in the entry point (scripts/train.py),
    # not here. This function assumes Horovod is already initialized if available.

    if _rank() == 0:
        debug(f'train size = {len(train_data):,} | val size = {len(validation_data):,}')

    torch.manual_seed(cfg.training.seed)

    save_dir = cfg.out_dir
    makedirs(save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    # Build model
    if _rank() == 0:
        debug('Building model')

    model = build_model(cfg)

    if _rank() == 0:
        debug(model)
        debug(f'Number of parameters = {param_count_all(model):,}')
        debug(f'Number of trainable parameters = {param_count(model):,}')

    model.to(cfg.training.device)

    # Optimizer
    lr_scaler = _world_size() if _is_distributed() and not cfg.training.use_adasum else 1
    if _is_distributed() and cfg.training.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

    lr = cfg.optimizer.lr * lr_scaler
    optimizer = init_optimizer(cfg.optimizer, model.parameters())
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    if _is_distributed():
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Checkpoint resume (before DistributedOptimizer wrapping)
    checkpoint_epoch = 0
    best_rmse = np.inf

    if cfg.test.checkpoint_path:
        if _rank() == 0:
            checkpoint = torch.load(cfg.test.checkpoint_path, map_location=cfg.training.device)
            debug(f'Loading model from {cfg.test.checkpoint_path}')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.training.device)
            checkpoint_epoch = checkpoint['epoch']
            best_rmse = checkpoint['best_loss']

        if _is_distributed():
            checkpoint_epoch = hvd.broadcast_object(checkpoint_epoch, root_rank=0)
            best_rmse = hvd.broadcast_object(best_rmse, root_rank=0)

    # Wrap optimizer with DistributedOptimizer AFTER checkpoint load
    if _is_distributed():
        compression = hvd.Compression.fp16 if cfg.training.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Adasum if cfg.training.use_adasum else hvd.Average,
            gradient_predivide_factor=cfg.training.gradient_predivide_factor,
            backward_passes_per_step=cfg.training.accumulate_grad_batches,
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Scheduler (optional — if scheduler.type is null, uses manual lr_decay only)
    scheduler = None
    if hasattr(cfg, 'scheduler') and getattr(cfg.scheduler, 'type', None) is not None:
        num_steps = len(train_data) // cfg.training.batch_size_train
        scheduler = init_scheduler(cfg.scheduler, optimizer, num_steps_per_epoch=num_steps)
        if _rank() == 0:
            debug(f'Using scheduler: {cfg.scheduler.type}')

    trainer = Trainer(model, optimizer, cfg)
    best_epoch = checkpoint_epoch
    start_epoch = best_epoch + 1

    iterator = trange if _rank() == 0 else range

    for epoch in iterator(start_epoch, cfg.training.epochs + 1):
        if _rank() == 0:
            debug(f'Epoch {epoch}')

        # Manual lr decay (default, matches original exp20 behavior)
        if scheduler is None and epoch % cfg.training.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= cfg.training.lr_decay

        rmse_train, rmse_ki, rmse_kd, rmse_ic50, rmse_ec50 = trainer.train(train_data, epoch)
        metrics = trainer.eval(validation_data)

        # Step scheduler if configured
        if scheduler is not None:
            scheduler.step()

        if _is_distributed():
            trainer.optimizer.synchronize()

        if _rank() == 0:
            debug(f'Train RMSE = {rmse_train:.6f}')
            writer.add_scalar('train_rmse', rmse_train, epoch)
            debug(f'Train Ki RMSE = {rmse_ki:.6f}')
            writer.add_scalar('train_ki_rmse', rmse_ki, epoch)
            debug(f'Train Kd RMSE = {rmse_kd:.6f}')
            writer.add_scalar('train_kd_rmse', rmse_kd, epoch)
            debug(f'Train IC50 RMSE = {rmse_ic50:.6f}')
            writer.add_scalar('train_ic50_rmse', rmse_ic50, epoch)
            debug(f'Train EC50 RMSE = {rmse_ec50:.6f}')
            writer.add_scalar('train_ec50_rmse', rmse_ec50, epoch)

            for metric_name, value in metrics['all'].items():
                debug(f'Validation {metric_name} mean = {value:.6f}')
                writer.add_scalar(f'Validation_{metric_name}_mean', value, epoch)

            for label_type in LABEL_NAMES:
                for metric_name, value in metrics[label_type].items():
                    debug(f'Validation {label_type} {metric_name} = {value:.6f}')
                    writer.add_scalar(f'Validation_{label_type}_{metric_name}', value, epoch)

            if metrics['all']['RMSE'] < best_rmse:
                debug(f'best RMSE mean update {best_rmse:.6f} -> {metrics["all"]["RMSE"]:.6f}')
                best_rmse = metrics['all']['RMSE']
                best_epoch = epoch
                debug(f'Save the model on epoch {best_epoch}!')
                model.to('cpu')
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_loss': best_rmse,
                }, os.path.join(cfg.out_dir, f'best_rmse_model_epoch{start_epoch}-.pth'))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_loss': best_rmse,
            }, os.path.join(cfg.out_dir, f'model_epoch{start_epoch}-.pth'))

            model.to(_local_rank() if _is_distributed() else cfg.training.device)

    if _rank() == 0:
        info(f'Best validation RMSE mean = {best_rmse:.6f} on epoch {best_epoch}')
        results = metrics
        results['best_epoch'] = best_epoch
        return results
    return None
