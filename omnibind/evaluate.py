"""Evaluation / testing module for OmniBind models.

Supports both single-GPU and distributed evaluation (Horovod).
"""

import gc
import json
import math
import os
import time
from logging import Logger
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from omegaconf import DictConfig
from lifelines.utils import concordance_index
from sklearn.metrics import (
    auc, mean_squared_error, precision_recall_curve,
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
)

from omnibind.data_utils import CPIDataset, collate_fn
from omnibind.model import build_model
from omnibind.utils import makedirs, param_count, param_count_all

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


def _rank() -> int:
    return hvd.rank() if _hvd_initialized() else 0


def _world_size() -> int:
    return hvd.size() if _hvd_initialized() else 1


def _is_distributed() -> bool:
    return _hvd_initialized() and hvd.size() > 1


def rmse(targets: List[float], preds: List[float]) -> float:
    """Compute root mean squared error."""
    return math.sqrt(mean_squared_error(targets, preds))


LABEL_NAMES = ['Ki', 'Kd', 'IC50', 'EC50']


class Tester:
    """Test evaluation handler."""

    def __init__(self, model: nn.Module, cfg: DictConfig, logger: Logger = None):
        self.cfg = cfg
        self.model = model
        self.logger = logger

        if self.logger is not None:
            self.debug, self.info = self.logger.debug, self.logger.info
        else:
            self.debug = self.info = print

    def test(self, dataset: CPIDataset) -> Dict[str, dict]:
        """Run test evaluation.

        Returns:
            Nested dict: {all: {RMSE,...}, Ki: {...}, Kd: {...}, IC50: {...}, EC50: {...}}.
        """
        self.model.eval()

        kwargs = {'num_workers': self.cfg.training.num_workers}
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        test_batch_size = getattr(self.cfg.test, 'batch_size', self.cfg.training.batch_size_valid)

        if _is_distributed():
            import torch.utils.data.distributed as ddp
            test_sampler = ddp.DistributedSampler(
                dataset, num_replicas=_world_size(), rank=_rank())
        else:
            test_sampler = None

        dataloader = DataLoader(
            dataset, batch_size=test_batch_size,
            sampler=test_sampler, shuffle=False,
            pin_memory=self.cfg.training.pin_memory,
            collate_fn=collate_fn, **kwargs,
        )

        T_Kis, P_Kis = [], []
        T_Kds, P_Kds = [], []
        T_IC50s, P_IC50s = [], []
        T_EC50s, P_EC50s = [], []

        torch.cuda.empty_cache()
        start_time = time.time()

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

                predicted_kis = predicted_kis.cpu().numpy()[kis_mask]
                predicted_kds = predicted_kds.cpu().numpy()[kds_mask]
                predicted_ic50s = predicted_ic50s.cpu().numpy()[ic50s_mask]
                predicted_ec50s = predicted_ec50s.cpu().numpy()[ec50s_mask]

                T_Kis.extend(kis_np)
                T_Kds.extend(kds_np)
                T_IC50s.extend(ic50s_np)
                T_EC50s.extend(ec50s_np)

                P_Kis.extend(predicted_kis)
                P_Kds.extend(predicted_kds)
                P_IC50s.extend(predicted_ic50s)
                P_EC50s.extend(predicted_ec50s)

                del compounds, adjs, aas, sas
                torch.cuda.empty_cache()
                gc.collect()

        elapsed = time.time() - start_time
        self.debug(f'Prediction completed in {elapsed:.1f}s')

        true_pred_pairs = {
            'Ki': (T_Kis, P_Kis),
            'Kd': (T_Kds, P_Kds),
            'IC50': (T_IC50s, P_IC50s),
            'EC50': (T_EC50s, P_EC50s),
        }

        if _is_distributed():
            return self._gather_and_compute_metrics(true_pred_pairs)
        else:
            return self._compute_metrics(true_pred_pairs)

    def _compute_metrics(self, true_pred_pairs: dict) -> dict:
        """Compute evaluation metrics from per-label predictions."""
        metrics = {'all': {}, 'Ki': {}, 'Kd': {}, 'IC50': {}, 'EC50': {}}

        for label_name in LABEL_NAMES:
            T, P = true_pred_pairs[label_name]
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

        # Compute 'all' as mean of valid per-label metrics
        for metric_name in ['RMSE', 'CINDEX', 'AUC', 'PRC', 'ACC', 'precision', 'recall', 'f1']:
            valid_values = [metrics[lbl][metric_name] for lbl in LABEL_NAMES
                            if not math.isnan(metrics[lbl][metric_name])]
            metrics['all'][metric_name] = (sum(valid_values) / len(valid_values)
                                           if valid_values else float('nan'))

        return metrics

    def _gather_and_compute_metrics(self, true_pred_pairs: dict) -> dict:
        """Gather predictions from all ranks and compute metrics on rank 0."""
        save_dir = os.path.join(self.cfg.out_dir, 'rank_predictions')
        os.makedirs(save_dir, exist_ok=True)

        rank = _rank()
        save_data = {}
        for label_name in LABEL_NAMES:
            T, P = true_pred_pairs[label_name]
            save_data[f'true_{label_name}'] = np.array(T)
            save_data[f'pred_{label_name}'] = np.array(P)

        np.savez_compressed(
            os.path.join(save_dir, f'predictions_rank{rank}.npz'),
            **save_data,
        )

        hvd.allreduce(torch.tensor(0), name='barrier')

        if rank == 0:
            gathered_pairs = {lbl: ([], []) for lbl in LABEL_NAMES}
            for r in range(_world_size()):
                data = np.load(os.path.join(save_dir, f'predictions_rank{r}.npz'))
                for label_name in LABEL_NAMES:
                    gathered_pairs[label_name][0].extend(data[f'true_{label_name}'])
                    gathered_pairs[label_name][1].extend(data[f'pred_{label_name}'])
            results = self._compute_metrics(gathered_pairs)
        else:
            results = {}

        results = hvd.broadcast_object(results, root_rank=0)
        return results


def run_testing(cfg: DictConfig, test_data: CPIDataset, logger: Logger = None) -> dict:
    """Run test evaluation pipeline.

    Args:
        cfg: Full Hydra config.
        test_data: Test dataset.
        logger: Optional logger.

    Returns:
        Dict of test metrics (only on rank 0).
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(cfg.training.seed)

    if _rank() == 0:
        debug(f'test size = {len(test_data):,}')

    save_dir = cfg.out_dir
    makedirs(save_dir)

    # Build model
    model = build_model(cfg)
    if _rank() == 0:
        debug(f'Number of parameters = {param_count_all(model):,}')

    model.to(cfg.training.device)

    # Load checkpoint
    if not cfg.test.checkpoint_path:
        if _rank() == 0:
            debug('No checkpoint provided. Cannot test without a trained model.')
        return None

    if _rank() == 0:
        checkpoint = torch.load(cfg.test.checkpoint_path, map_location=cfg.training.device)
        debug(f'Loading model from {cfg.test.checkpoint_path}')
        model.load_state_dict(checkpoint['model_state_dict'])

    if _is_distributed():
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    tester = Tester(model, cfg, logger)
    results = tester.test(test_data)

    if _rank() == 0:
        for label_type in ['all'] + LABEL_NAMES:
            for metric_name, value in results[label_type].items():
                if isinstance(value, (int, float)):
                    info(f'Test {label_type} {metric_name} = {value:.6f}')

        test_result_path = os.path.join(cfg.out_dir, 'test_results.json')
        with open(test_result_path, 'w') as f:
            json.dump(results, f, indent=4)
        info(f'Test results saved to {test_result_path}')

        return results
    return None
