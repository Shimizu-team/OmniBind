"""Training entry point for OmniBind.

Usage:
    # Single GPU
    python scripts/train.py dataset.data_dir=./data/processed/seed42

    # Distributed (Horovod)
    horovodrun -np 4 python scripts/train.py dataset.data_dir=./data/processed/seed42
"""

import os
import sys
import time
import json
import warnings
from typing import Callable, Dict
from logging import Logger

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnibind.utils import makedirs, timeit, create_logger
from omnibind.data_utils import CPIDataset

warnings.filterwarnings('ignore')

TRAIN_LOGGER_NAME = 'train'

# Optional Horovod
try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


@timeit(logger_name=TRAIN_LOGGER_NAME)
def main(cfg: DictConfig, train_func: Callable) -> None:

    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=cfg.out_dir, quiet=cfg.quiet)
    debug, info = (logger.debug, logger.info) if logger else (print, print)

    # Initialize Horovod (must be done before any hvd calls)
    if HAS_HOROVOD:
        hvd.init()
        if cfg.training.cuda:
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(cfg.training.seed)
        # Limit CPU threads per worker (match original HPC config)
        torch.set_num_threads(max(1, os.cpu_count() // (hvd.size() or 1)))

    rank = hvd.rank() if HAS_HOROVOD else 0

    # Load preprocessed data (flat directory with 8 npy files per split)
    if rank == 0:
        debug('Loading data')
    start = time.time()

    data_dir = cfg.dataset.data_dir

    compounds_train = np.load(os.path.join(data_dir, 'compounds_train.npy'), allow_pickle=True)
    adjancies_train = np.load(os.path.join(data_dir, 'adjancies_train.npy'), allow_pickle=True)
    aas_train = np.load(os.path.join(data_dir, 'aas_train.npy'), allow_pickle=True)
    sas_train = np.load(os.path.join(data_dir, 'sas_train.npy'), allow_pickle=True)
    ki_train = np.load(os.path.join(data_dir, 'ki_train.npy'), allow_pickle=True)
    kd_train = np.load(os.path.join(data_dir, 'kd_train.npy'), allow_pickle=True)
    ic50_train = np.load(os.path.join(data_dir, 'ic50_train.npy'), allow_pickle=True)
    ec50_train = np.load(os.path.join(data_dir, 'ec50_train.npy'), allow_pickle=True)

    compounds_valid = np.load(os.path.join(data_dir, 'compounds_valid.npy'), allow_pickle=True)
    adjancies_valid = np.load(os.path.join(data_dir, 'adjancies_valid.npy'), allow_pickle=True)
    aas_valid = np.load(os.path.join(data_dir, 'aas_valid.npy'), allow_pickle=True)
    sas_valid = np.load(os.path.join(data_dir, 'sas_valid.npy'), allow_pickle=True)
    ki_valid = np.load(os.path.join(data_dir, 'ki_valid.npy'), allow_pickle=True)
    kd_valid = np.load(os.path.join(data_dir, 'kd_valid.npy'), allow_pickle=True)
    ic50_valid = np.load(os.path.join(data_dir, 'ic50_valid.npy'), allow_pickle=True)
    ec50_valid = np.load(os.path.join(data_dir, 'ec50_valid.npy'), allow_pickle=True)

    elapsed = time.time() - start
    if rank == 0:
        info(f'{(elapsed // 3600):.0f}h{((elapsed % 3600) // 60):.0f}m{(elapsed % 3600 % 60):.0f}s to load data npy.')
        debug('finish Loading data')

    dataset_train = CPIDataset(compounds_train, adjancies_train, aas_train, sas_train,
                               ki_train, kd_train, ic50_train, ec50_train)
    dataset_valid = CPIDataset(compounds_valid, adjancies_valid, aas_valid, sas_valid,
                               ki_valid, kd_valid, ic50_valid, ec50_valid)

    if cfg.preprocessing.save_memory:
        dataset_train.save_memory(
            max_atom_len=cfg.preprocessing.max_atom_len,
            max_aa_len=cfg.preprocessing.max_aa_len)
        dataset_valid.save_memory(
            max_atom_len=cfg.preprocessing.max_atom_len,
            max_aa_len=cfg.preprocessing.max_aa_len)

    if cfg.training.debug:
        dataset_train = torch.utils.data.Subset(dataset_train, indices=list(range(16000)))
        dataset_valid = torch.utils.data.Subset(dataset_valid, indices=list(range(30000)))

    if rank == 0:
        info('Starting training and validation!')

    results = train_func(cfg, dataset_train, dataset_valid, logger)

    if rank == 0:
        info('Finished training and validation!')
        if results:
            file_path = os.path.abspath(os.path.join(cfg.out_dir, 'results.json'))
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            info(f'Results saved to {file_path}')


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def entry(cfg: DictConfig) -> None:
    from omnibind.train import run_training
    main(cfg=cfg, train_func=run_training)


if __name__ == "__main__":
    entry()
