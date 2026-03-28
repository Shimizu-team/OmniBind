"""Test evaluation entry point for OmniBind.

Usage:
    python scripts/test.py dataset.data_dir=./data/processed/seed42 \
        test.checkpoint_path=./checkpoints/best_model.pth

    # Distributed (Horovod)
    horovodrun -np 4 python scripts/test.py dataset.data_dir=./data/processed/seed42 \
        test.checkpoint_path=./checkpoints/best_model.pth
"""

import os
import sys
import time
import json
import warnings
from logging import Logger

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omnibind.utils import makedirs, timeit, create_logger
from omnibind.data_utils import CPIDataset

warnings.filterwarnings('ignore')

TEST_LOGGER_NAME = 'test'

# Optional Horovod
try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


@timeit(logger_name=TEST_LOGGER_NAME)
def main(cfg: DictConfig) -> None:
    from omnibind.evaluate import run_testing

    logger = create_logger(name=TEST_LOGGER_NAME, save_dir=cfg.out_dir, quiet=cfg.quiet)
    debug, info = (logger.debug, logger.info) if logger else (print, print)

    # Initialize Horovod
    if HAS_HOROVOD:
        hvd.init()
        if cfg.training.cuda:
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(cfg.training.seed)
        torch.set_num_threads(max(1, os.cpu_count() // (hvd.size() or 1)))

    rank = hvd.rank() if HAS_HOROVOD else 0

    # Load test data (flat directory with 8 npy files)
    if rank == 0:
        debug('Loading test data')
    start = time.time()

    data_dir = cfg.dataset.data_dir
    compounds_test = np.load(os.path.join(data_dir, 'compounds_test.npy'), allow_pickle=True)
    adjancies_test = np.load(os.path.join(data_dir, 'adjancies_test.npy'), allow_pickle=True)
    aas_test = np.load(os.path.join(data_dir, 'aas_test.npy'), allow_pickle=True)
    sas_test = np.load(os.path.join(data_dir, 'sas_test.npy'), allow_pickle=True)
    ki_test = np.load(os.path.join(data_dir, 'ki_test.npy'), allow_pickle=True)
    kd_test = np.load(os.path.join(data_dir, 'kd_test.npy'), allow_pickle=True)
    ic50_test = np.load(os.path.join(data_dir, 'ic50_test.npy'), allow_pickle=True)
    ec50_test = np.load(os.path.join(data_dir, 'ec50_test.npy'), allow_pickle=True)

    elapsed = time.time() - start
    if rank == 0:
        info(f'{(elapsed // 3600):.0f}h{((elapsed % 3600) // 60):.0f}m{(elapsed % 3600 % 60):.0f}s to load data npy.')
        debug('finished loading test data')

    dataset_test = CPIDataset(compounds_test, adjancies_test, aas_test, sas_test,
                              ki_test, kd_test, ic50_test, ec50_test)

    if cfg.preprocessing.save_memory:
        dataset_test.save_memory(
            max_atom_len=cfg.preprocessing.max_atom_len,
            max_aa_len=cfg.preprocessing.max_aa_len)

    if cfg.training.debug:
        dataset_test = torch.utils.data.Subset(dataset_test, indices=list(range(5000)))

    if rank == 0:
        info('Starting test evaluation!')

    results = run_testing(cfg, dataset_test, logger)

    if rank == 0:
        info('Finished test evaluation!')
        if results is not None:
            file_path = os.path.abspath(os.path.join(cfg.out_dir, 'test_results.json'))
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            info(f'Test results saved to {file_path}')


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def entry(cfg: DictConfig) -> None:
    main(cfg=cfg)


if __name__ == "__main__":
    entry()
