"""Utility functions for OmniBind."""

import logging
import math
import os
from datetime import timedelta
from functools import wraps
from time import time
from typing import Any, Callable

import torch.nn as nn


def makedirs(path: str, isfile: bool = False) -> None:
    """Create directory (or parent directory if isfile=True)."""
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """Create a logger with stream and file handlers.

    Args:
        name: Logger name.
        save_dir: Directory for log files (verbose.log + quiet.log).
        quiet: If True, stream handler only shows INFO+.

    Returns:
        Configured logger.
    """
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)
        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """Decorator that logs elapsed time of a function."""
    def timeit_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name else print
            info(f'Elapsed time = {delta}')
            return result
        return wrap
    return timeit_decorator


def param_count(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_count_all(model: nn.Module) -> int:
    """Count all parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def compute_pnorm(model: nn.Module) -> float:
    """Compute L2 norm of model parameters."""
    return math.sqrt(sum(p.norm().item() ** 2 for p in model.parameters()))


def compute_gnorm(model: nn.Module) -> float:
    """Compute L2 norm of model gradients."""
    return math.sqrt(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None))


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights with Xavier normal (2D) or zeros (1D)."""
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
