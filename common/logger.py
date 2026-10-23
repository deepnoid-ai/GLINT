import logging
import os
import sys
from datetime import datetime

import torch.distributed as dist
from transformers.trainer import logger


class CustomPrefixFilter(logging.Filter):
    def filter(self, record):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.msg = f"[{current_time} {record.levelname}] {record.msg}"
        return True


class MainProcessFilter(logging.Filter):
    def filter(self, record):
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True


def load_logger(logger=None, level=logging.INFO):
    if logger is None:
        logger = logging.getLogger("myLogger")

    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.filters.clear()
    logger.addFilter(CustomPrefixFilter())
    logger.addFilter(MainProcessFilter())

    return logger


def set_logger_file(filepath, logger):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_handler = logging.FileHandler(filepath, mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            if hasattr(f, "flush") and f != sys.stdout and f != sys.stderr:
                f.flush()

    def flush(self):
        for f in self.files:
            if hasattr(f, "flush"):
                f.flush()

    def isatty(self):
        return any(hasattr(f, "isatty") and f.isatty() for f in self.files)


def set_stdout_file(filepath):
    log_file = open(filepath, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)


def logger_file_setting(logger, filepath):
    set_logger_file(filepath, logger)
    set_stdout_file(filepath)


logger = load_logger(logger=logger, level=logging.INFO)
