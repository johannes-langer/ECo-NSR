import logging
import os
import sys
from logging.config import dictConfig
from pathlib import Path

import yaml

# find top directory
CWD = [p for p in Path(__file__).parents if p.stem == "eco-nsr"][0]
sys.path.append(CWD.as_posix())


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    _format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + _format + reset,
        logging.INFO: grey + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    _format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    def format(self, record):
        formatter = logging.Formatter(self._format)
        return formatter.format(record)


def init() -> None:
    """
    Read logging.yml
    """
    with open(os.path.join(CWD, "src", "logging.yml"), "r") as f:
        config = yaml.safe_load(f.read())
    dictConfig(config)


def log_only_console(logger: logging.Logger) -> logging.Logger:
    """
    Remove all handlers from the logger and add a new stream handler with a custom formatter.

    Parameters
    ----------
    logger : logging.Logger
        The logger to modify.

    Returns
    -------
    logging.Logger
        The modified logger.
    """
    # ~ remove all handlers ~
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # ~ add a new stream handler ~
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)

    return logger


def log_only_file(logger: logging.Logger, file_path: str) -> logging.Logger:
    """
    Remove all handlers from the logger and add a new file handler with a custom formatter.

    Parameters
    ----------
    logger : logging.Logger
        The logger to modify.
    file_path : str
        The path to the file to log to.

    Returns
    -------
    logging.Logger
        The modified logger.
    """
    # ~ remove all handlers ~
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # ~ add a new file handler ~
    file_handler = logging.FileHandler(filename=file_path)
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)

    return logger
