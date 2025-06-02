import logging
from colorlog import ColoredFormatter
import functools

import ours.utils as utils

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

LEVELS = [logging.CRITICAL, logging.ERROR,
          logging.WARNING, logging.INFO, logging.DEBUG]

def get_log_level(verbosity):
    return LEVELS[-3:][verbosity]

def set_log_level(log_level, logger=None):
    logging.root.setLevel(log_level)
    if logger is not None:
        logger.setLevel(log_level)
        for stream in logger.handlers:
            stream.setLevel(log_level)

def set_log_file(logger, filename):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler): 
            logger.removeHandler(handler)
    logger.addHandler( logging.FileHandler(filename) )

def set_log_format(logger, color=False):
    if color:
        formatter = ColoredFormatter(
            "%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", 
            style="%", datefmt="%Y-%m-%d %H:%M" )
    if logger is not None:
        for stream in logger.handlers:
            stream.setFormatter(formatter)

def configure_logger(log, log_level=logging.WARNING, filename=None):
    if filename is not None:
        log.addHandler( logging.FileHandler(filename) )
        set_log_format(log)
    else:
        log.addHandler( logging.StreamHandler() )
        set_log_format(log, color=True)
    set_log_level(log_level, log)

def create_logger(name, log_level=logging.WARNING, filename=None):
    log = logging.getLogger(name)
    configure_logger(log, log_level, filename)
    return log

def errors(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        log = logging.getLogger("errors")
        try:
            return function(*args, **kwargs)
        except Exception as e:
            if any([isinstance(handler, logging.FileHandler) for handler in log.handlers]):
                log.critical("Interrupted execution\n%s", e, exc_info=True)
            else:
                log.critical("Interrupted execution")
            raise
    return wrapper

get_stats_log = lambda name, array : (f"{name} = [ %s | %s | %s ] %s", *utils.get_stats(array))