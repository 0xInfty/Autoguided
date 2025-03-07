import logging
from colorlog import ColoredFormatter

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

def create_logger(log_level=logging.WARNING, filename=None):
    log = logging.getLogger('pythonConfig')
    if filename is not None:
        log.addHandler( logging.FileHandler(filename) )
        set_log_format(log)
    else:
        log.addHandler( logging.StreamHandler() )
        set_log_format(log, color=True)
    set_log_level(log_level, log)
    return log
