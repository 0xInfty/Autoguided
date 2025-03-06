import logging
from colorlog import ColoredFormatter

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def set_log_level(log_level, logger=None):
    logging.root.setLevel(log_level)
    if logger is not None:
        logger.setLevel(log_level)
        for stream in logger.handlers:
            stream.setLevel(log_level)

def set_log_format(logger):
    log_format = "%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(message)s"
    formatter = ColoredFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    if logger is not None:
        for stream in logger.handlers:
            stream.setFormatter(formatter)

def create_logger(log_level=logging.WARNING):
    log = logging.getLogger('pythonConfig')
    log.addHandler( logging.StreamHandler() )
    set_log_format(log)
    set_log_level(log_level, log)
    return log
