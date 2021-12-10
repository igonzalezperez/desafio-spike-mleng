"""
Define function which configures logger both to be written on file and printed in stderr.
"""
# %% Imports
import sys

from loguru import logger


# %% Functions and Classes
def logger_config(filepath: str = 'logs/.log', level: str = 'WARNING', **kwargs) -> None:
    """
    Define loggers for file and stderr.
    :param filepath: Where to save log output.
    :param level: Log level.
    :param kwargs: Any keyword arguments valid in logger.add() method.
    """
    fmt = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> |' \
          '<level>{level: <7}</level>| ' \
          '<level>{message}</level>'
    # Remove default logger
    logger.remove()
    # Add custom loggers for stderr and file
    logger.add(sys.stderr, level=level, format=fmt, **kwargs)
    logger.add(filepath, level=level, format=fmt, rotation='1GB', **kwargs)
    return


# %% Main
if __name__ == '__main__':
    log_level = 'DEBUG'
    logger_config(level=log_level)
    logger.debug('bug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')
