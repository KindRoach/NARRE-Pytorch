import logging

from utils.path_helper import ROOT_DIR

# create formatter, refer below link if you want add more information to log.
# https://docs.python.org/3/library/logging.html#logrecord-attributes
FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')


def get_logger(name: str):
    logger = logging.getLogger(name)

    #  if logger is newly created
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(FORMATTER)

        # add the handlers to the logger
        logger.addHandler(ch)

    return logger


def add_log_file(logger, path: str):
    # create file handler with INFO level
    fh = logging.FileHandler(ROOT_DIR.joinpath(path))
    fh.setLevel(logging.INFO)
    fh.setFormatter(FORMATTER)
    logger.addHandler(fh)


def remove_log_file(logger):
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]


logger = get_logger("DeepCoNN")

if __name__ == "__main__":
    logger.info("Info message.")
    add_log_file(logger, "log/test.log")
    logger.debug("Debug message.")
    logger.warning("Warning message.")
    remove_log_file(logger)
    logger.critical("Critical message.")
