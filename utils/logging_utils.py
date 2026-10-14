import os
import logging

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def rank_zero_filter(record):
    """Only let rank 0 log. The global rank comes from whichever launcher
    is in use -- SLURM (SLURM_PROCID), torchrun (RANK) or MPI (PMI_RANK) --
    and defaults to 0 for single-process runs. (Previously this read
    SLURM_PROCID unconditionally and crashed under any other launcher.)"""
    for var in ("SLURM_PROCID", "RANK", "PMI_RANK"):
        if var in os.environ:
            return int(os.environ[var]) == 0
    return True


# backwards-compatible alias (older name referred to the SLURM variable)
slurm_filter = rank_zero_filter


def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)
    root_logger = logging.getLogger()
    root_logger.addFilter(rank_zero_filter)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import torch
    import subprocess

    logging.info("--------------- Versions ---------------")
    logging.info(
        "git branch: " + str(subprocess.check_output(["git", "branch"]).strip())
    )
    logging.info(
        "git hash: "
        + str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip())
    )
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):  # pragma: no cover
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):  # pragma: no cover
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):  # pragma: no cover
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)
