import logging
import os


def setup_logger(log_dir: str, log_file: str = "run.log") -> logging.Logger:

    logger = logging.getLogger("ndi_logger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{log_dir}/{log_file}", mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="{asctime} - [{levelname}]: {message}",
        style="{",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
