import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

def configure_logging(level=logging.INFO):
    # Make sure the "logs" directory exists.
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    log_filename = os.path.join(log_dir, f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Set up the log handler
    file_handler = RotatingFileHandler(log_filename, maxBytes=10_000_000, backupCount=5)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Configure the logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )