import logging
import os
import sys

def setup_logging(level=logging.INFO, log_file="data/job_finder.log"):
    """
    Setup a simple, clean logging configuration.
    Logs to both console and a file.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Define a clear, readable format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Allow file to capture debug
    
    # Suppress noisy library logs unless in DEBUG mode
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("selenium").setLevel(logging.WARNING)

    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger

# Helper to get a module-specific logger
def get_logger(name):
    return logging.getLogger(name)
