import logging
import os

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Set up a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # prevent adding multiple handlers to the logger

    if logger.handlers:
        return logger
    
    # Console handler for logging debug and higher level messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File Handler for logging errors

    if log_file is None:
        os.makedirs('logs', exist_ok=True)
        log_file = f'logs/{name}_errors.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger