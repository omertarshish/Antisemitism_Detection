"""
Logging configuration for the project.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from .environment import get_project_root


def setup_logger(name, log_level=logging.INFO, log_to_file=True):
    """
    Set up and configure a logger.
    
    Args:
        name (str): Name of the logger
        log_level (int): Logging level
        log_to_file (bool): Whether to log to file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Set up file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = get_project_root() / 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
default_logger = setup_logger('antisemitism_detection')


def get_logger(name=None):
    """
    Get a logger instance.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the default logger.
    
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        return default_logger
    
    # Check if logger exists, create if it doesn't
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger
