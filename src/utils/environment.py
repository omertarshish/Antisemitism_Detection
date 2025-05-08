"""
Environment detection and path handling utilities.
"""

import os
import sys
from pathlib import Path


def detect_environment():
    """
    Detect the execution environment.
    
    Returns:
        str: 'slurm', 'colab', or 'local'
    """
    # Check for SLURM environment
    if os.getenv('SLURM_JOB_ID'):
        return 'slurm'
    
    # Check for Google Colab
    try:
        import google.colab
        return 'colab'
    except ImportError:
        # If not in Colab or SLURM, assume local
        return 'local'


def get_project_root():
    """
    Get the root directory of the project that works across environments.
    
    Returns:
        Path: The project root path
    """
    # Start with the current file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Navigate up to the project root (3 levels up from utils file)
    # utils -> src -> project_root
    project_root = current_dir.parent.parent
    
    # Verify this is the project root by checking for key directories
    if not (project_root / 'src').exists():
        # If not found, try to find it in the current working directory
        cwd = Path.cwd()
        if (cwd / 'src').exists():
            return cwd
        
        # If still not found, return the best guess
        print("Warning: Unable to determine project root with certainty.")
    
    return project_root


def get_data_path(data_type='raw'):
    """
    Get the appropriate data path for the current environment.
    
    Args:
        data_type (str): 'raw', 'processed', or 'results'
    
    Returns:
        Path: Path to the requested data directory
    """
    root = get_project_root()
    data_path = root / 'data' / data_type
    
    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    return data_path


def get_config_path():
    """
    Get the path to configuration files.
    
    Returns:
        Path: Path to the config directory
    """
    root = get_project_root()
    return root / 'config'
