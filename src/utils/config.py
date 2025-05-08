"""
Configuration handling utilities.
"""

import yaml
from pathlib import Path
from .environment import detect_environment, get_config_path


def load_config(config_name):
    """
    Load configuration with environment-specific overrides.
    
    Args:
        config_name (str): Name of the configuration file (without extension)
    
    Returns:
        dict: Configuration dictionary
    """
    env = detect_environment()
    config_path = get_config_path()
    
    # Base configuration path
    base_config_path = config_path / f'{config_name}.yaml'
    
    # Environment-specific configuration path
    env_config_path = config_path / 'environments' / env / f'{config_name}.yaml'
    
    # Check if base config exists
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment-specific config if it exists
    if env_config_path.exists():
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            # Deep update the configuration
            deep_update(config, env_config)
    
    return config


def deep_update(base_dict, update_dict):
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict (dict): Base dictionary to update
        update_dict (dict): Dictionary with values to update
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def save_config(config, config_name, environment=None):
    """
    Save a configuration to a file.
    
    Args:
        config (dict): Configuration dictionary
        config_name (str): Name of the configuration file (without extension)
        environment (str, optional): Environment to save to. If None, saves to base config.
    """
    config_path = get_config_path()
    
    if environment:
        # Save to environment-specific config
        config_file_path = config_path / 'environments' / environment / f'{config_name}.yaml'
        # Ensure the directory exists
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Save to base config
        config_file_path = config_path / f'{config_name}.yaml'
    
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
