"""
Utilities for handling antisemitism definitions.
"""

from pathlib import Path
import os
from ..utils.environment import get_config_path
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_definition(definition_name):
    """
    Load antisemitism definition text.
    
    Args:
        definition_name (str): Name of the definition to load (e.g., 'IHRA', 'JDA')
    
    Returns:
        str: The definition text
    """
    definition_path = get_config_path() / 'definitions' / f'{definition_name}.txt'
    
    if not definition_path.exists():
        raise FileNotFoundError(f"Definition file not found: {definition_path}")
    
    with open(definition_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_available_definitions():
    """
    List available antisemitism definitions.
    
    Returns:
        list: List of available definition names (without extension)
    """
    definition_dir = get_config_path() / 'definitions'
    
    # Create the directory if it doesn't exist
    os.makedirs(definition_dir, exist_ok=True)
    
    return [p.stem for p in definition_dir.glob('*.txt')]


def create_definition_prompt(definition_name, tweet_text):
    """
    Create a prompt for the model using the specified definition template.
    
    Args:
        definition_name (str): Name of the definition to use (e.g., 'IHRA', 'JDA')
        tweet_text (str): The tweet text to analyze
    
    Returns:
        str: The formatted prompt
    """
    # Load the definition template
    definition_template = load_definition(definition_name)
    
    # Format the template with the tweet text
    prompt = definition_template.format(text=tweet_text)
    
    return prompt


def save_definition(definition_name, definition_text):
    """
    Save a definition to a file.
    
    Args:
        definition_name (str): Name of the definition
        definition_text (str): The definition text
    """
    definition_dir = get_config_path() / 'definitions'
    os.makedirs(definition_dir, exist_ok=True)
    
    definition_path = definition_dir / f'{definition_name}.txt'
    
    with open(definition_path, 'w', encoding='utf-8') as f:
        f.write(definition_text)
    
    logger.info(f"Definition '{definition_name}' saved to {definition_path}")
