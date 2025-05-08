"""
Preprocessing utilities for tweet data.
"""

import pandas as pd
import re
import html
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_tweet_data(file_path):
    """
    Load tweet data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: DataFrame containing the tweet data
    """
    logger.info(f"Loading tweet data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} tweets")
        return df
    except Exception as e:
        logger.error(f"Error loading tweet data: {str(e)}")
        raise


def clean_tweet(text):
    """
    Clean and normalize tweet text.
    
    Args:
        text (str): Raw tweet text
    
    Returns:
        str: Cleaned tweet text
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbols but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_batches(df, batch_size):
    """
    Prepare batches of tweets for processing.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweet data
        batch_size (int): Size of each batch
    
    Returns:
        list: List of batch DataFrames
    """
    total_rows = len(df)
    batches = []
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end].copy()
        batches.append(batch_df)
    
    logger.info(f"Prepared {len(batches)} batches of size {batch_size}")
    return batches


def save_results(results, output_file):
    """
    Save processing results to a CSV file.
    
    Args:
        results (list): List of result dictionaries
        output_file (str): Path to save the output CSV file
    """
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def save_batch_results(batch_results, batch_id, output_dir):
    """
    Save batch results to a CSV file.
    
    Args:
        batch_results (list): List of result dictionaries for the batch
        batch_id (int): Batch identifier
        output_dir (str): Directory to save the batch results
    
    Returns:
        str: Path to the saved batch results file
    """
    try:
        batch_df = pd.DataFrame(batch_results)
        batch_file = f"{output_dir}/batch_{batch_id}.csv"
        batch_df.to_csv(batch_file, index=False)
        logger.info(f"Batch {batch_id} results saved to {batch_file}")
        return batch_file
    except Exception as e:
        logger.error(f"Error saving batch results: {str(e)}")
        raise
