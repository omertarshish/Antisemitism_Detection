#!/usr/bin/env python3
"""
Script for merging multiple batch CSV files from the temp_data/batches directory.
This utility helps combine 64 separate batch files into a single comprehensive result file.
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
from typing import List

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.environment import get_data_path
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def get_batch_files(directory: str = "batches") -> List[str]:
    """
    Get all batch CSV files from the specified directory within temp_data.

    Args:
        directory (str): Subdirectory name within temp_data containing batch files

    Returns:
        List[str]: List of file paths to batch CSV files, sorted numerically
    """
    temp_dir = get_data_path('temp_data')
    pattern = f"{directory}/batch_*.csv"
    files = glob.glob(str(temp_dir / pattern))

    # Sort files numerically by batch number
    def get_batch_number(file_path):
        filename = os.path.basename(file_path)
        return int(filename.replace("batch_", "").replace(".csv", ""))

    return sorted(files, key=get_batch_number)


def merge_batch_files(files: List[str]) -> pd.DataFrame:
    """
    Merge multiple batch files into a single DataFrame.

    Args:
        files (List[str]): List of file paths to merge

    Returns:
        pd.DataFrame: Merged DataFrame containing all batch data
    """
    if not files:
        raise ValueError("No batch files found to merge")

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            filename = os.path.basename(file)
            logger.info(f"Successfully read: {filename}")
        except Exception as e:
            logger.warning(f"Failed to read {file}: {str(e)}")

    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {len(dfs)} batch files into a single DataFrame with {len(merged_df)} rows")

    return merged_df


def save_merged_results(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Save the merged results to a CSV file.

    Args:
        df (pd.DataFrame): Merged DataFrame to save
        output_path (str, optional): Custom output path. If None, uses default location

    Returns:
        str: Path to the saved file
    """
    if output_path is None:
        results_dir = get_data_path('results')
        output_path = str(results_dir / "merged_batches.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Merged results saved to: {output_path}")

    return output_path


def main():
    """
    Main function to merge batch files from temp_data/batches directory.
    """
    try:
        # Get all batch files
        batch_files = get_batch_files()

        if not batch_files:
            logger.error("No batch files found in temp_data/batches directory")
            sys.exit(1)

        logger.info(f"Found {len(batch_files)} batch files to merge")

        # Merge batch files
        merged_df = merge_batch_files(batch_files)

        # Save merged results
        output_path = save_merged_results(merged_df)

        logger.info("Batch merging completed successfully")
        logger.info(f"Total records in merged file: {len(merged_df)}")
        logger.info(f"Output file: {output_path}")

    except Exception as e:
        logger.error(f"Error in batch merging process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()