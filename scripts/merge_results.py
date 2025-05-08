
# !/usr/bin/env python3
"""
Script for merging results from different model runs, including partial results from temp directory.
"""

import os
import sys
import glob
import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.dataset import merge_model_results
from src.utils.environment import get_data_path
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up and return the argument parser.
    """
    parser = argparse.ArgumentParser(description='Merge results from different model runs')
    parser.add_argument('--model1', type=str, required=False,
                        help='Path to the first model\'s results CSV')
    parser.add_argument('--model2', type=str, required=False,
                        help='Path to the second model\'s results CSV')
    parser.add_argument('--model1-name', type=str, default='7B',
                        help='Name identifier for the first model')
    parser.add_argument('--model2-name', type=str, default='8B',
                        help='Name identifier for the second model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the merged results')
    parser.add_argument('--temp-dir', action='store_true',
                        help='Merge partial results from temp directory')
    return parser


def get_temp_files(model_name: str) -> List[str]:
    """
    Get all partial result files for a specific model from the temp directory.

    Args:
        model_name (str): Name of the model to search for

    Returns:
        List[str]: List of file paths
    """
    temp_dir = get_data_path('data_temp')
    pattern = f"*{model_name}*.csv"
    files = glob.glob(str(temp_dir / pattern))
    return sorted(files)


def merge_temp_files(files: List[str]) -> pd.DataFrame:
    """
    Merge multiple partial result files into a single DataFrame.

    Args:
        files (List[str]): List of file paths to merge

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    if not files:
        raise ValueError("No files found to merge")

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Successfully read: {file}")
        except Exception as e:
            logger.warning(f"Failed to read {file}: {str(e)}")

    return pd.concat(dfs, ignore_index=True)


def save_results(combined_df: pd.DataFrame,
                 comparative_df: pd.DataFrame,
                 output_dir: str,
                 model1_name: str,
                 model2_name: str) -> Tuple[str, str]:
    """
    Save the merged results to CSV files.

    Args:
        combined_df (pd.DataFrame): Combined results DataFrame
        comparative_df (pd.DataFrame): Comparative results DataFrame
        output_dir (str): Directory to save results
        model1_name (str): Name of first model
        model2_name (str): Name of second model

    Returns:
        Tuple[str, str]: Paths to the saved files
    """
    os.makedirs(output_dir, exist_ok=True)

    combined_output = os.path.join(output_dir,
                                   f"combined_model_results_{model1_name}_vs_{model2_name}.csv")
    comparative_output = os.path.join(output_dir,
                                      f"comparative_model_results_{model1_name}_vs_{model2_name}.csv")

    combined_df.to_csv(combined_output, index=False)
    comparative_df.to_csv(comparative_output, index=False)

    return combined_output, comparative_output


def process_temp_directory(model1_name: str,
                           model2_name: str,
                           output_dir: Optional[str] = None) -> None:
    """
    Process and merge partial results from temp directory.

    Args:
        model1_name (str): Name of first model
        model2_name (str): Name of second model
        output_dir (Optional[str]): Output directory path
    """
    try:
        # Get partial results files
        model1_files = get_temp_files(model1_name)
        model2_files = get_temp_files(model2_name)

        if not model1_files or not model2_files:
            raise ValueError("No matching files found for one or both models")

        # Merge partial results
        model1_df = merge_temp_files(model1_files)
        model2_df = merge_temp_files(model2_files)

        # Save temporary complete files
        temp_dir = get_data_path('data_temp')
        model1_temp = str(temp_dir / f"temp_complete_{model1_name}.csv")
        model2_temp = str(temp_dir / f"temp_complete_{model2_name}.csv")

        model1_df.to_csv(model1_temp, index=False)
        model2_df.to_csv(model2_temp, index=False)

        # Merge results using existing function
        combined_df, comparative_df = merge_model_results(
            model1_temp, model2_temp, model1_name, model2_name
        )

        # Save final results
        if not output_dir:
            output_dir = str(get_data_path('results'))

        combined_path, comparative_path = save_results(
            combined_df, comparative_df, output_dir, model1_name, model2_name
        )

        logger.info(f"Merged datasets created successfully:")
        logger.info(f"1. {combined_path} - All results stacked")
        logger.info(f"2. {comparative_path} - Results side-by-side for comparison")

        # Clean up temporary complete files
        os.remove(model1_temp)
        os.remove(model2_temp)

    except Exception as e:
        logger.error(f"Error processing temp directory: {str(e)}")
        raise


def main():
    """
    Main function to merge results from different model runs.
    """
    parser = setup_argparser()
    args = parser.parse_args()

    try:
        if args.temp_dir:
            process_temp_directory(args.model1_name, args.model2_name, args.output_dir)
        else:
            if not args.model1 or not args.model2:
                raise ValueError("Both --model1 and --model2 are required when not using --temp-dir")

            output_dir = args.output_dir or str(get_data_path('results'))

            combined_df, comparative_df = merge_model_results(
                args.model1, args.model2, args.model1_name, args.model2_name
            )

            combined_path, comparative_path = save_results(
                combined_df, comparative_df, output_dir, args.model1_name, args.model2_name
            )

            logger.info(f"Merged datasets created successfully:")
            logger.info(f"1. {combined_path} - All results stacked")
            logger.info(f"2. {comparative_path} - Results side-by-side for comparison")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
