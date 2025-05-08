"""
Dataset handling utilities for antisemitism detection.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.environment import get_data_path

logger = get_logger(__name__)


class TweetDataset:
    """
    Class for handling tweet datasets.
    """
    
    def __init__(self, file_path=None, df=None):
        """
        Initialize the TweetDataset.
        
        Args:
            file_path (str, optional): Path to the CSV file
            df (pd.DataFrame, optional): DataFrame containing tweet data
        """
        if df is not None:
            self.df = df
        elif file_path is not None:
            self.df = self._load_file(file_path)
        else:
            self.df = pd.DataFrame()
        
        # Track changes
        self.file_path = file_path
        self.modified = False
    
    def _load_file(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def save(self, file_path=None):
        """
        Save the dataset to a CSV file.
        
        Args:
            file_path (str, optional): Path to save the CSV file. If None, uses the original file path.
        
        Returns:
            bool: True if successful, False otherwise
        """
        save_path = file_path or self.file_path
        
        if save_path is None:
            logger.error("No file path specified for saving")
            return False
        
        try:
            self.df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(self.df)} rows to {save_path}")
            self.modified = False
            self.file_path = save_path
            return True
        except Exception as e:
            logger.error(f"Error saving data to {save_path}: {str(e)}")
            return False
    
    def filter(self, **kwargs):
        """
        Filter the dataset based on column values.
        
        Args:
            **kwargs: Column-value pairs for filtering
        
        Returns:
            TweetDataset: New dataset with filtered data
        """
        filtered_df = self.df.copy()
        
        for column, value in kwargs.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
        
        logger.info(f"Filtered dataset from {len(self.df)} to {len(filtered_df)} rows")
        return TweetDataset(df=filtered_df)
    
    def split_by_batch(self, batch_size):
        """
        Split the dataset into batches.
        
        Args:
            batch_size (int): Size of each batch
        
        Returns:
            list: List of TweetDataset instances
        """
        n_batches = (len(self.df) + batch_size - 1) // batch_size
        batches = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.df))
            batch_df = self.df.iloc[start_idx:end_idx].copy()
            batches.append(TweetDataset(df=batch_df))
        
        logger.info(f"Split dataset into {len(batches)} batches of size {batch_size}")
        return batches
    
    def merge(self, *datasets):
        """
        Merge this dataset with other datasets.
        
        Args:
            *datasets: TweetDataset instances to merge
        
        Returns:
            TweetDataset: New dataset with merged data
        """
        dfs = [self.df]
        
        for dataset in datasets:
            if isinstance(dataset, TweetDataset):
                dfs.append(dataset.df)
            elif isinstance(dataset, pd.DataFrame):
                dfs.append(dataset)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged {len(dfs)} datasets into one with {len(merged_df)} rows")
        
        return TweetDataset(df=merged_df)
    
    def to_binary(self, column, positive_values=None):
        """
        Convert a column to binary (0/1) values.
        
        Args:
            column (str): Column name to convert
            positive_values (list, optional): Values to consider as positive (1)
        
        Returns:
            TweetDataset: New dataset with the binary column
        """
        result_df = self.df.copy()
        
        if column not in result_df.columns:
            logger.error(f"Column '{column}' not found in dataset")
            return self
        
        # Default positive values are 'Yes', 'yes', 'Y', 'y', '1', 1, True
        if positive_values is None:
            positive_values = ['Yes', 'yes', 'Y', 'y', '1', 1, True]
        
        # Create binary column
        binary_column = f"{column}_Binary"
        result_df[binary_column] = result_df[column].isin(positive_values).astype(int)
        
        logger.info(f"Converted column '{column}' to binary column '{binary_column}'")
        
        return TweetDataset(df=result_df)
    
    def add_column(self, column_name, values):
        """
        Add a new column to the dataset.
        
        Args:
            column_name (str): Name of the new column
            values (list or pd.Series): Values for the new column
        
        Returns:
            TweetDataset: Self with the new column
        """
        if len(values) != len(self.df):
            logger.error(f"Length mismatch: dataset has {len(self.df)} rows, but values has {len(values)} elements")
            return self
        
        self.df[column_name] = values
        self.modified = True
        logger.info(f"Added column '{column_name}' to dataset")
        
        return self
    
    def get_stats(self):
        """
        Get basic statistics about the dataset.
        
        Returns:
            dict: Dictionary with dataset statistics
        """
        stats = {
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict()
        }
        
        # Try to get statistics for binary columns
        binary_columns = [col for col in self.df.columns if col.endswith('_Binary')]
        if binary_columns:
            stats["binary_counts"] = {
                col: self.df[col].value_counts().to_dict() for col in binary_columns
            }
        
        return stats
    
    @property
    def shape(self):
        """
        Get the shape of the dataset.
        
        Returns:
            tuple: (num_rows, num_columns)
        """
        return self.df.shape
    
    def __len__(self):
        """
        Get the number of rows in the dataset.
        
        Returns:
            int: Number of rows
        """
        return len(self.df)


def merge_model_results(model1_file, model2_file, model1_name='7B', model2_name='8B'):
    """
    Merge results from two different model runs.
    
    Args:
        model1_file (str): Path to the first model's results CSV
        model2_file (str): Path to the second model's results CSV
        model1_name (str, optional): Name identifier for the first model
        model2_name (str, optional): Name identifier for the second model
    
    Returns:
        tuple: (combined_df, comparative_df) with different formats of the merged data
    """
    logger.info(f"Merging results from {model1_file} and {model2_file}")
    
    # Load datasets
    df_model1 = pd.read_csv(model1_file)
    df_model2 = pd.read_csv(model2_file)
    
    # Add model identifier column
    df_model1['Model'] = model1_name
    df_model2['Model'] = model2_name
    
    # Convert Yes/No decisions to 1/0
    for df in [df_model1, df_model2]:
        df['IHRA_Binary'] = (df['IHRA_Decision'] == 'Yes').astype(int)
        df['JDA_Binary'] = (df['JDA_Decision'] == 'Yes').astype(int)
    
    # Combine the datasets
    combined_df = pd.concat([df_model1, df_model2], ignore_index=True)
    
    # Create a version with one row per tweet (useful for comparing models)
    pivot_df = df_model1[['TweetID', 'Username', 'CreateDate', 'Biased', 'Keyword', 'Text', 
                          'IHRA_Binary', 'JDA_Binary']].copy()
    
    # Rename columns for clarity
    pivot_df.rename(columns={
        'IHRA_Binary': f'IHRA_Binary_{model1_name}',
        'JDA_Binary': f'JDA_Binary_{model1_name}'
    }, inplace=True)
    
    # Add columns for second model results
    model2_mapping = dict(zip(df_model2['TweetID'], zip(df_model2['IHRA_Binary'], df_model2['JDA_Binary'])))
    
    pivot_df[f'IHRA_Binary_{model2_name}'] = pivot_df['TweetID'].map(lambda x: model2_mapping.get(x, (np.nan, np.nan))[0])
    pivot_df[f'JDA_Binary_{model2_name}'] = pivot_df['TweetID'].map(lambda x: model2_mapping.get(x, (np.nan, np.nan))[1])
    
    # Calculate text length for complexity analysis
    pivot_df['text_length'] = pivot_df['Text'].str.len()
    
    logger.info(f"Created combined dataset with {len(combined_df)} rows and comparative dataset with {len(pivot_df)} rows")
    
    return combined_df, pivot_df
