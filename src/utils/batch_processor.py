"""
Utilities for batch processing.
"""

import os
import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Class for processing data in batches with parallel execution.
    """
    
    def __init__(self, max_workers=4, show_progress=True):
        """
        Initialize the batch processor.
        
        Args:
            max_workers (int): Maximum number of parallel workers
            show_progress (bool): Whether to show progress bars
        """
        self.max_workers = max_workers
        self.show_progress = show_progress
        logger.info(f"Initialized BatchProcessor with {max_workers} workers")
    
    def process_batches(self, batches, process_func, *args, **kwargs):
        """
        Process batches in parallel.
        
        Args:
            batches (list): List of batch items
            process_func (callable): Function to process each batch
            *args, **kwargs: Additional arguments for process_func
        
        Returns:
            list: Combined results from all batches
        """
        all_results = []
        total_batches = len(batches)
        
        logger.info(f"Processing {total_batches} batches with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_func, batch, batch_idx, *args, **kwargs): batch_idx 
                      for batch_idx, batch in enumerate(batches)}
            
            # Process as they complete
            if self.show_progress:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        batch_idx = futures[future]
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            else:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        batch_idx = futures[future]
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
        
        # Sort results by batch index if results are dictionaries with 'batch_idx' key
        if all_results and isinstance(all_results[0], dict) and 'batch_idx' in all_results[0]:
            all_results.sort(key=lambda x: x['batch_idx'])
        
        logger.info(f"Completed processing {len(all_results)}/{total_batches} batches")
        
        return all_results
    
    def process_with_retries(self, items, process_func, max_retries=3, retry_delay=2, *args, **kwargs):
        """
        Process items with retries on failure.
        
        Args:
            items (list): Items to process
            process_func (callable): Function to process each item
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries (exponential backoff)
            *args, **kwargs: Additional arguments for process_func
        
        Returns:
            list: Results from processing
        """
        results = []
        failed_items = []
        
        # Initial processing
        logger.info(f"Processing {len(items)} items with retries (max: {max_retries})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_func, item, *args, **kwargs): i for i, item in enumerate(items)}
            
            # Process as they complete
            if self.show_progress:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Initial processing"):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                    except Exception as e:
                        logger.warning(f"Item {idx} failed: {str(e)}")
                        failed_items.append((idx, items[idx]))
            else:
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                    except Exception as e:
                        logger.warning(f"Item {idx} failed: {str(e)}")
                        failed_items.append((idx, items[idx]))
        
        # Retry failed items
        remaining_items = failed_items
        for retry in range(max_retries):
            if not remaining_items:
                break
            
            logger.info(f"Retry {retry+1}/{max_retries}: Processing {len(remaining_items)} failed items")
            time.sleep(retry_delay * (2 ** retry))  # Exponential backoff
            
            retry_results = []
            next_remaining = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit retry tasks
                futures = {executor.submit(process_func, item, *args, **kwargs): idx 
                          for idx, item in remaining_items}
                
                # Process as they complete
                if self.show_progress:
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                                      desc=f"Retry {retry+1}"):
                        idx = futures[future]
                        try:
                            result = future.result()
                            retry_results.append((idx, result))
                        except Exception as e:
                            logger.warning(f"Item {idx} failed again: {str(e)}")
                            next_remaining.append((idx, dict(remaining_items)[idx]))
                else:
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                            retry_results.append((idx, result))
                        except Exception as e:
                            logger.warning(f"Item {idx} failed again: {str(e)}")
                            next_remaining.append((idx, dict(remaining_items)[idx]))
            
            # Add successful retries to results
            results.extend(retry_results)
            remaining_items = next_remaining
        
        # Log final status
        if remaining_items:
            logger.warning(f"{len(remaining_items)} items failed after all retries")
        
        # Sort by original index and extract just the results
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        return final_results


def combine_batch_results(result_files, output_file):
    """
    Combine multiple batch result files into a single file.
    
    Args:
        result_files (list): List of batch result file paths
        output_file (str): Path to save the combined results
    
    Returns:
        pd.DataFrame: Combined results
    """
    dfs = []
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        logger.error("No valid batch result files found")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save to output file
    try:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined {len(combined_df)} rows from {len(dfs)} batch files to {output_file}")
    except Exception as e:
        logger.error(f"Error saving combined results to {output_file}: {str(e)}")
    
    return combined_df
