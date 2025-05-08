#!/usr/bin/env python3
"""
Script for running antisemitism detection inference using Ollama.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.ollama_client import OllamaClient
from src.models.definitions import create_definition_prompt, get_available_definitions
from src.data.preprocessing import load_tweet_data, prepare_batches, save_results, save_batch_results
from src.utils.config import load_config
from src.utils.environment import get_data_path, detect_environment
from src.utils.logger import get_logger
from src.utils.batch_processor import BatchProcessor

# Initialize logger
logger = get_logger(__name__)


def process_tweet(args):
    """
    Process a single tweet and return the results.
    """
    tweet_row, definition_names, ollama_client = args
    
    # Extract tweet information
    tweet_id = tweet_row['ID']
    tweet_text = tweet_row['Text']
    
    result = {
        'TweetID': tweet_id,
        'Username': tweet_row['Username'],
        'CreateDate': tweet_row['CreateDate'],
        'Biased': tweet_row['Biased'],
        'Keyword': tweet_row['Keyword'],
        'Text': tweet_text
    }
    
    # Process with each definition
    for definition_name in definition_names:
        try:
            # Create the prompt for this definition
            prompt = create_definition_prompt(definition_name, tweet_text)
            
            # Get the model's response
            response = ollama_client.generate(prompt)
            
            # Extract decision and explanation
            decision, explanation = ollama_client.extract_decision_explanation(response)
            
            # Add results to the output
            result[f'{definition_name}_Decision'] = decision
            result[f'{definition_name}_Explanation'] = explanation
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet_id} with {definition_name}: {str(e)}")
            result[f'{definition_name}_Decision'] = "Error"
            result[f'{definition_name}_Explanation'] = str(e)
    
    return result


def process_batch(batch_df, batch_idx, definition_names, ollama_client, output_dir=None):
    """
    Process a batch of tweets.
    """
    batch_results = []
    errors = []
    
    # Process each tweet in the batch
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), 
                         desc=f"Batch {batch_idx+1}", leave=False):
        try:
            result = process_tweet((row, definition_names, ollama_client))
            batch_results.append(result)
        except Exception as e:
            logger.error(f"Error processing row {idx} in batch {batch_idx+1}: {str(e)}")
            errors.append((idx, str(e)))
    
    # Save batch results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        batch_file = os.path.join(output_dir, f"batch_{batch_idx+1}.csv")
        pd.DataFrame(batch_results).to_csv(batch_file, index=False)
        logger.info(f"Saved batch {batch_idx+1} results to {batch_file}")
    
    return {
        'batch_idx': batch_idx,
        'results': batch_results,
        'errors': errors
    }


def main():
    """
    Main function to run antisemitism detection inference.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run antisemitism detection inference using Ollama')
    parser.add_argument('--input', type=str, required=False,
                        help='Input CSV file with tweets (if not provided, will use the one from config)')
    parser.add_argument('--output', type=str, required=False,
                        help='Output CSV file for results (if not provided, will use the one from config)')
    parser.add_argument('--config', type=str, default='ollama_config',
                        help='Configuration file name (without extension)')
    parser.add_argument('--ip-port', type=str, required=False,
                        help='IP:PORT of the Ollama instance (overrides config)')
    parser.add_argument('--model', type=str, required=False,
                        help='Ollama model name to use (overrides config)')
    parser.add_argument('--definitions', type=str, default='IHRA,JDA',
                        help='Comma-separated list of definition names to use (default: IHRA,JDA)')
    parser.add_argument('--batch-size', type=int, required=False,
                        help='Batch size for processing (overrides config)')
    parser.add_argument('--max-workers', type=int, required=False,
                        help='Maximum number of parallel workers (overrides config)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tweets to process (for testing)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get environment info
    env = detect_environment()
    logger.info(f"Detected environment: {env}")
    
    # Parse definition names
    definition_names = args.definitions.split(',')
    available_definitions = get_available_definitions()
    
    # Validate definition names
    for definition_name in definition_names:
        if definition_name not in available_definitions:
            logger.error(f"Definition '{definition_name}' not found. Available definitions: {available_definitions}")
            return
    
    logger.info(f"Using definitions: {definition_names}")
    
    # Set input and output files
    input_file = args.input or config.get('input_file')
    if not input_file:
        input_file = str(get_data_path('raw') / 'tweets.csv')
    
    output_file = args.output or config.get('output_file')
    if not output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = args.model or config.get('model', 'default')
        model_short = model_name.split(':')[0]
        output_file = str(get_data_path('results') / f"analysis_{timestamp}_{model_short}.csv")
    
    # Set batch size and max workers
    batch_size = args.batch_size or config.get('batch_size', 100)
    max_workers = args.max_workers or config.get('max_workers', 4)
    
    # Set up temp directory for batch results
    temp_dir = os.environ.get('ANTISEMITISM_TEMP_DIR')
    if not temp_dir:
        temp_dir = config.get('temp_dir')
        if not temp_dir:
            temp_dir = str(get_data_path('processed') / 'batches')
    print(f"Using temp directory: {temp_dir}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize the Ollama client
    ip_port = args.ip_port or config.get('ip_port')
    model = args.model or config.get('model')
    ollama_client = OllamaClient(ip_port=ip_port, model=model)
    
    # Pull the model
    if not ollama_client.pull_model():
        logger.error("Failed to pull the model. Please check the Ollama server and try again.")
        return
    
    # Load tweet data
    try:
        df = load_tweet_data(input_file)
    except Exception as e:
        logger.error(f"Error loading tweet data: {str(e)}")
        return
    
    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to processing {args.limit} rows (for testing)")
    
    # Prepare batches
    batches = prepare_batches(df, batch_size)
    
    # Set up batch processor
    batch_processor = BatchProcessor(max_workers=max_workers)
    
    # Process batches
    logger.info(f"Processing {len(batches)} batches with {max_workers} workers")
    
    start_time = time.time()
    batch_results = batch_processor.process_batches(batches, process_batch, 
                                                   definition_names, ollama_client, temp_dir)
    end_time = time.time()
    
    # Combine results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result['results'])
    
    # Save final results
    save_results(all_results, output_file)
    
    # Print statistics
    total_tweets = len(all_results)
    total_time = end_time - start_time
    tweets_per_second = total_tweets / total_time if total_time > 0 else 0
    
    logger.info(f"Processed {total_tweets} tweets in {total_time:.2f} seconds ({tweets_per_second:.2f} tweets/sec)")
    
    # Print definition-specific statistics
    for definition_name in definition_names:
        yes_count = sum(1 for r in all_results if r.get(f'{definition_name}_Decision') == 'Yes')
        no_count = sum(1 for r in all_results if r.get(f'{definition_name}_Decision') == 'No')
        error_count = sum(1 for r in all_results if r.get(f'{definition_name}_Decision') == 'Error')
        
        logger.info(f"{definition_name}: {yes_count} antisemitic, {no_count} not antisemitic, {error_count} errors")
    
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
