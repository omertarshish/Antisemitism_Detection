"""
Error analysis utilities for antisemitism detection.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from ..utils.logger import get_logger

logger = get_logger(__name__)


def identify_false_positives(df, ground_truth_col, prediction_col):
    """
    Identify false positive cases.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        ground_truth_col (str): Column name for ground truth
        prediction_col (str): Column name for predictions
    
    Returns:
        pd.DataFrame: DataFrame with only false positive cases
    """
    # Ensure binary values
    df_binary = df.copy()
    
    # Identify false positives (ground truth = 0, prediction = 1)
    false_positives = df_binary[(df_binary[ground_truth_col] == 0) & 
                               (df_binary[prediction_col] == 1)]
    
    logger.info(f"Identified {len(false_positives)} false positives out of {len(df)} samples")
    
    return false_positives


def identify_false_negatives(df, ground_truth_col, prediction_col):
    """
    Identify false negative cases.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        ground_truth_col (str): Column name for ground truth
        prediction_col (str): Column name for predictions
    
    Returns:
        pd.DataFrame: DataFrame with only false negative cases
    """
    # Ensure binary values
    df_binary = df.copy()
    
    # Identify false negatives (ground truth = 1, prediction = 0)
    false_negatives = df_binary[(df_binary[ground_truth_col] == 1) & 
                               (df_binary[prediction_col] == 0)]
    
    logger.info(f"Identified {len(false_negatives)} false negatives out of {len(df)} samples")
    
    return false_negatives


def analyze_by_length(df, prediction_col, ground_truth_col='Biased', bins=None, labels=None):
    """
    Analyze error rates by text length.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        prediction_col (str): Column name for predictions
        ground_truth_col (str): Column name for ground truth
        bins (list, optional): Bins for text length
        labels (list, optional): Labels for bins
    
    Returns:
        pd.DataFrame: DataFrame with error rates by length bin
    """
    # Create a copy of the dataframe
    analysis_df = df.copy()
    
    # Ensure text_length column exists
    if 'text_length' not in analysis_df.columns:
        analysis_df['text_length'] = analysis_df['Text'].str.len()
    
    # Default bins if not provided
    if bins is None:
        bins = [0, 80, 120, 160, 200, 240, 280, 320, 400, 600, 1000]
    
    # Default labels if not provided
    if labels is None:
        labels = ['0-80', '81-120', '121-160', '161-200', '201-240',
                  '241-280', '281-320', '321-400', '401-600', '601+']
    
    # Create length bins
    analysis_df['length_bin'] = pd.cut(analysis_df['text_length'], bins=bins, labels=labels)
    
    # Calculate error metrics by length bin
    length_stats = []
    
    for length_bin in labels:
        bin_df = analysis_df[analysis_df['length_bin'] == length_bin]
        if len(bin_df) == 0:
            continue
        
        # Filter non-antisemitic tweets (ground truth = 0)
        non_antisemitic = bin_df[bin_df[ground_truth_col] == 0]
        if len(non_antisemitic) > 0:
            # Count false positives
            fp = non_antisemitic[non_antisemitic[prediction_col] == 1]
            fp_rate = len(fp) / len(non_antisemitic)
        else:
            fp_rate = 0
        
        # Filter antisemitic tweets (ground truth = 1)
        antisemitic = bin_df[bin_df[ground_truth_col] == 1]
        if len(antisemitic) > 0:
            # Count false negatives
            fn = antisemitic[antisemitic[prediction_col] == 0]
            fn_rate = len(fn) / len(antisemitic)
        else:
            fn_rate = 0
        
        # Overall error rate
        if len(bin_df) > 0:
            error_count = len(fp) + len(fn) if 'fp' in locals() and 'fn' in locals() else 0
            error_rate = error_count / len(bin_df)
        else:
            error_rate = 0
        
        length_stats.append({
            'Length_Bin': length_bin,
            'Tweet_Count': len(bin_df),
            'Non_Antisemitic_Count': len(non_antisemitic) if 'non_antisemitic' in locals() else 0,
            'Antisemitic_Count': len(antisemitic) if 'antisemitic' in locals() else 0,
            'False_Positive_Count': len(fp) if 'fp' in locals() else 0,
            'False_Negative_Count': len(fn) if 'fn' in locals() else 0,
            'False_Positive_Rate': fp_rate,
            'False_Negative_Rate': fn_rate,
            'Error_Rate': error_rate
        })
    
    # Create DataFrame
    length_df = pd.DataFrame(length_stats)
    
    logger.info(f"Completed length-based error analysis across {len(length_df)} bins")
    
    return length_df


def analyze_by_keyword(df, prediction_col, ground_truth_col='Biased', min_count=5):
    """
    Analyze error rates by keyword.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        prediction_col (str): Column name for predictions
        ground_truth_col (str): Column name for ground truth
        min_count (int): Minimum count to include a keyword
    
    Returns:
        pd.DataFrame: DataFrame with error rates by keyword
    """
    # Ensure 'Keyword' column exists
    if 'Keyword' not in df.columns:
        logger.error("Keyword column not found in dataframe")
        return pd.DataFrame()
    
    # Calculate error metrics by keyword
    keyword_stats = []
    
    for keyword in df['Keyword'].unique():
        keyword_df = df[df['Keyword'] == keyword]
        if len(keyword_df) < min_count:  # Skip keywords with too few examples
            continue
        
        # Filter ground truth categories
        non_antisemitic = keyword_df[keyword_df[ground_truth_col] == 0]
        antisemitic = keyword_df[keyword_df[ground_truth_col] == 1]
        
        # Calculate false positive rate
        if len(non_antisemitic) > 0:
            fp = non_antisemitic[non_antisemitic[prediction_col] == 1]
            fp_rate = len(fp) / len(non_antisemitic)
        else:
            fp_rate = 0
        
        # Calculate false negative rate
        if len(antisemitic) > 0:
            fn = antisemitic[antisemitic[prediction_col] == 0]
            fn_rate = len(fn) / len(antisemitic)
        else:
            fn_rate = 0
        
        # Overall error rate
        error_count = (len(fp) if 'fp' in locals() else 0) + (len(fn) if 'fn' in locals() else 0)
        error_rate = error_count / len(keyword_df)
        
        keyword_stats.append({
            'Keyword': keyword,
            'Tweet_Count': len(keyword_df),
            'Non_Antisemitic_Count': len(non_antisemitic),
            'Antisemitic_Count': len(antisemitic),
            'False_Positive_Count': len(fp) if 'fp' in locals() else 0,
            'False_Negative_Count': len(fn) if 'fn' in locals() else 0,
            'False_Positive_Rate': fp_rate,
            'False_Negative_Rate': fn_rate,
            'Error_Rate': error_rate
        })
    
    # Create DataFrame and sort by error rate
    keyword_df = pd.DataFrame(keyword_stats)
    keyword_df = keyword_df.sort_values('Error_Rate', ascending=False)
    
    logger.info(f"Completed keyword-based error analysis across {len(keyword_df)} keywords")
    
    return keyword_df


def compare_model_errors(df, model1_col, model2_col, ground_truth_col='Biased'):
    """
    Compare errors between two model predictions.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions from both models
        model1_col (str): Column name for first model's predictions
        model2_col (str): Column name for second model's predictions
        ground_truth_col (str): Column name for ground truth
    
    Returns:
        dict: Dictionary with error comparison statistics
    """
    # Create binary flags for false positives and false negatives
    df = df.copy()
    
    # False positives (prediction = 1, ground truth = 0)
    df[f'{model1_col}_FP'] = ((df[ground_truth_col] == 0) & (df[model1_col] == 1)).astype(int)
    df[f'{model2_col}_FP'] = ((df[ground_truth_col] == 0) & (df[model2_col] == 1)).astype(int)
    
    # False negatives (prediction = 0, ground truth = 1)
    df[f'{model1_col}_FN'] = ((df[ground_truth_col] == 1) & (df[model1_col] == 0)).astype(int)
    df[f'{model2_col}_FN'] = ((df[ground_truth_col] == 1) & (df[model2_col] == 0)).astype(int)
    
    # Compare false positives
    fp_fixed = ((df[f'{model1_col}_FP'] == 1) & (df[f'{model2_col}_FP'] == 0)).sum()
    fp_introduced = ((df[f'{model1_col}_FP'] == 0) & (df[f'{model2_col}_FP'] == 1)).sum()
    fp_persistent = ((df[f'{model1_col}_FP'] == 1) & (df[f'{model2_col}_FP'] == 1)).sum()
    
    # Compare false negatives
    fn_fixed = ((df[f'{model1_col}_FN'] == 1) & (df[f'{model2_col}_FN'] == 0)).sum()
    fn_introduced = ((df[f'{model1_col}_FN'] == 0) & (df[f'{model2_col}_FN'] == 1)).sum()
    fn_persistent = ((df[f'{model1_col}_FN'] == 1) & (df[f'{model2_col}_FN'] == 1)).sum()
    
    # Create result dictionary
    results = {
        'fp_comparison': {
            'fixed': fp_fixed,
            'introduced': fp_introduced,
            'persistent': fp_persistent
        },
        'fn_comparison': {
            'fixed': fn_fixed,
            'introduced': fn_introduced,
            'persistent': fn_persistent
        },
        'fp_net_improvement': fp_fixed - fp_introduced,
        'fn_net_improvement': fn_fixed - fn_introduced,
        'overall_net_improvement': (fp_fixed + fn_fixed) - (fp_introduced + fn_introduced)
    }
    
    # Extract examples for each category
    results['fp_fixed_examples'] = df[(df[f'{model1_col}_FP'] == 1) & (df[f'{model2_col}_FP'] == 0)]
    results['fp_introduced_examples'] = df[(df[f'{model1_col}_FP'] == 0) & (df[f'{model2_col}_FP'] == 1)]
    results['fn_fixed_examples'] = df[(df[f'{model1_col}_FN'] == 1) & (df[f'{model2_col}_FN'] == 0)]
    results['fn_introduced_examples'] = df[(df[f'{model1_col}_FN'] == 0) & (df[f'{model2_col}_FN'] == 1)]
    
    # Print summary
    print("\n===== Error Comparison =====")
    print(f"False Positives Fixed:      {fp_fixed}")
    print(f"False Positives Introduced: {fp_introduced}")
    print(f"False Positives Persistent: {fp_persistent}")
    print(f"Net FP Improvement:         {results['fp_net_improvement']}")
    print()
    print(f"False Negatives Fixed:      {fn_fixed}")
    print(f"False Negatives Introduced: {fn_introduced}")
    print(f"False Negatives Persistent: {fn_persistent}")
    print(f"Net FN Improvement:         {results['fn_net_improvement']}")
    print()
    print(f"Overall Net Improvement:    {results['overall_net_improvement']}")
    
    logger.info(f"Completed error comparison between {model1_col} and {model2_col}")
    
    return results


def get_error_examples(df, prediction_col, ground_truth_col='Biased', error_type='both', max_examples=10):
    """
    Get example texts for error cases.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        prediction_col (str): Column name for predictions
        ground_truth_col (str): Column name for ground truth
        error_type (str): Type of errors to get ('fp', 'fn', or 'both')
        max_examples (int): Maximum number of examples to return
    
    Returns:
        dict: Dictionary with error examples
    """
    examples = {}
    
    # Get false positive examples
    if error_type in ['fp', 'both']:
        fp_df = df[(df[ground_truth_col] == 0) & (df[prediction_col] == 1)]
        fp_examples = fp_df.sample(min(max_examples, len(fp_df)))
        examples['false_positives'] = fp_examples[['Text', 'Keyword']].to_dict('records')
    
    # Get false negative examples
    if error_type in ['fn', 'both']:
        fn_df = df[(df[ground_truth_col] == 1) & (df[prediction_col] == 0)]
        fn_examples = fn_df.sample(min(max_examples, len(fn_df)))
        examples['false_negatives'] = fn_examples[['Text', 'Keyword']].to_dict('records')
    
    return examples
