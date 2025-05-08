#!/usr/bin/env python3
"""
Script for running analysis on antisemitism detection results.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.analysis.metrics import calculate_metrics, display_metrics, compare_models, calculate_improvement
from src.analysis.error_analysis import (analyze_by_length, analyze_by_keyword, 
                                        compare_model_errors, get_error_examples)
from src.analysis.topic_modeling import analyze_topics_for_false_positives, generate_topic_report
from src.visualization.plots import (plot_metrics_comparison, plot_error_rates_by_length,
                                    plot_error_rates_by_keyword, plot_topic_distribution,
                                    plot_error_changes, create_confusion_matrix_plot)
from src.utils.config import load_config
from src.utils.environment import get_data_path
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def main():
    """
    Main function to run analysis on antisemitism detection results.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run analysis on antisemitism detection results')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with comparative model results')
    parser.add_argument('--config', type=str, default='analysis_config',
                        help='Configuration file name (without extension)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis results')
    parser.add_argument('--model1-col', type=str, default='IHRA_Binary_7B',
                        help='Column name for first model predictions')
    parser.add_argument('--model2-col', type=str, default='IHRA_Binary_8B',
                        help='Column name for second model predictions')
    parser.add_argument('--ground-truth-col', type=str, default='Biased',
                        help='Column name for ground truth')
    parser.add_argument('--topic-analysis', action='store_true',
                        help='Perform topic modeling analysis on false positives')
    parser.add_argument('--n-topics', type=int, default=None,
                        help='Number of topics for topic modeling (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.get('output_dir', str(get_data_path('results')))
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    plots_dir = config.get('plots_dir', os.path.join(output_dir, 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded dataset with {len(df)} entries")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Extract column names
    model1_col = args.model1_col
    model2_col = args.model2_col
    ground_truth_col = args.ground_truth_col
    
    # Convert column names to friendly names for display
    model1_name = model1_col.replace('_Binary', '')
    model2_name = model2_col.replace('_Binary', '')
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    model1_metrics = calculate_metrics(df[ground_truth_col], df[model1_col], model1_name)
    model2_metrics = calculate_metrics(df[ground_truth_col], df[model2_col], model2_name)
    
    # Display metrics
    display_metrics(model1_metrics)
    display_metrics(model2_metrics)
    
    # Compare models
    comparison_data = compare_models([model1_metrics, model2_metrics])
    
    # Calculate improvement
    improvement_data = calculate_improvement(model1_metrics, model2_metrics)
    
    # Create metrics comparison plot
    plot_metrics_comparison([model1_metrics, model2_metrics], plots_dir, 
                           f"{model1_name} vs {model2_name} Metrics Comparison")
    
    # Create confusion matrix plots
    create_confusion_matrix_plot(df[ground_truth_col], df[model1_col], plots_dir, model1_name)
    create_confusion_matrix_plot(df[ground_truth_col], df[model2_col], plots_dir, model2_name)
    
    # Analyze errors by length
    logger.info("Analyzing error rates by tweet length...")
    
    length_bins = config.get('error_analysis', {}).get('analyze_by_length', {}).get('bins')
    length_labels = config.get('error_analysis', {}).get('analyze_by_length', {}).get('labels')
    
    length_df1 = analyze_by_length(df, model1_col, ground_truth_col, bins=length_bins, labels=length_labels)
    length_df2 = analyze_by_length(df, model2_col, ground_truth_col, bins=length_bins, labels=length_labels)
    
    # Plot error rates by length
    plot_error_rates_by_length([length_df1, length_df2], [model1_name, model2_name], plots_dir, 'fp')
    plot_error_rates_by_length([length_df1, length_df2], [model1_name, model2_name], plots_dir, 'fn')
    
    # Analyze errors by keyword
    logger.info("Analyzing error rates by keyword...")
    
    min_count = config.get('error_analysis', {}).get('analyze_by_keyword', {}).get('min_count', 5)
    top_n = config.get('error_analysis', {}).get('analyze_by_keyword', {}).get('top_n', 15)
    
    keyword_df1 = analyze_by_keyword(df, model1_col, ground_truth_col, min_count=min_count)
    keyword_df2 = analyze_by_keyword(df, model2_col, ground_truth_col, min_count=min_count)
    
    # Plot error rates by keyword
    plot_error_rates_by_keyword(keyword_df1, 'fp', top_n, plots_dir, model1_name)
    plot_error_rates_by_keyword(keyword_df2, 'fp', top_n, plots_dir, model2_name)
    
    # Compare model errors
    logger.info("Comparing model errors...")
    error_comparison = compare_model_errors(df, model1_col, model2_col, ground_truth_col)
    
    # Plot error changes
    plot_error_changes(error_comparison, plots_dir, f"Error Changes from {model1_name} to {model2_name}")
    
    # Perform topic analysis if requested
    if args.topic_analysis:
        logger.info("Performing topic modeling on false positives...")
        
        # Set number of topics
        n_topics = args.n_topics or config.get('topic_analysis', {}).get('n_topics', 5)
        
        # Analyze topics for each model
        topic_results1 = analyze_topics_for_false_positives(
            df, model1_col, ground_truth_col, n_topics=n_topics
        )
        
        topic_results2 = analyze_topics_for_false_positives(
            df, model2_col, ground_truth_col, n_topics=n_topics
        )
        
        # Generate topic reports
        if topic_results1:
            generate_topic_report(topic_results1, model1_name, output_dir)
            plot_topic_distribution(topic_results1['topics'], plots_dir, model1_name)
        
        if topic_results2:
            generate_topic_report(topic_results2, model2_name, output_dir)
            plot_topic_distribution(topic_results2['topics'], plots_dir, model2_name)
    
    # Save error examples
    error_examples_dir = os.path.join(output_dir, 'error_examples')
    os.makedirs(error_examples_dir, exist_ok=True)
    
    # Save examples of fixed and introduced errors
    if 'fp_fixed_examples' in error_comparison:
        error_comparison['fp_fixed_examples'].to_csv(
            os.path.join(error_examples_dir, 'fp_fixed_examples.csv'), index=False
        )
    
    if 'fp_introduced_examples' in error_comparison:
        error_comparison['fp_introduced_examples'].to_csv(
            os.path.join(error_examples_dir, 'fp_introduced_examples.csv'), index=False
        )
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    logger.info(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
