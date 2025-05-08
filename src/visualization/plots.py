"""
Visualization utilities for antisemitism detection analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set up better styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")


def save_figure(fig, filename, output_dir=None, formats=None):
    """
    Save a figure to file.
    
    Args:
        fig: Matplotlib figure
        filename (str): Base filename without extension
        output_dir (str, optional): Directory to save the figure
        formats (list, optional): List of formats to save (default: ['png'])
    
    Returns:
        dict: Dictionary with paths to saved files
    """
    if output_dir is None:
        output_dir = 'plots'
    
    if formats is None:
        formats = ['png']
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in all requested formats
    saved_files = {}
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        saved_files[fmt] = filepath
    
    logger.info(f"Saved figure to {', '.join(saved_files.values())}")
    
    return saved_files


def plot_metrics_comparison(metrics_list, output_dir=None, title=None):
    """
    Create a bar chart comparing metrics from multiple models.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        output_dir (str, optional): Directory to save the plot
        title (str, optional): Title for the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Extract model names and metrics
    model_names = [metrics.get('model', f'Model {i+1}') for i, metrics in enumerate(metrics_list)]
    
    # Metrics to plot
    metrics_to_plot = ['precision', 'specificity', 'f05', 'f1', 'recall']
    metric_labels = ['Precision', 'Specificity', 'F0.5 Score', 'F1 Score', 'Recall']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up positions
    x = np.arange(len(metric_labels))
    width = 0.8 / len(model_names)
    
    # Plot bars for each model
    for i, metrics in enumerate(metrics_list):
        position = x + width * (i - len(model_names) / 2 + 0.5)
        bars = ax.bar(position, [metrics[m] for m in metrics_to_plot], width, label=model_names[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.3f}", ha='center', va='bottom', fontsize=8)
    
    # Set up labels and title
    ax.set_ylabel('Score')
    plot_title = title or 'Performance Metrics Comparison'
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = title.lower().replace(' ', '_') if title else 'metrics_comparison'
        save_figure(fig, filename, output_dir)
    
    return fig


def plot_error_rates_by_length(length_dfs, model_names=None, output_dir=None, error_type='fp'):
    """
    Create a line plot of error rates by text length.
    
    Args:
        length_dfs (list): List of DataFrames from analyze_by_length
        model_names (list, optional): List of model names
        output_dir (str, optional): Directory to save the plot
        error_type (str): Type of error to plot ('fp', 'fn', or 'both')
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(length_dfs))]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot for each model
    markers = ['o', 's', '^', 'd', 'x', '*', 'P', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    for i, df in enumerate(length_dfs):
        model_name = model_names[i]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        if error_type == 'fp' or error_type == 'both':
            ax.plot(df['Length_Bin'], df['False_Positive_Rate'], 
                    marker=marker, linestyle=linestyle, 
                    label=f"{model_name} (FP)" if error_type == 'both' else model_name)
        
        if error_type == 'fn' or error_type == 'both':
            ax.plot(df['Length_Bin'], df['False_Negative_Rate'], 
                    marker=marker, linestyle=':', 
                    label=f"{model_name} (FN)" if error_type == 'both' else model_name)
    
    # Set up labels and title
    ax.set_xlabel('Tweet Length (characters)')
    if error_type == 'fp':
        ax.set_ylabel('False Positive Rate')
        title = 'False Positive Rate by Tweet Length'
    elif error_type == 'fn':
        ax.set_ylabel('False Negative Rate')
        title = 'False Negative Rate by Tweet Length'
    else:
        ax.set_ylabel('Error Rate')
        title = 'Error Rates by Tweet Length'
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = f"{error_type}_by_length"
        save_figure(fig, filename, output_dir)
    
    return fig


def plot_error_rates_by_keyword(keyword_df, error_type='fp', top_n=15, output_dir=None, model_name=None):
    """
    Create a horizontal bar chart of error rates by keyword.
    
    Args:
        keyword_df (pd.DataFrame): DataFrame from analyze_by_keyword
        error_type (str): Type of error to plot ('fp', 'fn', or 'error')
        top_n (int): Number of top keywords to include
        output_dir (str, optional): Directory to save the plot
        model_name (str, optional): Name of the model for the title
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Sort by the appropriate error rate and select top N
    if error_type == 'fp':
        sorted_df = keyword_df.sort_values('False_Positive_Rate', ascending=False).head(top_n)
        rate_col = 'False_Positive_Rate'
        count_col = 'False_Positive_Count'
        total_col = 'Non_Antisemitic_Count'
        title_prefix = 'False Positive Rate'
    elif error_type == 'fn':
        sorted_df = keyword_df.sort_values('False_Negative_Rate', ascending=False).head(top_n)
        rate_col = 'False_Negative_Rate'
        count_col = 'False_Negative_Count'
        total_col = 'Antisemitic_Count'
        title_prefix = 'False Negative Rate'
    else:  # error
        sorted_df = keyword_df.sort_values('Error_Rate', ascending=False).head(top_n)
        rate_col = 'Error_Rate'
        count_col = 'False_Positive_Count'  # Will add FN later
        total_col = 'Tweet_Count'
        title_prefix = 'Overall Error Rate'
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = ax.barh(sorted_df['Keyword'], sorted_df[rate_col], color=sns.color_palette()[0])
    
    # Add count labels
    for i, bar in enumerate(bars):
        row = sorted_df.iloc[i]
        
        if error_type == 'error':
            # For overall error, show both FP and FN
            count_text = f"{row['False_Positive_Count']}FP + {row['False_Negative_Count']}FN/{row[total_col]}"
        else:
            count_text = f"{row[count_col]}/{row[total_col]}"
        
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                count_text, va='center', fontsize=9)
    
    # Set up labels and title
    ax.set_xlabel(f'{title_prefix}')
    ax.set_ylabel('Keyword')
    
    model_label = f" - {model_name}" if model_name else ""
    ax.set_title(f'{title_prefix} by Keyword{model_label}')
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set reasonable x-axis limit
    max_rate = sorted_df[rate_col].max()
    ax.set_xlim(0, min(1.0, max_rate * 1.2))
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = f"{error_type}_by_keyword"
        if model_name:
            filename += f"_{model_name.lower().replace(' ', '_')}"
        save_figure(fig, filename, output_dir)
    
    return fig


def plot_topic_distribution(topics, output_dir=None, model_name=None):
    """
    Create a horizontal bar chart of topic distribution.
    
    Args:
        topics (list): List of topic dictionaries
        output_dir (str, optional): Directory to save the plot
        model_name (str, optional): Name of the model for the title
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create a dataframe from topics
    topic_df = pd.DataFrame({
        'Topic': [f"Topic {i+1}: {' '.join(topic['words'][:3])}" for i, topic in enumerate(topics)],
        'Count': [topic['tweet_count'] for topic in topics],
        'Percentage': [topic['percentage'] for topic in topics]
    })
    
    # Sort by count
    topic_df = topic_df.sort_values('Count', ascending=False)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = ax.barh(topic_df['Topic'], topic_df['Count'], color=sns.color_palette()[2])
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        percentage = topic_df.iloc[i]['Percentage'] * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{percentage:.1f}%", va='center')
    
    # Set up labels and title
    ax.set_xlabel('Number of False Positive Tweets')
    ax.set_ylabel('Topic')
    
    model_label = f" - {model_name}" if model_name else ""
    ax.set_title(f'False Positive Topics{model_label}')
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = "fp_topics"
        if model_name:
            filename += f"_{model_name.lower().replace(' ', '_')}"
        save_figure(fig, filename, output_dir)
    
    return fig


def plot_error_changes(changes_data, output_dir=None, title="Error Changes"):
    """
    Create a bar chart showing error changes between models.
    
    Args:
        changes_data (dict): Dictionary with error change data
        output_dir (str, optional): Directory to save the plot
        title (str): Title for the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Extract categories and counts
    categories = list(changes_data['fp_comparison'].keys())
    fp_counts = [changes_data['fp_comparison'][cat] for cat in categories]
    fn_counts = [changes_data['fn_comparison'][cat] for cat in categories]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up positions
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, fp_counts, width, label='False Positives')
    bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives')
    
    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    str(int(height)), ha='center', va='bottom')
    
    # Set up labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Errors')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = "error_changes"
        save_figure(fig, filename, output_dir)
    
    return fig


def create_confusion_matrix_plot(y_true, y_pred, output_dir=None, model_name=None):
    """
    Create a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir (str, optional): Directory to save the plot
        model_name (str, optional): Name of the model for the title
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Calculate confusion matrix
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    # Calculate percentages
    cm_percent = cm.copy()
    for i in range(2):
        cm_percent.iloc[i, 0] = cm.iloc[i, 0] / cm.iloc[i, 2] * 100
        cm_percent.iloc[i, 1] = cm.iloc[i, 1] / cm.iloc[i, 2] * 100
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm.iloc[0:2, 0:2], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            text = f"{cm_percent.iloc[i, j]:.1f}%"
            ax.text(j + 0.5, i + 0.7, text, ha='center', va='center', fontsize=9, color='black')
    
    # Set up labels and title
    model_label = f" - {model_name}" if model_name else ""
    ax.set_title(f'Confusion Matrix{model_label}')
    
    # Add some padding
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        filename = "confusion_matrix"
        if model_name:
            filename += f"_{model_name.lower().replace(' ', '_')}"
        save_figure(fig, filename, output_dir)
    
    return fig
