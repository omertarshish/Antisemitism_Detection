"""
Performance metrics for antisemitism detection.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, matthews_corrcoef
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_fbeta(y_true, y_pred, beta=0.5):
    """
    Calculate F-beta score with beta=0.5 to emphasize precision over recall.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta value (default: 0.5)
    
    Returns:
        float: F-beta score
    """
    return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)


def calculate_metrics(y_true, y_pred, model_name=None):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels (0/1 where 1=antisemitic)
        y_pred: Predicted labels (0/1 where 1=antisemitic)
        model_name: Name of the model/definition (optional)
    
    Returns:
        dict: Dictionary of metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Focus metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 1 - false positive rate
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f05 = calculate_fbeta(y_true, y_pred, beta=0.5)
    
    # False positive rate
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # Additional metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Total number of predicted positives
    predicted_positives = tp + fp
    
    # Create metrics dictionary
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'f05': f05,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'false_positive_rate': fp_rate,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'total_predicted_positive': predicted_positives,
        'total_actual_positive': tp + fn,
        'total_actual_negative': tn + fp,
        'total_samples': tp + tn + fp + fn
    }
    
    return metrics


def display_metrics(metrics, detailed=True):
    """
    Display metrics in a readable format.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        detailed: Whether to display detailed metrics
    """
    model_name = metrics.get('model', 'Model')
    
    print(f"\n===== Performance Metrics for {model_name} =====")
    
    # Core metrics
    print(
        f"Precision:          {metrics['precision']:.4f} - Of tweets classified as antisemitic, "
        f"{metrics['precision'] * 100:.1f}% were actually antisemitic"
    )
    print(
        f"Specificity:        {metrics['specificity']:.4f} - Correctly identified "
        f"{metrics['specificity'] * 100:.1f}% of non-antisemitic tweets"
    )
    print(
        f"False Positive Rate: {metrics['false_positive_rate']:.4f} - Incorrectly classified "
        f"{metrics['false_positive_rate'] * 100:.1f}% of non-antisemitic tweets as antisemitic"
    )
    print(
        f"F0.5 Score:         {metrics['f05']:.4f} - Weighted harmonic mean of precision and recall "
        f"(precision weighted more)"
    )
    print(
        f"F1 Score:           {metrics['f1']:.4f} - Balanced harmonic mean of precision and recall"
    )
    print(
        f"Recall/Sensitivity: {metrics['recall']:.4f} - Found {metrics['recall'] * 100:.1f}% "
        f"of all antisemitic tweets"
    )
    
    if detailed:
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"Matthews Corr Coef: {metrics['mcc']:.4f}")
    
    # Error analysis
    print("\nError Analysis:")
    print(
        f"False Positives:    {metrics['false_positive']} tweets "
        f"({metrics['false_positive'] / metrics['total_predicted_positive'] * 100:.1f}% "
        f"of all antisemitic predictions)"
    )
    print(f"False Negatives:    {metrics['false_negative']} tweets")
    print(f"Total Errors:       {metrics['false_positive'] + metrics['false_negative']} tweets")
    
    # Overall statistics
    if detailed:
        print("\nOverall Statistics:")
        print(f"Total Samples:      {metrics['total_samples']}")
        print(f"Actual Positive:    {metrics['total_actual_positive']}")
        print(f"Actual Negative:    {metrics['total_actual_negative']}")
        print(f"Predicted Positive: {metrics['total_predicted_positive']}")
        print(f"Predicted Negative: {metrics['total_samples'] - metrics['total_predicted_positive']}")


def compare_models(metrics_list):
    """
    Compare metrics from multiple models.
    
    Args:
        metrics_list: List of metrics dictionaries from calculate_metrics()
    
    Returns:
        dict: Dictionary with comparison data
    """
    # Extract model names
    model_names = [metrics.get('model', f'Model {i+1}') for i, metrics in enumerate(metrics_list)]
    
    # Create comparison data for key metrics
    comparison_data = {
        'Model': model_names,
        'Precision': [metrics['precision'] for metrics in metrics_list],
        'Specificity': [metrics['specificity'] for metrics in metrics_list],
        'False Positive Rate': [metrics['false_positive_rate'] for metrics in metrics_list],
        'F0.5 Score': [metrics['f05'] for metrics in metrics_list],
        'F1 Score': [metrics['f1'] for metrics in metrics_list],
        'Recall': [metrics['recall'] for metrics in metrics_list],
        'Matthews Correlation': [metrics['mcc'] for metrics in metrics_list]
    }
    
    # Print comparison table
    print("\n===== Model Comparison =====")
    col_width = max(len(col) for col in comparison_data.keys()) + 2
    
    # Print header
    print("".join(col.ljust(col_width) for col in comparison_data.keys()))
    
    # Print rows
    for i in range(len(model_names)):
        row = []
        for col in comparison_data.keys():
            if col == 'Model':
                row.append(comparison_data[col][i].ljust(col_width))
            else:
                row.append(f"{comparison_data[col][i]:.4f}".ljust(col_width))
        print("".join(row))
    
    return comparison_data


def calculate_improvement(baseline_metrics, new_metrics):
    """
    Calculate relative improvement between two sets of metrics.
    
    Args:
        baseline_metrics: Baseline metrics dictionary
        new_metrics: New metrics dictionary to compare against baseline
    
    Returns:
        dict: Dictionary with improvement percentages
    """
    # Metrics to compare
    metrics_to_compare = [
        'precision', 'specificity', 'false_positive_rate', 'f05', 'f1', 'recall', 'mcc'
    ]
    
    # Calculate improvements (for false_positive_rate, lower is better)
    improvements = {}
    
    for metric in metrics_to_compare:
        baseline_value = baseline_metrics[metric]
        new_value = new_metrics[metric]
        
        # Avoid division by zero
        if abs(baseline_value) < 0.0001:
            baseline_value = 0.0001
        
        # For false_positive_rate, improvement is reduction
        if metric == 'false_positive_rate':
            improvement = (baseline_value - new_value) / baseline_value
        else:
            improvement = (new_value - baseline_value) / baseline_value
        
        improvements[metric] = improvement
    
    # Print improvements
    print("\n===== Relative Improvement =====")
    for metric, improvement in improvements.items():
        direction = "decrease" if (metric == 'false_positive_rate' and improvement > 0) else "increase"
        print(f"{metric.capitalize()}: {improvement:.2%} {direction}")
    
    return improvements
