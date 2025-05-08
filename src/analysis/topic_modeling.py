"""
Topic modeling utilities for antisemitism detection analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import os
from ..utils.logger import get_logger

logger = get_logger(__name__)


def prepare_text_data(texts, max_df=0.95, min_df=2, stop_words='english'):
    """
    Prepare text data for topic modeling.
    
    Args:
        texts (list): List of text strings
        max_df (float): Ignore terms that appear in more than this fraction of documents
        min_df (int or float): Ignore terms that appear in fewer than this number/fraction of documents
        stop_words (str or list): Stop words to filter out
    
    Returns:
        tuple: (document-term matrix, vectorizer)
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    dtm = vectorizer.fit_transform(texts)
    
    logger.info(f"Created document-term matrix with {dtm.shape[1]} features from {len(texts)} documents")
    
    return dtm, vectorizer


def run_lda_topic_modeling(dtm, vectorizer, n_topics=5, random_state=42):
    """
    Perform LDA topic modeling.
    
    Args:
        dtm: Document-term matrix
        vectorizer: Vectorizer used to create the document-term matrix
        n_topics (int): Number of topics to identify
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (lda_model, topics_dict)
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Perform LDA topic modeling
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=100,
        learning_method='online'
    )
    
    # Fit the model
    lda.fit(dtm)
    
    # Extract top words per topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'words': top_words,
            'weights': topic[top_words_idx].tolist()
        })
    
    logger.info(f"Completed LDA topic modeling with {n_topics} topics")
    
    return lda, topics


def assign_topics_to_documents(lda_model, dtm):
    """
    Assign topics to documents.
    
    Args:
        lda_model: Fitted LDA model
        dtm: Document-term matrix
    
    Returns:
        tuple: (topic_assignments, dominant_topics)
    """
    # Transform documents to topic space
    topic_distributions = lda_model.transform(dtm)
    
    # Get dominant topic for each document
    dominant_topics = topic_distributions.argmax(axis=1)
    
    # Count documents per topic
    topic_counts = np.bincount(dominant_topics, minlength=lda_model.n_components)
    
    # Calculate percentages
    topic_percentages = topic_counts / len(dominant_topics)
    
    # Create a dictionary of topic distributions
    topic_info = {
        'distributions': topic_distributions,
        'dominant_topics': dominant_topics,
        'topic_counts': topic_counts,
        'topic_percentages': topic_percentages
    }
    
    logger.info(f"Assigned topics to {len(dominant_topics)} documents")
    
    return topic_info


def analyze_topics_for_false_positives(df, prediction_col, ground_truth_col='Biased', n_topics=5):
    """
    Perform topic modeling on false positive tweets.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        prediction_col (str): Column name for predictions
        ground_truth_col (str): Column name for ground truth
        n_topics (int): Number of topics to identify
    
    Returns:
        dict: Dictionary with topic modeling results
    """
    # Identify false positives
    false_positives = df[(df[ground_truth_col] == 0) & (df[prediction_col] == 1)]
    
    if len(false_positives) < 10:
        logger.warning(f"Too few false positives for topic modeling: {len(false_positives)}")
        return None
    
    # Get texts for analysis
    texts = false_positives['Text'].tolist()
    
    # Prepare document-term matrix
    dtm, vectorizer = prepare_text_data(texts)
    
    # Run LDA topic modeling
    lda_model, topics = run_lda_topic_modeling(dtm, vectorizer, n_topics=n_topics)
    
    # Assign topics to documents
    topic_info = assign_topics_to_documents(lda_model, dtm)
    
    # Add topic assignments to the dataframe
    false_positives = false_positives.copy()
    false_positives['Topic'] = topic_info['dominant_topics']
    
    # Update topic info with tweet counts
    for i, topic in enumerate(topics):
        topic['tweet_count'] = topic_info['topic_counts'][i]
        topic['percentage'] = topic_info['topic_percentages'][i]
    
    # Prepare topic examples
    topic_examples = {}
    for topic_id in range(n_topics):
        # Get examples for this topic
        topic_tweets = false_positives[false_positives['Topic'] == topic_id]
        
        # Sort by topic probability (if available)
        topic_examples[topic_id] = topic_tweets.sample(min(5, len(topic_tweets))).to_dict('records')
    
    # Create result dictionary
    results = {
        'topics': topics,
        'topic_info': topic_info,
        'false_positives': false_positives,
        'topic_examples': topic_examples
    }
    
    logger.info(f"Completed topic modeling for {len(false_positives)} false positives")
    
    return results


def generate_topic_report(topic_results, model_name=None, output_dir=None):
    """
    Generate an HTML report for topic modeling results.
    
    Args:
        topic_results (dict): Results from analyze_topics_for_false_positives()
        model_name (str, optional): Name of the model/definition
        output_dir (str, optional): Directory to save the HTML report
    
    Returns:
        str: HTML report content
    """
    if not topic_results:
        return "No topic modeling results available."
    
    topics = topic_results['topics']
    topic_examples = topic_results['topic_examples']
    
    model_label = f" for {model_name}" if model_name else ""
    
    # Create HTML content
    html_content = f"<h1>False Positive Analysis{model_label}</h1>\n"
    html_content += "<p>Topics identified in false positive tweets</p>\n"
    
    for i, topic in enumerate(topics):
        html_content += f"<h2>Topic {i + 1} ({topic['tweet_count']} tweets, {topic['percentage'] * 100:.1f}%)</h2>\n"
        
        # Top words with weights
        html_content += "<h3>Top words:</h3>\n<ul>\n"
        for word, weight in zip(topic['words'], topic['weights']):
            html_content += f"<li>{word} ({weight:.3f})</li>\n"
        html_content += "</ul>\n"
        
        # Example tweets
        html_content += "<h3>Example tweets:</h3>\n<ul>\n"
        for sample in topic_examples.get(i, []):
            sample_text = sample.get('Text', '').replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<li>{sample_text}</li>\n"
        html_content += "</ul>\n"
    
    # Save to file if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"fp_topic_analysis{'_' + model_name if model_name else ''}.html"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Topic report saved to {filepath}")
    
    return html_content
