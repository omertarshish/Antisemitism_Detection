# RAG-Enhanced Antisemitism Detection
# This module implements a Retrieval Augmented Generation approach to improve
# antisemitism detection by incorporating full tweet context

import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Import LangChain components for RAG implementation
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.models.ollama_client import OllamaClient
from src.models.definitions import create_definition_prompt, get_available_definitions

# Initialize logger
logger = get_logger(__name__)

class TwitterContextRetriever:
    """
    Class for retrieving and processing the full context of tweets.
    Extracts thread information, replies, and quoted content when available.
    """

    def __init__(self, api_key=None, api_secret=None, bearer_token=None):
        """
        Initialize the context retriever with API credentials.

        Args:
            api_key (str, optional): Twitter API key
            api_secret (str, optional): Twitter API secret
            bearer_token (str, optional): Twitter API bearer token
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.bearer_token = bearer_token
        self.session = requests.Session()

        # Set up headers for API requests if credentials provided
        if bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})

        logger.info("TwitterContextRetriever initialized")

    def extract_tweet_context(self, tweet_id, tweet_text=None, username=None):
        """
        Extract the full context of a tweet including thread, replies and quoted content.

        Args:
            tweet_id (str): The unique identifier of the tweet
            tweet_text (str, optional): The text content of the tweet if already available
            username (str, optional): The username of the tweet author if already available

        Returns:
            dict: A dictionary containing the full context of the tweet
        """
        try:
            # Prepare base context structure
            context = {
                "original_tweet": tweet_text or "",
                "original_author": username or "",
                "thread_tweets": [],      # Previous tweets in the same thread
                "replies": [],            # Replies to this tweet
                "quoted_tweets": [],      # Tweets quoted by this tweet
                "parent_tweet": None,     # The tweet this is replying to
                "referenced_articles": [] # Any news articles or external content referenced
            }

            # If we have API credentials, use the Twitter API
            if self.bearer_token:
                # Construct the API URL for tweet lookup
                url = f"https://api.twitter.com/2/tweets/{tweet_id}"
                params = {
                    "expansions": "referenced_tweets.id,entities.mentions.username,attachments.media_keys",
                    "tweet.fields": "conversation_id,created_at,in_reply_to_user_id,text",
                    "user.fields": "username,name",
                    "media.fields": "url"
                }

                # Make the API request
                response = self.session.get(url, params=params)

                # Process the response if successful
                if response.status_code == 200:
                    data = response.json()

                    # Extract thread information, replies, etc.
                    # Implementation would depend on the Twitter API structure

                    logger.debug(f"Successfully retrieved context for tweet {tweet_id}")

                # API rate limiting handling
                elif response.status_code == 429:
                    # Handle rate limiting
                    reset_time = int(response.headers.get('x-rate-limit-reset', 0))
                    sleep_time = max(reset_time - time.time(), 0) + 1
                    logger.warning(f"Rate limited. Sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)

                else:
                    # Log other API errors
                    logger.error(f"API error {response.status_code}: {response.text}")

            # If no API credentials or API request failed, try to extract from tweet URL
            # This is a fallback method and may be less reliable
            else:
                # Construct tweet URL
                tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"

                # Extract any URLs from the tweet text that might provide context
                if tweet_text:
                    # Very basic URL extraction - in production, use a more robust method
                    words = tweet_text.split()
                    for word in words:
                        if word.startswith("http://") or word.startswith("https://"):
                            # Check if it's a news article or other context
                            parsed_url = urlparse(word)
                            if parsed_url.netloc not in ["twitter.com", "t.co"]:
                                context["referenced_articles"].append(word)

                logger.debug(f"Constructed context for tweet {tweet_id} without API access")

            return context

        except Exception as e:
            logger.error(f"Error extracting context for tweet {tweet_id}: {str(e)}")
            return context


class RAGTweetProcessor:
    """
    Implements Retrieval Augmented Generation for antisemitism detection in tweets.
    Uses vector storage to maintain contextual information about tweets.
    """

    def __init__(self, config_name="rag_config"):
        """
        Initialize the RAG processor with configuration.

        Args:
            config_name (str): Name of the configuration file to load
        """
        # Load configuration
        self.config = load_config(config_name)

        # Initialize context retriever
        api_key = self.config.get("twitter_api_key")
        api_secret = self.config.get("twitter_api_secret")
        bearer_token = self.config.get("twitter_bearer_token")

        self.context_retriever = TwitterContextRetriever(
            api_key=api_key,
            api_secret=api_secret,
            bearer_token=bearer_token
        )

        # Initialize vector store path
        self.vector_store_path = self.config.get("vector_store_path", "data/vector_stores/tweet_contexts")

        # Initialize embeddings model
        embedding_model = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Set up vector store if it exists
        self.vector_store = None
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
                logger.info(f"Loaded existing vector store from {self.vector_store_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {str(e)}")

        # Initialize LLM for RAG
        ollama_base_url = self.config.get("ollama_base_url", "http://localhost:11434")
        ollama_model = self.config.get("ollama_model", "llama3")

        try:
            self.llm = Ollama(base_url=ollama_base_url, model=ollama_model)
            logger.info(f"Initialized Ollama LLM with model {ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            self.llm = None

        # Set up RAG chain if both vector store and LLM are available
        self.rag_chain = None
        if self.vector_store and self.llm:
            try:
                self.rag_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                )
                logger.info("Initialized RAG chain successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG chain: {str(e)}")

    def build_vector_store(self, dataframe):
        """
        Build or update the vector store with tweet context information.

        Args:
            dataframe (pd.DataFrame): DataFrame containing tweets with context

        Returns:
            bool: Success status
        """
        try:
            documents = []

            # Process each tweet to extract context and create documents
            for index, row in tqdm(dataframe.iterrows(), total=len(dataframe),
                                  desc="Building vector store"):
                tweet_id = row['TweetID']
                tweet_text = row['Text']
                username = row.get('Username', '')

                # Extract context
                context = self.context_retriever.extract_tweet_context(
                    tweet_id=tweet_id,
                    tweet_text=tweet_text,
                    username=username
                )

                # Format the context into a document
                full_context = f"Original Tweet: {tweet_text}\n\n"

                if context["original_author"]:
                    full_context += f"Author: {context['original_author']}\n\n"

                if context["parent_tweet"]:
                    full_context += f"Replying to: {context['parent_tweet']}\n\n"

                if context["thread_tweets"]:
                    full_context += "Previous Tweets in Thread:\n"
                    for thread_tweet in context["thread_tweets"]:
                        full_context += f"- {thread_tweet}\n"

                if context["replies"]:
                    full_context += "\nReplies:\n"
                    for reply in context["replies"]:
                        full_context += f"- {reply}\n"

                if context["quoted_tweets"]:
                    full_context += "\nQuoted Tweets:\n"
                    for quote in context["quoted_tweets"]:
                        full_context += f"- {quote}\n"

                if context["referenced_articles"]:
                    full_context += "\nReferenced Content:\n"
                    for article in context["referenced_articles"]:
                        full_context += f"- {article}\n"

                # Create document with metadata
                document = {
                    "page_content": full_context,
                    "metadata": {
                        "tweet_id": tweet_id,
                        "author": username,
                        "date": row.get('CreateDate', ''),
                        "is_antisemitic": row.get('Biased', 0),
                        "keyword": row.get('Keyword', '')
                    }
                }

                documents.append(document)

            # Text splitter for handling longer contexts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = text_splitter.split_documents(documents)

            # Create new vector store
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

            # Save the vector store
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)

            logger.info(f"Vector store built and saved to {self.vector_store_path}")

            # Initialize RAG chain with new vector store
            if self.llm:
                self.rag_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                )

            return True

        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            return False

    def analyze_tweet_with_context(self, tweet_text, tweet_id=None, definition="IHRA"):
        """
        Analyze a tweet with its full context to determine if it's antisemitic.

        Args:
            tweet_text (str): The text content of the tweet
            tweet_id (str, optional): The unique identifier of the tweet
            definition (str): The antisemitism definition to use (IHRA or JDA)

        Returns:
            dict: Analysis results including decision and explanation
        """
        # Check if RAG chain is available
        if not self.rag_chain:
            logger.warning("RAG chain not available. Using simpler analysis.")
            # Fall back to basic analysis without RAG
            return self._analyze_without_rag(tweet_text, definition)

        try:
            # Select the right definition text
            if definition == "IHRA":
                definition_text = """
                IHRA Definition: Antisemitism is a certain perception of Jews, which may be expressed as hatred toward Jews.
                Rhetorical and physical manifestations of antisemitism are directed toward Jewish or non-Jewish individuals
                and/or their property, toward Jewish community institutions and religious facilities.
                """
            elif definition == "JDA":
                definition_text = """
                JDA Definition: Antisemitism is discrimination, prejudice, hostility or violence against Jews as Jews (or
                Jewish institutions as Jewish).
                """
            else:
                definition_text = f"Using {definition} definition of antisemitism."

            # Construct the query with definition
            query = f"""
            Analyze the following tweet and determine if it's antisemitic according to the {definition} definition:

            Tweet ID: {tweet_id or 'Unknown'}
            Tweet Text: {tweet_text}

            {definition_text}

            Pay special attention to:
            1. The full context of the tweet, including any thread it's part of
            2. Whether it's reporting on antisemitism rather than being antisemitic itself
            3. Whether it's calling out antisemitism
            4. Whether it uses antisemitic tropes or stereotypes
            5. Whether it's criticizing Israeli policy (which isn't necessarily antisemitic) vs delegitimizing Israel's existence

            Provide a detailed explanation with your reasoning, and clearly state whether the tweet is antisemitic or not.
            Format your answer with a clear "Decision: Yes/No" at the beginning.
            """

            # Run the RAG query
            result = self.rag_chain.run(query)

            # Extract decision and explanation
            decision = "Unclear"
            explanation = result

            if "Decision: Yes" in result.upper():
                decision = "Yes"
            elif "Decision: No" in result.upper():
                decision = "No"

            return {
                "decision": decision,
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error in RAG analysis: {str(e)}")
            # Fall back to basic analysis
            return self._analyze_without_rag(tweet_text, definition)

    def _analyze_without_rag(self, tweet_text, definition="IHRA"):
        """
        Fallback method when RAG isn't available.

        Args:
            tweet_text (str): The text content of the tweet
            definition (str): The antisemitism definition to use

        Returns:
            dict: Analysis results
        """
        try:
            # Create simple client without context enhancement
            ollama_client = OllamaClient(
                ip_port=self.config.get("ollama_ip_port"),
                model=self.config.get("ollama_model", "llama3")
            )

            # Create the prompt
            prompt = create_definition_prompt(definition, tweet_text)

            # Get response
            response = ollama_client.generate(prompt)

            # Extract decision and explanation
            decision, explanation = ollama_client.extract_decision_explanation(response)

            return {
                "decision": decision,
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return {
                "decision": "Error",
                "explanation": f"Analysis failed: {str(e)}"
            }

    def process_dataset(self, input_file, output_file, definitions=None, limit=None):
        """
        Process an entire dataset of tweets using RAG-enhanced analysis.

        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to the output CSV file
            definitions (list, optional): List of definitions to use
            limit (int, optional): Limit the number of tweets to process

        Returns:
            bool: Success status
        """
        try:
            # Load the dataset
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} tweets from {input_file}")

            # Apply limit if specified
            if limit:
                df = df.head(limit)
                logger.info(f"Limited to processing {limit} tweets")

            # Use default definitions if none provided
            if not definitions:
                definitions = ["IHRA", "JDA"]

            # Process each tweet
            results = []

            # Get number of workers from config
            max_workers = self.config.get("max_workers", 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for index, row in df.iterrows():
                    tweet_id = row['TweetID']
                    tweet_text = row['Text']

                    # Create base result with original data
                    result = {
                        'TweetID': tweet_id,
                        'Username': row['Username'],
                        'CreateDate': row['CreateDate'],
                        'Biased': row['Biased'],
                        'Keyword': row['Keyword'],
                        'Text': tweet_text
                    }

                    # Submit analysis tasks for each definition
                    for definition in definitions:
                        future = executor.submit(
                            self.analyze_tweet_with_context,
                            tweet_text=tweet_text,
                            tweet_id=tweet_id,
                            definition=definition
                        )
                        futures.append((future, result, definition))

                # Process completed futures
                for future, result, definition in tqdm(futures, desc="Processing tweets"):
                    try:
                        analysis = future.result()
                        result[f'{definition}_Decision'] = analysis['decision']
                        result[f'{definition}_Explanation'] = analysis['explanation']
                    except Exception as e:
                        logger.error(f"Error processing tweet {result['TweetID']} with {definition}: {str(e)}")
                        result[f'{definition}_Decision'] = "Error"
                        result[f'{definition}_Explanation'] = str(e)

                    # Add to results if not already there
                    if result not in results:
                        results.append(result)

            # Save results
            results_df = pd.DataFrame(results)

            # Add binary columns for easier analysis
            for definition in definitions:
                results_df[f'{definition}_Binary'] = results_df[f'{definition}_Decision'].map({
                    'Yes': 1,
                    'No': 0,
                    'Error': 0,
                    'Unclear': 0
                })

            # Save to file
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")

            # Log statistics
            for definition in definitions:
                yes_count = results_df[results_df[f'{definition}_Decision'] == 'Yes'].shape[0]
                no_count = results_df[results_df[f'{definition}_Decision'] == 'No'].shape[0]
                error_count = results_df[results_df[f'{definition}_Decision'] == 'Error'].shape[0]

                logger.info(f"{definition}: {yes_count} antisemitic, {no_count} not antisemitic, {error_count} errors")

            return True

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='RAG-enhanced antisemitism detection')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with tweets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file for results')
    parser.add_argument('--config', type=str, default='rag_config',
                        help='Configuration file name (without extension)')
    parser.add_argument('--build-store', action='store_true',
                        help='Build or rebuild the vector store before processing')
    parser.add_argument('--definitions', type=str, default='IHRA,JDA',
                        help='Comma-separated list of definition names to use')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tweets to process')

    args = parser.parse_args()

    # Initialize processor
    processor = RAGTweetProcessor(config_name=args.config)

    # Build vector store if requested
    if args.build_store:
        logger.info("Building vector store...")
        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)
        processor.build_vector_store(df)

    # Process the dataset
    definitions = args.definitions.split(',')
    processor.process_dataset(
        input_file=args.input,
        output_file=args.output,
        definitions=definitions,
        limit=args.limit
    )

    logger.info("Processing complete")