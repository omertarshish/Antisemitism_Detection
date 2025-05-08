"""
Client for interacting with Ollama API on various environments.
"""

import requests
import time
import json
from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, ip_port=None, model=None, config_name="ollama_config"):
        """
        Initialize the Ollama client.
        
        Args:
            ip_port (str, optional): IP and port of the Ollama server (e.g., '192.168.1.1:11434')
            model (str, optional): Name of the model to use
            config_name (str, optional): Name of the configuration file to load
        """
        # Load configuration
        config = load_config(config_name)
        
        # Use provided values or fall back to config
        self.ip_port = ip_port or config.get("ip_port", "localhost:11434")
        self.model = model or config.get("model", "deepseek-r1:7b")
        
        # Set up base URL
        self.base_url = f"http://{self.ip_port}/api"
        
        logger.info(f"Initialized Ollama client for {self.model} at {self.ip_port}")
    
    def pull_model(self):
        """
        Pull the model from Ollama if not already present.
        
        Returns:
            bool: True if successful, False otherwise
        """
        url = f"{self.base_url}/pull"
        payload = {"model": self.model}
        
        try:
            logger.info(f"Pulling model {self.model}...")
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {self.model}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
    
    def generate(self, prompt, max_retries=3, temperature=0.7, stream=False):
        """
        Send a generation request to Ollama.
        
        Args:
            prompt (str): The input prompt for the model
            max_retries (int, optional): Maximum number of retries on failure
            temperature (float, optional): Sampling temperature
            stream (bool, optional): Whether to stream the response
        
        Returns:
            str: The model's response
        """
        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating response (attempt {attempt+1}/{max_retries})...")
                response = requests.post(url, json=payload, timeout=180)
                
                if response.status_code == 200:
                    # Parse the JSON response
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    logger.warning(f"Error: {response.status_code}, {response.text}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 20 * (attempt + 1)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    return f"ERROR: Status code {response.status_code}"
            except Exception as e:
                logger.warning(f"Exception: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                return f"ERROR: {str(e)}"
    
    def extract_decision_explanation(self, response):
        """
        Extract the decision and explanation from the model's response.
        
        Args:
            response (str): The model's response
        
        Returns:
            tuple: (decision, explanation)
        """
        try:
            # Try to find the formatted response
            if "DECISION:" in response and "EXPLANATION:" in response:
                decision_part = response.split("DECISION:")[1].split("EXPLANATION:")[0].strip()
                explanation_part = response.split("EXPLANATION:")[1].strip()
                
                # Clean up the decision to just get Yes/No
                decision = "Yes" if "yes" in decision_part.lower() else "No" if "no" in decision_part.lower() else "Unclear"
                
                return decision, explanation_part
            else:
                # For unformatted responses, try to extract a yes/no
                decision = "Yes" if "yes" in response.lower()[:50] else "No" if "no" in response.lower()[:50] else "Unclear"
                return decision, response
        except Exception as e:
            logger.error(f"Error extracting decision: {str(e)}")
            return "Error", str(e)
