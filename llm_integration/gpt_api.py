#!/usr/bin/env python3
"""
GPT-5 API Client
=================

Client for querying GPT-5 (OpenAI) for causal discovery meta-knowledge.

Setup:
    export OPENAI_API_KEY="your-api-key"

    Or pass directly:
    client = GPTClient(api_key="your-key")

Usage:
    from llm_integration.gpt_api import GPTClient

    client = GPTClient()
    response = client.query(prompt, temperature=0.7)
"""

import os
import time
import re
from typing import Dict, Tuple, Optional
from openai import OpenAI


class GPTClient:
    """Client for GPT-5 API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        """
        Initialize GPT client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model identifier (default: gpt-5)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter.\n"
                "Get API key from: https://platform.openai.com/api-keys"
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        print(f"GPT client initialized with model: {self.model}")

    def query(self,
             prompt: str,
             temperature: float = 0.7,
             max_tokens: int = 1024,
             retry_attempts: int = 3) -> str:
        """
        Query GPT with a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            retry_attempts: Number of retry attempts on failure

        Returns:
            Response text from GPT
        """
        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting with exponential backoff
                if "rate limit" in error_msg:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # Handle other API errors
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt == retry_attempts - 1:
                    raise

                time.sleep(1)  # Brief pause before retry

        raise RuntimeError("Failed to get response after all retry attempts")

    def parse_response(self, response: str) -> Dict[str, Tuple[float, float]]:
        """
        Parse GPT's response to extract metric ranges.

        Expected format:
        METRIC_NAME: min_value - max_value
        
        Example:
        F1_Score: 0.75 - 0.85
        Precision: 0.70 - 0.80
        Recall: 0.60 - 0.90

        Args:
            response: Raw response from GPT

        Returns:
            Dict mapping metric names to (min_val, max_val) tuples
        """
        metrics = {}
        
        # Pattern to extract metric ranges
        pattern = r'(\w+(?:_\w+)*)\s*:\s*([0-9.]+)\s*[-–—]\s*([0-9.]+)'
        
        matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
        
        for metric_name, min_val_str, max_val_str in matches:
            try:
                min_val = float(min_val_str)
                max_val = float(max_val_str)
                
                # Ensure min <= max
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                
                # Store normalized metric name
                metric_key = metric_name.lower().replace('score', '').replace('_', '')
                metrics[metric_key] = (min_val, max_val)
                
            except ValueError:
                continue
        
        # If no metrics found, try alternative parsing
        if not metrics:
            print("Warning: No metrics found with standard pattern. Trying alternative parsing...")
            
            # Try to find any numbers that might be ranges
            alt_pattern = r'([0-9.]+)\s*[-–—]\s*([0-9.]+)'
            number_matches = re.findall(alt_pattern, response)
            
            # Common metrics to look for
            common_metrics = ['f1', 'precision', 'recall', 'accuracy', 'auc']
            
            for i, (min_val_str, max_val_str) in enumerate(number_matches):
                if i < len(common_metrics):
                    try:
                        min_val = float(min_val_str)
                        max_val = float(max_val_str)
                        if min_val > max_val:
                            min_val, max_val = max_val, min_val
                        metrics[common_metrics[i]] = (min_val, max_val)
                    except ValueError:
                        continue

        return metrics

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the GPT model.

        Returns:
            Dictionary with model information
        """
        return {
            "provider": "OpenAI",
            "model": self.model,
            "api_version": "v1",
            "supports_chat": True,
            "supports_streaming": True,
        }