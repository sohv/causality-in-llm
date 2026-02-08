#!/usr/bin/env python3
"""
DeepSeek-R1 API Client
=======================

Client for querying DeepSeek-R1 for causal discovery meta-knowledge.

DeepSeek-R1 is accessible through DeepSeek's API using OpenAI-compatible format.

Setup:
    export DEEPSEEK_API_KEY="your-api-key"
    export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # optional override

    Or pass directly:
    client = DeepSeekClient(api_key="your-key")

Usage:
    from llm_integration.deepseek_api import DeepSeekClient

    client = DeepSeekClient()
    response = client.query(prompt, temperature=0.7)
"""

import os
import time
import re
from typing import Dict, Tuple, Optional
from openai import OpenAI


class DeepSeekClient:
    """Client for DeepSeek-R1 via OpenAI-compatible API."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "deepseek-reasoner"):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            base_url: Base URL for API (or set DEEPSEEK_BASE_URL env var)
            model: Model identifier (default: deepseek-reasoner)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment "
                "variable or pass api_key parameter.\n"
                "Get API key from: https://platform.deepseek.com/api_keys"
            )

        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        print(f"DeepSeek client initialized: model={self.model}, base_url={self.base_url}")

    def query(self,
              prompt: str,
              temperature: float = 0.7,
              max_tokens: int = 1024,
              retry_attempts: int = 3) -> str:
        """
        Query DeepSeek-R1 with a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            retry_attempts: Number of retry attempts on failure

        Returns:
            Response text from DeepSeek
        """
        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # DeepSeek-R1 includes reasoning traces, extract final answer
                content = response.choices[0].message.content
                
                # If response contains reasoning markers, extract the final answer
                if "<think>" in content and "</think>" in content:
                    # Extract content after reasoning
                    parts = content.split("</think>")
                    if len(parts) > 1:
                        return parts[-1].strip()
                
                return content

            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting with exponential backoff
                if "rate limit" in error_msg or "too many requests" in error_msg:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # Handle quota exceeded
                if "quota" in error_msg:
                    print(f"API quota exceeded: {e}")
                    raise

                # Handle model availability
                if "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
                    print(f"Model not available, trying fallback: {e}")
                    # Try with fallback model
                    if self.model != "deepseek-chat":
                        self.model = "deepseek-chat"
                        continue

                # Handle other API errors
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt == retry_attempts - 1:
                    raise

                time.sleep(1)  # Brief pause before retry

        raise RuntimeError("Failed to get response after all retry attempts")

    def parse_response(self, response: str) -> Dict[str, Tuple[float, float]]:
        """
        Parse DeepSeek's response to extract metric ranges.

        Expected format:
        METRIC_NAME: min_value - max_value
        
        Example:
        F1_Score: 0.75 - 0.85
        Precision: 0.70 - 0.80
        Recall: 0.60 - 0.90

        Args:
            response: Raw response from DeepSeek

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
        Get information about the DeepSeek model.

        Returns:
            Dictionary with model information
        """
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "api_version": "v1",
            "supports_chat": True,
            "supports_reasoning": True,
            "notes": "DeepSeek-R1 includes reasoning traces in responses"
        }