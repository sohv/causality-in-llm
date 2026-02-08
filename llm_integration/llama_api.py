#!/usr/bin/env python3
"""
Llama API Client (via Together AI)
====================================

Client for querying Llama (Meta) for causal discovery meta-knowledge.

Uses Together AI's OpenAI-compatible API.

Setup:
    export LLAMA_API_KEY="your-together-api-key"
    export LLAMA_BASE_URL="https://api.together.xyz/v1"
    export LLAMA_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"

Usage:
    from llm_integration.llama_api import LlamaClient

    client = LlamaClient()
    response = client.query(prompt, temperature=0.7)
"""

import os
import time
import re
from typing import Dict, Tuple, Optional
from openai import OpenAI


class LlamaClient:
    """Client for Llama via Together AI."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Initialize Llama client.

        Args:
            api_key: Together AI API key (or set LLAMA_API_KEY env var)
            base_url: Base URL (default: Together AI)
            model: Model identifier (default: Llama-3.3-70B-Instruct-Turbo)
        """
        self.api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Llama API key required. Set LLAMA_API_KEY environment variable "
                "or pass api_key parameter.\n"
                "Together AI: https://api.together.xyz/settings/api-keys"
            )

        self.base_url = base_url or os.getenv(
            "LLAMA_BASE_URL", "https://api.together.xyz/v1"
        )
        self.model = model or os.getenv(
            "LLAMA_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        print(f"Llama client initialized: model={self.model}, base_url={self.base_url}")

    def query(self,
              prompt: str,
              temperature: float = 0.7,
              max_tokens: int = 1024,
              retry_attempts: int = 3) -> str:
        """
        Query Llama with a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            retry_attempts: Number of retry attempts on failure

        Returns:
            Response text from Llama
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
                error_str = str(e).lower()
                if 'rate' in error_str or '429' in error_str or 'quota' in error_str:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif attempt == retry_attempts - 1:
                    print(f"Failed after {retry_attempts} attempts: {e}")
                    raise
                else:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    time.sleep(1)

        raise RuntimeError("Failed to get response after all retry attempts")

    def parse_response(self, response: str) -> Dict[str, Tuple[float, float]]:
        """
        Parse Llama's response to extract metric ranges.

        Args:
            response: Raw response text

        Returns:
            Dictionary mapping metric names to (lower, upper) tuples
        """
        results = {}

        patterns = {
            'precision': r'Precision:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'recall': r'Recall:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'f1': r'F1[- ]?[Ss]core:\s*\(([0-9.]+),\s*([0-9.]+)\)|F1:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'shd': r'SHD:\s*\(([0-9.]+),\s*([0-9.]+)\)'
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if metric == 'f1':
                    groups = [g for g in match.groups() if g is not None]
                    if len(groups) >= 2:
                        results[metric] = (float(groups[0]), float(groups[1]))
                else:
                    results[metric] = (float(match.group(1)), float(match.group(2)))

        expected_metrics = {'precision', 'recall', 'f1', 'shd'}
        missing = expected_metrics - set(results.keys())
        if missing:
            print(f"Warning: Failed to parse metrics: {missing}")
            print(f"Response was:\n{response}\n")

        return results

    def query_and_parse(self, prompt: str, **kwargs) -> Dict[str, Tuple[float, float]]:
        """Query Llama and parse the response in one step."""
        response = self.query(prompt, **kwargs)
        return self.parse_response(response)


if __name__ == "__main__":
    import sys

    if not os.getenv("LLAMA_API_KEY"):
        print("Error: LLAMA_API_KEY environment variable not set")
        print("\n  export LLAMA_API_KEY='your-together-api-key'")
        sys.exit(1)

    print("Testing Llama API client...")
    print("=" * 80)

    client = LlamaClient()

    test_prompt = """You are an expert in causal discovery algorithms.

Given:
- Dataset: Titanic (7 variables, social science)
- Algorithm: PC (constraint-based)
- Sample size: 1000

Estimate performance ranges:
Precision: (?, ?)
Recall: (?, ?)
F1: (?, ?)
SHD: (?, ?)

Provide only numerical ranges."""

    print(f"\nTest prompt:\n{test_prompt}\n")
    print("Querying Llama...")

    try:
        response = client.query(test_prompt, temperature=0.7)
        print(f"\nRaw response:\n{response}\n")

        parsed = client.parse_response(response)
        print("\nParsed results:")
        for metric, (lower, upper) in parsed.items():
            print(f"  {metric:10s}: ({lower:.4f}, {upper:.4f})")

        print("\nLlama API client working correctly!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
