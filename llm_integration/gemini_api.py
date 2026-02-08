#!/usr/bin/env python3
"""
Gemini 1.5 Pro API Client
==========================

Client for querying Gemini (Google) for causal discovery meta-knowledge.

Setup:
    export GOOGLE_API_KEY="your-api-key"

    Or pass directly:
    client = GeminiClient(api_key="your-key")

Usage:
    from llm_integration import GeminiClient

    client = GeminiClient()
    response = client.query(prompt, temperature=0.7)
"""

import os
import time
import re
from typing import Dict, Tuple, Optional
import google.generativeai as genai


class GeminiClient:
    """Client for Gemini 1.5 Pro API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model identifier (default: gemini-1.5-pro)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

        print(f"Gemini client initialized with model: {self.model_name}")

    def query(self,
             prompt: str,
             temperature: float = 0.7,
             max_tokens: int = 1024,
             retry_attempts: int = 3) -> str:
        """
        Query Gemini with a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            retry_attempts: Number of retry attempts on failure

        Returns:
            Response text from Gemini
        """
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for attempt in range(retry_attempts):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                return response.text

            except Exception as e:
                # Check for rate limiting
                error_str = str(e).lower()
                if 'quota' in error_str or 'rate' in error_str or '429' in error_str:
                    wait_time = 2 ** attempt  # Exponential backoff
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
        Parse Gemini's response to extract metric ranges.

        Expected format:
            Precision: (0.70, 0.85)
            Recall: (0.65, 0.80)
            F1: (0.60, 0.75)
            SHD: (5, 12)

        Args:
            response: Raw response text from Gemini

        Returns:
            Dictionary mapping metric names to (lower, upper) tuples
        """
        results = {}

        # Patterns for each metric
        patterns = {
            'precision': r'Precision:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'recall': r'Recall:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'f1': r'F1[- ]?[Ss]core:\s*\(([0-9.]+),\s*([0-9.]+)\)|F1:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            'shd': r'SHD:\s*\(([0-9.]+),\s*([0-9.]+)\)'
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                # Handle F1 pattern with multiple capture groups
                if metric == 'f1':
                    groups = [g for g in match.groups() if g is not None]
                    if len(groups) >= 2:
                        lower = float(groups[0])
                        upper = float(groups[1])
                        results[metric] = (lower, upper)
                else:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    results[metric] = (lower, upper)

        # Validate we got all metrics
        expected_metrics = {'precision', 'recall', 'f1', 'shd'}
        missing = expected_metrics - set(results.keys())
        if missing:
            print(f"Warning: Failed to parse metrics: {missing}")
            print(f"Response was:\n{response}\n")

        return results

    def query_and_parse(self, prompt: str, **kwargs) -> Dict[str, Tuple[float, float]]:
        """
        Query Gemini and parse the response in one step.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for query()

        Returns:
            Parsed metrics dictionary
        """
        response = self.query(prompt, **kwargs)
        return self.parse_response(response)


if __name__ == "__main__":
    # Test the client
    import sys

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)

    print("Testing Gemini API client...")
    print("="*80)

    client = GeminiClient()

    # Simple test prompt
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
    print("Querying Gemini 1.5 Pro...")

    try:
        response = client.query(test_prompt, temperature=0.7)
        print(f"\nRaw response:\n{response}\n")

        parsed = client.parse_response(response)
        print("\nParsed results:")
        for metric, (lower, upper) in parsed.items():
            print(f"  {metric:10s}: ({lower:.4f}, {upper:.4f})")

        print("\n✓ Gemini API client working correctly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
