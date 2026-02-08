#!/usr/bin/env python3
"""
Claude 3.5 Sonnet API Client
=============================

Client for querying Claude (Anthropic) for causal discovery meta-knowledge.

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

    Or pass directly:
    client = ClaudeClient(api_key="your-key")

Usage:
    from llm_integration import ClaudeClient

    client = ClaudeClient()
    response = client.query(prompt, temperature=0.7)
"""

import os
import time
import re
from typing import Dict, Tuple, Optional
import anthropic


class ClaudeClient:
    """Client for Claude 3.5 Sonnet API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model identifier (default: claude-3-5-sonnet-20241022)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        print(f"Claude client initialized with model: {self.model}")

    def query(self,
             prompt: str,
             temperature: float = 0.7,
             max_tokens: int = 1024,
             retry_attempts: int = 3) -> str:
        """
        Query Claude with a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            retry_attempts: Number of retry attempts on failure

        Returns:
            Response text from Claude
        """
        for attempt in range(retry_attempts):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                response_text = message.content[0].text
                return response_text

            except anthropic.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

            except anthropic.APIError as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt == retry_attempts - 1:
                    raise

            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

        raise RuntimeError("Failed to get response after all retry attempts")

    def parse_response(self, response: str) -> Dict[str, Tuple[float, float]]:
        """
        Parse Claude's response to extract metric ranges.

        Expected format:
            Precision: (0.70, 0.85)
            Recall: (0.65, 0.80)
            F1: (0.60, 0.75)
            SHD: (5, 12)

        Args:
            response: Raw response text from Claude

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
        Query Claude and parse the response in one step.

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

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    print("Testing Claude API client...")
    print("="*80)

    client = ClaudeClient()

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
    print("Querying Claude 3.5 Sonnet...")

    try:
        response = client.query(test_prompt, temperature=0.7)
        print(f"\nRaw response:\n{response}\n")

        parsed = client.parse_response(response)
        print("\nParsed results:")
        for metric, (lower, upper) in parsed.items():
            print(f"  {metric:10s}: ({lower:.4f}, {upper:.4f})")

        print("\n✓ Claude API client working correctly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
