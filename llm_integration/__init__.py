"""
Multi-LLM Integration for Meta-Knowledge Evaluation
====================================================

This module provides unified interfaces to multiple LLMs for querying
causal discovery algorithm performance.

Supported LLMs:
1. Claude 3.5 Sonnet (Anthropic)
2. Gemini 1.5 Pro (Google)
3. Qwen 2.5 72B (Alibaba - via Together AI / DashScope)
4. Llama 3.3 70B (Meta - via Together AI / Groq)
5. GPT-5 (OpenAI)
6. DeepSeek R1

Key features:
- Unified API for all LLMs
- Rate limiting and error handling
- Prompt template support
- Response parsing and validation
"""

from .claude_api import ClaudeClient
from .gemini_api import GeminiClient
from .qwen_api import QwenClient
from .llama_api import LlamaClient
from .gpt_api import GPTClient
from .deepseek_api import DeepSeekClient
from .multi_llm_runner import MultiLLMRunner, run_multi_llm_experiments

__all__ = [
    'ClaudeClient',
    'GeminiClient',
    'QwenClient',
    'LlamaClient',
    'GPTClient',
    'DeepSeekClient',
    'MultiLLMRunner',
    'run_multi_llm_experiments'
]
