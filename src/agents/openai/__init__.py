"""
OpenAI Agent Package

Provides the sample OpenAI agent implementation using OpenAI Realtime API.
"""

from .config import OpenAISettings, get_openai_settings
from .sample_agent import SampleOpenAIAgent

__all__ = [
    "OpenAISettings",
    "SampleOpenAIAgent",
    "get_openai_settings",
]
