"""
OpenAI Agent Package

Provides the sample OpenAI agent implementation using OpenAI Realtime API.
"""

from .sample_agent import SampleOpenAIAgent
from .config import OpenAISettings, get_openai_settings

__all__ = [
    "SampleOpenAIAgent",
    "OpenAISettings",
    "get_openai_settings",
]
