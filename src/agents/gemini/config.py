"""
Gemini Agent Configuration

Gemini-specific settings for the sample Gemini agent.
These settings are only loaded when using the Gemini agent.
"""

from functools import lru_cache
from typing import Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory for .env file (project root)
_CONFIG_DIR = Path(__file__).parent.parent.parent.parent


class GeminiSettings(BaseSettings):
    """
    Gemini-specific settings for the sample agent.
    
    Loaded from environment variables when the Gemini agent is used.
    """
    
    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_voice: Literal[
        "Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"
    ] = "Puck"
    # Thinking budget: 0=disabled, -1=dynamic, 1-32768=fixed token budget
    gemini_thinking_budget: int = -1
    # Enable Google Search grounding for real-time information
    gemini_google_search_grounding: bool = False
    # Proactive audio: model can decide not to respond if content is not relevant
    gemini_proactive_audio: bool = False
    # Context window compression: enables longer sessions (beyond 15min audio-only limit)
    gemini_context_window_compression: bool = True


@lru_cache
def get_gemini_settings() -> GeminiSettings:
    """
    Get cached Gemini settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return GeminiSettings()
