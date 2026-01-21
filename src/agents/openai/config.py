"""
OpenAI Agent Configuration

OpenAI-specific settings for the sample OpenAI agent.
These settings are only loaded when using the OpenAI agent.
"""

from functools import lru_cache
from typing import Literal, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory for .env file (project root)
_CONFIG_DIR = Path(__file__).parent.parent.parent.parent


class OpenAISettings(BaseSettings):
    """
    OpenAI-specific settings for the sample agent.
    
    Loaded from environment variables when the OpenAI agent is used.
    """
    
    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-realtime-preview"
    openai_voice: Literal[
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"
    ] = "alloy"
    
    # Transcription model for user speech
    # Options: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe, gpt-4o-transcribe-latest
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    
    # Voice Activity Detection (VAD) type
    # Options: server_vad, semantic_vad
    openai_vad_type: Literal["server_vad", "semantic_vad"] = "server_vad"
    
    # VAD threshold (0.0 to 1.0) - only for server_vad
    openai_vad_threshold: float = 0.5
    
    # Silence duration before turn ends (ms) - only for server_vad
    openai_vad_silence_duration_ms: int = 300
    
    # Noise reduction type (near_field, far_field, or None to disable)
    openai_noise_reduction: Optional[Literal["near_field", "far_field"]] = "near_field"


@lru_cache
def get_openai_settings() -> OpenAISettings:
    """
    Get cached OpenAI settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return OpenAISettings()
