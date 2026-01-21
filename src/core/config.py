"""
Application Configuration

Centralized configuration using Pydantic Settings for type-safe
environment variable management with validation.

This module contains ONLY vendor-agnostic settings.
Vendor-specific settings (OpenAI, Gemini) are managed by their respective agents.
"""

from functools import lru_cache
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory containing this file, then go up to avatar_chat_server/
_CONFIG_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    This class contains only VENDOR-AGNOSTIC settings.
    """
    
    model_config = SettingsConfigDict(
        env_file=_CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Assistant Configuration (shared across all agents)
    assistant_instructions: str = (
        "You are a helpful and friendly AI assistant. Be concise in your responses."
    )
    
    # Wav2Arkit Model Configuration (ONNX CPU-only)
    onnx_model_path: str = "./pretrained_models/wav2arkit_cpu.onnx"
    
    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    debug: bool = False

    # Authentication Configuration
    auth_enabled: bool = True
    auth_secret_key: str = ""
    auth_token_ttl: int = 3600
    auth_allowed_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175"
    auth_enable_rate_limiting: bool = True
    
    # Agent Configuration
    agent_type: str = "sample_openai"  # "sample_openai", "sample_gemini", "remote"
    agent_url: Optional[str] = None  # URL for remote agent (e.g., "ws://agent-service:8080/ws")
    
    # Audio Configuration (vendor-agnostic)
    # Note: Widget sends 24kHz audio. This is used for Wav2Arkit processing.
    input_sample_rate: int = 24000      # Input audio sample rate (widget format)
    wav2arkit_sample_rate: int = 16000  # Wav2Arkit model expects 16kHz
    blendshape_fps: int = 30            # Output blendshape frame rate
    audio_chunk_duration: float = 0.5   # 0.5 second chunks for Wav2Arkit processing


class AudioConstants:
    """Audio processing constants derived from settings."""

    def __init__(self, settings: Settings):
        self.input_sample_rate = settings.input_sample_rate
        self.wav2arkit_sample_rate = settings.wav2arkit_sample_rate
        self.blendshape_fps = settings.blendshape_fps
        self.audio_chunk_duration = settings.audio_chunk_duration

        # Derived values
        self.frame_interval_ms = 1000 / self.blendshape_fps
        self.samples_per_frame = self.input_sample_rate // self.blendshape_fps
        self.bytes_per_frame = self.samples_per_frame * 2  # PCM16 = 2 bytes


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once
    and reused throughout the application lifecycle.
    """
    return Settings()


@lru_cache
def get_audio_constants() -> AudioConstants:
    """Get cached audio constants derived from settings."""
    return AudioConstants(get_settings())


def get_allowed_origins() -> list[str]:
    """Parse allowed origins from comma-separated string."""
    settings = get_settings()
    if not settings.auth_allowed_origins:
        return []
    return [origin.strip() for origin in settings.auth_allowed_origins.split(',') if origin.strip()]
