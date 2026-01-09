"""
Application Configuration

Centralized configuration using Pydantic Settings for type-safe
environment variable management with validation.
"""

from functools import lru_cache
from typing import Literal, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory containing this file, then go up to avatar_chat_server/
_CONFIG_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    Example: OPENAI_API_KEY=sk-xxx or openai_api_key=sk-xxx
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
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"
    ] = "alloy"
    
    # Assistant Configuration
    assistant_instructions: str = (
        "You are a helpful and friendly AI assistant. Be concise in your responses."
    )
    
    # Audio2Expression Model Configuration
    model_path: str = "./pretrained_models/lam_audio2exp_streaming.tar"
    identity_idx: int = 10  # 0-11, affects expression style
    use_gpu: bool = True
    
    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    debug: bool = False

    # Authentication Configuration
    auth_enabled: bool = True  # Enable/disable authentication
    auth_secret_key: str = ""  # HMAC secret key (generate with: openssl rand -hex 32)
    auth_token_ttl: int = 3600  # Token TTL in seconds (1 hour)
    auth_allowed_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175"  # Comma-separated
    auth_enable_rate_limiting: bool = True
    
    # Audio Configuration (constants, but configurable if needed)
    openai_sample_rate: int = 24000    # OpenAI Realtime API uses 24kHz
    audio2exp_sample_rate: int = 16000 # Audio2Expression model expects 16kHz
    blendshape_fps: int = 30           # Output blendshape frame rate
    audio_chunk_duration: float = 1.0  # 1 second chunks for Audio2Expression processing


class AudioConstants:
    """Audio processing constants derived from settings."""
    
    def __init__(self, settings: Settings):
        self.openai_sample_rate = settings.openai_sample_rate
        self.audio2exp_sample_rate = settings.audio2exp_sample_rate
        self.blendshape_fps = settings.blendshape_fps
        self.audio_chunk_duration = settings.audio_chunk_duration
        
        # Derived values
        self.frame_interval_ms = 1000 / self.blendshape_fps
        self.samples_per_frame = self.openai_sample_rate // self.blendshape_fps
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
