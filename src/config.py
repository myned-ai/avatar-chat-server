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
    
    # OpenAI Configuration (for sample agent)
    openai_api_key: str
    openai_model: str = "gpt-4o-realtime-preview"
    openai_voice: Literal[
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"
    ] = "alloy"
    
    # Gemini Configuration (for sample agent)
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

    # Assistant Configuration
    assistant_instructions: str = (
        "You are a helpful and friendly AI assistant. Be concise in your responses."
    )
    
    # Wav2Arkit Model Configuration (ONNX CPU-only)
    onnx_model_path: str = "./pretrained_models/wav2arkit_cpu.onnx"  # ONNX model (CPU optimized)
    
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
    
    # Agent Configuration
    agent_type: str = "sample_openai"  # "sample_openai", "sample_gemini", "remote"
    agent_url: Optional[str] = None  # URL for remote agent (e.g., "ws://agent-service:8080/ws")
    
    # Audio Configuration (constants, but configurable if needed)
    # Note: Widget sends 24kHz audio. OpenAI uses 24kHz, Gemini expects 16kHz (resampled internally)
    openai_sample_rate: int = 24000     # OpenAI Realtime API uses 24kHz (also widget format)
    wav2arkit_sample_rate: int = 16000  # Wav2Arkit model expects 16kHz
    blendshape_fps: int = 30            # Output blendshape frame rate
    audio_chunk_duration: float = 0.5   # 0.5 second chunks for Wav2Arkit processing


class AudioConstants:
    """Audio processing constants derived from settings."""

    def __init__(self, settings: Settings):
        self.openai_sample_rate = settings.openai_sample_rate
        self.wav2arkit_sample_rate = settings.wav2arkit_sample_rate
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
