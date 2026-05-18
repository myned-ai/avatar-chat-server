"""
Application Configuration

Centralized configuration using Pydantic Settings for type-safe
environment variable management with validation.

This module contains ONLY vendor-agnostic settings.
Vendor-specific settings (OpenAI, Gemini) are managed by their respective agents.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory containing this file, then go up to avatar_chat_server/
_CONFIG_DIR = Path(__file__).parent.parent.parent


def _resolve_models_path(p: str) -> Path:
    """Resolve a model-bundle path against the repo root if relative."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (_CONFIG_DIR / pp).resolve()


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
    assistant_instructions: str = "You are a helpful and friendly AI assistant. Be concise in your responses."

    # Audio2Face-3D Model Bundles (CPU-only ONNX, downloaded from HuggingFace).
    # `a2f_model_dir` = nvidia/Audio2Face-3D-v2.3.1-James (public)
    # `a2e_model_dir` = nvidia/Audio2Emotion-v2.2 (gated — needs HF_TOKEN once).
    # Paths are resolved relative to the repo root (parent of src/) so they
    # work whether you launch the server from `src/` or the repo root.
    # Set a2e_model_dir to "" to disable A2E and run with neutral emotion only.
    a2f_model_dir: str = "src/pretrained_models/a2f"
    a2e_model_dir: str = "src/pretrained_models/a2e"

    @property
    def a2f_model_dir_resolved(self) -> Path:
        return _resolve_models_path(self.a2f_model_dir)

    @property
    def a2e_model_dir_resolved(self) -> Path | None:
        if not self.a2e_model_dir:
            return None
        return _resolve_models_path(self.a2e_model_dir)

    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    use_ssl: bool = False
    debug: bool = False
    debug_audio_capture: bool = False  # Save incoming audio to files for debugging

    # Knowledge Base Configuration
    # Can be a local file path (e.g. "data/knowledge.md") or a URL
    knowledge_base_source: str | None = None

    # Authentication Configuration
    auth_enabled: bool = False
    auth_secret_key: str = ""
    auth_token_ttl: int = 3600
    auth_allowed_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175"
    auth_enable_rate_limiting: bool = True

    # Agent Configuration
    agent_type: str = "sample_openai"  # "sample_openai", "sample_gemini", "remote"
    agent_url: str | None = None  # URL for remote agent (e.g., "ws://agent-service:8080/ws")

    # Audio Configuration (vendor-agnostic)
    # Note: Widget sends 24kHz audio. This is used for Wav2Arkit processing.
    input_sample_rate: int = 24000  # Input audio sample rate (widget format)
    output_sample_rate: int = 24000  # Output audio sample rate (for playback and lip-sync)
    wav2arkit_sample_rate: int = 16000  # Wav2Arkit model expects 16kHz
    blendshape_fps: int = 30  # Output blendshape frame rate
    audio_chunk_duration: float = 0.5  # 0.5 second chunks for Wav2Arkit processing

    # Transcript timing estimation
    # Used to calculate text offsets for transcript deltas
    # Typical values: slow=12, normal=16, fast=20 chars/sec
    transcript_chars_per_second: float = 16.0

    @property
    def frame_interval_ms(self) -> float:
        return 1000 / self.blendshape_fps

    @property
    def samples_per_frame(self) -> int:
        return self.input_sample_rate // self.blendshape_fps

    @property
    def bytes_per_frame(self) -> int:
        return self.samples_per_frame * 2  # PCM16 = 2 bytes


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once
    and reused throughout the application lifecycle.
    """
    return Settings()


def get_allowed_origins() -> list[str]:
    """Parse allowed origins from comma-separated string."""
    settings = get_settings()
    if not settings.auth_allowed_origins:
        return []
    return [origin.strip() for origin in settings.auth_allowed_origins.split(",") if origin.strip()]
