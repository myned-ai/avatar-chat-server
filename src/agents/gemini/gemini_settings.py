"""
Gemini Agent Configuration

Gemini-specific settings for the sample Gemini agent.
These settings are only loaded when using the Gemini agent.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    gemini_api_version: str = "v1alpha"
    gemini_model: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    gemini_voice: Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"] = "Leda"
    # Thinking budget: 0=disabled, -1=dynamic, 1-32768=fixed token budget
    gemini_thinking_budget: int = -1
    # Enable Google Search grounding for real-time information
    gemini_google_search_grounding: bool = False
    # Proactive audio: model can decide not to respond if content is not relevant
    # DISABLED to prevent early cutoff/interruption of user speech (user report: immediate response)
    gemini_proactive_audio: bool = False
    # Context window compression: enables longer sessions (beyond 15min audio-only limit)
    gemini_context_window_compression: bool = True
    
    # Transcript timing (chars per second)
    # Gemini voice is slightly slower than OpenAI, so we retard the transcript cursor
    # Default global is 16.0. Slower value = truncated text stays longer/appears slower.
    gemini_transcript_speed: float = 15.0
    
    # Audio Input Configuration
    # Sample rate to send to Gemini (default 16000 for best recognition)
    # Reverting to 16000 to match working simple_gemini_test.py
    gemini_input_sample_rate: int = 16000

    # VAD Configuration
    # Sensitivity: "sensitivity_unspecified", "start_sensitivity_low", "start_sensitivity_medium", "start_sensitivity_high"
    # Defaults to UNSPECIFIED (Balanced)
    gemini_vad_start_sensitivity: str = "START_SENSITIVITY_LOW"
    gemini_vad_end_sensitivity: str = "END_SENSITIVITY_LOW"

    # Turn Coverage: "TURN_COVERAGE_UNSPECIFIED", "TURN_INCLUDES_ONLY_ACTIVITY", "TURN_INCLUDES_ALL_INPUT"
    # Default: UNSPECIFIED (Only Activity)
    gemini_turn_coverage: str = "TURN_INCLUDES_ALL_INPUT"


@lru_cache
def get_gemini_settings() -> GeminiSettings:
    """
    Get cached Gemini settings.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return GeminiSettings()
