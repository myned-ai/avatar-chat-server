"""
Abstract Agent Interface

Defines the interface for conversational AI agents.
Supports both local (in-process) and remote (inter-container) implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ConversationState:
    """State tracking for an active conversation."""

    session_id: str | None = None
    is_responding: bool = False
    transcript_buffer: str = ""
    audio_done: bool = False  # Track if response audio is complete


class BaseAgent(ABC):
    """
    Abstract base class for conversational AI agents.

    Agents handle real-time voice/text interactions and provide
    event-driven callbacks for audio, transcripts, and state changes.
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the agent service."""
        pass

    @property
    def input_sample_rate(self) -> int:
        """
        Preferred input sample rate for this agent.
        Defaults to 24000 (OpenAI standard).
        """
        return 24000

    @property
    def output_sample_rate(self) -> int:
        """
        Preferred output sample rate for this agent.
        Defaults to 24000 (OpenAI standard).
        """
        return 24000

    @property
    @abstractmethod
    def transcript_speed(self) -> float:
        """
        Transcript character emission speed (chars/sec).
        Used by the client to smooth out text rendering.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> ConversationState:
        """Get current conversation state."""
        pass

    @abstractmethod
    def set_event_handlers(
        self,
        on_audio_delta: Callable[[bytes], Awaitable[None]] | None = None,
        on_transcript_delta: Callable[
            [str, str, str | None, str | None], Awaitable[None]
        ]
        | None = None,
        on_response_start: Callable[[str], Awaitable[None]] | None = None,
        on_response_end: Callable[[str, str | None], Awaitable[None]] | None = None,
        on_user_transcript: Callable[[str, str], Awaitable[None]] | None = None,
        on_interrupted: Callable[[], Awaitable[None]] | None = None,
        on_error: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        """
        Set event handler callbacks.

        Args:
            on_audio_delta: Called with audio bytes when agent responds
            on_transcript_delta: Called with text during streaming response
            on_response_start: Called when agent starts responding (with session_id)
            on_response_end: Called when agent finishes responding (with full transcript)
            on_user_transcript: Called with transcribed user speech
            on_interrupted: Called when user interrupts
            on_error: Called on errors
        """
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the agent service."""
        pass

    @abstractmethod
    def send_text_message(self, text: str) -> None:
        """
        Send a text message to the agent.

        Args:
            text: Text message content
        """
        pass

    def cancel_response(self) -> None:
        """
        Cancel the current response generation.
        Optional implementation - base does nothing.
        """
        # Default implementation does nothing - subclasses may override
        return

    @abstractmethod
    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Append audio to the input buffer.

        Args:
            audio_bytes: PCM16 audio bytes
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the agent service."""
        pass
