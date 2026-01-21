"""
Sample OpenAI Agent

Monolithic implementation of the agent interface using OpenAI Realtime API.
This is the sample agent that ships with the chat-server.
"""

import asyncio
import time
from typing import Callable, Dict, Optional, Any

from ..base_agent import BaseAgent, ConversationState
from .config import get_openai_settings
from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class SampleOpenAIAgent(BaseAgent):
    """
    Sample agent implementation using OpenAI Realtime API.

    Wraps the RealtimeClient to provide a clean agent interface
    for voice-based conversation with the AI assistant.
    """

    def __init__(self):
        """
        Initialize the OpenAI agent.

        Loads OpenAI-specific settings from environment variables.
        """
        self._settings = get_settings()  # Core settings (assistant_instructions, debug)
        self._openai_settings = get_openai_settings()  # OpenAI-specific settings
        self._client: Optional[Any] = None
        self._connected = False
        self._state = ConversationState()

        # Event callbacks (set by the router)
        self._on_audio_delta: Optional[Callable] = None
        self._on_transcript_delta: Optional[Callable] = None
        self._on_response_start: Optional[Callable] = None
        self._on_response_end: Optional[Callable] = None
        self._on_user_transcript: Optional[Callable] = None
        self._on_interrupted: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Current response item ID for cancellation
        self._current_item_id: Optional[str] = None
        self._response_cancelled: bool = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to OpenAI Realtime API."""
        return self._connected

    @property
    def state(self) -> ConversationState:
        """Get current conversation state."""
        return self._state

    def set_event_handlers(
        self,
        on_audio_delta: Optional[Callable] = None,
        on_transcript_delta: Optional[Callable] = None,
        on_response_start: Optional[Callable] = None,
        on_response_end: Optional[Callable] = None,
        on_user_transcript: Optional[Callable] = None,
        on_interrupted: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """
        Set event handler callbacks.

        Args:
            on_audio_delta: Called with audio bytes when AI responds
            on_transcript_delta: Called with text during streaming response
            on_response_start: Called when AI starts responding
            on_response_end: Called when AI finishes responding, with full transcript
            on_user_transcript: Called with transcribed user speech
            on_interrupted: Called when user interrupts
            on_error: Called on errors
        """
        self._on_audio_delta = on_audio_delta
        self._on_transcript_delta = on_transcript_delta
        self._on_response_start = on_response_start
        self._on_response_end = on_response_end
        self._on_user_transcript = on_user_transcript
        self._on_interrupted = on_interrupted
        self._on_error = on_error

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        if self._connected:
            return

        logger.info("Connecting to OpenAI Realtime API")

        from .realtime_client import RealtimeClient

        self._client = RealtimeClient(
            api_key=self._openai_settings.openai_api_key,
            model=self._openai_settings.openai_model,
            debug=False,  # Disable RealtimeClient debug logging to avoid spamming with base64 audio
        )

        # Setup event handlers
        self._setup_events()

        # Connect and wait for session
        await self._client.connect()
        await self._client.wait_for_session_created()

        # Configure session
        self._client.update_session(
            instructions=self._settings.assistant_instructions,
            voice=self._openai_settings.openai_voice,
            modalities=["text", "audio"],
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription={"model": "whisper-1"},
            turn_detection={
                "type": "server_vad",
                "threshold": 0.2,
                "prefix_padding_ms": 100,
                "silence_duration_ms": 300,
            },
        )

        self._connected = True
        logger.info(f"Connected to OpenAI Realtime API (model: {self._openai_settings.openai_model}, voice: {self._openai_settings.openai_voice})")

    def _setup_events(self) -> None:
        """Setup event handlers for the Realtime client."""
        client = self._client

        # Handle conversation updates
        client.on("conversation.updated", self._handle_conversation_updated)
        client.on("conversation.item.appended", self._handle_item_appended)
        client.on("conversation.item.completed", self._handle_item_completed)
        client.on("conversation.item.input_transcription.completed", self._handle_user_transcript)
        client.on("conversation.interrupted", self._handle_interrupted)

        # Handle response.audio.done event (signals all audio has been sent)
        client.realtime.on("server.response.audio.done", self._handle_audio_done)

        # Handle errors
        client.on("error", self._handle_error)

        # Debug logging
        if self._settings.debug:
            client.on("realtime.event", self._handle_debug_event)

    def _handle_debug_event(self, event: Dict) -> None:
        """Debug handler for all realtime events."""
        source = event.get("source", "unknown")
        evt = event.get("event", {})
        event_type = evt.get("type", "unknown")

        skip_events = [
            "input_audio_buffer.append",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "response.audio.delta",
            "response.audio_transcript.delta",
        ]

        if source == "server" and event_type not in skip_events:
            logger.debug(f"[{source}] {event_type}")

    def _handle_conversation_updated(self, event: Dict) -> None:
        """Handle conversation updates (delta events)."""
        # Early exit if cancelled - check FIRST before any processing
        if self._response_cancelled:
            return

        delta = event.get("delta", {})

        # Stream audio delta - check cancelled flag again right before processing
        if delta and "audio" in delta:
            if self._response_cancelled:
                return

            audio_data = delta["audio"]
            if isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                audio_bytes = audio_data.tobytes()

            if self._on_audio_delta and not self._response_cancelled:
                asyncio.create_task(self._on_audio_delta(audio_bytes))

        # Stream transcript delta
        if delta and "transcript" in delta:
            if self._response_cancelled:
                return

            transcript_delta = delta["transcript"]

            # Get role and item ids from the event if available
            item = event.get("item", {})
            role = item.get("role", "assistant")
            item_id = item.get("id")
            previous_item_id = item.get("previous_item_id") or event.get("previous_item_id")

            self._state.transcript_buffer += transcript_delta

            if self._on_transcript_delta and not self._response_cancelled:
                asyncio.create_task(
                    self._on_transcript_delta(transcript_delta, role, item_id, previous_item_id)
                )

    def _handle_audio_done(self, event: Dict) -> None:
        """Handle response.audio.done event - all audio has been sent."""
        logger.debug("Audio done received")
        self._state.audio_done = True

    def _handle_item_appended(self, event: Dict) -> None:
        """Handle new conversation items."""
        item = event.get("item", {})

        if item.get("role") == "assistant":
            self._current_item_id = item.get("id")
            self._response_cancelled = False  # Reset cancel flag for new response
            self._state.session_id = f"session_{int(time.time() * 1000)}"
            self._state.is_responding = True
            self._state.transcript_buffer = ""
            self._state.audio_done = False  # Reset audio done flag

            if self._on_response_start:
                asyncio.create_task(self._on_response_start(self._state.session_id))

    async def _wait_for_audio_done(self, timeout: float = 3.0) -> bool:
        """Wait for audio_done flag to be set, with timeout."""
        start = asyncio.get_event_loop().time()
        while not self._state.audio_done:
            if asyncio.get_event_loop().time() - start > timeout:
                logger.warning("Timeout waiting for audio_done")
                return False
            await asyncio.sleep(0.05)
        return True

    def _handle_item_completed(self, event: Dict) -> None:
        """Handle completed conversation items."""
        item = event.get("item", {})
        role = item.get("role", "")

        if role == "assistant":

            # Wait for audio_done before triggering response_end
            async def wait_and_complete():
                await self._wait_for_audio_done(timeout=3.0)

                # Don't send response_end if:
                # 1. We were interrupted (response_cancelled is True)
                if self._response_cancelled:
                    logger.debug("Skipping response_end - was interrupted")
                    return

                transcript = item.get("formatted", {}).get("transcript", "")
                self._state.is_responding = False
                self._current_item_id = None

                if transcript:
                    logger.debug(f"Assistant: {transcript}")

                if self._on_response_end:
                    await self._on_response_end(transcript, item.get("id"))

            asyncio.create_task(wait_and_complete())

    def _handle_user_transcript(self, event: Dict) -> None:
        """Handle user's transcribed speech."""
        transcript = event.get("transcript", "")

        # Get role from the item
        item = event.get("item", {})
        role = item.get("role", "user")

        if transcript:
            logger.debug(f"{role.capitalize()}: {transcript}")
            # Note: Response cancellation is now handled in _handle_interrupted
            # which fires on speech_started (before transcript is available)
            if self._on_user_transcript:
                asyncio.create_task(self._on_user_transcript(transcript, role))

    def _handle_interrupted(self, event: Dict) -> None:
        """Handle conversation interruption."""
        logger.info("Conversation interrupted - stopping audio processing immediately")

        # Set cancelled flag IMMEDIATELY to stop processing any pending audio deltas
        # This MUST be set before anything else to ensure no more audio is processed
        self._response_cancelled = True
        self._state.is_responding = False
        self._state.audio_done = True  # Mark audio as done to prevent waiting
        self._current_item_id = None

        # Call interrupt handler synchronously-ish (as a high-priority task)
        # This ensures the ChatConnectionManager receives the interrupt ASAP
        if self._on_interrupted:
            # Create task but also try to run it immediately
            task = asyncio.create_task(self._on_interrupted())
            # Log for debugging
            logger.debug(f"Interrupt handler task created: {task}")

    def _handle_error(self, error: Dict) -> None:
        """Handle errors from the Realtime API."""
        logger.error(f"Realtime API error: {error}")

        if self._on_error:
            asyncio.create_task(self._on_error(error))

    def send_text_message(self, text: str) -> None:
        """
        Send a text message to the assistant.

        Args:
            text: Text message content
        """
        if not self._connected or not self._client:
            return

        # Cancel any ongoing response before sending new message (text-based interruption)
        if self._state.is_responding:
            logger.debug("Cancelling ongoing response due to text message")
            self._client.cancel_response()
            self._response_cancelled = True
            self._state.is_responding = False
            self._current_item_id = None
            if self._on_interrupted:
                asyncio.create_task(self._on_interrupted())

        logger.debug(f"User text: {text}")
        self._client.send_user_message_content([
            {"type": "input_text", "text": text}
        ])

    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Append audio to the input buffer.

        Args:
            audio_bytes: PCM16 audio bytes
        """
        if not self._connected or not self._client:
            return

        self._client.append_input_audio(audio_bytes)

    async def disconnect(self) -> None:
        """Disconnect from OpenAI Realtime API."""
        if self._client and self._connected:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error requesting client disconnect: {e}")

            # Wait briefly for underlying realtime websocket to close
            try:
                timeout = 3.0
                poll_interval = 0.05
                waited = 0.0
                realtime_api = getattr(self._client, 'realtime', None)
                while realtime_api and getattr(realtime_api, 'is_connected', lambda: False)() and waited < timeout:
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                if realtime_api and getattr(realtime_api, 'is_connected', lambda: False)():
                    logger.warning('Realtime API did not close within timeout')
            except Exception as e:
                logger.warning(f"Error while waiting for realtime disconnect: {e}")

            self._connected = False
            logger.info("Disconnected from OpenAI Realtime API")
