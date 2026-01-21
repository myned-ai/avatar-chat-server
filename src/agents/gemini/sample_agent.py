"""
Sample Gemini Agent

Monolithic implementation of the agent interface using Google Gemini Live API.
This is the sample agent that ships with the chat-server.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

import numpy as np

# Import Gemini SDK
from google import genai
from google.genai import types

from core.config import get_settings
from core.logger import get_logger

from ..base_agent import BaseAgent, ConversationState
from .config import get_gemini_settings

logger = get_logger(__name__)


class SampleGeminiAgent(BaseAgent):
    """
    Sample agent implementation using Google Gemini Live API.

    Wraps the Gemini Live API client to provide a clean agent interface
    for voice-based conversation with the AI assistant.
    """

    def __init__(self):
        """
        Initialize the Gemini agent.

        Loads Gemini-specific settings from environment variables.
        """
        self._settings = get_settings()  # Core settings (assistant_instructions, debug)
        self._gemini_settings = get_gemini_settings()  # Gemini-specific settings
        self._client: genai.Client | None = None
        self._session: Any | None = None
        self._connected = False
        self._state = ConversationState()

        # Event callbacks
        self._on_audio_delta: Callable | None = None
        self._on_transcript_delta: Callable | None = None
        self._on_response_start: Callable | None = None
        self._on_response_end: Callable | None = None
        self._on_user_transcript: Callable | None = None
        self._on_interrupted: Callable | None = None
        self._on_error: Callable | None = None

        # Interruption handling
        self._response_cancelled = False
        self._current_turn_id: str | None = None

        # Background tasks
        self._receive_task: asyncio.Task | None = None
        self._session_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if connected to Gemini Live API."""
        return self._connected

    @property
    def state(self) -> ConversationState:
        """Get current conversation state."""
        return self._state

    def set_event_handlers(
        self,
        on_audio_delta: Callable | None = None,
        on_transcript_delta: Callable | None = None,
        on_response_start: Callable | None = None,
        on_response_end: Callable | None = None,
        on_user_transcript: Callable | None = None,
        on_interrupted: Callable | None = None,
        on_error: Callable | None = None,
    ) -> None:
        """
        Set event handler callbacks.

        Args:
            on_audio_delta: Called with audio bytes when agent responds
            on_transcript_delta: Called with text during streaming response
            on_response_start: Called when agent starts responding
            on_response_end: Called when agent finishes responding, with full transcript
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
        """Connect to Gemini Live API."""
        if self._connected:
            return

        logger.info("Connecting to Gemini Live API")

        try:
            # Initialize client with API key
            self._client = genai.Client(api_key=self._gemini_settings.gemini_api_key)

            # Build tools list
            tools = []
            if self._gemini_settings.gemini_google_search_grounding:
                tools.append(types.Tool(google_search=types.GoogleSearch()))
                logger.info("Google Search grounding enabled")

            # Configure thinking mode (for models that support it)
            thinking_config = None
            if self._gemini_settings.gemini_thinking_budget != 0:
                thinking_config = types.ThinkingConfig(thinking_budget=self._gemini_settings.gemini_thinking_budget)
                logger.info(f"Thinking mode enabled (budget: {self._gemini_settings.gemini_thinking_budget})")

            # Configure context window compression for longer sessions
            context_window_compression = None
            if self._gemini_settings.gemini_context_window_compression:
                context_window_compression = types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow())
                logger.info("Context window compression enabled (longer sessions)")

            # Configure proactivity
            proactivity = None
            if self._gemini_settings.gemini_proactive_audio:
                proactivity = types.ProactivityConfig(proactive_audio=True)
                logger.info("Proactive audio enabled")

            # Configure Live API session
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._gemini_settings.gemini_voice)
                    )
                ),
                system_instruction=types.Content(parts=[types.Part(text=self._settings.assistant_instructions)]),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=False,
                        start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                        prefix_padding_ms=100,
                        silence_duration_ms=300,
                    )
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                tools=tools if tools else None,
                generation_config=types.GenerationConfig(thinking_config=thinking_config) if thinking_config else None,
                context_window_compression=context_window_compression,
                proactivity=proactivity,
            )

            # Connect using async context - we manage the session manually
            self._session = await self._client.aio.live.connect(
                model=self._gemini_settings.gemini_model,
                config=config,
            ).__aenter__()

            self._connected = True

            # Start listening for responses in background
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info(
                f"Connected to Gemini Live API (model: {self._gemini_settings.gemini_model}, voice: {self._gemini_settings.gemini_voice})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live API: {e}")
            self._connected = False
            raise

    async def _receive_loop(self) -> None:
        """Background loop to receive and process responses from Gemini."""
        try:
            while self._connected and self._session:
                try:
                    async for response in self._session.receive():
                        await self._handle_response(response)
                except Exception as e:
                    if self._connected:
                        logger.error(f"Error receiving from Gemini: {e}")
                        if self._on_error:
                            await self._on_error({"error": str(e)})
                    break
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            if self._on_error:
                await self._on_error({"error": str(e)})

    async def _handle_response(self, response: Any) -> None:
        """Handle a response from Gemini Live API."""
        try:
            # Skip processing if already cancelled (from previous interruption)
            if self._response_cancelled:
                # But still check for new response start to reset the flag
                server_content = getattr(response, "server_content", None)
                if server_content:
                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn and not self._state.is_responding:
                        # New response starting, reset cancelled flag
                        self._response_cancelled = False
                    else:
                        return  # Skip processing cancelled response data

            server_content = getattr(response, "server_content", None)
            if not server_content:
                return

            # Check for interruption FIRST - this must be handled immediately
            if getattr(server_content, "interrupted", False):
                logger.info("Gemini detected interruption - stopping immediately")
                # Set cancelled flag FIRST to stop any pending processing
                self._response_cancelled = True
                self._state.is_responding = False
                self._current_turn_id = None
                # Call interrupt handler immediately
                if self._on_interrupted:
                    await self._on_interrupted()
                return

            # Handle model turn (audio/text response)
            model_turn = getattr(server_content, "model_turn", None)
            if model_turn:
                # Signal response start if this is a new turn
                if not self._state.is_responding:
                    self._state.session_id = f"session_{int(time.time() * 1000)}"
                    self._current_turn_id = self._state.session_id
                    self._state.is_responding = True
                    self._state.transcript_buffer = ""
                    self._response_cancelled = False
                    if self._on_response_start:
                        await self._on_response_start(self._state.session_id)

                # Process parts (audio and text)
                parts = getattr(model_turn, "parts", [])
                for part in parts:
                    if self._response_cancelled:
                        break

                    # Handle audio data
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data:
                        audio_data = getattr(inline_data, "data", None)
                        if audio_data and isinstance(audio_data, bytes):
                            if self._on_audio_delta and not self._response_cancelled:
                                await self._on_audio_delta(audio_data)

                    # Handle text data
                    text = getattr(part, "text", None)
                    if text:
                        self._state.transcript_buffer += text
                        if self._on_transcript_delta and not self._response_cancelled:
                            # Use current turn id as item id so router can correlate events
                            await self._on_transcript_delta(text, "assistant", self._current_turn_id, None)

            # Handle output transcription (assistant's speech as text)
            output_transcription = getattr(server_content, "output_transcription", None)
            if output_transcription:
                text = getattr(output_transcription, "text", "")
                if text:
                    self._state.transcript_buffer += text
                    if self._on_transcript_delta and not self._response_cancelled:
                        await self._on_transcript_delta(text, "assistant", self._current_turn_id, None)

            # Handle input transcription (user's speech as text)
            input_transcription = getattr(server_content, "input_transcription", None)
            if input_transcription:
                text = getattr(input_transcription, "text", "")
                if text and self._on_user_transcript:
                    logger.debug(f"User: {text}")
                    await self._on_user_transcript(text)

            # Check for turn completion
            turn_complete = getattr(server_content, "turn_complete", False)
            if turn_complete and self._state.is_responding:
                transcript = self._state.transcript_buffer
                self._state.is_responding = False
                self._current_turn_id = None
                if self._on_response_end:
                    await self._on_response_end(transcript, self._current_turn_id)
                logger.debug(f"Assistant: {transcript}")

        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}")

    def send_text_message(self, text: str) -> None:
        """
        Send a text message to the agent.

        Args:
            text: Text message content
        """
        if not self._connected or not self._session:
            return

        # Cancel any ongoing response before sending new message
        if self._state.is_responding:
            logger.debug("Cancelling ongoing response due to text message")
            self._response_cancelled = True
            self._state.is_responding = False
            self._current_turn_id = None
            if self._on_interrupted:
                asyncio.create_task(self._on_interrupted())

        logger.debug(f"User text: {text}")
        asyncio.create_task(self._send_text_async(text))

    async def _send_text_async(self, text: str) -> None:
        """Send text message asynchronously."""
        try:
            async with self._session_lock:
                if self._session:
                    await self._session.send(input=text, end_of_turn=True)
        except Exception as e:
            logger.error(f"Error sending text: {e}")

    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Append audio to the input buffer.

        The widget sends 24kHz audio (OpenAI format), but Gemini expects 16kHz.
        We resample here to maintain compatibility with the existing widget.

        Args:
            audio_bytes: PCM16 audio bytes (24kHz from widget)
        """
        if not self._connected or not self._session:
            return

        # Resample 24kHz -> 16kHz (widget sends OpenAI format, Gemini needs 16kHz)
        audio_16k = self._resample_24k_to_16k(audio_bytes)
        asyncio.create_task(self._send_audio_async(audio_16k))

    def _resample_24k_to_16k(self, audio_bytes: bytes) -> bytes:
        """Resample PCM16 audio from 24kHz to 16kHz using linear interpolation."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # 16000/24000 = 2/3 ratio
        out_len = int(len(audio) * 2 / 3)
        indices = np.linspace(0, len(audio) - 1, out_len)
        resampled = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))
        return resampled.astype(np.int16).tobytes()

    async def _send_audio_async(self, audio_bytes: bytes) -> None:
        """Send audio data asynchronously."""
        try:
            async with self._session_lock:
                if self._session:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                    )
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Gemini Live API."""
        if not self._connected:
            return

        logger.info("Disconnecting from Gemini Live API")
        self._connected = False

        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close session
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing Gemini session: {e}")
            self._session = None

        logger.info("Disconnected from Gemini Live API")
