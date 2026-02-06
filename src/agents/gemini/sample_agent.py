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

from core.logger import get_logger
from core.settings import get_settings

from ..base_agent import BaseAgent, ConversationState
from .gemini_settings import get_gemini_settings

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
        # Background tasks
        self._receive_task: asyncio.Task | None = None
        self._session_lock = asyncio.Lock()
        
        # Interruption synchronization
        self._interruption_lock = asyncio.Lock()
        self._interruption_in_progress = False

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

        # 1. Validate API Key
        if not self._gemini_settings.gemini_api_key:
            logger.error("GEMINI_API_KEY not found in settings")
            raise ValueError("GEMINI_API_KEY is missing")

        # Initialize client if needed
        if not self._client:
            self._client = genai.Client(
                api_key=self._gemini_settings.gemini_api_key,
                http_options={"api_version": "v1alpha"}
            )

        # Create connection event to wait for successful connection
        self._connection_ready = asyncio.Event()
        
        # Start the connection loop in background
        self._receive_task = asyncio.create_task(self._run_connection_loop())
        
        # Wait for connection to be established
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
            if not self._connected:
                # If event set but not connected, meant it failed
                raise ConnectionError("Failed to establish connection to Gemini")
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for Gemini connection")
            if self._receive_task:
                self._receive_task.cancel()
            raise ConnectionError("Connection timeout")

    async def _run_connection_loop(self) -> None:
        """
        Background task that maintains the Gemini session.
        Uses `async with` to ensure proper resource management.
        Auto-reconnects if the session closes unexpectedly.
        """
        config = self._build_live_config()
        model_id = self._gemini_settings.gemini_model

        logger.info(f"Starting connection loop for model: {model_id}")

        while True:
            try:
                # Connect using async context manager - keeping it alive for the duration of the session
                logger.info("Initiating connection to Gemini Live API...")
                async with self._client.aio.live.connect(model=model_id, config=config) as session:
                    self._session = session
                    self._connected = True
                    self._connection_ready.set()
                    
                    # [FIX] Reset conversation state on new connection to prevent stale flags
                    self._state.is_responding = False
                    self._state.transcript_buffer = ""
                    self._response_cancelled = False
                    self._current_turn_id = None
                    
                    logger.info("Gemini Live API Connected and Session Active")
                    
                    # Run receive loop inside the context
                    try:
                        while True: # Keep loop running until session ends or error
                            async for response in session.receive():
                                await self._handle_response(response)
                            
                            logger.info("Gemini receive iterator finished. Re-entering receive loop (Session Active).")
                            # Do NOT break. Loop back to call session.receive() again.
                            await asyncio.sleep(0.01) 

                    except asyncio.CancelledError:
                        logger.info("Connection loop cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"Error in receive loop: {e}", exc_info=True)
                        if self._on_error:
                            await self._on_error({"error": str(e)})

            except asyncio.CancelledError:
                logger.info("Gemini connection task cancelled. Stopping reconnect loop.")
                break
            except Exception as e:
                logger.error(f"Connection failed: {e}. Retrying in 2s...", exc_info=True)
                if self._on_error:
                    await self._on_error({"error": f"Connection failed: {e}"})
                
                # Reset connection state before retrying
                self._connected = False
                self._session = None
                self._connection_ready.clear() # Reset event for next attempt? 
                # Actually, connect() waits for this. If it's already set, connect() returns.
                # If we are reconnecting, connect() is already returned. This is fine.
                
                await asyncio.sleep(2) # Backoff before reconnect
            finally:
                # Cleanup for this attempt (or final cleanup if broken loop)
                self._session = None
                self._connected = False
                if not self._receive_task or self._receive_task.cancelled():
                     logger.info("Gemini connection loop cleanup")

        self._connected = False
        self._session = None
        self._connection_ready.set() # Ensure any waiters unblock (though they observe not connected)

    def _build_live_config(self) -> dict | types.LiveConnectConfig:
        """Build the configuration object for Gemini Live."""
        # Build tools list
        tools = []
        if self._gemini_settings.gemini_google_search_grounding:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Configure Live API session
        # Configure Live API session (Using dict to match verified simple_gemini_test.py)
        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": self._gemini_settings.gemini_voice
                    }
                }
            },
            # Configs at root level (verified working)
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "system_instruction": {
                # "parts": [{"text": self._settings.assistant_instructions}]
                # [DEBUG] Disabled to match simple_gemini_test.py for stability check
                "parts": []
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
                    "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
                    "prefix_padding_ms": 100,
                    "silence_duration_ms": 300,
                }
            },
            "generation_config": {
                "thinking_config": {
                     "include_thoughts": False
                }
            }
        }
        
        # [DEBUG] Disable tools to match simple_gemini_test.py
        # if tools:
        #      config["tools"] = tools

        return config

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
                # logger.debug("Response missing server_content")
                return



            # Check for interruption FIRST - this must be handled immediately
            if getattr(server_content, "interrupted", False):
                logger.info("Gemini detected interruption - triggering robust handler")
                # Call robust interrupt handler
                asyncio.create_task(self._handle_interruption_signal())
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
                                # logger.debug(f"Received audio chunk from Gemini ({len(audio_data)} bytes)")
                                # Send raw 24kHz audio directly (Client expects 24kHz)
                                await self._on_audio_delta(audio_data)

                    # Handle text data (IGNORING: This contains "Thinking" process in preview model)
                    text = getattr(part, "text", None)
                    if text:
                        # print(f"DEBUG: Ignored part.text: {text[:50]}...")
                        pass
                    #    self._state.transcript_buffer += text
                    #    if self._on_transcript_delta and not self._response_cancelled:
                    #        await self._on_transcript_delta(text, "assistant", self._current_turn_id, None)

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
            logger.error(f"Error handling Gemini response: {e}", exc_info=True)

    async def _handle_interruption_signal(self) -> None:
        """
        Handle conversation interruption with production-grade locking and error handling.
        """
        # [FAST PATH] Check if already in progress without acquiring lock first
        if self._interruption_in_progress:
            logger.debug(f"Session {self._state.session_id}: Interruption already in progress, ignoring duplicate")
            return

        # [ACQUIRE LOCK] With timeout to prevent deadlocks
        try:
            async def acquire_lock_with_timeout():
                async with self._interruption_lock:
                    return True
            
            await asyncio.wait_for(acquire_lock_with_timeout(), timeout=1.0)
            
            async with self._interruption_lock:
                    # [DOUBLE-CHECK] Inside the lock, verify again
                    if self._interruption_in_progress:
                        return

                    self._interruption_in_progress = True
                    logger.info(f"Session {self._state.session_id}: Starting interruption handling")

                    try:
                        # Send interrupt message to client IMMEDIATELY
                        try:
                            if self._on_interrupted:
                                await self._on_interrupted()
                            logger.info(f"Session {self._state.session_id}: Interruption signal sent to client")
                        except Exception as e:
                            logger.error(f"Session {self._state.session_id}: Failed to send interrupt message: {e}")
                            return

                        # Cancel current response if not already cancelled
                        self.cancel_response()

                        logger.info(f"Session {self._state.session_id}: Interruption handling completed successfully")

                    finally:
                        self._interruption_in_progress = False

        except asyncio.TimeoutError:
            logger.error(f"Session {self._state.session_id}: Interruption handling lock timeout")
            self._interruption_in_progress = False
        except Exception as e:
            logger.error(f"Session {self._state.session_id}: Unexpected error: {e}", exc_info=True)
            self._interruption_in_progress = False

    def cancel_response(self) -> None:
        """
        Explicitly cancel the current response.
        Used by the server when the user interrupts explicitly.
        """
        logger.debug("Cancelling ongoing response (internal state)")
        self._response_cancelled = True
        self._state.is_responding = False
        self._current_turn_id = None
        # Gemini specific: We don't have a client.cancel_response() method,
        # so we rely on ignoring subsequent events via the _response_cancelled flag.

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
            asyncio.create_task(self._handle_interruption_signal())

        logger.debug(f"User text: {text}")
        asyncio.create_task(self._send_text_async(text))

    async def _send_text_async(self, text: str) -> None:
        """Send text message asynchronously."""
        try:
            async with self._session_lock:
                if self._session:
                    await self._session.send(input=text, end_of_turn=True)
        except Exception as e:
            logger.error(f"Error sending text: {e}", exc_info=True)

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

        # Gemini natively expects 16kHz input, but can accept 24kHz via mime_type
        # Our widget sends 24kHz. For simplicity, we send 24kHz and label it.
        # However, to be safe with v1alpha strictness, we might still want to resample input if recognition fails.
        # For now, let's try sending raw 24k and see if Gemini complains. 
        # Actually, simpler: Let's assume input needs to be 16k for best recognition?
        # NO, user wants high freq output. Input is separate.
        # But wait, input logic was also resampling.
        
    def _resample_24k_to_16k(self, audio_bytes: bytes) -> bytes:
        """
        Simple resampler from 24kHz to 16kHz (3:2 ratio).
        Uses simple linear interpolation/decimation for speed.
        """
        # 3 input samples -> 2 output samples
        # Very distinct logic: 24000 / 16000 = 1.5
        # We can just take every 1.5th sample? No, better to use numpy or simple slicing if possible.
        # But we don't assume numpy is available in the minimal env (though it is imported as np).
        # SampleGeminiAgent imports numpy as np at top of file.
        
        try:
             audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
             # Resample: 24000 -> 16000 is factor of 2/3
             # Create indices for 16k based on 24k length
             num_samples = len(audio_np)
             new_num_samples = int(num_samples * 16000 / 24000)
             
             # Linear interpolation
             indices = np.linspace(0, num_samples - 1, new_num_samples)
             resampled_np = np.interp(indices, np.arange(num_samples), audio_np).astype(np.int16)
             
             return resampled_np.tobytes()
        except Exception as e:
             logger.error(f"Error resampling audio: {e}")
             return audio_bytes # Fallback to original

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

        # [STABILITY FIX] Gemini Live API prefers 16kHz input.
        # simple_gemini_test.py sends 16kHz and is stable.
        # Server was sending 24kHz and getting Remote Close.
        # Restoring resampling for INPUT only.
        
        # Resample 24k -> 16k
        audio_16k = self._resample_24k_to_16k(audio_bytes)
        
        asyncio.create_task(self._send_audio_async(audio_16k))

    async def _send_audio_async(self, audio_bytes: bytes) -> None:
        """Send audio data asynchronously."""
        try:
            async with self._session_lock:
                if self._session:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=audio_bytes, mime_type="audio/pcm")
                    )
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)

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

        # Close session - managed by _run_connection_loop context manager on cancellation
        self._session = None

        logger.info("Disconnected from Gemini Live API")
