"""
Sample Gemini Agent

Monolithic implementation of the agent interface using Google Gemini Live API.
This is the sample agent that ships with the chat-server.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

# Import Gemini SDK
from google.genai import Client, types

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
        self._client: Client | None = None
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
        
        # Interruption synchronization
        self._interruption_lock = asyncio.Lock()
        self._interruption_in_progress = False
        self._current_user_transcript: str = "" # Buffer for fragmented user input
        
        # Dynamic VAD Settings (Initialized from config, can be updated)
        self._vad_start_sensitivity = self._gemini_settings.gemini_vad_start_sensitivity
        self._vad_end_sensitivity = self._gemini_settings.gemini_vad_end_sensitivity
        self._reconnect_requested = False

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

    @property
    def input_sample_rate(self) -> int:
        # Use configured input rate (now 16000 to match working simple test)
        return self._gemini_settings.gemini_input_sample_rate

    @property
    def output_sample_rate(self) -> int:
        """Get negotiated output sample rate."""
        return self._gemini_settings.gemini_output_sample_rate

    @property
    def transcript_speed(self) -> float:
        """Get transcript speed (chars/sec) for this agent."""
        return self._gemini_settings.gemini_transcript_speed

    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Append audio to the input buffer.
        
        Args:
            audio_bytes: PCM16 audio bytes
        """
        if not self._connected or not self._session:
            print(f"[Agent] Not connected, dropping {len(audio_bytes)} bytes")
            return

       
        # Check if resampling is needed
        # We assume input is matching configured input_sample_rate (negotiated by ChatSession)
        # Gemini expects 16kHz usually.
        # If we are already sending 16k, pass through.
        
        # Note: If we need resampling in the future, we should implement a generic _resample(bytes, src, dst)
        # For now, since we aligned everything to 16k, we pass through.
        target_rate = 16000 # Gemini native
        current_rate = self.input_sample_rate
        
        if current_rate != target_rate:
             # Implement resampling if needed, but for now we expect 16k=16k
             pass
        
        asyncio.create_task(self._send_audio_async(audio_bytes))

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
            # Use v1alpha to match working simple_gemini_test.py
            self._client = Client(
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
        model_id = self._gemini_settings.gemini_model

        logger.info(f"Starting connection loop for model: {model_id}")

        while True:
            try:
                # [FIX] Reset reconnect request flag at start of loop to prevent infinite cycling
                # This ensures that a previous request is cleared and only new requests trigger break
                self._reconnect_requested = False
                
                # [FIX] Build config INSIDE loop to apply dynamic changes (e.g. VAD)
                config = self._build_live_config()
                
                # Connect using async context manager - keeping it alive for the duration of the session
                logger.info("Initiating connection to Gemini Live API...")
                async with self._client.aio.live.connect(model=model_id, config=config) as session: # type: ignore
                    self._session = session
                    self._connected = True
                    self._connection_ready.set()
                    
                    # Reset conversation state on new connection to prevent stale flags
                    self._state.is_responding = False
                    self._state.transcript_buffer = ""
                    self._response_cancelled = False
                    self._current_turn_id = None
                    
                    logger.info("Gemini Live API Connected and Session Active")
                    
                    # Run receive loop inside the context
                    try:
                        while not self._reconnect_requested: # Keep loop running until session ends or error or reconnect requested
                            async for response in session.receive():
                                await self._handle_response(response)
                                if self._reconnect_requested:
                                    logger.info("Reconnect requested - breaking receive loop")
                                    break
                            
                            if self._reconnect_requested:
                                break

                            logger.info("Gemini receive iterator finished. Re-entering receive loop (Session Active).")
                            # Do NOT break. Loop back to call session.receive() again.
                            await asyncio.sleep(0.01)
                        
                        if self._reconnect_requested:
                           logger.info("Closing session for reconnect...")
                           # Async context exit will close session
 

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
                # [CRITICAL] Check for Auth errors (401) to prevent infinite loops
                error_str = str(e)
                if "401" in error_str or "Unauthorized" in error_str or "Unauthenticated" in error_str:
                    logger.critical(f"Authentication failed: {e}. Stopping connection loop to prevent infinite retries.")
                    if self._on_error:
                        await self._on_error({"error": f"Authentication failed: {e}. Please check your API key."})
                    break

                logger.error(f"Connection failed: {e}. Retrying in 2s...", exc_info=True)
                if self._on_error:
                    await self._on_error({"error": f"Connection failed: {e}"})
                
                # Reset connection state before retrying
                self._connected = False
                self._session = None
                self._connection_ready.clear()
                
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
        
        if self._reconnect_requested:
             # Reset flag if we exited loop due to request (loop handles retry logic usually)
             self._reconnect_requested = False


    def _build_live_config(self) -> dict:
        """
        Builds the LiveConnectConfig dictionary for the session.
        Simplified to match the working pattern in simple_gemini_test.py.
        """
        
        # Determine audio format
        # Gemini usually expects 16kHz or 24kHz.
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
            # [DEBUG] Disabled to match simple_gemini_test.py for stability check
            "system_instruction": {
                "parts": []
            },
            
            # [CRITICAL] Commenting out realtime_input_config to MATCH simple_gemini_test.py EXACTLY
            # This relies on server-side defaults which are confirmed working.
            # "realtime_input_config": {
            #     "automatic_activity_detection": {
            #         "start_of_speech_sensitivity": self._vad_start_sensitivity,
            #         "end_of_speech_sensitivity": self._vad_end_sensitivity,
            #     },
            #     "turn_coverage": self._gemini_settings.gemini_turn_coverage,
            # },
            
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

    async def _handle_response(self, response: types.LiveServerMessage) -> None:
        """Handle a response from Gemini Live API."""
        # try:
        #      # [DEBUG] Log raw response structure to trace audio flow
        #      # logger.debug(f"[GEMINI RX] Response keys: {[k for k in dir(response) if not k.startswith('_')]}")
        #      pass
        # except:
        #      pass

        try:
            server_content = response.server_content
            if not server_content:
                return

            # [ALWAYS] Handle input transcription (user's speech)
            # This must be processed regardless of agent state (responding, cancelled, etc.)
            if server_content.input_transcription:
                text = server_content.input_transcription.text
                if text:
                     # Buffer the text
                     self._current_user_transcript += text

            # [ALWAYS] Check for interruption
            if server_content.interrupted:
                logger.info("Gemini detected interruption - canceling immediately")
                
                # Update state SYNCHRONOUSLY
                self._response_cancelled = True
                self._state.is_responding = False
                self._current_turn_id = None
                
                asyncio.create_task(self._handle_interruption_signal())
                return

            # [CONDITIONAL] Handle Agent Response (Audio/Text)
            # If cancelled, we only proceed if this packet marks the start of a NEW turn.
            if self._response_cancelled:
                # [DEBUG] Log packet in cancelled state
                mt = server_content.model_turn
                ot = server_content.output_transcription
                tc = server_content.turn_complete
                logger.info(f"CANCELLED STATE: Ignoring packet. MT={bool(mt)}, OT={bool(ot)}, TC={tc}")

                # Check for new response start (Audio OR Text)
                if (mt or ot) and not self._state.is_responding:
                     # New response starting, reset cancelled flag
                     self._response_cancelled = False
                else:
                    return  # Skip processing cancelled response data



            # Handle model turn (audio/text response)
            model_turn = server_content.model_turn
            output_transcription = server_content.output_transcription
            
            # Check for turn start signal (from either audio OR text)
            # If we are not responding, and we receive content, we must start a new turn.
            # This handles the case where text arrives before audio after an interruption.
            if (model_turn or output_transcription) and not self._state.is_responding:
                # [USER TURN COMPLETE]
                # Since the model is responding, the user's turn is effectively over.
                # Send the accumulated user transcript as "final" to close the user bubble.
                if self._current_user_transcript and self._on_user_transcript:
                        logger.debug(f"Finalizing User Transcript: {self._current_user_transcript}")
                        await self._on_user_transcript(self._current_user_transcript)
                        self._current_user_transcript = ""

                self._state.session_id = f"session_{int(time.time() * 1000)}"
                self._current_turn_id = self._state.session_id
                self._state.is_responding = True
                self._state.transcript_buffer = ""
                self._response_cancelled = False
                if self._on_response_start:
                    await self._on_response_start(self._state.session_id)

            if model_turn:
                # Process parts (audio and text)
                parts = model_turn.parts or []
                for part in parts:
                    if self._response_cancelled:
                        break

                    # Handle audio data
                    inline_data = part.inline_data
                    if inline_data:
                        audio_data = inline_data.data
                        if audio_data:
                            if self._on_audio_delta and not self._response_cancelled:
                                # logger.debug(f"Received audio chunk from Gemini ({len(audio_data)} bytes)")
                                # Send raw 24kHz audio directly (Client expects 24kHz)
                                await self._on_audio_delta(audio_data)

                    # Handle text data (IGNORING: This contains "Thinking" process in preview model)
                    text = part.text
                    if text:
                        logger.debug(f"DEBUG: Ignored part.text: {text[:50]}...")
                        # pass

            # Handle output transcription (assistant's speech as text)
            output_transcription = server_content.output_transcription
            if output_transcription:
                text = output_transcription.text
                if text:
                    # logger.debug(f"[{self._state.session_id}] Gemini Output Transcription: '{text}'")
                    self._state.transcript_buffer += text
                    if self._on_transcript_delta and not self._response_cancelled:
                        await self._on_transcript_delta(text, "assistant", self._current_turn_id, None)

            # Check for turn completion
            if server_content.turn_complete and self._state.is_responding:
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

    def update_vad_settings(self, start_sensitivity: int, end_sensitivity: int) -> None:
        """
        Update VAD sensitivity settings and trigger a fast reconnect to apply them.
        Only reconnects if values actually change.
        
        Args:
            start_sensitivity: New start sensitivity
            end_sensitivity: New end sensitivity
        """
        if (self._vad_start_sensitivity == start_sensitivity and
            self._vad_end_sensitivity == end_sensitivity):
            # No change, avoid disruptive reconnect
            logger.debug("VAD Settings unchanged, skipping reconnect.")
            return

        logger.info(f"Updating VAD Settings: Start={start_sensitivity}, End={end_sensitivity}")
        self._vad_start_sensitivity = start_sensitivity
        self._vad_end_sensitivity = end_sensitivity
        
        if self._connected:
            logger.info("Triggering reconnect to apply VAD settings...")
            self._reconnect_requested = True

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

    # Removed duplicate append_audio and _resample_audio methods that were causing conflicts

    async def _send_text_async(self, text: str) -> None:
        """
        Send text message to Gemini Live API.
        
        Args:
            text: Text content
        """
        if not self._session:
            return

        try:
            # Send text input
            # Use self._session.send(..., end_of_turn=True) for text
            await self._session.send(
                input=text,
                end_of_turn=True
            )
        except Exception as e:
            logger.error(f"Error sending text to Gemini: {e}")
            if self._on_error:
                await self._on_error({"error": str(e)})

    async def _send_audio_async(self, audio_bytes: bytes) -> None:
        """
        Send audio bytes to Gemini Live API.
        
        Args:
            audio_bytes: PCM16 audio bytes
        """
        if not self._session:
            return

        try:
            # Send audio data
            # Using v1alpha style with types.Blob
            # Explicitly setting rate=16000 as per user request/docs
            mime_type = "audio/pcm;rate=16000"
            await self._session.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type=mime_type)
            )
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
            if self._on_error:
                await self._on_error({"error": str(e)})

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
