"""
OpenAI Realtime Service

Service layer for OpenAI Realtime API integration.
Handles voice-to-voice conversation with the AI assistant.
"""

import asyncio
import time
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field

from config import Settings
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationState:
    """State tracking for an active conversation."""
    session_id: Optional[str] = None
    is_responding: bool = False
    transcript_buffer: str = ""
    audio_done: bool = False  # Track if response.audio.done has been received


class OpenAIRealtimeService:
    """
    Service for OpenAI Realtime API interactions.
    
    Wraps the RealtimeClient to provide a clean service interface
    for voice-based conversation with the AI assistant.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the OpenAI Realtime service.
        
        Args:
            settings: Application settings containing API key and configuration
        """
        self.settings = settings
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
        
        from realtime_client import RealtimeClient
        
        self._client = RealtimeClient(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            debug=self.settings.debug,
        )
        
        # Setup event handlers
        self._setup_events()
        
        # Connect and wait for session
        await self._client.connect()
        await self._client.wait_for_session_created()
        
        # Configure session
        self._client.update_session(
            instructions=self.settings.assistant_instructions,
            voice=self.settings.openai_voice,
            modalities=["text", "audio"],
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription={"model": "whisper-1"},
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
        )
        
        self._connected = True
        logger.info(f"Connected to OpenAI Realtime API (voice: {self.settings.openai_voice})")
    
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
        if self.settings.debug:
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
        delta = event.get("delta", {})
        
        # Stream audio delta
        if delta and "audio" in delta:
            audio_data = delta["audio"]
            if isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                audio_bytes = audio_data.tobytes()
            
            if self._on_audio_delta:
                asyncio.create_task(self._on_audio_delta(audio_bytes))
        
        # Stream transcript delta
        if delta and "transcript" in delta:
            transcript_delta = delta["transcript"]
            self._state.transcript_buffer += transcript_delta
            
            if self._on_transcript_delta:
                asyncio.create_task(self._on_transcript_delta(transcript_delta))
    
    def _handle_audio_done(self, event: Dict) -> None:
        """Handle response.audio.done event - all audio has been sent."""
        logger.debug("Audio done received")
        self._state.audio_done = True
    
    def _handle_item_appended(self, event: Dict) -> None:
        """Handle new conversation items."""
        item = event.get("item", {})

        if item.get("role") == "assistant":
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
                transcript = item.get("formatted", {}).get("transcript", "")
                self._state.is_responding = False

                if transcript:
                    logger.debug(f"Assistant: {transcript}")

                if self._on_response_end:
                    await self._on_response_end(transcript)
            
            asyncio.create_task(wait_and_complete())
    
    def _handle_user_transcript(self, event: Dict) -> None:
        """Handle user's transcribed speech."""
        transcript = event.get("transcript", "")

        if transcript:
            logger.debug(f"User: {transcript}")

            if self._on_user_transcript:
                asyncio.create_task(self._on_user_transcript(transcript))
    
    def _handle_interrupted(self, event: Dict) -> None:
        """Handle conversation interruption."""
        logger.debug("Conversation interrupted")
        self._state.is_responding = False

        if self._on_interrupted:
            asyncio.create_task(self._on_interrupted())
    
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
            self._client.disconnect()
            self._connected = False
            logger.info("Disconnected from OpenAI Realtime API")


# Singleton instance
_openai_service: Optional[OpenAIRealtimeService] = None


def get_openai_service(settings: Optional[Settings] = None) -> OpenAIRealtimeService:
    """
    Get or create the OpenAI service singleton.
    
    Args:
        settings: Application settings (uses defaults if not provided)
        
    Returns:
        OpenAIRealtimeService instance
    """
    global _openai_service
    
    if _openai_service is None:
        from config import get_settings
        settings = settings or get_settings()
        _openai_service = OpenAIRealtimeService(settings)
    
    return _openai_service
