"""
Remote Agent

Implementation of the agent interface for remote (inter-container) agents.
Connects to an external agent service via WebSocket and proxies calls/events.
"""

import asyncio
import json
from collections.abc import Callable

import websockets

from core.logger import get_logger
from core.settings import get_settings

from .base_agent import BaseAgent, ConversationState

logger = get_logger(__name__)


class RemoteAgent(BaseAgent):
    """
    Remote agent implementation that connects to an external agent service.

    Assumes the remote service implements a WebSocket API compatible with
    the chat router's expectations (e.g., same event messages).
    """

    def __init__(self):
        """
        Initialize the remote agent.

        Loads settings from environment variables.
        """
        self._settings = get_settings()
        self._ws: websockets.WebSocketServerProtocol | None = None
        self._connected = False
        self._state = ConversationState()
        self._event_loop = asyncio.get_event_loop()

        # Event callbacks
        self._on_audio_delta: Callable | None = None
        self._on_transcript_delta: Callable | None = None
        self._on_response_start: Callable | None = None
        self._on_response_end: Callable | None = None
        self._on_user_transcript: Callable | None = None
        self._on_interrupted: Callable | None = None
        self._on_error: Callable | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to remote agent."""
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
            on_response_end: Called when agent finishes responding
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
        """Connect to remote agent WebSocket."""
        if self._connected:
            return

        agent_url = self._settings.agent_url
        if not agent_url:
            raise ValueError("AGENT_URL not configured for remote agent")

        logger.info(f"Connecting to remote agent at {agent_url}")

        try:
            self._ws = await websockets.connect(agent_url)
            self._connected = True
            # Start listening for events
            asyncio.create_task(self._listen_for_events())
            logger.info("Connected to remote agent")
        except Exception as e:
            logger.error(f"Failed to connect to remote agent: {e}")
            raise

    async def _listen_for_events(self) -> None:
        """Listen for events from remote agent and dispatch to handlers."""
        try:
            async for message in self._ws:
                try:
                    event = json.loads(message)
                    await self._handle_remote_event(event)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from remote agent: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Remote agent connection closed")
            self._connected = False
        except Exception as e:
            logger.error(f"Error listening to remote agent: {e}")
            self._connected = False
            if self._on_error:
                await self._on_error({"error": str(e)})

    async def _handle_remote_event(self, event: dict) -> None:
        """Handle events received from remote agent."""
        event_type = event.get("type")

        if event_type == "audio_delta" and self._on_audio_delta:
            # Assume audio is base64 or bytes
            audio_data = event.get("data", "")
            if isinstance(audio_data, str):
                import base64

                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            await self._on_audio_delta(audio_bytes)

        elif event_type == "transcript_delta" and self._on_transcript_delta:
            # Support itemId/previousItemId propagation from remote agents
            await self._on_transcript_delta(
                event.get("delta", ""),
                event.get("role", "assistant"),
                event.get("itemId"),
                event.get("previousItemId"),
            )

        elif event_type == "response_start" and self._on_response_start:
            await self._on_response_start(event.get("session_id", ""))

        elif event_type == "response_end" and self._on_response_end:
            await self._on_response_end(event.get("transcript", ""), event.get("itemId"))

        elif event_type == "user_transcript" and self._on_user_transcript:
            await self._on_user_transcript(event.get("transcript", ""))

        elif event_type == "interrupted" and self._on_interrupted:
            await self._on_interrupted()

        elif event_type == "error" and self._on_error:
            await self._on_error(event.get("error", {}))

        # Update state based on events
        if event_type == "response_start":
            self._state.session_id = event.get("session_id")
            self._state.is_responding = True
            self._state.transcript_buffer = ""
        elif event_type == "response_end" or event_type == "interrupted":
            self._state.is_responding = False

    def send_text_message(self, text: str) -> None:
        """
        Send a text message to the remote agent.

        Args:
            text: Text message content
        """
        if not self._connected or not self._ws:
            return

        message = {"type": "text", "data": text}
        asyncio.create_task(self._send_message(message))

    def append_audio(self, audio_bytes: bytes) -> None:
        """
        Append audio to the remote agent's input buffer.

        Args:
            audio_bytes: PCM16 audio bytes
        """
        if not self._connected or not self._ws:
            return

        import base64

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        message = {"type": "audio", "data": audio_b64}
        asyncio.create_task(self._send_message(message))

    async def _send_message(self, message: dict) -> None:
        """Send a message to the remote agent."""
        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message to remote agent: {e}")
            if self._on_error:
                await self._on_error({"error": str(e)})

    async def disconnect(self) -> None:
        """Disconnect from remote agent."""
        if self._ws and self._connected:
            await self._ws.close()
            self._connected = False
            logger.info("Disconnected from remote agent")
