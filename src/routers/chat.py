"""
Chat WebSocket Router

Handles WebSocket connections for real-time voice chat with
synchronized audio and facial animation blendshapes.
"""

import asyncio
import base64
import orjson
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import get_settings, get_audio_constants
from services import get_audio2exp_service, get_openai_service
from logger import get_logger

logger = get_logger(__name__)


router = APIRouter(tags=["chat"])


@dataclass
class ClientState:
    """State for a connected frontend client."""
    websocket: WebSocket
    user_id: str = ""
    is_streaming_audio: bool = False


class ChatConnectionManager:
    """
    Manages WebSocket connections and orchestrates real-time chat.
    
    Coordinates between:
    - Frontend WebSocket clients
    - OpenAI Realtime API (voice-to-voice AI)
    - Audio2Expression model (audio-to-blendshape inference)
    """
    
    def __init__(self):
        self.clients: Dict[WebSocket, ClientState] = {}
        self._settings = get_settings()
        self._audio_constants = get_audio_constants()
        self._audio2exp_service = get_audio2exp_service()
        self._openai_service = get_openai_service()
        
        # Audio and frame processing state
        self._audio_buffer: bytearray = bytearray()
        # Bounded queues prevent memory exhaustion and provide backpressure
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=90)  # 3 seconds @ 30 FPS
        self._audio_chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=10)  # 10 chunks (~10 seconds)
        
        # Tracking state
        self._current_session_id: Optional[str] = None
        self._speech_start_time: float = 0
        self._total_frames_emitted: int = 0
        self._total_audio_received: float = 0
        self._blendshape_frame_idx: int = 0
        self._speech_ended: bool = False
        
        # Background tasks
        self._frame_emit_task: Optional[asyncio.Task] = None
        self._inference_task: Optional[asyncio.Task] = None
        
        # Setup OpenAI event handlers
        self._setup_openai_handlers()
    
    def _setup_openai_handlers(self) -> None:
        """Setup event handlers for OpenAI service."""
        self._openai_service.set_event_handlers(
            on_audio_delta=self._handle_audio_delta,
            on_response_start=self._handle_response_start,
            on_response_end=self._handle_response_end,
            on_user_transcript=self._handle_user_transcript,
            on_transcript_delta=self._handle_transcript_delta,
            on_interrupted=self._handle_interrupted,
        )
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.clients[websocket] = ClientState(websocket=websocket)
        
        # Ensure OpenAI is connected
        if not self._openai_service.is_connected:
            await self._openai_service.connect()

        logger.info(f"Client connected: {websocket.client}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Handle client disconnection."""
        if websocket in self.clients:
            del self.clients[websocket]
            logger.info(f"Client disconnected: {websocket.client}")
    
    async def broadcast(self, message: Dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        # Use orjson for 3-5x faster JSON serialization
        message_str = orjson.dumps(message).decode('utf-8')
        tasks = [
            client.websocket.send_text(message_str)
            for client in self.clients.values()
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    # ==================== OpenAI Event Handlers ====================
    
    async def _handle_response_start(self, session_id: str) -> None:
        """Handle AI response start."""
        self._current_session_id = session_id
        self._speech_start_time = time.time()
        self._total_frames_emitted = 0
        self._total_audio_received = 0
        self._blendshape_frame_idx = 0
        self._speech_ended = False
        
        # Clear queues
        self._audio_buffer.clear()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._audio_chunk_queue.empty():
            try:
                self._audio_chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset Audio2Exp context
        if self._audio2exp_service.is_available:
            self._audio2exp_service.reset_context()
        
        # Start background tasks
        if self._frame_emit_task is None or self._frame_emit_task.done():
            self._frame_emit_task = asyncio.create_task(self._emit_frames())
        if self._inference_task is None or self._inference_task.done():
            self._inference_task = asyncio.create_task(self._inference_worker())
        
        await self.broadcast({"type": "avatar_state", "state": "Responding"})
        await self.broadcast({
            "type": "audio_start",
            "sessionId": session_id,
            "sampleRate": self._audio_constants.openai_sample_rate,
            "format": "audio/pcm16",
            "timestamp": int(time.time() * 1000),
        })
    
    async def _handle_audio_delta(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from OpenAI."""
        if not self._audio2exp_service.is_available:
            # No Audio2Exp - send audio directly (no blendshapes)
            await self.broadcast({
                "type": "audio_chunk",
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "sessionId": self._current_session_id,
                "timestamp": int(time.time() * 1000),
            })
            return
        
        # Buffer audio for Audio2Expression inference
        self._audio_buffer.extend(audio_bytes)
        
        chunk_samples = len(audio_bytes) // 2
        chunk_duration = chunk_samples / self._audio_constants.openai_sample_rate
        self._total_audio_received += chunk_duration
        
        # Calculate buffer duration
        buffer_samples = len(self._audio_buffer) // 2
        buffer_duration = buffer_samples / self._audio_constants.openai_sample_rate
        
        # When we have 1 second, queue for processing
        if buffer_duration >= self._audio_constants.audio_chunk_duration:
            chunk_bytes_size = int(
                self._audio_constants.audio_chunk_duration
                * self._audio_constants.openai_sample_rate
                * 2
            )
            chunk_bytes = bytes(self._audio_buffer[:chunk_bytes_size])
            self._audio_buffer = bytearray(self._audio_buffer[chunk_bytes_size:])
            await self._audio_chunk_queue.put(chunk_bytes)
    
    async def _handle_response_end(self, transcript: str) -> None:
        """Handle AI response end."""
        # Wait for audio to stabilize (OpenAI Realtime API can send response.done
        # before all audio.delta events are received - this is a known issue)
        # Increased wait time and iterations to ensure we catch the final audio chunks
        last_audio = self._total_audio_received
        stable_count = 0
        max_wait_iterations = 40  # Max 40 * 50ms = 2 seconds wait
        iterations = 0
        while stable_count < 10 and iterations < max_wait_iterations:  # Need 10 stable checks (500ms)
            await asyncio.sleep(0.05)
            iterations += 1
            if self._total_audio_received == last_audio:
                stable_count += 1
            else:
                stable_count = 0
                last_audio = self._total_audio_received
        
        # Flush remaining audio - ensure ALL buffered audio is processed
        buffer_samples = len(self._audio_buffer) // 2
        if buffer_samples > 0 and self._audio2exp_service.is_available:
            # Use a longer minimum sample threshold to ensure we don't lose short final words
            min_samples = int(0.3 * self._audio_constants.openai_sample_rate)  # 300ms minimum
            remaining_bytes = bytes(self._audio_buffer)
            if buffer_samples < min_samples:
                # Pad with silence to ensure the model processes the full audio
                padding = bytes((min_samples - buffer_samples) * 2)
                remaining_bytes = remaining_bytes + padding
            self._audio_buffer.clear()
            await self._audio_chunk_queue.put(remaining_bytes)
            logger.debug(f"Flushing final audio: {buffer_samples} samples ({buffer_samples / self._audio_constants.openai_sample_rate:.3f}s)")
        
        self._speech_ended = True
        
        # Wait for workers to finish
        if self._inference_task and not self._inference_task.done():
            try:
                await asyncio.wait_for(self._inference_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._inference_task.cancel()
        
        if self._frame_emit_task and not self._frame_emit_task.done():
            try:
                timeout = max(3.0, self._frame_queue.qsize() / self._audio_constants.blendshape_fps + 1.0)
                await asyncio.wait_for(self._frame_emit_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._frame_emit_task.cancel()
        
        # Log summary
        speech_duration = time.time() - self._speech_start_time if self._speech_start_time else 0
        expected_frames = int(self._total_audio_received * self._audio_constants.blendshape_fps)
        logger.debug(f"Speech complete: {speech_duration:.2f}s, {self._total_frames_emitted} frames (expected ~{expected_frames})")
        
        await self.broadcast({
            "type": "audio_end",
            "sessionId": self._current_session_id,
            "timestamp": int(time.time() * 1000),
        })
        
        if transcript:
            await self.broadcast({
                "type": "transcript_done",
                "text": transcript,
                "role": "assistant",
                "timestamp": int(time.time() * 1000),
            })
        
        await self.broadcast({"type": "avatar_state", "state": "Listening"})
        
        # Reset state
        self._current_session_id = None
        self._speech_ended = False
        self._inference_task = None
        self._frame_emit_task = None
    
    async def _handle_transcript_delta(self, text: str) -> None:
        """Handle streaming transcript from AI."""
        await self.broadcast({
            "type": "transcript_delta",
            "text": text,
            "role": "assistant",
            "timestamp": int(time.time() * 1000),
        })
    
    async def _handle_user_transcript(self, transcript: str) -> None:
        """Handle transcribed user speech."""
        await self.broadcast({
            "type": "transcript_done",
            "text": transcript,
            "role": "user",
            "timestamp": int(time.time() * 1000),
        })
    
    async def _handle_interrupted(self) -> None:
        """Handle conversation interruption."""
        await self.broadcast({
            "type": "interrupt",
            "timestamp": int(time.time() * 1000),
        })
    
    # ==================== Background Workers ====================
    
    async def _inference_worker(self) -> None:
        """Process audio chunks through Audio2Expression model."""
        chunk_count = 0
        try:
            while True:
                try:
                    audio_bytes = await asyncio.wait_for(
                        self._audio_chunk_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    if self._speech_ended and self._audio_chunk_queue.empty():
                        break
                    continue
                
                chunk_count += 1
                frames = self._audio2exp_service.process_audio_chunk(audio_bytes)

                if frames:
                    logger.debug(f"Chunk {chunk_count}: {len(audio_bytes) // 2 / self._audio_constants.openai_sample_rate:.3f}s -> {len(frames)} frames")
                    
                    for frame in frames:
                        await self._frame_queue.put(frame)
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Inference worker error: {e}")
    
    async def _emit_frames(self) -> None:
        """Emit synchronized audio+blendshape frames to clients."""
        try:
            while True:
                try:
                    frame_data = await asyncio.wait_for(
                        self._frame_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    if self._speech_ended and self._frame_queue.empty():
                        logger.debug(f"Frame emission complete: {self._total_frames_emitted} frames")
                        break
                    continue
                
                # Send synchronized frame (audio already base64 encoded by worker)
                await self.broadcast({
                    "type": "sync_frame",
                    "weights": frame_data["weights"],
                    "audio": frame_data["audio"],  # Already base64 encoded
                    "sessionId": self._current_session_id,
                    "timestamp": int(time.time() * 1000),
                    "frameIndex": self._blendshape_frame_idx,
                })
                
                self._blendshape_frame_idx += 1
                self._total_frames_emitted += 1

                if self._total_frames_emitted % 30 == 0:
                    elapsed = time.time() - self._speech_start_time
                    logger.debug(f"Sent {self._total_frames_emitted} sync_frames in {elapsed:.2f}s")
                    
        except asyncio.CancelledError:
            # Drain remaining frames
            while not self._frame_queue.empty():
                try:
                    frame_data = self._frame_queue.get_nowait()
                    await self.broadcast({
                        "type": "sync_frame",
                        "weights": frame_data["weights"],
                        "audio": frame_data["audio"],  # Already base64 encoded
                        "sessionId": self._current_session_id,
                        "timestamp": int(time.time() * 1000),
                        "frameIndex": self._blendshape_frame_idx,
                    })
                    self._blendshape_frame_idx += 1
                    self._total_frames_emitted += 1
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.warning(f"Frame emission error: {e}")
    
    # ==================== Client Message Handling ====================
    
    async def handle_message(self, websocket: WebSocket, data: Dict) -> None:
        """Handle incoming message from client."""
        client = self.clients.get(websocket)
        if not client:
            return
        
        msg_type = data.get("type")
        
        if msg_type == "text":
            # Text message from user
            text = data.get("data", "")
            self._openai_service.send_text_message(text)
        
        elif msg_type == "audio_stream_start":
            client.is_streaming_audio = True
            client.user_id = data.get("userId", "unknown")
            logger.debug(f"Audio stream started from {client.user_id}")
        
        elif msg_type == "audio":
            if client.is_streaming_audio:
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    self._openai_service.append_audio(audio_bytes)
        
        elif msg_type == "audio_stream_end":
            client.is_streaming_audio = False
            logger.debug(f"Audio stream ended from {client.user_id}")
        
        elif msg_type == "ping":
            await websocket.send_json({
                "type": "pong",
                "timestamp": int(time.time() * 1000),
            })


# Singleton manager
manager = ChatConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice chat.

    Handles bidirectional communication between the frontend
    and AI services (OpenAI + Audio2Expression).
    """
    # Import here to avoid circular dependency
    from main import auth_middleware

    session_id = str(uuid.uuid4())

    # Authenticate WebSocket connection
    settings = get_settings()
    if settings.auth_enabled and auth_middleware:
        is_authenticated, error = await auth_middleware.authenticate_websocket(
            websocket, session_id
        )
        if not is_authenticated:
            await websocket.close(code=1008, reason=error)
            return

    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)
