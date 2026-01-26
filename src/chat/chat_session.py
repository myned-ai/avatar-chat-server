import asyncio
import base64
import time
from typing import Any

import orjson
from fastapi import WebSocket

from core.logger import get_logger
from core.settings import Settings
from services import Wav2ArkitService, create_agent_instance

logger = get_logger(__name__)


class ChatSession:
    """
    Represents a single active conversation session.
    Holds all state specific to one user connection.
    """
    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        settings: Settings,
        wav2arkit_service: Wav2ArkitService,
    ):
        logger.info(f"Initializing ChatSession for client: {session_id}")

        self.websocket = websocket
        self.session_id = session_id
        self.settings = settings
        self.wav2arkit_service = wav2arkit_service

        # Unique Agent Instance per Session
        self.agent = create_agent_instance()

        # Client State
        self.is_streaming_audio = False
        self.user_id = ""
        self.is_active = True

        # Audio and frame processing state
        self.audio_buffer: bytearray = bytearray()
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=120)
        self.audio_chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=15)

        # Tracking state
        self.current_turn_id: str | None = None
        self.speech_start_time: float = 0
        self.actual_audio_start_time: float = 0
        self.total_frames_emitted: int = 0
        self.total_audio_received: float = 0
        self.blendshape_frame_idx: int = 0
        self.speech_ended: bool = False
        self.is_interrupted: bool = False
        self.first_audio_received: bool = False

        # Background tasks
        self.frame_emit_task: asyncio.Task | None = None
        self.inference_task: asyncio.Task | None = None

        self._setup_agent_handlers()

    def _setup_agent_handlers(self) -> None:
        """Setup event handlers specific to this session's agent."""
        self.agent.set_event_handlers(
            on_audio_delta=self._handle_audio_delta,
            on_response_start=self._handle_response_start,
            on_response_end=self._handle_response_end,
            on_user_transcript=self._handle_user_transcript,
            on_transcript_delta=self._handle_transcript_delta,
            on_interrupted=self._handle_interrupted,
        )

    async def start(self) -> None:
        """Initialize connection to agent."""
        if not self.agent.is_connected:
            await self.agent.connect()

    async def stop(self) -> None:
        """Cleanup resources."""
        self.is_active = False
        self.is_interrupted = True
        
        # Cancel tasks
        if self.inference_task and not self.inference_task.done():
            self.inference_task.cancel()
            try:
                await self.inference_task
            except asyncio.CancelledError:
                pass
        
        if self.frame_emit_task and not self.frame_emit_task.done():
            self.frame_emit_task.cancel()
            try:
                await self.frame_emit_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect Agent
        await self.agent.disconnect()
        logger.info(f"Session {self.session_id} stopped.")

    async def send_json(self, message: dict[str, Any]) -> None:
        """Send JSON to this specific client."""
        if not self.is_active:
            return

        try:
            message_str = orjson.dumps(message).decode("utf-8")
            await self.websocket.send_text(message_str)
        except Exception as e:
            # Silence expected errors on disconnect
            if "Unexpected ASGI message" in str(e) or "websocket.close" in str(e):
                logger.debug(f"Socket closed while sending to {self.session_id}: {e}")
            else:
                logger.error(f"Error sending to client {self.session_id}: {e}")

    async def _handle_response_start(self, session_id: str) -> None:
        """Handle AI response start."""
        self.current_turn_id = f"turn_{int(time.time() * 1000)}_{session_id[:8]}"
        self.speech_start_time = time.time()
        self.actual_audio_start_time = 0
        self.total_frames_emitted = 0
        self.total_audio_received = 0
        self.blendshape_frame_idx = 0
        self.speech_ended = False
        self.is_interrupted = False
        self.first_audio_received = False

        # Clear queues
        self.audio_buffer.clear()
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.audio_chunk_queue.empty():
            try:
                self.audio_chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if self.wav2arkit_service.is_available:
             self.wav2arkit_service.reset_context()

        # Send start event BEFORE starting tasks to ensure client is ready to receive frames
        await self.send_json({"type": "avatar_state", "state": "Responding"})
        await self.send_json(
            {
                "type": "audio_start",
                "sessionId": session_id,
                "turnId": self.current_turn_id,
                "sampleRate": self.settings.input_sample_rate,
                "format": "audio/pcm16",
                "timestamp": int(time.time() * 1000),
            }
        )

        # Start background tasks
        if self.frame_emit_task is None or self.frame_emit_task.done():
            self.frame_emit_task = asyncio.create_task(self._emit_frames())
        if self.inference_task is None or self.inference_task.done():
            self.inference_task = asyncio.create_task(self._inference_worker())

    async def _handle_audio_delta(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from agent."""
        if self.is_interrupted:
            return

        if not self.first_audio_received:
            self.first_audio_received = True
            self.actual_audio_start_time = time.time()

        if not self.wav2arkit_service.is_available:
            await self.send_json(
                {
                    "type": "audio_chunk",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "sessionId": self.session_id,
                    "timestamp": int(time.time() * 1000),
                }
            )
            return

        self.audio_buffer.extend(audio_bytes)
        
        chunk_samples = len(audio_bytes) // 2
        chunk_duration = chunk_samples / self.settings.input_sample_rate
        self.total_audio_received += chunk_duration
        
        buffer_samples = len(self.audio_buffer) // 2
        buffer_duration = buffer_samples / self.settings.input_sample_rate

        if buffer_duration >= self.settings.audio_chunk_duration:
            chunk_bytes_size = int(
                self.settings.audio_chunk_duration * self.settings.input_sample_rate * 2
            )
            chunk_bytes = bytes(self.audio_buffer[:chunk_bytes_size])
            self.audio_buffer = bytearray(self.audio_buffer[chunk_bytes_size:])

            if self.is_interrupted:
                return

            await self.audio_chunk_queue.put(chunk_bytes)

    async def _handle_response_end(self, transcript: str, item_id: str | None = None) -> None:
        """Handle AI response end."""
        if self.is_interrupted:
            return

        # Wait for audio to stabilize
        last_audio = self.total_audio_received
        stable_count = 0
        max_wait_iterations = 60
        iterations = 0
        while stable_count < 15 and iterations < max_wait_iterations:
            await asyncio.sleep(0.05)
            iterations += 1
            if self.total_audio_received == last_audio:
                stable_count += 1
            else:
                stable_count = 0
                last_audio = self.total_audio_received

        if self.is_interrupted:
            return

        # Flush remaining audio
        buffer_samples = len(self.audio_buffer) // 2
        if buffer_samples > 0 and self.wav2arkit_service.is_available:
            min_samples = int(0.3 * self.settings.input_sample_rate)
            remaining_bytes = bytes(self.audio_buffer)
            if buffer_samples < min_samples:
                padding = bytes((min_samples - buffer_samples) * 2)
                remaining_bytes = remaining_bytes + padding
            self.audio_buffer.clear()
            await self.audio_chunk_queue.put(remaining_bytes)

        while not self.audio_chunk_queue.empty() and not self.is_interrupted:
            await asyncio.sleep(0.05)

        self.speech_ended = True

        # Wait for workers
        if self.inference_task and not self.inference_task.done():
            try:
                await asyncio.wait_for(self.inference_task, timeout=8.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        if self.frame_emit_task and not self.frame_emit_task.done():
            try:
                remaining_frames = self.frame_queue.qsize()
                timeout = max(5.0, remaining_frames / self.settings.blendshape_fps + 2.0)
                await asyncio.wait_for(self.frame_emit_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        await self.send_json(
            {
                "type": "audio_end",
                "sessionId": self.session_id,
                "turnId": self.current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

        if transcript:
            msg = {
                "type": "transcript_done",
                "text": transcript,
                "role": "assistant",
                "turnId": self.current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
            if item_id:
                msg["itemId"] = item_id
            await self.send_json(msg)

        await self.send_json({"type": "avatar_state", "state": "Listening"})

        self.current_turn_id = None
        self.speech_ended = False
        self.inference_task = None
        self.frame_emit_task = None

    async def _handle_transcript_delta(
        self,
        text: str,
        role: str = "assistant",
        item_id: str | None = None,
        previous_item_id: str | None = None,
    ) -> None:
        turn_id = self.current_turn_id if role == "assistant" else f"user_{int(time.time() * 1000)}"
        msg = {
            "type": "transcript_delta",
            "text": text,
            "role": role,
            "turnId": turn_id,
            "sessionId": self.session_id,
            "timestamp": int(time.time() * 1000),
        }
        if item_id:
            msg["itemId"] = item_id
        if previous_item_id:
            msg["previousItemId"] = previous_item_id
        await self.send_json(msg)

    async def _handle_user_transcript(self, transcript: str, role: str = "user") -> None:
        user_turn_id = f"{role}_{int(time.time() * 1000)}"
        await self.send_json(
            {
                "type": "transcript_done",
                "text": transcript,
                "role": role,
                "turnId": user_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _handle_interrupted(self) -> None:
        self.is_interrupted = True
        interrupted_turn_id = self.current_turn_id

        await self.send_json({"type": "interrupt", "timestamp": int(time.time() * 1000)})
        logger.debug(f"Session {self.session_id}: Interruption signal sent to client")
        await self.send_json({"type": "avatar_state", "state": "Listening"})

        # Halt Upstream Agent synchronously
        try:
            # Direct call, not create_task, because it is synchronous
            self.agent.cancel_response()
        except Exception as e:
            logger.warning(f"Failed to cancel agent response: {e}")

        self.audio_buffer.clear()
        
        # Flush synchronous queues
        while not self.audio_chunk_queue.empty():
            try:
                self.audio_chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Proper Task Cancellation
        tasks_to_cancel = []
        if self.inference_task and not self.inference_task.done():
            self.inference_task.cancel()
            tasks_to_cancel.append(self.inference_task)
        
        if self.frame_emit_task and not self.frame_emit_task.done():
            self.frame_emit_task.cancel()
            tasks_to_cancel.append(self.frame_emit_task)
            
        if tasks_to_cancel:
            # Wait for tasks to clean up (swallow CancelledError)
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self.inference_task = None
        self.frame_emit_task = None
        self.speech_ended = True
        self.current_turn_id = None

        if interrupted_turn_id:
            await self.send_json(
                {
                    "type": "transcript_done",
                    "text": "",
                    "role": "assistant",
                    "turnId": interrupted_turn_id,
                    "interrupted": True,
                    "timestamp": int(time.time() * 1000),
                }
            )

    async def _inference_worker(self) -> None:
        """Process audio chunks through Wav2Arkit model."""
        try:
            logger.info(f"Session {self.session_id}: Inference worker started")
            loop = asyncio.get_event_loop()
            
            while True:
                if self.is_interrupted:
                    break
                
                try:
                    audio_bytes = await asyncio.wait_for(
                        self.audio_chunk_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    if self.speech_ended and self.audio_chunk_queue.empty():
                        break
                    continue
                
                if self.is_interrupted:
                    break

                # Run inference in default executor
                try:
                    # Use None executor to use the default loop executor (ThreadPool)
                    frames = await loop.run_in_executor(
                        None, self.wav2arkit_service.process_audio_chunk, audio_bytes
                    )
                except Exception as e:
                    logger.error(f"Inference processing failed: {e}", exc_info=True)
                    # Prevent CPU spin loop on persistent error
                    await asyncio.sleep(0.1)
                    continue

                if self.is_interrupted:
                    break

                if frames:
                    for frame in frames:
                        if self.is_interrupted:
                            break
                        await self.frame_queue.put(frame)
                else:
                    logger.warning("Inference returned no frames")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Session {self.session_id} inference worker fatal error: {e}", exc_info=True)

    async def _emit_frames(self) -> None:
        """Emit synchronized frames."""
        try:
            while True:
                if self.is_interrupted:
                    break

                # Pacing logic
                elapsed_time = time.time() - self.speech_start_time
                target_frames = int(elapsed_time * self.settings.blendshape_fps)

                if self.total_frames_emitted >= target_frames:
                    await asyncio.sleep(0.005)
                    continue

                try:
                    frame_data = await asyncio.wait_for(
                        self.frame_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    if self.speech_ended and self.frame_queue.empty():
                        break
                    continue
                
                if self.is_interrupted:
                    break

                if self.blendshape_frame_idx % 30 == 0:
                    audio_b64 = frame_data.get("audio", "")
                    audio_size = len(audio_b64)
                    logger.debug(f"Session {self.session_id}: Sending sync_frame {self.blendshape_frame_idx} (Active). Audio payload: {audio_size} chars")

                await self.send_json(
                    {
                        "type": "sync_frame",
                        "weights": frame_data["weights"],
                        "audio": frame_data["audio"],
                        "sessionId": self.session_id,
                        "turnId": self.current_turn_id,
                        "timestamp": int(time.time() * 1000),
                        "frameIndex": self.blendshape_frame_idx,
                    }
                )

                self.blendshape_frame_idx += 1
                self.total_frames_emitted += 1

        except asyncio.CancelledError:
            if not self.is_interrupted:
                # Drain
                while not self.frame_queue.empty():
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        await self.send_json(
                            {
                                "type": "sync_frame",
                                "weights": frame_data["weights"],
                                "audio": frame_data["audio"],
                                "sessionId": self.session_id,
                                "turnId": self.current_turn_id,
                                "timestamp": int(time.time() * 1000),
                                "frameIndex": self.blendshape_frame_idx,
                            }
                        )
                        self.blendshape_frame_idx += 1
                        self.total_frames_emitted += 1
                    except asyncio.QueueEmpty:
                        break

    async def process_message(self, data: dict[str, Any]) -> None:
        """Handle incoming message from client."""
        msg_type = data.get("type")

        if msg_type == "text":
            self.agent.send_text_message(data.get("data", ""))

        elif msg_type == "audio_stream_start":
            self.is_streaming_audio = True
            self.user_id = data.get("userId", "unknown")

        elif msg_type == "audio":
            if self.is_streaming_audio:
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    self.agent.append_audio(audio_bytes)

        elif msg_type == "interrupt":
            await self._handle_interrupted()

        elif msg_type == "ping":
            await self.send_json({
                "type": "pong",
                "timestamp": int(time.time() * 1000),
            })
