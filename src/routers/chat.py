"""
Chat WebSocket Router

Handles WebSocket connections for real-time voice chat with
synchronized audio and facial animation blendshapes.
"""

import asyncio
import base64
import time
import uuid
from dataclasses import dataclass

import orjson
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import get_audio_constants, get_settings
from core.logger import get_logger
from services import get_agent, get_wav2arkit_service

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
    - Agent service (voice-to-voice AI)
    - Wav2Arkit model (audio-to-blendshape inference)
    """

    def __init__(self):
        self.clients: dict[WebSocket, ClientState] = {}
        self._settings = get_settings()
        self._audio_constants = get_audio_constants()
        self._wav2arkit_service = get_wav2arkit_service()
        self._agent = get_agent()

        # Audio and frame processing state
        self._audio_buffer: bytearray = bytearray()
        # Bounded queues prevent memory exhaustion and provide backpressure
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=120)  # 4 seconds @ 30 FPS
        self._audio_chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=15)  # 15 chunks (~15 seconds)

        # Tracking state
        self._current_session_id: str | None = None
        self._current_turn_id: str | None = None  # Unique ID for each turn (for transcript correlation)
        self._speech_start_time: float = 0
        self._actual_audio_start_time: float = 0  # Time when first audio actually arrives
        self._total_frames_emitted: int = 0
        self._total_audio_received: float = 0
        self._blendshape_frame_idx: int = 0
        self._speech_ended: bool = False
        self._is_interrupted: bool = False  # Synchronous flag to stop processing immediately
        self._first_audio_received: bool = False  # Track if we've received any audio for this response

        # Background tasks
        self._frame_emit_task: asyncio.Task | None = None
        self._inference_task: asyncio.Task | None = None

        # Setup agent event handlers
        self._setup_agent_handlers()

    def _setup_agent_handlers(self) -> None:
        """Setup event handlers for agent service."""
        self._agent.set_event_handlers(
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

        # Ensure agent is connected
        if not self._agent.is_connected:
            await self._agent.connect()

        logger.info(f"Client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle client disconnection. If this was the last client,
        gracefully shut down background workers and disconnect the agent
        (closing the OpenAI Realtime connection).
        """
        if websocket in self.clients:
            del self.clients[websocket]
            logger.info(f"Client disconnected: {websocket.client}")

        # If no clients remain, perform graceful shutdown of agent and workers
        if not self.clients:
            logger.info("No clients remain — shutting down agent connection and workers")

            # Signal interruption to stop any in-flight processing
            self._is_interrupted = True
            self._speech_ended = True

            # Clear audio buffer
            buffer_size = len(self._audio_buffer)
            self._audio_buffer.clear()
            logger.debug(f"Cleared audio buffer ({buffer_size} bytes)")

            # Clear audio chunk queue
            chunks_cleared = 0
            while not self._audio_chunk_queue.empty():
                try:
                    self._audio_chunk_queue.get_nowait()
                    chunks_cleared += 1
                except asyncio.QueueEmpty:
                    break
            logger.debug(f"Cleared audio chunk queue ({chunks_cleared} chunks)")

            # Clear frame queue
            frames_cleared = 0
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                    frames_cleared += 1
                except asyncio.QueueEmpty:
                    break
            logger.debug(f"Cleared frame queue ({frames_cleared} frames)")

            # Cancel background tasks
            if self._inference_task and not self._inference_task.done():
                self._inference_task.cancel()
                try:
                    await self._inference_task
                except asyncio.CancelledError:
                    pass
                self._inference_task = None
                logger.debug("Cancelled inference task")

            if self._frame_emit_task and not self._frame_emit_task.done():
                self._frame_emit_task.cancel()
                try:
                    await self._frame_emit_task
                except asyncio.CancelledError:
                    pass
                self._frame_emit_task = None
                logger.debug("Cancelled frame emit task")

            # Reset all state fields
            self._current_session_id = None
            self._current_turn_id = None
            self._speech_start_time = 0
            self._actual_audio_start_time = 0
            self._total_frames_emitted = 0
            self._total_audio_received = 0
            self._blendshape_frame_idx = 0
            self._first_audio_received = False

            # Disconnect the agent (closes OpenAI connection and clears its buffers)
            try:
                await self._agent.disconnect()
                logger.info("Agent disconnected - OpenAI connection closed")
            except Exception as e:
                logger.warning(f"Error disconnecting agent: {e}")

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        # Use orjson for faster JSON serialization
        message_str = orjson.dumps(message).decode("utf-8")
        tasks = [client.websocket.send_text(message_str) for client in self.clients.values()]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ==================== Agent Event Handlers ====================

    async def _handle_response_start(self, session_id: str) -> None:
        """Handle AI response start."""
        self._current_session_id = session_id
        self._current_turn_id = f"turn_{int(time.time() * 1000)}_{session_id[:8]}"
        self._speech_start_time = time.time()
        self._actual_audio_start_time = 0  # Will be set when first audio arrives
        self._total_frames_emitted = 0
        self._total_audio_received = 0
        self._blendshape_frame_idx = 0
        self._speech_ended = False
        self._is_interrupted = False  # Reset interrupt flag for new response
        self._first_audio_received = False

        # DEBUG: Log response start
        logger.info(f"Response started (session: {session_id}, turn: {self._current_turn_id})")

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

        # Reset Wav2Arkit context
        if self._wav2arkit_service.is_available:
            self._wav2arkit_service.reset_context()

        # Start background tasks
        if self._frame_emit_task is None or self._frame_emit_task.done():
            self._frame_emit_task = asyncio.create_task(self._emit_frames())
        if self._inference_task is None or self._inference_task.done():
            self._inference_task = asyncio.create_task(self._inference_worker())

        await self.broadcast({"type": "avatar_state", "state": "Responding"})
        await self.broadcast(
            {
                "type": "audio_start",
                "sessionId": session_id,
                "turnId": self._current_turn_id,
                "sampleRate": self._audio_constants.input_sample_rate,
                "format": "audio/pcm16",
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _handle_audio_delta(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from agent."""
        # Early exit if interrupted - check synchronously before any processing
        if self._is_interrupted:
            return

        time.time()

        # Track when first audio actually arrives (for accurate frame timing)
        if not self._first_audio_received:
            self._first_audio_received = True
            self._actual_audio_start_time = time.time()
            logger.debug("First audio received, resetting speech start time")

        if not self._wav2arkit_service.is_available:
            # No Wav2Arkit - send audio directly (no blendshapes)
            await self.broadcast(
                {
                    "type": "audio_chunk",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "sessionId": self._current_session_id,
                    "timestamp": int(time.time() * 1000),
                }
            )
            return

        # Buffer audio for Wav2Arkit inference
        self._audio_buffer.extend(audio_bytes)

        chunk_samples = len(audio_bytes) // 2
        chunk_duration = chunk_samples / self._audio_constants.input_sample_rate
        self._total_audio_received += chunk_duration

        # Calculate buffer duration
        buffer_samples = len(self._audio_buffer) // 2
        buffer_duration = buffer_samples / self._audio_constants.input_sample_rate

        # DEBUG: Log audio received (show total received so far)
        logger.debug(
            f"Audio: +{chunk_duration * 1000:.0f}ms (total={self._total_audio_received:.2f}s, buffer={buffer_duration:.2f}s)"
        )

        # When we have enough audio, queue for processing
        # Use consistent chunk duration to maintain timing alignment
        if buffer_duration >= self._audio_constants.audio_chunk_duration:
            chunk_bytes_size = int(
                self._audio_constants.audio_chunk_duration * self._audio_constants.input_sample_rate * 2
            )
            chunk_bytes = bytes(self._audio_buffer[:chunk_bytes_size])
            self._audio_buffer = bytearray(self._audio_buffer[chunk_bytes_size:])

            # Check if interrupted before queuing
            if self._is_interrupted:
                return

            # Queue for processing - use blocking put to ensure no audio is dropped
            # The queue is sized to handle normal processing rates
            await self._audio_chunk_queue.put(chunk_bytes)

            # DEBUG: Log when we queue a chunk for inference
            logger.debug(
                f"Queued {self._audio_constants.audio_chunk_duration:.1f}s chunk for inference (queue_size={self._audio_chunk_queue.qsize()})"
            )

    async def _handle_response_end(self, transcript: str, item_id: str | None = None) -> None:
        """Handle AI response end."""
        # Don't process response end if we were interrupted
        if self._is_interrupted:
            logger.debug("Skipping response end handling - was interrupted")
            return

        # Wait for audio to stabilize (OpenAI Realtime API can send response.done
        # before all audio.delta events are received - this is a known issue)
        # Increased wait time and iterations to ensure we catch the final audio chunks
        last_audio = self._total_audio_received
        stable_count = 0
        max_wait_iterations = 60  # Max 60 * 50ms = 3 seconds wait
        iterations = 0
        while stable_count < 15 and iterations < max_wait_iterations:  # Need 15 stable checks (750ms)
            await asyncio.sleep(0.05)
            iterations += 1
            if self._total_audio_received == last_audio:
                stable_count += 1
            else:
                stable_count = 0
                last_audio = self._total_audio_received

        # Check if interrupted during wait
        if self._is_interrupted:
            return

        # Flush remaining audio - ensure ALL buffered audio is processed
        buffer_samples = len(self._audio_buffer) // 2
        if buffer_samples > 0 and self._wav2arkit_service.is_available:
            # Use a longer minimum sample threshold to ensure we don't lose short final words
            min_samples = int(0.3 * self._audio_constants.input_sample_rate)  # 300ms minimum
            remaining_bytes = bytes(self._audio_buffer)
            if buffer_samples < min_samples:
                # Pad with silence to ensure the model processes the full audio
                padding = bytes((min_samples - buffer_samples) * 2)
                remaining_bytes = remaining_bytes + padding
            self._audio_buffer.clear()

            # Queue final audio - use blocking put to ensure it's processed
            await self._audio_chunk_queue.put(remaining_bytes)
            logger.debug(
                f"Flushing final audio: {buffer_samples} samples ({buffer_samples / self._audio_constants.input_sample_rate:.3f}s)"
            )

        # Signal that speech has ended - but wait for queues to drain first
        # Don't set _speech_ended until audio queue is empty to ensure all audio is processed
        while not self._audio_chunk_queue.empty() and not self._is_interrupted:
            await asyncio.sleep(0.05)

        self._speech_ended = True

        # Wait for workers to finish with longer timeouts
        if self._inference_task and not self._inference_task.done():
            try:
                # Wait longer to ensure all frames are generated
                await asyncio.wait_for(self._inference_task, timeout=8.0)
            except asyncio.TimeoutError:
                logger.warning("Inference task timed out during response end")
                self._inference_task.cancel()
            except asyncio.CancelledError:
                pass

        if self._frame_emit_task and not self._frame_emit_task.done():
            try:
                # Dynamic timeout based on remaining frames
                remaining_frames = self._frame_queue.qsize()
                timeout = max(5.0, remaining_frames / self._audio_constants.blendshape_fps + 2.0)
                await asyncio.wait_for(self._frame_emit_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Frame emit task timed out with {self._frame_queue.qsize()} frames remaining")
                # Don't cancel - let it drain remaining frames
                pass
            except asyncio.CancelledError:
                pass

        # Log summary
        speech_duration = time.time() - self._speech_start_time if self._speech_start_time else 0
        expected_frames = int(self._total_audio_received * self._audio_constants.blendshape_fps)
        frame_deficit = expected_frames - self._total_frames_emitted
        logger.info(
            f"Speech complete: {speech_duration:.2f}s, "
            f"audio_received={self._total_audio_received:.2f}s, "
            f"frames_emitted={self._total_frames_emitted} (expected ~{expected_frames}, deficit={frame_deficit}), "
            f"audio_queue={self._audio_chunk_queue.qsize()}, frame_queue={self._frame_queue.qsize()}"
        )

        await self.broadcast(
            {
                "type": "audio_end",
                "sessionId": self._current_session_id,
                "turnId": self._current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

        if transcript:
            msg = {
                "type": "transcript_done",
                "text": transcript,
                "role": "assistant",
                "turnId": self._current_turn_id,
                "timestamp": int(time.time() * 1000),
            }
            if item_id:
                msg["itemId"] = item_id

            await self.broadcast(msg)

        await self.broadcast({"type": "avatar_state", "state": "Listening"})

        # Reset state
        self._current_session_id = None
        self._current_turn_id = None
        self._speech_ended = False
        self._inference_task = None
        self._frame_emit_task = None

    async def _handle_transcript_delta(
        self,
        text: str,
        role: str = "assistant",
        item_id: str | None = None,
        previous_item_id: str | None = None,
    ) -> None:
        """Handle streaming transcript from AI.

        Supports optional `item_id` and `previous_item_id` so clients can
        correlate completions to the correct input item as recommended by
        realtime transcription docs.
        """
        # Use appropriate turn ID based on role
        turn_id = self._current_turn_id if role == "assistant" else f"user_{int(time.time() * 1000)}"

        msg = {
            "type": "transcript_delta",
            "text": text,
            "role": role,
            "turnId": turn_id,
            "sessionId": self._current_session_id,
            "timestamp": int(time.time() * 1000),
        }

        if item_id:
            msg["itemId"] = item_id
        if previous_item_id:
            msg["previousItemId"] = previous_item_id

        await self.broadcast(msg)

    async def _handle_user_transcript(self, transcript: str, role: str = "user") -> None:
        """Handle transcribed user speech."""
        logger.info(f"[INTERRUPT DEBUG] User transcript received at {time.time():.3f}: '{transcript}'")

        # Generate a unique turn ID for user transcript to help client correlate
        user_turn_id = f"{role}_{int(time.time() * 1000)}"
        await self.broadcast(
            {
                "type": "transcript_done",
                "text": transcript,
                "role": role,
                "turnId": user_turn_id,
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _handle_interrupted(self) -> None:
        """Handle conversation interruption."""
        interrupt_start = time.time()
        logger.info(f"[INTERRUPT DEBUG] chat.py _handle_interrupted called at {interrupt_start:.3f}")

        # Set interrupt flag FIRST to stop any in-flight processing immediately
        # This is checked synchronously in _handle_audio_delta and workers
        self._is_interrupted = True

        # Capture current turn id so we can tell clients which assistant turn ended
        interrupted_turn_id = self._current_turn_id

        # CRITICAL: Send interrupt signal to widget IMMEDIATELY before any cleanup
        # This minimizes latency - widget can stop audio playback while we do cleanup
        logger.info(
            f"[INTERRUPT DEBUG] Sending interrupt message to widget at {time.time():.3f} (+{(time.time() - interrupt_start) * 1000:.1f}ms)"
        )
        await self.broadcast(
            {
                "type": "interrupt",
                "timestamp": int(time.time() * 1000),
            }
        )
        logger.info(
            f"[INTERRUPT DEBUG] Interrupt message sent at {time.time():.3f} (+{(time.time() - interrupt_start) * 1000:.1f}ms)"
        )
        await self.broadcast({"type": "avatar_state", "state": "Listening"})

        # Clear audio buffer to prevent any buffered audio from being processed
        self._audio_buffer.clear()

        # Clear any queued audio chunks to prevent further processing
        while not self._audio_chunk_queue.empty():
            try:
                self._audio_chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear frame queue to stop any pending blendshape frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel ongoing inference if active
        if self._inference_task and not self._inference_task.done():
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None

        # Cancel frame emission task to stop broadcasting immediately
        if self._frame_emit_task and not self._frame_emit_task.done():
            self._frame_emit_task.cancel()
            try:
                await self._frame_emit_task
            except asyncio.CancelledError:
                pass
            self._frame_emit_task = None

        # Reset state
        self._speech_ended = True  # Signal to any remaining workers to stop
        self._current_session_id = None
        self._current_turn_id = None

        # Emit explicit transcript_done for assistant to ensure clients finalize bubbles
        if interrupted_turn_id:
            await self.broadcast(
                {
                    "type": "transcript_done",
                    "text": "",
                    "role": "assistant",
                    "turnId": interrupted_turn_id,
                    "interrupted": True,
                    "timestamp": int(time.time() * 1000),
                }
            )

        logger.info(
            f"[INTERRUPT DEBUG] Interrupt handling complete at {time.time():.3f} (+{(time.time() - interrupt_start) * 1000:.1f}ms total)"
        )

    # ==================== Background Workers ====================

    async def _inference_worker(self) -> None:
        """Process audio chunks through Wav2Arkit model."""
        chunk_count = 0
        total_inference_time = 0.0
        import concurrent.futures

        try:
            while True:
                # Check for interruption at start of each iteration
                if self._is_interrupted:
                    logger.debug("Inference worker stopping due to interruption")
                    break

                try:
                    dequeue_start = time.time()
                    audio_bytes = await asyncio.wait_for(
                        self._audio_chunk_queue.get(),
                        timeout=0.1,
                    )
                    dequeue_wait = (time.time() - dequeue_start) * 1000
                except asyncio.TimeoutError:
                    if self._speech_ended and self._audio_chunk_queue.empty():
                        # DEBUG: Log final inference stats
                        avg_inference_time = (total_inference_time / chunk_count * 1000) if chunk_count > 0 else 0
                        logger.debug(f"Inference complete: {chunk_count} chunks, avg={avg_inference_time:.1f}ms/chunk")
                        break
                    continue

                # Check for interruption after dequeue
                if self._is_interrupted:
                    break

                chunk_count += 1
                chunk_duration = len(audio_bytes) // 2 / self._audio_constants.input_sample_rate

                # DEBUG: Time the inference
                inference_start = time.time()
                # Run inference in thread pool to avoid blocking async loop
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    frames = await loop.run_in_executor(
                        executor, self._wav2arkit_service.process_audio_chunk, audio_bytes
                    )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Check for interruption after inference
                if self._is_interrupted:
                    break

                # DEBUG: Log inference performance
                rtf = inference_time / chunk_duration  # Real-Time Factor
                logger.debug(
                    f"Inference #{chunk_count}: {len(audio_bytes)} bytes ({chunk_duration:.3f}s) -> "
                    f"{len(frames) if frames else 0} frames in {inference_time * 1000:.1f}ms "
                    f"(RTF={rtf:.3f}, dequeue_wait={dequeue_wait:.1f}ms, "
                    f"frame_queue={self._frame_queue.qsize()})"
                )

                if frames:
                    enqueue_start = time.time()
                    for frame in frames:
                        # Check for interruption during frame enqueue
                        if self._is_interrupted:
                            break

                        # Use blocking put to ensure no frames are dropped
                        await self._frame_queue.put(frame)

                    enqueue_time = (time.time() - enqueue_start) * 1000

                    # DEBUG: Warn if frame queue is getting full
                    queue_usage = self._frame_queue.qsize() / self._frame_queue.maxsize
                    if queue_usage > 0.8:
                        logger.warning(
                            f"Frame queue high: {self._frame_queue.qsize()}/{self._frame_queue.maxsize} "
                            f"({queue_usage * 100:.0f}% full)"
                        )

                    logger.debug(
                        f"Enqueued {len(frames)} frames in {enqueue_time:.1f}ms "
                        f"(queue_size={self._frame_queue.qsize()})"
                    )

        except asyncio.CancelledError:
            logger.debug("Inference worker cancelled")
        except Exception as e:
            logger.warning(f"Inference worker error: {e}")

    async def _emit_frames(self) -> None:
        """
        Emit synchronized audio+blendshape frames to clients at 30 FPS.

        Uses time-based pacing to maintain smooth 30 FPS emission rate,
        preventing frames from being sent faster than they can be consumed.
        Only emits frames when we're behind the expected schedule based on
        elapsed real-time since speech started.
        """
        frame_times = []
        last_emit_time = time.time()
        target_frame_interval = 1.0 / self._audio_constants.blendshape_fps  # 33.33ms for 30 FPS

        try:
            while True:
                # Check for interruption at start of each iteration
                if self._is_interrupted:
                    logger.debug("Frame emission stopping due to interruption")
                    break

                try:
                    # Calculate how many frames SHOULD have been emitted by now
                    elapsed_time = time.time() - self._speech_start_time
                    target_frames = int(elapsed_time * self._audio_constants.blendshape_fps)

                    # Only emit if we're behind schedule
                    if self._total_frames_emitted >= target_frames:
                        # We're ahead of schedule, wait a bit before checking again
                        await asyncio.sleep(0.005)  # 5ms wait
                        continue

                    # Try to get frame with timeout
                    dequeue_start = time.time()
                    frame_data = await asyncio.wait_for(
                        self._frame_queue.get(),
                        timeout=0.1,
                    )
                    dequeue_wait = (time.time() - dequeue_start) * 1000
                except asyncio.TimeoutError:
                    if self._speech_ended and self._frame_queue.empty():
                        # DEBUG: Log emission stats
                        avg_emit_interval = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0
                        logger.debug(
                            f"Frame emission complete: {self._total_frames_emitted} frames, "
                            f"avg_interval={avg_emit_interval:.1f}ms"
                        )
                        break
                    continue

                # Check for interruption after dequeue
                if self._is_interrupted:
                    break

                # DEBUG: Time the broadcast
                broadcast_start = time.time()
                await self.broadcast(
                    {
                        "type": "sync_frame",
                        "weights": frame_data["weights"],
                        "audio": frame_data["audio"],  # Already base64 encoded
                        "sessionId": self._current_session_id,
                        "turnId": self._current_turn_id,
                        "timestamp": int(time.time() * 1000),
                        "frameIndex": self._blendshape_frame_idx,
                    }
                )
                broadcast_time = (time.time() - broadcast_start) * 1000

                # Track inter-frame timing
                current_time = time.time()
                frame_interval = current_time - last_emit_time
                frame_times.append(frame_interval)
                last_emit_time = current_time

                self._blendshape_frame_idx += 1
                self._total_frames_emitted += 1

                # DEBUG: Log every 30 frames (1 second)
                if self._total_frames_emitted % 30 == 0:
                    elapsed = time.time() - self._speech_start_time
                    expected_frames = int(self._total_audio_received * self._audio_constants.blendshape_fps)
                    lag = expected_frames - self._total_frames_emitted

                    # Calculate recent frame rate
                    recent_intervals = frame_times[-30:] if len(frame_times) >= 30 else frame_times
                    avg_interval = sum(recent_intervals) / len(recent_intervals) if recent_intervals else 0
                    current_fps = 1.0 / avg_interval if avg_interval > 0 else 0

                    logger.debug(
                        f"Emitted {self._total_frames_emitted} frames in {elapsed:.2f}s "
                        f"(fps={current_fps:.1f}, lag={lag} frames, "
                        f"queue={self._frame_queue.qsize()}, broadcast={broadcast_time:.1f}ms)"
                    )

                # DEBUG: Warn if emission is significantly slower than target
                if frame_interval > target_frame_interval * 1.5:  # More than 50% slower than 33ms
                    logger.warning(
                        f"Slow frame emission: {frame_interval * 1000:.1f}ms (expected {target_frame_interval * 1000:.1f}ms), "
                        f"dequeue_wait={dequeue_wait:.1f}ms, broadcast={broadcast_time:.1f}ms"
                    )

        except asyncio.CancelledError:
            # Only drain remaining frames if NOT interrupted (interruption means stop immediately)
            if not self._is_interrupted:
                logger.debug(f"Frame emission cancelled, draining {self._frame_queue.qsize()} remaining frames")
                while not self._frame_queue.empty():
                    try:
                        frame_data = self._frame_queue.get_nowait()
                        await self.broadcast(
                            {
                                "type": "sync_frame",
                                "weights": frame_data["weights"],
                                "audio": frame_data["audio"],  # Already base64 encoded
                                "sessionId": self._current_session_id,
                                "turnId": self._current_turn_id,
                                "timestamp": int(time.time() * 1000),
                                "frameIndex": self._blendshape_frame_idx,
                            }
                        )
                        self._blendshape_frame_idx += 1
                        self._total_frames_emitted += 1
                    except asyncio.QueueEmpty:
                        break
            else:
                logger.debug("Frame emission cancelled due to interruption, discarding remaining frames")
        except Exception as e:
            logger.warning(f"Frame emission error: {e}")

    # ==================== Client Message Handling ====================

    async def handle_message(self, websocket: WebSocket, data: dict) -> None:
        """Handle incoming message from client."""
        client = self.clients.get(websocket)
        if not client:
            return

        msg_type = data.get("type")

        if msg_type == "text":
            # Text message from user
            text = data.get("data", "")
            self._agent.send_text_message(text)

        elif msg_type == "audio_stream_start":
            client.is_streaming_audio = True
            client.user_id = data.get("userId", "unknown")
            logger.debug(f"Audio stream started from {client.user_id}")

        elif msg_type == "audio":
            if client.is_streaming_audio:
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    self._agent.append_audio(audio_bytes)

        elif msg_type == "audio_stream_end":
            client.is_streaming_audio = False
            logger.debug(f"Audio stream ended from {client.user_id}")

        elif msg_type == "ping":
            await websocket.send_json(
                {
                    "type": "pong",
                    "timestamp": int(time.time() * 1000),
                }
            )


# Singleton manager
manager = ChatConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice chat.

    Handles bidirectional communication between the frontend
    and AI services (OpenAI + Wav2Arkit).
    """
    # Import here to avoid circular dependency
    from main import auth_middleware

    session_id = str(uuid.uuid4())

    # Authenticate WebSocket connection
    settings = get_settings()
    if settings.auth_enabled and auth_middleware:
        is_authenticated, error = await auth_middleware.authenticate_websocket(websocket, session_id)
        if not is_authenticated:
            await websocket.close(code=1008, reason=error)
            return

    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
