import asyncio
import base64
import concurrent.futures
import time
from pathlib import Path
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
        
        # Determine negotiated input sample rate
        self.input_sample_rate = self.settings.input_sample_rate
        # Check if agent has a preference (BaseAgent property)
        if hasattr(self.agent, "input_sample_rate"):
             self.input_sample_rate = self.agent.input_sample_rate
             logger.info(f"Session {self.session_id}: Negotiated input sample rate: {self.input_sample_rate}")

        # Determine negotiated output sample rate
        self.output_sample_rate = self.settings.output_sample_rate
        # Check if agent has a preference
        if hasattr(self.agent, "output_sample_rate"):
             self.output_sample_rate = self.agent.output_sample_rate
             logger.info(f"Session {self.session_id}: Negotiated output sample rate: {self.output_sample_rate}")

        # Client State
        self.is_streaming_audio = False
        self.user_id = ""
        
        # Audio Debugging
        self.debug_file_path = None
        if self.settings.debug_audio_capture:
            timestamp = int(time.time())
            self.debug_file_path = Path(f"debug_input_{self.session_id[:8]}_{timestamp}.pcm")
            logger.info(f"Session {self.session_id}: Debug audio capture ENABLED -> {self.debug_file_path}")
        self.is_active = True

        # Audio and frame processing state
        self.audio_buffer: bytearray = bytearray()
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=120)
        self.audio_chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=15)

        # Tracking state
        self.current_turn_id: str | None = None
        self.current_turn_session_id: str | None = None  # Turn-level session ID from agent
        self.first_audio_received: bool = False

        # Track accumulated text for accurate interruption cutting
        self.current_turn_text: str = ""
        
        # Where we are in the text stream (time-wise)
        self.virtual_cursor_text_ms = 0.0
        
        # Calibration constant for transcript timing (configurable via settings)
        self.chars_per_second = settings.transcript_chars_per_second
        # Check if agent has a preference
        if hasattr(self.agent, "transcript_speed"):
             self.chars_per_second = self.agent.transcript_speed
             logger.info(f"Session {self.session_id}: Using agent-specific transcript speed: {self.chars_per_second} chars/sec")
        
        # Background tasks
        self.frame_emit_task: asyncio.Task | None = None
        self.inference_task: asyncio.Task | None = None
        
        # Thread safety and re-entry prevention during interruptions
        self._interruption_lock = asyncio.Lock()
  
        # Dedicated thread pool for heavier inference tasks
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._setup_agent_handlers()

    def _setup_agent_handlers(self) -> None:
        """Setup event handlers specific to this session's agent."""
        self.agent.set_event_handlers(
            on_audio_delta=self._handle_audio_delta,
            on_response_start=self._handle_response_start,
            on_response_end=self._handle_response_end,
            on_user_transcript=self._handle_user_transcript,
            on_transcript_delta=self._handle_transcript_delta,
            on_interrupted=self._handle_interrupted
        )

    async def start(self) -> None:
        """Initialize connection to agent."""
        
        # [PROTOCOL] Configure client with negotiated sample rate
        await self.send_json({
            "type": "config",
            "audio": {
                "inputSampleRate": self.input_sample_rate
            }
        })
        
        if not self.agent.is_connected:
            await self.agent.connect()

    async def stop(self) -> None:
        """Cleanup resources."""
        self.is_active = False
        self.is_interrupted = True
        
        await self._cancel_and_wait_task(self.inference_task)
        await self._cancel_and_wait_task(self.frame_emit_task)
        
        # Shutdown executor
        self._inference_executor.shutdown(wait=False)

        # Disconnect Agent
        await self.agent.disconnect()
        logger.info(f"Session {self.session_id} stopped.")
        
        # Shutdown executor
        self._inference_executor.shutdown(wait=False)

        # Disconnect Agent
        await self.agent.disconnect()
        logger.info(f"Session {self.session_id} stopped.")

    async def _cancel_and_wait_task(self, task: asyncio.Task | None) -> None:
        """Helper to safely cancel and await a task."""
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error during task cancellation: {e}")

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

    async def _drain_queue(self, queue: asyncio.Queue) -> None:
        """Helper to flush an asyncio queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _handle_response_start(self, session_id: str) -> None:
        """Handle AI response start."""
        logger.info(f"Session {self.session_id}: Response start received, creating turn for session {session_id}")
        self.current_turn_session_id = session_id  # Store turn-level session ID
        self.current_turn_id = f"turn_{int(time.time() * 1000)}_{session_id[:8]}"
        self.speech_start_time = time.time()
        self.actual_audio_start_time = 0
        self.total_frames_emitted = 0
        self.total_audio_received = 0
        self.blendshape_frame_idx = 0
        self.speech_ended = False
        self.is_interrupted = False
        self.first_audio_received = False
        self.current_turn_text = ""

        # Clear queues
        self.audio_buffer.clear()
        await self._drain_queue(self.frame_queue)
        await self._drain_queue(self.audio_chunk_queue)

        if self.wav2arkit_service.is_available:
             self.wav2arkit_service.reset_context()

        # Send start event BEFORE starting tasks to ensure client is ready to receive frames
        logger.debug(f"Session {self.session_id}: Sending audio_start with turnId={self.current_turn_id}, sessionId={self.current_turn_session_id}")
        await self.send_json({"type": "avatar_state", "state": "Responding"})
        await self.send_json(
            {
                "type": "audio_start",
                "sessionId": self.current_turn_session_id,
                "turnId": self.current_turn_id,
                "sampleRate": self.output_sample_rate,
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

        # Drop packets if worker tasks are dead (prevents queue blocking/freezing)
        if self.inference_task is None or self.inference_task.done():
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
        chunk_duration = chunk_samples / self.input_sample_rate
        self.total_audio_received += chunk_duration
        
        buffer_samples = len(self.audio_buffer) // 2
        buffer_duration = buffer_samples / self.input_sample_rate

        if buffer_duration >= self.settings.audio_chunk_duration:
            chunk_bytes_size = int(
                self.settings.audio_chunk_duration * self.input_sample_rate * 2
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
            min_samples = int(0.3 * self.input_sample_rate)
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
        
        # Check interruption before final waits/sends
        if self.is_interrupted:
            return

        if self.frame_emit_task and not self.frame_emit_task.done():
            try:
                remaining_frames = self.frame_queue.qsize()
                timeout = max(5.0, remaining_frames / self.settings.blendshape_fps + 2.0)
                await asyncio.wait_for(self.frame_emit_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        if self.is_interrupted:
            return

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
        
        if role == "assistant":
            self.current_turn_text += text
            
            # Calculate heuristic timestamp
            # We assume the text starts "now" at the end of the previous text
            char_duration_ms = (len(text) / self.chars_per_second) * 1000
            
            start_offset = self.virtual_cursor_text_ms
            end_offset = start_offset + char_duration_ms
            
            self.virtual_cursor_text_ms += char_duration_ms
        else:
            start_offset = 0
            end_offset = 0

        turn_id = self.current_turn_id if role == "assistant" else f"user_{int(time.time() * 1000)}"
        msg = {
            "type": "transcript_delta",
            "text": text,
            "role": role,
            "turnId": turn_id,
            "sessionId": self.session_id,
            "timestamp": int(time.time() * 1000),
            "startOffset": int(start_offset),
            "endOffset": int(end_offset)
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

    async def _calculate_truncated_text(self) -> str:
        """Helper to calculate truncated text based on audio duration."""
        try:
            seconds_spoken = self.total_frames_emitted / self.settings.blendshape_fps
            estimated_chars = int(seconds_spoken * 16)  # English avg: ~16 chars/sec
            
            final_text = self.current_turn_text
            if len(final_text) > estimated_chars:
                # Try to cut at word boundary
                search_buffer = 10
                cut_index = final_text.find(" ", estimated_chars)
                
                if cut_index != -1 and cut_index - estimated_chars < search_buffer:
                    final_text = final_text[:cut_index] + "..."
                else:
                    final_text = final_text[:estimated_chars] + "..."
            
            logger.debug(f"Session {self.session_id}: Text truncated to {len(final_text)} chars ({seconds_spoken:.2f}s spoken)")
            return final_text
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error computing text truncation: {e}")
            return self.current_turn_text

    async def _execute_interruption_sequence(self) -> None:
        """
        Executes the core cleanup logic inside the interruption lock.
        Separated for clarity and maintenance.
        """
        # Double-check inside lock
        if self.is_interrupted:
            return

        self.is_interrupted = True
        interrupted_turn_id = self.current_turn_id
        logger.debug(f"Session {self.session_id}: Cancellation sequence (turn_id: {interrupted_turn_id})")

        # 1. Cancel Agent Response
        try:
            self.agent.cancel_response()
        except Exception as e:
            logger.debug(f"Session {self.session_id}: Agent cancel: {e}")

        # 2. Clear Local Buffers
        self.audio_buffer.clear()
        
        # 3. Drain Queues
        await self._drain_queue(self.audio_chunk_queue)
        await self._drain_queue(self.frame_queue)

        # 4. Cancel Tasks
        await self._cancel_and_wait_task(self.inference_task)
        await self._cancel_and_wait_task(self.frame_emit_task)
        
        self.inference_task = None
        self.frame_emit_task = None

        # 5. Reset State
        self.speech_ended = True
        self.current_turn_id = None
        
        # 6. Calc Truncation & Send
        final_text = await self._calculate_truncated_text()
        
        try:
            await self.send_json({
                "type": "transcript_done",
                "text": final_text,
                "role": "assistant",
                "turnId": interrupted_turn_id,
                "interrupted": True,
                "timestamp": int(time.time() * 1000)
            })
            logger.info(f"Session {self.session_id}: Interruption complete")

        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to send transcript_done: {e}")

        # Critical: Re-enable processing
        self.is_interrupted = False

    async def _handle_interrupted(self) -> None:
        """
        Production-grade interruption handler.
        """
        # Fast-path check
        if self.is_interrupted:
            return

        # [MODIFIED] Calculate offset immediately for the instant message
        current_offset_ms = 0
        if self.settings.blendshape_fps > 0:
             current_offset_ms = int((self.total_frames_emitted / self.settings.blendshape_fps) * 1000)

        # Immediate client notification
        try:
            await self.send_json({
                "type": "interrupt", 
                "timestamp": int(time.time() * 1000),
                "turnId": self.current_turn_id,   # Add context
                "offsetMs": current_offset_ms     # Add exact cut point
            })
            await self.send_json({"type": "avatar_state", "state": "Listening"})
        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to send interrupt: {e}")
            return

        # Lock acquisition with timeout
        try:
            async def _locked_execution():
                async with self._interruption_lock:
                    await self._execute_interruption_sequence()

            await asyncio.wait_for(_locked_execution(), timeout=1.0)

        except asyncio.TimeoutError:
            logger.error(f"Session {self.session_id}: TIMEOUT unlocking interrupted state")
            self.is_interrupted = False
        except Exception as e:
            logger.error(f"Session {self.session_id}: EXCEPTION in interruption: {e}", exc_info=True)
            self.is_interrupted = False

    async def on_error(self, error: Any) -> None:
        """Handle error events from the agent."""
        logger.error(f"Session {self.session_id}: Agent error: {error}")

    async def _inference_worker(self) -> None:
        """Process audio chunks through Wav2Arkit model."""
        try:
            logger.info(f"Session {self.session_id}: Inference worker started")
            loop = asyncio.get_event_loop()
            
            while True:
                if self.is_interrupted:
                    await asyncio.sleep(0.01)
                    continue
                
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
                    await asyncio.sleep(0.01)
                    continue

                # Run inference in default executor
                try:
                    # Use dedicated executor to prevent blocking
                    frames = await loop.run_in_executor(
                        self._inference_executor, 
                        self.wav2arkit_service.process_audio_chunk, 
                        audio_bytes,
                        self.output_sample_rate
                    )
                except Exception as e:
                    logger.error(f"Inference processing failed: {e}", exc_info=True)
                    # Prevent CPU spin loop on persistent error
                    await asyncio.sleep(0.1)
                    continue

                if self.is_interrupted:
                    await asyncio.sleep(0.01)
                    continue

                if frames:
                    for frame in frames:
                        if self.is_interrupted:
                            # Break inner loop but continue outer loop
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
                    await asyncio.sleep(0.01)
                    continue

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
                    await asyncio.sleep(0.01)
                    continue

                if self.blendshape_frame_idx % 30 == 0:
                    audio_b64 = frame_data.get("audio", "")
                    audio_size = len(audio_b64)
                    logger.debug(f"Session {self.session_id}: Sending sync_frame {self.blendshape_frame_idx} (Active). Audio payload: {audio_size} chars")

                await self.send_json(
                    {
                        "type": "sync_frame",
                        "weights": frame_data["weights"],
                        "audio": frame_data["audio"],
                        "sessionId": self.current_turn_session_id,
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
                                "sessionId": self.current_turn_session_id,
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
            # Unstick session: If user starts speaking, previous interruption is over.
            # This ensures we recover even if OpenAI errored out in the previous turn.
            self.is_interrupted = False

        elif msg_type == "audio":
            if self.is_streaming_audio:
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Debug: Save raw input audio
                    if self.debug_file_path:
                        try:
                            with open(self.debug_file_path, "ab") as f:
                                f.write(audio_bytes)
                        except Exception as e:
                            logger.warning(f"Failed to write debug audio: {e}")

                    self.agent.append_audio(audio_bytes)

        elif msg_type == "interrupt":
            logger.info(f"Session {self.session_id}: Received explicit interrupt command from client")
            await self._handle_interrupted()

        elif msg_type == "ping":
            await self.send_json({
                "type": "pong",
                "timestamp": int(time.time() * 1000),
            })
