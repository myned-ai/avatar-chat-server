"""
Conversation management for Realtime API.

Holds conversation history and performs event validation.
Tracks items, responses, and audio buffers.
"""

import copy

import numpy as np

from .utils import RealtimeUtils


class RealtimeConversation:
    """
    Holds conversation history and performs event validation for RealtimeAPI.
    """

    # OpenAI Realtime API uses 24kHz sample rate
    DEFAULT_FREQUENCY = 24000

    def __init__(self):
        self.clear()
        self._setup_event_processors()

    def _setup_event_processors(self):
        """Setup event processors for different event types."""
        self.event_processors = {
            "conversation.item.created": self._process_item_created,
            "conversation.item.truncated": self._process_item_truncated,
            "conversation.item.deleted": self._process_item_deleted,
            "conversation.item.input_audio_transcription.completed": self._process_input_audio_transcription_completed,
            "input_audio_buffer.speech_started": self._process_speech_started,
            "input_audio_buffer.speech_stopped": self._process_speech_stopped,
            "response.created": self._process_response_created,
            "response.output_item.added": self._process_response_output_item_added,
            "response.output_item.done": self._process_response_output_item_done,
            "response.content_part.added": self._process_response_content_part_added,
            "response.audio_transcript.delta": self._process_response_audio_transcript_delta,
            "response.audio.delta": self._process_response_audio_delta,
            "response.text.delta": self._process_response_text_delta,
            "response.function_call_arguments.delta": self._process_response_function_call_arguments_delta,
            "error": self._process_error,
        }

    def clear(self) -> bool:
        """
        Clears the conversation history and resets to default.

        Returns:
            True on success
        """
        self.item_lookup: dict[str, dict] = {}
        self.items: list[dict] = []
        self.response_lookup: dict[str, dict] = {}
        self.responses: list[dict] = []
        self.queued_speech_items: dict[str, dict] = {}
        self.queued_transcript_items: dict[str, dict] = {}
        self.queued_input_audio: np.ndarray | None = None
        return True

    def queue_input_audio(self, input_audio: np.ndarray) -> np.ndarray:
        """
        Queue input audio for manual speech event.

        Args:
            input_audio: Int16 audio array

        Returns:
            The queued audio
        """
        self.queued_input_audio = input_audio
        return input_audio

    def process_event(self, event: dict, *args) -> tuple[dict | None, dict | None]:
        """
        Process an event from the WebSocket server and compose items.

        Args:
            event: The event object from the server
            *args: Additional arguments (e.g., input_audio_buffer)

        Returns:
            Tuple of (item, delta) - item may be None for some events
        """
        if "event_id" not in event:
            raise ValueError(f'Missing "event_id" on event: {event}')
        if "type" not in event:
            raise ValueError(f'Missing "type" on event: {event}')

        event_type = event["type"]
        processor = self.event_processors.get(event_type)

        if not processor:
            raise ValueError(f'Missing conversation event processor for "{event_type}"')

        return processor(event, *args)

    def get_item(self, item_id: str) -> dict | None:
        """
        Retrieves an item by id.

        Args:
            item_id: The ID of the item

        Returns:
            The item or None if not found
        """
        return self.item_lookup.get(item_id)

    def get_items(self) -> list[dict]:
        """
        Returns all conversation items.

        Returns:
            List of all items
        """
        return self.items.copy()

    # Event Processors

    def _process_item_created(self, event: dict, *args) -> tuple[dict, None]:
        """Process conversation.item.created event."""
        item = event["item"]
        # Deep copy values
        new_item = copy.deepcopy(item)

        if new_item["id"] not in self.item_lookup:
            self.item_lookup[new_item["id"]] = new_item
            self.items.append(new_item)

        # Initialize formatted fields
        new_item["formatted"] = {
            "audio": np.array([], dtype=np.int16),
            "text": "",
            "transcript": "",
        }

        # If we have a speech item, can populate audio
        if new_item["id"] in self.queued_speech_items:
            queued = self.queued_speech_items.pop(new_item["id"])
            if "audio" in queued:
                new_item["formatted"]["audio"] = queued["audio"]

        # Populate formatted text if it comes out on creation
        if new_item.get("content"):
            text_content = [
                c.get("text") or c.get("transcript", "")
                for c in new_item["content"]
                if c.get("type") in ("input_text", "text")
            ]
            new_item["formatted"]["text"] = "\n".join(text_content)

        # If we have a transcript item queued, populate it
        if new_item["id"] in self.queued_transcript_items:
            queued = self.queued_transcript_items.pop(new_item["id"])
            new_item["formatted"]["transcript"] = queued.get("transcript", "")

        # Handle specific item types
        if new_item.get("type") == "message":
            if new_item.get("role") == "user":
                new_item["status"] = "completed"
                # Check for input_audio content
                if new_item.get("content"):
                    for _i, content in enumerate(new_item["content"]):
                        if content.get("type") == "input_audio":
                            new_item["formatted"]["audio"] = self.queued_input_audio or np.array([], dtype=np.int16)
                            new_item["formatted"]["transcript"] = content.get("transcript", "")
                            self.queued_input_audio = None
            else:
                new_item["status"] = "in_progress"
        elif new_item.get("type") == "function_call":
            new_item["formatted"]["tool"] = {
                "type": "function",
                "name": new_item.get("name", ""),
                "call_id": new_item.get("call_id", ""),
                "arguments": "",
            }
            new_item["status"] = "in_progress"
        elif new_item.get("type") == "function_call_output":
            new_item["status"] = "completed"
            new_item["formatted"]["output"] = new_item.get("output", "")

        return (new_item, None)

    def _process_item_truncated(self, event: dict, *args) -> tuple[dict, None]:
        """Process conversation.item.truncated event."""
        item_id = event["item_id"]
        audio_end_ms = event["audio_end_ms"]

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'item.truncated: Item "{item_id}" not found')

        end_index = int((audio_end_ms * self.DEFAULT_FREQUENCY) / 1000)
        item["formatted"]["transcript"] = ""
        item["formatted"]["audio"] = item["formatted"]["audio"][:end_index]

        return (item, None)

    def _process_item_deleted(self, event: dict, *args) -> tuple[dict, None]:
        """Process conversation.item.deleted event."""
        item_id = event["item_id"]

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'item.deleted: Item "{item_id}" not found')

        del self.item_lookup[item["id"]]
        if item in self.items:
            self.items.remove(item)

        return (item, None)

    def _process_input_audio_transcription_completed(self, event: dict, *args) -> tuple[dict | None, dict | None]:
        """Process conversation.item.input_audio_transcription.completed event."""
        item_id = event["item_id"]
        content_index = event.get("content_index", 0)
        transcript = event.get("transcript", "")

        item = self.item_lookup.get(item_id)
        # Use single space for empty transcript
        formatted_transcript = transcript or " "

        if not item:
            # Can receive transcripts in VAD mode before item.created
            self.queued_transcript_items[item_id] = {"transcript": formatted_transcript}
            return (None, None)
        else:
            if item.get("content") and len(item["content"]) > content_index:
                item["content"][content_index]["transcript"] = transcript
            item["formatted"]["transcript"] = formatted_transcript
            return (item, {"transcript": transcript})

    def _process_speech_started(self, event: dict, *args) -> tuple[None, None]:
        """Process input_audio_buffer.speech_started event."""
        item_id = event["item_id"]
        audio_start_ms = event["audio_start_ms"]
        self.queued_speech_items[item_id] = {"audio_start_ms": audio_start_ms}
        return (None, None)

    def _process_speech_stopped(self, event: dict, *args) -> tuple[None, None]:
        """Process input_audio_buffer.speech_stopped event."""
        item_id = event["item_id"]
        audio_end_ms = event["audio_end_ms"]
        input_audio_buffer = args[0] if args else None

        if item_id not in self.queued_speech_items:
            self.queued_speech_items[item_id] = {"audio_start_ms": audio_end_ms}

        speech = self.queued_speech_items[item_id]
        speech["audio_end_ms"] = audio_end_ms

        if input_audio_buffer is not None and len(input_audio_buffer) > 0:
            start_index = int((speech["audio_start_ms"] * self.DEFAULT_FREQUENCY) / 1000)
            end_index = int((speech["audio_end_ms"] * self.DEFAULT_FREQUENCY) / 1000)
            speech["audio"] = input_audio_buffer[start_index:end_index]

        return (None, None)

    def _process_response_created(self, event: dict, *args) -> tuple[None, None]:
        """Process response.created event."""
        response = event["response"]
        if response["id"] not in self.response_lookup:
            self.response_lookup[response["id"]] = response
            self.responses.append(response)
        return (None, None)

    def _process_response_output_item_added(self, event: dict, *args) -> tuple[dict, None]:
        """Process response.output_item.added event."""
        response_id = event["response_id"]
        item = event["item"]

        # Register item in item_lookup so subsequent deltas can find it
        new_item = copy.deepcopy(item)
        if "formatted" not in new_item:
            new_item["formatted"] = {
                "audio": np.array([], dtype=np.int16),
                "text": "",
                "transcript": "",
            }

        if new_item["id"] not in self.item_lookup:
            self.item_lookup[new_item["id"]] = new_item
            self.items.append(new_item)
        
        # Use the stored item reference
        stored_item = self.item_lookup[new_item["id"]]

        response = self.response_lookup.get(response_id)
        if not response:
            raise ValueError(f'response.output_item.added: Response "{response_id}" not found')

        if "output" not in response:
            response["output"] = []
        response["output"].append(stored_item)

        return (stored_item, None)

    def _process_response_output_item_done(self, event: dict, *args) -> tuple[dict | None, None]:
        """Process response.output_item.done event."""
        item = event.get("item")
        if not item:
            return (None, None)

        found_item = self.item_lookup.get(item["id"])
        if not found_item:
            return (None, None)

        found_item["status"] = item.get("status", "completed")
        return (found_item, None)

    def _process_response_content_part_added(self, event: dict, *args) -> tuple[dict | None, None]:
        """Process response.content_part.added event."""
        item_id = event["item_id"]
        part = event["part"]

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'response.content_part.added: Item "{item_id}" not found')

        if "content" not in item:
            item["content"] = []
        item["content"].append(part)

        return (item, None)

    def _process_response_audio_transcript_delta(self, event: dict, *args) -> tuple[dict, dict]:
        """Process response.audio_transcript.delta event."""
        item_id = event["item_id"]
        delta = event.get("delta", "")

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'response.audio_transcript.delta: Item "{item_id}" not found')

        item["formatted"]["transcript"] = item["formatted"].get("transcript", "") + delta

        return (item, {"transcript": delta})

    def _process_response_audio_delta(self, event: dict, *args) -> tuple[dict, dict]:
        """Process response.audio.delta event."""
        item_id = event["item_id"]
        delta = event.get("delta", "")

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'response.audio.delta: Item "{item_id}" not found')

        # Decode base64 audio and append
        audio_bytes = RealtimeUtils.base64_to_array_buffer(delta)
        append_values = np.frombuffer(audio_bytes, dtype=np.int16)

        item["formatted"]["audio"] = RealtimeUtils.merge_int16_arrays(item["formatted"]["audio"], append_values)

        return (item, {"audio": append_values})

    def _process_response_text_delta(self, event: dict, *args) -> tuple[dict, dict]:
        """Process response.text.delta event."""
        item_id = event["item_id"]
        delta = event.get("delta", "")

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'response.text.delta: Item "{item_id}" not found')

        item["formatted"]["text"] = item["formatted"].get("text", "") + delta

        return (item, {"text": delta})

    def _process_response_function_call_arguments_delta(self, event: dict, *args) -> tuple[dict, dict]:
        """Process response.function_call_arguments.delta event."""
        item_id = event["item_id"]
        delta = event.get("delta", "")

        item = self.item_lookup.get(item_id)
        if not item:
            raise ValueError(f'response.function_call_arguments.delta: Item "{item_id}" not found')

        if "formatted" not in item:
            item["formatted"] = {}
        if "tool" not in item["formatted"]:
            item["formatted"]["tool"] = {"arguments": ""}

        item["formatted"]["tool"]["arguments"] += delta

        return (item, {"arguments": delta})

    def _process_error(self, event: dict, *args) -> tuple[dict, None]:
        """
        Process error event from the Realtime API.

        Error events have the structure:
        {
            "event_id": "...",
            "type": "error",
            "error": {
                "type": "invalid_request_error" | "server_error" | ...,
                "code": "...",
                "message": "...",
                "param": "..." | null,
                "event_id": "..." | null  # The event_id that caused the error
            }
        }

        Returns:
            Tuple of (error_item, None) where error_item contains formatted error info
        """
        error_data = event.get("error", {})

        error_item = {
            "id": event.get("event_id", ""),
            "type": "error",
            "error": {
                "type": error_data.get("type", "unknown_error"),
                "code": error_data.get("code"),
                "message": error_data.get("message", "Unknown error occurred"),
                "param": error_data.get("param"),
                "event_id": error_data.get("event_id"),
            },
            "formatted": {
                "text": error_data.get("message", "Unknown error occurred"),
                "error_type": error_data.get("type", "unknown_error"),
                "error_code": error_data.get("code"),
            },
        }

        return (error_item, None)
