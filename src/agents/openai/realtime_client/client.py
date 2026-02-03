"""
High-level Realtime API Client.

Primary abstraction for interfacing with the Realtime API.
Enables rapid application development with a simplified control flow.

Features:
- Automatic session management
- Conversation state tracking
- Tool/function calling support
- Audio streaming helpers
- Custom events for application flow
"""

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from .api import RealtimeAPI
from .conversation import RealtimeConversation
from .event_handler import RealtimeEventHandler
from .utils import RealtimeUtils

# Type definitions
ToolDefinition = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], Any]


class RealtimeClient(RealtimeEventHandler):
    """
    High-level client for the OpenAI Realtime API.
    
    Provides:
    - Simplified session management
    - Automatic conversation state tracking
    - Tool/function calling support
    - Helper methods for audio streaming
    - Custom events: conversation.updated, conversation.item.appended,
      conversation.item.completed, conversation.interrupted
    """
    
    def __init__(
        self,
        model: str,
        url: str | None = None,
        api_key: str | None = None,
        debug: bool = False
    ):
        """
        Create a new RealtimeClient instance.
        
        Args:
            url: WebSocket URL (defaults to OpenAI's Realtime API)
            api_key: OpenAI API key
            model: Model to use for the session
            debug: Enable debug logging
        """
        super().__init__()
        
        self.model = model
        
        # Default session configuration
        self.default_session_config = {
            'modalities': ['text', 'audio'],
            'instructions': '',
            'voice': 'alloy',
            'speed': 1.0,
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_transcription': None,
            'turn_detection': None,
            'tools': [],
            'tool_choice': 'auto',
            'temperature': 0.8,
            'max_response_output_tokens': 4096,
        }
        
        # Default VAD configuration
        self.default_server_vad_config = {
            'type': 'server_vad',
            'threshold': 0.5,
            'prefix_padding_ms': 300,
            'silence_duration_ms': 200,
        }
        
        # Transcription models
        self.transcription_models = [{'model': 'whisper-1'}]
        
        # Initialize state
        self.session_config: dict[str, Any] = {}
        self.tools: dict[str, dict[str, Any]] = {}  # name -> {definition, handler}
        self.session_created = False
        self.input_audio_buffer = np.array([], dtype=np.int16)
        self._response_cancelled = False  # Flag to stop dispatching audio after interruption
        self._is_responding = False  # Flag to track if there's an active response
        
        # Create underlying API and conversation handlers
        self.realtime = RealtimeAPI(url=url, api_key=api_key, debug=debug)
        self.conversation = RealtimeConversation()
        
        # Initialize configuration
        self._reset_config()
        self._add_api_event_handlers()
    
    def _reset_config(self) -> bool:
        """
        Resets session config and conversation config to default.
        
        Returns:
            True on success
        """
        self.session_created = False
        self.tools = {}
        self.session_config = copy.deepcopy(self.default_session_config)
        self.input_audio_buffer = np.array([], dtype=np.int16)
        return True
    
    def _add_api_event_handlers(self) -> bool:
        """
        Sets up event handlers for a fully-functional application control flow.
        
        Returns:
            True on success
        """
        # Event logging handlers
        def on_client_event(event):
            realtime_event = {
                'time': asyncio.get_event_loop().time(),
                'source': 'client',
                'event': event,
            }
            self.dispatch('realtime.event', realtime_event)
        
        def on_server_event(event):
            realtime_event = {
                'time': asyncio.get_event_loop().time(),
                'source': 'server',
                'event': event,
            }
            self.dispatch('realtime.event', realtime_event)
        
        self.realtime.on('client.*', on_client_event)
        self.realtime.on('server.*', on_server_event)
        
        # Handle session created
        def on_session_created(event):
            self.session_created = True
        
        self.realtime.on('server.session.created', on_session_created)
        
        # Setup for application control flow
        def handler(event, *args):
            """Process event through conversation."""
            try:
                item, delta = self.conversation.process_event(event, *args)
                return {'item': item, 'delta': delta}
            except Exception as e:
                print(f"Error processing event: {e}")
                return {'item': None, 'delta': None}
        
        def handler_with_dispatch(event, *args):
            """Process event and dispatch conversation.updated."""
            # Skip dispatching audio deltas if response was cancelled (interrupted)
            event_type = event.get('type', '')
            if self._response_cancelled and 'audio' in event_type and 'delta' in event_type:
                return {'item': None, 'delta': None}

            result = handler(event, *args)
            item = result.get('item')
            delta = result.get('delta')
            if item:
                self.dispatch('conversation.updated', {'item': item, 'delta': delta})
            return result
        
        async def call_tool(tool: dict[str, Any]):
            """Call a registered tool and send the result."""
            try:
                arguments = json.loads(tool.get('arguments', '{}'))
                tool_config = self.tools.get(tool['name'])
                
                if not tool_config:
                    raise ValueError(f'Tool "{tool["name"]}" has not been added')
                
                result = tool_config['handler'](arguments)
                
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    result = await result
                
                self.realtime.send('conversation.item.create', {
                    'item': {
                        'type': 'function_call_output',
                        'call_id': tool['call_id'],
                        'output': json.dumps(result),
                    }
                })
            except Exception as e:
                self.realtime.send('conversation.item.create', {
                    'item': {
                        'type': 'function_call_output',
                        'call_id': tool['call_id'],
                        'output': json.dumps({'error': str(e)}),
                    }
                })
            
            self.create_response()
        
        # Register handlers for conversation events
        def on_response_created(event):
            # Reset cancelled flag and mark as responding when new response starts
            self._response_cancelled = False
            self._is_responding = True
            handler(event)

        self.realtime.on('server.response.created', on_response_created)
        
        def on_response_done(event):
            # Mark as no longer responding when response completes
            self._is_responding = False
            # Note: Don't call handler(event) - response.done doesn't need conversation processing

        self.realtime.on('server.response.done', on_response_done)
        self.realtime.on('server.response.output_item.added', handler)
        self.realtime.on('server.response.content_part.added', handler)
        
        def on_speech_started(event):
            # ALWAYS dispatch the interrupt event.
            # Even if the server is "done" generating, the client might still be playing
            # buffered audio. The application layer needs this signal to clear that buffer.
            self.dispatch('conversation.interrupted')

            # Set the cancelled flag to stop processing any lingering network packets
            self._response_cancelled = True

            # Only send the cancel command upstream if OpenAi is actually generating
            if self._is_responding:
                self.realtime.send('response.cancel')

            # Process the event
            handler(event)

        self.realtime.on('server.input_audio_buffer.speech_started', on_speech_started)
        
        def on_speech_stopped(event):
            handler(event, self.input_audio_buffer)
        
        self.realtime.on('server.input_audio_buffer.speech_stopped', on_speech_stopped)
        
        # Handlers to update application state
        def on_item_created(event):
            result = handler_with_dispatch(event)
            item = result.get('item')
            self.dispatch('conversation.item.appended', {'item': item})
        
        self.realtime.on('server.conversation.item.created', on_item_created)
        
        self.realtime.on('server.conversation.item.truncated', handler_with_dispatch)
        self.realtime.on('server.conversation.item.deleted', handler_with_dispatch)
        
        def on_input_transcription_completed(event):
            result = handler_with_dispatch(event)
            item = result.get('item')
            delta = result.get('delta')
            if item and delta and delta.get('transcript'):
                self.dispatch('conversation.item.input_transcription.completed', {
                    'item': item,
                    'transcript': delta['transcript']
                })
        
        self.realtime.on('server.conversation.item.input_audio_transcription.completed', on_input_transcription_completed)
        
        def on_input_transcription_failed(event):
            """Handle input audio transcription failure."""
            # Dispatch the failure event so applications can handle it
            self.dispatch('conversation.item.input_transcription.failed', {
                'item_id': event.get('item_id'),
                'content_index': event.get('content_index'),
                'error': event.get('error', {}),
            })
        
        self.realtime.on('server.conversation.item.input_audio_transcription.failed', on_input_transcription_failed)
        
        self.realtime.on('server.response.audio_transcript.delta', handler_with_dispatch)
        self.realtime.on('server.response.audio.delta', handler_with_dispatch)
        self.realtime.on('server.response.text.delta', handler_with_dispatch)
        self.realtime.on('server.response.function_call_arguments.delta', handler_with_dispatch)
        
        # Handle completed function calls
        def on_output_item_done(event):
            result = handler_with_dispatch(event)
            item = result.get('item')
            if item and item.get('status') == 'completed':
                self.dispatch('conversation.item.completed', {'item': item})
            if item and item.get('formatted', {}).get('tool'):
                asyncio.create_task(call_tool(item['formatted']['tool']))
        
        self.realtime.on('server.response.output_item.done', on_output_item_done)
        
        # Handle error events from OpenAI
        def on_error(event):
            """Handle error events from the Realtime API."""
            # Dispatch the error event to the application layer
            self.dispatch('error', event)
        
        self.realtime.on('server.error', on_error)
        
        return True
    
    def is_connected(self) -> bool:
        """
        Check if the client is connected and session has started.
        
        Returns:
            True if connected
        """
        return self.realtime.is_connected()
    
    def reset(self) -> bool:
        """
        Resets the client instance entirely: disconnects and clears active config.
        
        Returns:
            True on success
        """
        self.disconnect()
        self.clear_event_handlers()
        self.realtime.clear_event_handlers()
        self._reset_config()
        self._add_api_event_handlers()
        return True
    
    async def connect(self) -> bool:
        """
        Connects to the Realtime WebSocket API.
        Updates session config and conversation config.
        
        Returns:
            True on success
        """
        if self.is_connected():
            raise ValueError("Already connected, use .disconnect() first")
        
        await self.realtime.connect(model=self.model)
        #self.update_session() <-- Avoid sending default values unnecessarily
        return True
    
    async def wait_for_session_created(self) -> bool:
        """
        Waits for a session.created event to be executed before proceeding.
        
        Returns:
            True when session is created
        """
        if not self.is_connected():
            raise ValueError("Not connected, use .connect() first")
        
        while not self.session_created:
            await asyncio.sleep(0.001)
        
        return True
    
    def disconnect(self):
        """Disconnects from the Realtime API and clears the conversation history."""
        self.session_created = False
        if self.realtime.is_connected():
            self.realtime.disconnect()
        self.conversation.clear()
    
    def get_turn_detection_type(self) -> str | None:
        """
        Gets the active turn detection mode.
        
        Returns:
            "server_vad" or None
        """
        turn_detection = self.session_config.get('turn_detection')
        if turn_detection:
            return turn_detection.get('type')
        return None
    
    def add_tool(self, definition: ToolDefinition, handler: ToolHandler) -> dict[str, Any]:
        """
        Add a tool and handler.
        
        Args:
            definition: Tool definition with name, description, parameters
            handler: Function to call when tool is invoked
            
        Returns:
            dict with definition and handler
        """
        if 'name' not in definition:
            raise ValueError("Tool definition must have a 'name'")
        
        name = definition['name']
        
        if name in self.tools:
            raise ValueError(f'Tool "{name}" already added. Use .remove_tool("{name}") first.')
        
        self.tools[name] = {
            'definition': definition,
            'handler': handler,
        }
        
        self.update_session()
        
        return self.tools[name]
    
    def remove_tool(self, name: str) -> bool:
        """
        Removes a tool.
        
        Args:
            name: Name of the tool to remove
            
        Returns:
            True on success
        """
        if name in self.tools:
            del self.tools[name]
        
        return True
    
    def delete_item(self, item_id: str) -> bool:
        """
        Deletes an item from the conversation.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True on success
        """
        self.realtime.send('conversation.item.delete', {'item_id': item_id})
        return True
    
    def update_session(
        self,
        modalities: list[str]| None = None,
        instructions: str|None = None,
        voice: str|None = None,
        speed: float|None = None,
        input_audio_format: str|None = None,
        output_audio_format: str|None = None,
        input_audio_transcription: dict|None = None,
        input_audio_noise_reduction: dict|None = None,
        turn_detection: dict|None = None,
        tools: list[ToolDefinition]| None = None,
        tool_choice: str|None = None,
        temperature: float|None = None,
        max_response_output_tokens: int|None = None,
    ) -> bool:
        """
        Updates session configuration.
        
        If the client is not yet connected, will save details and
        instantiate upon connection.
        
        Args:
            modalities: List of modalities ('text', 'audio')
            instructions: System instructions
            voice: Voice for audio output
            speed: Voice speed (0.25 to 1.5, where 1.0 is normal)
            input_audio_format: Format for input audio
            output_audio_format: Format for output audio
            input_audio_transcription: Transcription config
            input_audio_noise_reduction: Noise reduction config (e.g., {"type": "near_field"})
            turn_detection: Turn detection config (or None to disable)
            tools: List of tool definitions
            tool_choice: Tool choice mode ('auto', 'none', etc.)
            temperature: Sampling temperature
            max_response_output_tokens: Max tokens for response
            
        Returns:
            True on success
        """
        # Update local config
        if modalities is not None:
            self.session_config['modalities'] = modalities
        if instructions is not None:
            self.session_config['instructions'] = instructions
        if voice is not None:
            self.session_config['voice'] = voice
        if speed is not None:
            self.session_config['speed'] = speed
        if input_audio_format is not None:
            self.session_config['input_audio_format'] = input_audio_format
        if output_audio_format is not None:
            self.session_config['output_audio_format'] = output_audio_format
        if input_audio_transcription is not None:
            self.session_config['input_audio_transcription'] = input_audio_transcription
        if input_audio_noise_reduction is not None:
            self.session_config['input_audio_noise_reduction'] = input_audio_noise_reduction
        if turn_detection is not None:
            self.session_config['turn_detection'] = turn_detection
        if tools is not None:
            self.session_config['tools'] = tools
        if tool_choice is not None:
            self.session_config['tool_choice'] = tool_choice
        if temperature is not None:
            self.session_config['temperature'] = temperature
        if max_response_output_tokens is not None:
            self.session_config['max_response_output_tokens'] = max_response_output_tokens
        
        # Handle turn_detection special case for VAD
        if turn_detection is not None:
            if turn_detection.get('type') == 'server_vad':
                self.session_config['turn_detection'] = {
                    **self.default_server_vad_config,
                    **turn_detection,
                }
        
        # Build tools list from session config and registered tools
        use_tools = []
        
        for tool_def in (tools or []):
            definition = {'type': 'function', **tool_def}
            if definition.get('name') in self.tools:
                raise ValueError(f'Tool "{definition["name"]}" has already been defined')
            use_tools.append(definition)
        
        for _, tool_config in self.tools.items():
            use_tools.append({
                'type': 'function',
                **tool_config['definition'],
            })
        
        # Build session object
        session = {**self.session_config}
        session['tools'] = use_tools
        
        # Send update if connected
        if self.realtime.is_connected():
            self.realtime.send('session.update', {'session': session})
        
        return True
    
    def send_user_message_content(
        self,
        content: list[dict[str, Any]]
    ) -> bool:
        """
        Sends user message content and generates a response.
        
        Args:
            content: List of content parts, e.g.:
                [{"type": "input_text", "text": "Hello"}]
                [{"type": "input_audio", "audio": "<base64>"}]
                
        Returns:
            True on success
        """
        if content:
            for c in content:
                if c.get('type') == 'input_audio':
                    audio = c.get('audio')
                    # Convert numpy arrays to base64
                    if isinstance(audio, (np.ndarray, bytes)):
                        c['audio'] = RealtimeUtils.array_buffer_to_base64(audio)
            
            self.realtime.send('conversation.item.create', {
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': content,
                }
            })
        
        self.create_response()
        return True
    
    def append_input_audio(self, audio_data: bytes | np.ndarray) -> bool:
        """
        Appends user audio to the existing audio buffer.
        
        Args:
            audio_data: Int16 audio data (bytes or numpy array)
            
        Returns:
            True on success
        """
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = audio_data
        
        if len(audio_array) > 0:
            self.realtime.send('input_audio_buffer.append', {
                'audio': RealtimeUtils.array_buffer_to_base64(audio_array),
            })
            
            self.input_audio_buffer = RealtimeUtils.merge_int16_arrays(
                self.input_audio_buffer,
                audio_array
            )
        
        return True
    
    def create_response(self) -> bool:
        """
        Forces a model response generation.
        
        Returns:
            True on success
        """
        # If no VAD and there's pending audio, commit it first
        if self.get_turn_detection_type() is None and len(self.input_audio_buffer) > 0:
            self.realtime.send('input_audio_buffer.commit')
            self.conversation.queue_input_audio(self.input_audio_buffer)
            self.input_audio_buffer = np.array([], dtype=np.int16)
        
        self.realtime.send('response.create')
        return True
    
    def cancel_response(self, item_id: str|None = None, sample_count: int = 0) -> dict[str, Any]:
        """
        Cancels the ongoing server generation and truncates ongoing generation.
        
        Args:
            item_id: ID of the item to cancel (if None, just cancels generation)
            sample_count: Number of samples to keep (for truncation)
            
        Returns:
            Dict with item if truncated, empty dict otherwise
        """
        if not item_id:
            self.realtime.send('response.cancel')
            return {}
        
        item = self.conversation.get_item(item_id)
        if not item:
            raise ValueError(f'Could not find item "{item_id}"')
        
        if item.get('type') != 'message':
            raise ValueError('Can only cancel "message" items')
        if item.get('role') != 'assistant':
            raise ValueError('Can only cancel assistant messages')
        if item.get('status') == 'completed':
            raise ValueError(f'Cannot cancel completed message "{item_id}"')
        
        self.realtime.send('response.cancel')
        
        item.get('formatted', {}).get('audio', np.array([], dtype=np.int16))
        audio_end_ms = int((sample_count / self.conversation.DEFAULT_FREQUENCY) * 1000)
        
        self.realtime.send('conversation.item.truncate', {
            'item_id': item_id,
            'content_index': 0,
            'audio_end_ms': audio_end_ms,
        })
        
        return {'item': item}
    
    async def wait_for_next_item(self) -> dict[str, Any]:
        """
        Utility for waiting for the next conversation.item.appended event.
        
        Returns:
            Dict with the appended item
        """
        event = await self.wait_for_next('conversation.item.appended')
        return {'item': event.get('item')}
    
    async def wait_for_next_completed_item(self) -> dict[str, Any]:
        """
        Utility for waiting for the next conversation.item.completed event.
        
        Returns:
            Dict with the completed item
        """
        event = await self.wait_for_next('conversation.item.completed')
        return {'item': event.get('item')}
