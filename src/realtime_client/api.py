"""
Low-level Realtime API WebSocket interface.

Thin wrapper over WebSocket for connecting to the OpenAI Realtime API.
Handles connection, authentication, and sending/receiving events.
"""

import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

import websockets
from websockets.client import WebSocketClientProtocol

from .event_handler import RealtimeEventHandler
from .utils import RealtimeUtils


class RealtimeAPI(RealtimeEventHandler):
    """
    Low-level WebSocket interface for the OpenAI Realtime API.
    
    Use this for direct API access without higher-level abstractions.
    For most use cases, prefer RealtimeClient.
    """
    
    DEFAULT_URL = "wss://api.openai.com/v1/realtime"
    DEFAULT_MODEL = "gpt-4o-realtime-preview"
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Create a new RealtimeAPI instance.
        
        Args:
            url: WebSocket URL (defaults to OpenAI's Realtime API)
            api_key: OpenAI API key
            debug: Enable debug logging
        """
        super().__init__()
        self.url = url or self.DEFAULT_URL
        self.api_key = api_key
        self.debug = debug
        self.ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
    
    def is_connected(self) -> bool:
        """
        Check if the WebSocket is connected.
        
        Returns:
            True if connected
        """
        if self.ws is None:
            return False
        # websockets >= 12.0 uses 'state' instead of 'open'
        try:
            from websockets.protocol import State
            return self.ws.state == State.OPEN
        except (ImportError, AttributeError):
            # Fallback for older versions or different connection types
            return hasattr(self.ws, 'open') and self.ws.open
    
    def log(self, *args) -> bool:
        """
        Writes WebSocket logs to console.
        
        Args:
            *args: Arguments to log
            
        Returns:
            True on success
        """
        if self.debug:
            date = datetime.now().isoformat()
            formatted_args = []
            for arg in args:
                if isinstance(arg, dict):
                    formatted_args.append(json.dumps(arg, indent=2))
                else:
                    formatted_args.append(str(arg))
            print(f"[Websocket/{date}]", *formatted_args)
        return True
    
    async def connect(self, model: Optional[str] = None) -> bool:
        """
        Connects to the Realtime API WebSocket Server.
        
        Args:
            model: Model to use (defaults to gpt-4o-realtime-preview)
            
        Returns:
            True on successful connection
            
        Raises:
            ValueError: If already connected
            ConnectionError: If connection fails
        """
        model = model or self.DEFAULT_MODEL
        
        if not self.api_key and self.url == self.DEFAULT_URL:
            print(f'Warning: No apiKey provided for connection to "{self.url}"')
        
        if self.is_connected():
            raise ValueError("Already connected")
        
        # Build the connection URL with model parameter
        ws_url = f"{self.url}?model={model}" if model else self.url
        
        # Set up headers for authentication
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["OpenAI-Beta"] = "realtime=v1"
        
        try:
            self.log(f'Connecting to "{ws_url}"')
            
            self.ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            )
            
            self.log(f'Connected to "{self.url}"')
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            return True
            
        except Exception as e:
            self.log(f'Connection failed: {e}')
            raise ConnectionError(f'Could not connect to "{self.url}": {e}')
    
    async def _receive_loop(self):
        """Internal loop for receiving WebSocket messages."""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    event_type = data.get('type', 'unknown')
                    self.receive(event_type, data)
                except json.JSONDecodeError as e:
                    self.log(f'Failed to parse message: {e}')
                except Exception as e:
                    self.log(f'Error processing message: {e}')
        except websockets.exceptions.ConnectionClosed as e:
            self.log(f'Disconnected from "{self.url}"')
            self.dispatch('close', {'error': False, 'reason': str(e)})
        except Exception as e:
            self.log(f'Error in receive loop: {e}')
            self.dispatch('close', {'error': True, 'reason': str(e)})
        finally:
            self.ws = None
    
    def disconnect(self) -> bool:
        """
        Disconnects from the Realtime API server.
        
        Returns:
            True on success
        """
        if self.ws:
            asyncio.create_task(self._close_connection())
        return True
    
    async def _close_connection(self):
        """Internal method to close the WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
    
    def receive(self, event_name: str, event: Dict[str, Any]) -> bool:
        """
        Receives an event from WebSocket and dispatches it.
        
        Dispatches as "server.{event_name}" and "server.*" events.
        
        Args:
            event_name: The type of event
            event: The event data
            
        Returns:
            True on success
        """
        self.log('received:', event_name, event)
        self.dispatch(f'server.{event_name}', event)
        self.dispatch('server.*', event)
        return True
    
    def send(self, event_name: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Sends an event to WebSocket.
        
        Dispatches as "client.{event_name}" and "client.*" events.
        
        Args:
            event_name: The type of event to send
            data: The event data
            
        Returns:
            True on success
            
        Raises:
            ConnectionError: If not connected
            ValueError: If data is not a dict
        """
        if not self.is_connected():
            raise ConnectionError("RealtimeAPI is not connected")
        
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("data must be a dict")
        
        event = {
            'event_id': RealtimeUtils.generate_id('evt_'),
            'type': event_name,
            **data
        }
        
        self.dispatch(f'client.{event_name}', event)
        self.dispatch('client.*', event)
        
        self.log('sent:', event_name, event)
        
        # Send asynchronously
        asyncio.create_task(self._send_message(json.dumps(event)))
        
        return True
    
    async def _send_message(self, message: str):
        """Internal method to send a message over the WebSocket."""
        if self.ws and self.is_connected():
            await self.ws.send(message)
    
    async def send_async(self, event_name: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Sends an event to WebSocket (async version).
        
        Args:
            event_name: The type of event to send
            data: The event data
            
        Returns:
            True on success
        """
        if not self.is_connected():
            raise ConnectionError("RealtimeAPI is not connected")
        
        data = data or {}
        event = {
            'event_id': RealtimeUtils.generate_id('evt_'),
            'type': event_name,
            **data
        }
        
        self.dispatch(f'client.{event_name}', event)
        self.dispatch('client.*', event)
        
        self.log('sent:', event_name, event)
        
        await self.ws.send(json.dumps(event))
        
        return True
