"""
OpenAI Realtime API Client for Python

A Python implementation of the OpenAI Realtime API client,
adapted from the official JavaScript reference client.

Usage:
    from openai_realtime import RealtimeClient

    client = RealtimeClient(api_key="your-api-key")
    await client.connect()
    
    client.on("conversation.updated", lambda event: print(event))
    client.send_user_message_content([{"type": "input_text", "text": "Hello!"}])
"""

from .utils import RealtimeUtils
from .event_handler import RealtimeEventHandler
from .conversation import RealtimeConversation
from .api import RealtimeAPI
from .client import RealtimeClient

__all__ = [
    "RealtimeUtils",
    "RealtimeEventHandler", 
    "RealtimeConversation",
    "RealtimeAPI",
    "RealtimeClient",
]

__version__ = "0.1.0"
