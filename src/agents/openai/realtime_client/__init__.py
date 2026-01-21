"""
OpenAI Realtime API Client for Python

A Python implementation of the OpenAI Realtime API client,
adapted from the official JavaScript reference client.

Usage:
    from agents.openai.realtime_client import RealtimeClient

    client = RealtimeClient(api_key="your-api-key")
    await client.connect()

    client.on("conversation.updated", lambda event: print(event))
    client.send_user_message_content([{"type": "input_text", "text": "Hello!"}])
"""

from .api import RealtimeAPI
from .client import RealtimeClient
from .conversation import RealtimeConversation
from .event_handler import RealtimeEventHandler
from .utils import RealtimeUtils

__all__ = [
    "RealtimeAPI",
    "RealtimeClient",
    "RealtimeConversation",
    "RealtimeEventHandler",
    "RealtimeUtils",
]

__version__ = "0.1.0"
