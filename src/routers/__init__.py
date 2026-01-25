"""
Routers Package

Contains FastAPI router modules for:
- Chat WebSocket endpoint
"""

from routers.chat_router import chat_router as chat_router

__all__ = ["chat_router"]
