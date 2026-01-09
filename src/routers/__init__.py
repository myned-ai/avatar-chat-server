"""
Routers Package

Contains FastAPI router modules for:
- Chat WebSocket endpoint
"""

from routers.chat import router as chat_router

__all__ = ["chat_router"]
