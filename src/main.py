"""
Avatar Chat Server

FastAPI application for real-time voice-to-avatar interaction.
Combines AI agents for voice conversation with Wav2Arkit
model for synchronized facial animation.

Supports multiple agent backends:
- sample_openai: OpenAI Realtime API
- sample_gemini: Google Gemini Live API
- remote: External agent service

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Or run directly:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from core.config import get_settings, get_allowed_origins
from core.logger import setup_logging, get_logger
from routers import chat_router
from services import get_wav2arkit_service, get_agent
from auth import AuthMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for proper resource management.
    """
    settings = get_settings()

    # Setup logging based on settings
    setup_logging("DEBUG" if settings.debug else "INFO")

    # Startup
    logger.info("=" * 60)
    logger.info("Avatar Chat Server Starting")
    logger.info("=" * 60)
    logger.info(f"WebSocket endpoint: ws://{settings.server_host}:{settings.server_port}/ws")
    logger.info(f"Agent type: {settings.agent_type}")
    logger.info(f"Wav2Arkit model: {settings.onnx_model_path}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Auth: {'Enabled' if settings.auth_enabled else 'Disabled'}")
    if settings.auth_enabled:
        allowed_origins = get_allowed_origins()
        logger.info(f"Allowed Origins: {', '.join(allowed_origins)}")
    logger.info("=" * 60)

    # Initialize services (lazy loading - will connect on first request)
    get_wav2arkit_service()
    
    yield

    # Shutdown
    logger.info("Shutting down...")
    agent = get_agent()
    await agent.disconnect()


# Create FastAPI application
app = FastAPI(
    title="Avatar Chat Server",
    description="""
    Real-time voice-to-avatar interaction server.
    
    ## Features
    
    - **Voice AI**: OpenAI Realtime API for natural voice conversations
    - **Facial Animation**: Wav2Arkit model for synchronized blendshapes
    - **WebSocket Streaming**: Real-time audio and animation data
    
    ## WebSocket Protocol
    
    Connect to `/ws` for real-time communication.
    
    ### Client → Server Messages
    
    - `{"type": "text", "data": "Hello"}` - Send text message
    - `{"type": "audio_stream_start", "userId": "user1"}` - Start audio streaming
    - `{"type": "audio", "data": "<base64>"}` - Send audio chunk (PCM16, 24kHz)
    - `{"type": "audio_stream_end"}` - End audio streaming
    - `{"type": "ping"}` - Heartbeat
    
    ### Server → Client Messages
    
    - `{"type": "audio_start", ...}` - AI started responding
    - `{"type": "sync_frame", "weights": {...}, "audio": "<base64>"}` - Synchronized frame
    - `{"type": "audio_end", ...}` - AI finished responding
    - `{"type": "transcript_delta", "text": "...", "role": "assistant"}` - Streaming text
    - `{"type": "transcript_done", "text": "...", "role": "user|assistant"}` - Complete transcript
    - `{"type": "avatar_state", "state": "Listening|Responding"}` - Avatar state change
    - `{"type": "interrupt"}` - User interrupted
    - `{"type": "pong"}` - Heartbeat response
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
settings = get_settings()
allowed_origins = get_allowed_origins() if settings.auth_enabled else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize authentication middleware
auth_middleware: Optional[AuthMiddleware] = None
if settings.auth_enabled:
    # Generate secret key if not provided
    auth_secret = settings.auth_secret_key or secrets.token_hex(32)
    if not settings.auth_secret_key:
        logger.warning("No AUTH_SECRET_KEY set. Using auto-generated key (not suitable for production)")
        logger.warning(f"Generated key: {auth_secret}")
        logger.warning(f"Add this to your .env file: AUTH_SECRET_KEY={auth_secret}")

    auth_middleware = AuthMiddleware(
        allowed_origins=allowed_origins,
        secret_key=auth_secret,
        token_ttl=settings.auth_token_ttl,
        enable_rate_limiting=settings.auth_enable_rate_limiting
    )

# Include routers
app.include_router(chat_router)


@app.post("/api/auth/token")
async def get_auth_token(request: Request):
    """
    Generate authentication token for WebSocket connection.

    The widget should call this endpoint before connecting to WebSocket.
    Origin is validated and a signed token is returned.

    Returns:
        {"token": "base64-encoded-token", "ttl": 3600}

    Raises:
        403: If origin is not allowed or auth is disabled
    """
    if not settings.auth_enabled or not auth_middleware:
        raise HTTPException(status_code=403, detail="Authentication not enabled")

    # Get origin from request headers
    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=400, detail="Missing Origin header")

    # Generate token for this origin
    token = auth_middleware.generate_token_for_origin(origin)
    if not token:
        raise HTTPException(status_code=403, detail="Origin not allowed")

    return {
        "token": token,
        "ttl": settings.auth_token_ttl,
        "origin": origin
    }


@app.get("/")
async def root():
    """Root endpoint with server information."""
    settings = get_settings()
    return {
        "name": "Avatar Chat Server",
        "version": "1.0.0",
        "status": "running",
        "websocket": f"ws://{settings.server_host}:{settings.server_port}/ws",
        "agent_type": settings.agent_type,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    wav2arkit_service = get_wav2arkit_service()
    agent = get_agent()
    
    return {
        "status": "healthy",
        "services": {
            "wav2arkit": "available" if wav2arkit_service.is_available else "unavailable",
            "agent": "connected" if agent.is_connected else "disconnected",
        },
    }


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
