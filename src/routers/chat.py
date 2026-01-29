
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from chat import ConnectionManager, get_connection_manager
from core.logger import get_logger
from core.settings import Settings, get_settings
from services import Wav2ArkitService, get_wav2arkit_service

logger = get_logger(__name__)


chat_router = APIRouter(tags=["chat"])

@chat_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    wav2arkit_service: Wav2ArkitService = Depends(get_wav2arkit_service),
    chat_connection_manager: ConnectionManager = Depends(get_connection_manager),
) -> None:

    session_id = str(uuid.uuid4())

    # Auth check...
    from main import auth_middleware

    if settings.auth_enabled and auth_middleware:
        is_authenticated, error = await auth_middleware.authenticate_websocket(
            websocket, session_id
        )
        if not is_authenticated:
            await websocket.close(code=1008, reason=error)
            return

    # Connect creates the specific session
    await chat_connection_manager.connect(websocket, session_id, settings, wav2arkit_service)

    try:
        while True:
            data = await websocket.receive_json()
            # Pass data to the specific manager/session
            await chat_connection_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        await chat_connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in chat handler: {type(e).__name__}: {e}", exc_info=True)
        await chat_connection_manager.disconnect(websocket)
