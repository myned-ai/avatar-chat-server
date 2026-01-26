import secrets

from core.logger import get_logger
from core.settings import get_allowed_origins, get_settings

from .middleware import AuthMiddleware

logger = get_logger(__name__)

_auth_middleware: AuthMiddleware | None = None

def get_auth_middleware() -> AuthMiddleware | None:
    """
    Get or create the global AuthMiddleware instance.
    """
    global _auth_middleware
    
    if _auth_middleware:
        return _auth_middleware
        
    settings = get_settings()
    
    if not settings.auth_enabled:
        return None

    allowed_origins = get_allowed_origins()
    
    # Generate secret key if not provided
    auth_secret = settings.auth_secret_key or secrets.token_hex(32)
    if not settings.auth_secret_key:
        logger.warning("No AUTH_SECRET_KEY set. Using auto-generated key (not suitable for production)")
        logger.warning(f"Generated key: {auth_secret}")
        logger.warning(f"Add this to your .env file: AUTH_SECRET_KEY={auth_secret}")

    _auth_middleware = AuthMiddleware(
        allowed_origins=allowed_origins,
        secret_key=auth_secret,
        token_ttl=settings.auth_token_ttl,
        enable_rate_limiting=settings.auth_enable_rate_limiting,
    )
    
    return _auth_middleware
