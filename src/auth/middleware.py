"""
Authentication Middleware

4-Layer Security Implementation:
1. Origin Validation - Whitelist check
2. HMAC Token Verification - Signature validation
3. Rate Limiting - Token bucket algorithm
4. Monitoring - Audit logging
"""

import logging
from typing import List, Optional
from fastapi import WebSocket, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .token_manager import TokenManager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AuthMiddleware:
    def __init__(
        self,
        allowed_origins: List[str],
        secret_key: str,
        token_ttl: int = 3600,
        enable_rate_limiting: bool = True
    ):
        """
        Initialize Authentication Middleware

        Args:
            allowed_origins: List of allowed origin domains
            secret_key: Secret key for HMAC token signing
            token_ttl: Token time-to-live in seconds
            enable_rate_limiting: Enable rate limiting
        """
        self.allowed_origins = set(allowed_origins)
        self.token_manager = TokenManager(secret_key, token_ttl)
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None

        logger.info(f"AuthMiddleware initialized with {len(allowed_origins)} allowed origins")

    def validate_origin(self, origin: Optional[str]) -> tuple[bool, Optional[str]]:
        """
        Validate request origin against whitelist

        Args:
            origin: Origin header from request

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not origin:
            return False, "Missing Origin header"

        # Allow localhost for development
        if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
            return True, None

        # Check against whitelist
        if origin not in self.allowed_origins:
            logger.warning(f"Rejected origin: {origin}")
            return False, f"Origin not allowed: {origin}"

        return True, None

    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        session_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Authenticate WebSocket connection

        Expected query parameters:
        - token: HMAC-signed token
        - origin: Origin domain (also from headers)

        Args:
            websocket: WebSocket connection
            session_id: Unique session identifier

        Returns:
            Tuple of (is_authenticated, error_message)
        """
        # Get origin from headers
        origin = websocket.headers.get("origin")

        # Layer 1: Origin Validation
        is_valid, error = self.validate_origin(origin)
        if not is_valid:
            logger.warning(f"Origin validation failed for {origin}: {error}")
            return False, error

        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.warning(f"Missing token for origin {origin}")
            return False, "Missing authentication token"

        # Layer 2: HMAC Token Verification
        is_valid, error = self.token_manager.verify_token(token, origin)
        if not is_valid:
            logger.warning(f"Token verification failed for {origin}: {error}")
            return False, error

        # Layer 3: Rate Limiting
        if self.rate_limiter:
            is_allowed, error = self.rate_limiter.check_rate_limit(origin, session_id)
            if not is_allowed:
                logger.warning(f"Rate limit exceeded for {origin}, session {session_id}: {error}")
                return False, error

        # Layer 4: Monitoring (successful auth)
        logger.info(f"WebSocket authenticated: origin={origin}, session={session_id}")

        return True, None

    def generate_token_for_origin(self, origin: str) -> Optional[str]:
        """
        Generate authentication token for a given origin

        This is typically called by an HTTP endpoint that the widget
        calls before establishing WebSocket connection.

        Args:
            origin: Origin domain

        Returns:
            HMAC-signed token or None if origin not allowed
        """
        is_valid, error = self.validate_origin(origin)
        if not is_valid:
            logger.warning(f"Token generation refused for {origin}: {error}")
            return None

        token = self.token_manager.generate_token(origin)
        logger.info(f"Generated token for origin: {origin}")
        return token

    def get_stats(self) -> dict:
        """Get authentication statistics"""
        stats = {
            "allowed_origins": len(self.allowed_origins),
            "rate_limiter_enabled": self.rate_limiter is not None
        }

        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()

        return stats
