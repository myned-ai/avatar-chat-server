"""
HMAC Token Manager

Generates and verifies HMAC-SHA256 signed tokens to ensure
that WebSocket connections are authorized by the server.

Token Format: base64(timestamp:signature)
where signature = HMAC-SHA256(timestamp + origin, secret_key)
"""

import hmac
import hashlib
import base64
import time
from typing import Optional, Tuple


class TokenManager:
    def __init__(self, secret_key: str, token_ttl: int = 3600):
        """
        Initialize Token Manager

        Args:
            secret_key: Secret key for HMAC signing (should be strong random string)
            token_ttl: Token time-to-live in seconds (default: 1 hour)
        """
        self.secret_key = secret_key.encode('utf-8')
        self.token_ttl = token_ttl

    def generate_token(self, origin: str) -> str:
        """
        Generate HMAC-signed token for a given origin

        Args:
            origin: Origin domain (e.g., 'https://example.com')

        Returns:
            Base64-encoded token string
        """
        timestamp = str(int(time.time()))
        message = f"{timestamp}:{origin}"

        signature = hmac.new(
            self.secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        token_data = f"{timestamp}:{signature}"
        return base64.b64encode(token_data.encode('utf-8')).decode('utf-8')

    def verify_token(self, token: str, origin: str) -> Tuple[bool, Optional[str]]:
        """
        Verify HMAC-signed token

        Args:
            token: Base64-encoded token
            origin: Origin domain to verify against

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Decode token
            token_data = base64.b64decode(token.encode('utf-8')).decode('utf-8')
            timestamp_str, provided_signature = token_data.split(':', 1)
            timestamp = int(timestamp_str)

            # Check expiration
            current_time = int(time.time())
            if current_time - timestamp > self.token_ttl:
                return False, "Token expired"

            # Verify signature
            message = f"{timestamp_str}:{origin}"
            expected_signature = hmac.new(
                self.secret_key,
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(provided_signature, expected_signature):
                return False, "Invalid signature"

            return True, None

        except (ValueError, base64.binascii.Error) as e:
            return False, f"Invalid token format: {str(e)}"
