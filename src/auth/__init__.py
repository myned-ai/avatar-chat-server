"""
Authentication Module

4-Layer Security System:
1. Origin Validation - Allow only whitelisted domains
2. HMAC Token Verification - Signed tokens prevent unauthorized access
3. Rate Limiting - Per-domain and per-session limits
4. Monitoring - Log all auth attempts and anomalies
"""

from .middleware import AuthMiddleware
from .token_manager import TokenManager
from .rate_limiter import RateLimiter

__all__ = ['AuthMiddleware', 'TokenManager', 'RateLimiter']
