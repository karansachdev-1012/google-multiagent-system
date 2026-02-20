"""
Enhanced Security Module for Multi-Agent System

Provides:
- Input sanitization
- Per-endpoint rate limiting
- Prompt injection detection
"""

import re
import time
import hashlib
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# Common XSS patterns
XSS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'<iframe',
    r'<object',
    r'<embed',
    r'<applet',
    r'data:text/html',
]

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
    r"(--|;|'|\"|%27|%22)",
    r"(\bOR\b.*=.*\bOR\b)",
    r"(\bAND\b.*=.*\bAND\b)",
]

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r"(ignore (all|previous|above) (instructions|prompts?|rules?))",
    r"(forget (all|your|yourself) (instructions|rules?|guidelines?|system.?prompt))",
    r"(you (are now|have become|must act as) (a|an))",
    r"(system:?.*\n)",
    r"(#system|#admin|#root)",
    r"(new (instruction|prompt|command):)",
    r"(override (your|all) (safety|security|content) (filters?|checks?|guidelines?))",
    r"(bypass (restrictions?|limitations?|filters?))",
    r"(disable (safety|security) (protocols?|measures?))",
]


class InputSanitizer:
    """
    Sanitizes user input to prevent XSS and injection attacks.
    """
    
    def __init__(self):
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in XSS_PATTERNS]
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]
    
    def sanitize(self, text: str) -> str:
        """Sanitize input text."""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # HTML escape
        text = self._escape_html(text)
        
        # Remove XSS patterns
        for pattern in self.xss_patterns:
            text = pattern.sub('', text)
        
        # Trim and normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters."""
        replacements = {
            '&': '&amp;',
            '<': '<',
            '>': '>',
            '"': '"',
            "'": '&#x27;',
            '/': '&#x2F;',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    def check_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns."""
        for pattern in self.sql_patterns:
            if pattern.search(text):
                return True
        return False
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary values."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        return sanitized


# Global sanitizer instance
input_sanitizer = InputSanitizer()


class PromptInjectionDetector:
    """
    Detects prompt injection attempts.
    
    Identifies attempts to override or manipulate agent behavior
    through specially crafted prompts.
    """
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]
        self.blocked_phrases: Set[str] = set()
        self.suspicious_count = 0
        self.detection_log: List[Dict[str, Any]] = []
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for prompt injection attempts.
        
        Returns:
            Dictionary with detection results
        """
        if not text:
            return {
                "is_suspicious": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "recommendation": "pass"
            }
        
        matched_patterns = []
        total_matches = 0
        
        for i, pattern in enumerate(self.patterns):
            matches = pattern.findall(text)
            if matches:
                matched_patterns.append({
                    "pattern_id": i,
                    "matches": matches
                })
                total_matches += len(matches)
        
        # Calculate suspicion score
        is_suspicious = total_matches > 0
        confidence = min(total_matches * 0.3, 1.0)
        
        if is_suspicious:
            self.suspicious_count += 1
            self.detection_log.append({
                "timestamp": time.time(),
                "text_length": len(text),
                "matches": total_matches,
                "patterns": matched_patterns
            })
            
            # Keep only last 100 detections
            if len(self.detection_log) > 100:
                self.detection_log = self.detection_log[-100:]
        
        recommendation = "block" if confidence > 0.7 else "warn" if confidence > 0.3 else "pass"
        
        return {
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "matched_patterns": matched_patterns,
            "recommendation": recommendation
        }
    
    def add_blocked_phrase(self, phrase: str):
        """Add a phrase to the blocked list."""
        self.blocked_phrases.add(phrase.lower())
    
    def check_blocked_phrases(self, text: str) -> bool:
        """Check if text contains any blocked phrases."""
        text_lower = text.lower()
        for phrase in self.blocked_phrases:
            if phrase in text_lower:
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_suspicious": self.suspicious_count,
            "blocked_phrases": len(self.blocked_phrases),
            "recent_detections": len(self.detection_log),
            "detection_log": self.detection_log[-10:]  # Last 10
        }


# Global detector instance
prompt_detector = PromptInjectionDetector()


class PerEndpointRateLimiter:
    """
    Per-endpoint rate limiting with configurable limits.
    
    Allows different rate limits for different endpoints.
    """
    
    def __init__(self):
        self.endpoint_limits: Dict[str, Dict[str, Any]] = {}
        self.requests: Dict[str, List[float]] = defaultdict(list)
        
        # Default limits
        self.set_limit("default", requests_per_minute=60, requests_per_hour=1000)
        self.set_limit("health", requests_per_minute=120, requests_per_hour=5000)
        self.set_limit("query", requests_per_minute=30, requests_per_hour=500)
        self.set_limit("admin", requests_per_minute=10, requests_per_hour=100)
    
    def set_limit(
        self,
        endpoint: str,
        requests_per_minute: int,
        requests_per_hour: int
    ):
        """Set rate limit for an endpoint."""
        self.endpoint_limits[endpoint] = {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour
        }
    
    def is_allowed(
        self,
        endpoint: str,
        user_id: str = "anonymous"
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Returns:
            (allowed: bool, info: dict)
        """
        key = f"{user_id}:{endpoint}"
        now = time.time()
        
        # Get limits for endpoint
        limits = self.endpoint_limits.get(endpoint, self.endpoint_limits.get("default"))
        
        # Clean old requests
        cutoff_minute = now - 60
        cutoff_hour = now - 3600
        
        if key in self.requests:
            self.requests[key] = [
                t for t in self.requests[key]
                if t > cutoff_hour
            ]
        else:
            self.requests[key] = []
        
        recent_minute = [t for t in self.requests[key] if t > cutoff_minute]
        recent_hour = self.requests[key]
        
        # Check limits
        if len(recent_minute) >= limits["per_minute"]:
            return False, {
                "reason": "rate_limit_exceeded",
                "limit_type": "per_minute",
                "limit": limits["per_minute"],
                "retry_after": 60 - (now - min(recent_minute)) if recent_minute else 60
            }
        
        if len(recent_hour) >= limits["per_hour"]:
            return False, {
                "reason": "rate_limit_exceeded",
                "limit_type": "per_hour",
                "limit": limits["per_hour"],
                "retry_after": 3600 - (now - min(recent_hour)) if recent_hour else 3600
            }
        
        # Allow request
        self.requests[key].append(now)
        
        return True, {
            "remaining_minute": limits["per_minute"] - len(recent_minute) - 1,
            "remaining_hour": limits["per_hour"] - len(recent_hour) - 1
        }
    
    def get_usage(self, endpoint: str, user_id: str = "anonymous") -> Dict[str, int]:
        """Get current usage for an endpoint."""
        key = f"{user_id}:{endpoint}"
        now = time.time()
        
        if key not in self.requests:
            return {"per_minute": 0, "per_hour": 0}
        
        cutoff_minute = now - 60
        cutoff_hour = now - 3600
        
        recent_minute = [t for t in self.requests[key] if t > cutoff_minute]
        recent_hour = self.requests[key]
        
        return {
            "per_minute": len(recent_minute),
            "per_hour": len(recent_hour)
        }


# Global rate limiter
per_endpoint_limiter = PerEndpointRateLimiter()


def sanitize_input(text: str) -> str:
    """Convenience function to sanitize input."""
    return input_sanitizer.sanitize(text)


def check_prompt_injection(text: str) -> Dict[str, Any]:
    """Convenience function to check for prompt injection."""
    return prompt_detector.analyze(text)


def check_rate_limit(
    endpoint: str,
    user_id: str = "anonymous"
) -> tuple[bool, Dict[str, Any]]:
    """Convenience function to check rate limit."""
    return per_endpoint_limiter.is_allowed(endpoint, user_id)


def secure_endpoint(endpoint_name: str):
    """
    Decorator to secure an endpoint with all security measures.
    
    Applies:
    - Input sanitization
    - Rate limiting
    - Prompt injection detection (for query endpoints)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get user ID from kwargs or use default
            user_id = kwargs.get('user_id', 'anonymous')
            
            # Check rate limit
            allowed, rate_info = check_rate_limit(endpoint_name, user_id)
            if not allowed:
                return {
                    "status": "error",
                    "error": "rate_limit_exceeded",
                    "details": rate_info
                }
            
            # Sanitize string inputs
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str) and key not in ['user_id', 'session_id']:
                    sanitized_kwargs[key] = sanitize_input(value)
                elif isinstance(value, dict):
                    sanitized_kwargs[key] = input_sanitizer.sanitize_dict(value)
                else:
                    sanitized_kwargs[key] = value
            
            # Check prompt injection for query endpoints
            if endpoint_name == "query" and 'query' in sanitized_kwargs:
                injection_result = check_prompt_injection(sanitized_kwargs['query'])
                if injection_result['recommendation'] == 'block':
                    logger.warning(
                        f"Prompt injection blocked for user {user_id}",
                        extra=injection_result
                    )
                    return {
                        "status": "error",
                        "error": "invalid_input",
                        "message": "Your input contains potentially unsafe content."
                    }
            
            # Execute function
            return func(*args, **sanitized_kwargs)
        
        return wrapper
    return decorator


# Export all security components
__all__ = [
    "InputSanitizer",
    "PromptInjectionDetector",
    "PerEndpointRateLimiter",
    "input_sanitizer",
    "prompt_detector",
    "per_endpoint_limiter",
    "sanitize_input",
    "check_prompt_injection",
    "check_rate_limit",
    "secure_endpoint"
]
