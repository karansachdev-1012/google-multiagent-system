"""
Security and Robustness Module for Multi-Agent System

This module provides security measures, input validation, rate limiting,
error handling, and monitoring capabilities for the agent system.
"""

import logging
import time
import hashlib
import re
from functools import wraps
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configuration for security measures."""
    max_request_length: int = 100000
    max_requests_per_minute: int = 600
    max_requests_per_hour: int = 10000
    enable_content_filtering: bool = True
    enable_rate_limiting: bool = True
    blocked_keywords: list = None

    def __post_init__(self):
        if self.blocked_keywords is None:
            self.blocked_keywords = [
                'hack', 'exploit', 'malware', 'virus', 'attack',
                'illegal', 'criminal', 'terrorist', 'weapon'
            ]

class RateLimiter:
    """Rate limiting implementation."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests = defaultdict(list)  # user_id -> list of timestamps

    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed based on rate limits."""
        if not self.config.enable_rate_limiting:
            return True

        now = time.time()
        user_requests = self.requests[user_id]

        # Clean old requests
        cutoff_minute = now - 60
        cutoff_hour = now - 3600

        user_requests[:] = [t for t in user_requests if t > cutoff_hour]

        # Check limits
        recent_minute = [t for t in user_requests if t > cutoff_minute]
        recent_hour = user_requests

        if len(recent_minute) >= self.config.max_requests_per_minute:
            return False
        if len(recent_hour) >= self.config.max_requests_per_hour:
            return False

        # Add current request
        user_requests.append(now)
        return True

class ContentFilter:
    """Content filtering and validation."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for word in config.blocked_keywords
        ]

    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate and filter input text."""
        if not text or not isinstance(text, str):
            return {
                'valid': False,
                'error': 'Invalid input: must be non-empty string'
            }

        if len(text) > self.config.max_request_length:
            return {
                'valid': False,
                'error': f'Input too long: {len(text)} > {self.config.max_request_length}'
            }

        if self.config.enable_content_filtering:
            for pattern in self.blocked_patterns:
                if pattern.search(text):
                    return {
                        'valid': False,
                        'error': 'Content contains blocked keywords'
                    }

        return {
            'valid': True,
            'filtered_text': self.sanitize_input(text)
        }

    def sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # Remove potentially dangerous characters
        text = re.sub(r'[<>]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text

class ErrorHandler:
    """Centralized error handling and recovery."""

    @staticmethod
    def handle_agent_error(error: Exception, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent execution errors."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'agent': agent_name,
            'context': context,
            'timestamp': time.time()
        }

        logger.error(f"Agent error in {agent_name}: {error}", extra=error_info)

        # Return user-friendly error response
        return {
            'status': 'error',
            'message': f'Sorry, there was an issue with the {agent_name}. Please try again.',
            'error_id': hashlib.md5(str(error_info).encode()).hexdigest()[:8]
        }

    @staticmethod
    def handle_tool_error(error: Exception, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution errors."""
        logger.warning(f"Tool error in {tool_name}: {error}")

        return {
            'status': 'error',
            'message': f'The {tool_name} tool encountered an issue. Using fallback information.',
            'fallback': True
        }

class MonitoringSystem:
    """System monitoring and analytics."""

    def __init__(self):
        self.metrics = defaultdict(int)
        self.start_time = time.time()

    def record_metric(self, metric_name: str, value: int = 1):
        """Record a metric."""
        self.metrics[metric_name] += value

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'total_requests': self.metrics.get('requests', 0),
            'agent_calls': self.metrics.get('agent_calls', 0),
            'tool_calls': self.metrics.get('tool_calls', 0),
            'errors': self.metrics.get('errors', 0),
            'average_response_time': self.metrics.get('total_response_time', 0) / max(self.metrics.get('requests', 1), 1)
        }

# Global instances
security_config = SecurityConfig()
rate_limiter = RateLimiter(security_config)
content_filter = ContentFilter(security_config)
error_handler = ErrorHandler()
monitor = MonitoringSystem()

def secure_agent_call(user_id: str = "anonymous"):
    """Decorator for secure agent function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting
            if not rate_limiter.is_allowed(user_id):
                return {
                    'status': 'error',
                    'message': 'Rate limit exceeded. Please try again later.'
                }

            # Input validation
            query = kwargs.get('query', args[0] if args else '')
            validation = content_filter.validate_input(query)
            if not validation['valid']:
                return {
                    'status': 'error',
                    'message': validation['error']
                }

            # Monitoring
            monitor.record_metric('requests')
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success metrics
                response_time = time.time() - start_time
                monitor.record_metric('total_response_time', int(response_time * 1000))

                return result

            except Exception as e:
                # Record error metrics
                monitor.record_metric('errors')
                return error_handler.handle_agent_error(e, func.__name__, {
                    'args': args,
                    'kwargs': kwargs,
                    'user_id': user_id
                })

        return wrapper
    return decorator

def secure_tool_call(tool_name: str):
    """Decorator for secure tool function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor.record_metric('tool_calls')

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                return error_handler.handle_tool_error(e, tool_name, {
                    'args': args,
                    'kwargs': kwargs
                })

        return wrapper
    return decorator