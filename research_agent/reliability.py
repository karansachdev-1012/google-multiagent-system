"""
Reliability Module for Multi-Agent System

Provides:
- CircuitBreaker for API resilience
- AgentMemory for conversation context
- ToolError for standardized error handling
- Async tool support
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 2         # Successes to close circuit
    timeout: float = 30.0              # Seconds before trying half-open
    excluded_exceptions: tuple = ()      # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit breaker implementation for API resilience.
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    def _can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN - allow one request through
        return True
    
    def _record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
    
    def _record_failure(self, exception: Exception):
        """Record failed execution."""
        # Don't count excluded exceptions
        if isinstance(exception, self.config.excluded_exceptions):
            return
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is OPEN"
            )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class ConversationMessage:
    """A single message in conversation history."""
    role: str  # "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    """
    Simple in-memory conversation context for agents.
    
    Maintains conversation history with configurable limits.
    """
    
    def __init__(self, max_messages: int = 50, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: deque = deque(maxlen=max_messages)
        self.session_id: Optional[str] = None
        self.created_at = time.time()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
    
    def get_context(self, include_last_n: int = None) -> str:
        """Get formatted context from recent messages."""
        messages = list(self.messages)
        if include_last_n:
            messages = messages[-include_last_n:]
        
        context_parts = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def get_messages(self) -> List[ConversationMessage]:
        """Get all messages."""
        return list(self.messages)
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return max(1, len(text) // 4)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_tokens = sum(
            self.estimate_tokens(m.content) for m in self.messages
        )
        return {
            "message_count": len(self.messages),
            "estimated_tokens": total_tokens,
            "session_id": self.session_id,
            "created_at": self.created_at
        }


class ToolError(Exception):
    """
    Standardized error class for tool failures.
    
    Provides consistent error structure across all tools.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str = "",
        error_type: str = "unknown_error",
        recoverable: bool = True,
        details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type
        self.recoverable = recoverable
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.error_type,
            "message": str(self),
            "tool_name": self.tool_name,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ToolErrorType:
    """Standard error types for tools."""
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    PARSING_ERROR = "parsing_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to add circuit breaker to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to add retry logic to functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# Global circuit breakers for common services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get state of all circuit breakers."""
    return {
        name: cb.get_state() 
        for name, cb in _circuit_breakers.items()
    }
