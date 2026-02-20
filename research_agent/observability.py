"""
Observability Module for Multi-Agent System

Provides:
- Structured logging
- Metrics collection
- Health check endpoints
- Request batching
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import threading
import uuid

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class StructuredLogger:
    """
    Structured logger for consistent log format.
    
    Logs in JSON format for easy parsing by log aggregators.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def add_context(self, **kwargs):
        """Add context to all subsequent log entries."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context."""
        self.context.clear()
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            **self.context,
            **kwargs
        }
        return json.dumps(log_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def log_agent_call(self, agent_name: str, query: str, duration_ms: float, success: bool):
        """Log agent call with structured data."""
        self.info(
            "Agent call completed",
            agent_name=agent_name,
            query_length=len(query),
            duration_ms=round(duration_ms, 2),
            success=success
        )
    
    def log_tool_call(self, tool_name: str, args: Dict, duration_ms: float, success: bool):
        """Log tool call with structured data."""
        self.info(
            "Tool call completed",
            tool_name=tool_name,
            args_keys=list(args.keys()) if args else [],
            duration_ms=round(duration_ms, 2),
            success=success
        )


# Global structured logger
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    
    Thread-safe metrics collection with aggregation.
    """
    
    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                tags=tags or {}
            )
            self._metrics[name].append(metric)
            self._cleanup_old_metrics(name)
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += amount
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        with self._lock:
            return self._gauges.get(name)
    
    def get_metrics(self, name: str, last_n: int = None) -> List[Dict]:
        """Get recent metrics."""
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            if last_n:
                metrics = metrics[-last_n:]
            return [asdict(m) for m in metrics]
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get aggregated statistics for a metric."""
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "sum": sum(values)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "metrics": {}
            }
            
            for name in self._metrics.keys():
                result["metrics"][name] = self.get_statistics(name)
            
            return result
    
    def _cleanup_old_metrics(self, name: str):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_minutes * 60)
        metrics = self._metrics[name]
        
        while metrics and metrics[0].timestamp < cutoff_time:
            metrics.popleft()
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()


# Global metrics collector
metrics_collector = MetricsCollector()


def track_duration(metric_name: str):
    """Decorator to track function execution duration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                metrics_collector.record_metric(
                    f"{metric_name}_duration_ms",
                    duration_ms
                )
                metrics_collector.increment_counter(f"{metric_name}_success")
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                metrics_collector.record_metric(
                    f"{metric_name}_duration_ms",
                    duration_ms
                )
                metrics_collector.increment_counter(f"{metric_name}_errors")
                raise
        return wrapper
    return decorator


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthCheckRegistry:
    """
    Registry for health checks.
    
    Collects health status from various components.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
    
    def register(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function."""
        self._checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        overall_status = "healthy"
        
        for name, check_func in self._checks.items():
            try:
                check = check_func()
                results["checks"][name] = asdict(check)
                
                if check.status == "unhealthy":
                    overall_status = "unhealthy"
                elif check.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                results["checks"][name] = {
                    "name": name,
                    "status": "unhealthy",
                    "message": str(e)
                }
                overall_status = "unhealthy"
        
        results["status"] = overall_status
        return results


# Global health check registry
health_checks = HealthCheckRegistry()


def default_health_check() -> HealthCheck:
    """Default system health check."""
    return HealthCheck(
        name="system",
        status="healthy",
        message="System operational"
    )


# Register default health check
health_checks.register("system", default_health_check)


class RequestBatcher:
    """
    Batches similar requests together for efficiency.
    
    Useful for reducing API calls when multiple requests
    can be processed together.
    """
    
    def __init__(self, batch_size: int = 10, max_wait_ms: float = 100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._pending: Dict[str, List] = defaultdict(list)
        self._lock = threading.Lock()
    
    def add(self, key: str, item: Any) -> List[Any]:
        """Add item to batch. Returns batch if full."""
        with self._lock:
            self._pending[key].append(item)
            
            if len(self._pending[key]) >= self.batch_size:
                batch = self._pending[key].copy()
                self._pending[key].clear()
                return batch
        
        return None
    
    def get_pending(self, key: str) -> List[Any]:
        """Get all pending items for a key."""
        with self._lock:
            return self._pending[key].copy()
    
    def clear(self, key: str = None):
        """Clear pending items."""
        with self._lock:
            if key:
                self._pending[key].clear()
            else:
                self._pending.clear()


def get_health_status() -> Dict[str, Any]:
    """Get overall health status."""
    return health_checks.run_checks()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return metrics_collector.get_all_stats()
