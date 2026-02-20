"""
UX Improvements Module for Multi-Agent System

Provides:
- Streaming response support
- Conversation context manager
- Response formatting options
- Agent confidence scoring
"""

import time
import json
from typing import Dict, Any, Optional, List, Callable, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio


class ResponseFormat(Enum):
    """Response format options."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    CONSOLE = "console"


@dataclass
class ConfidenceScore:
    """Agent confidence score for responses."""
    score: float  # 0.0 to 1.0
    confidence_level: str  # "high", "medium", "low"
    factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.score >= 0.8:
            self.confidence_level = "high"
        elif self.score >= 0.5:
            self.confidence_level = "medium"
        else:
            self.confidence_level = "low"


class StreamingResponse:
    """
    Handles streaming responses for better UX.
    
    Provides real-time feedback to users as responses
    are being generated.
    """
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self._index = 0
    
    def __iter__(self) -> Iterator[str]:
        for chunk in self.chunks:
            yield chunk
    
    def __aiter__(self) -> AsyncIterator[str]:
        """Async iterator for streaming."""
        async def async_generator():
            for chunk in self.chunks:
                yield chunk
                await asyncio.sleep(0.01)  # Small delay for streaming effect
        return async_generator()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks": self.chunks,
            "total_chunks": len(self.chunks),
            "total_length": sum(len(c) for c in self.chunks)
        }


class ResponseFormatter:
    """
    Formats agent responses in various formats.
    
    Supports multiple output formats for different use cases.
    """
    
    def __init__(self, format: ResponseFormat = ResponseFormat.MARKDOWN):
        self.format = format
    
    def format_response(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Format response based on selected format."""
        if self.format == ResponseFormat.JSON:
            return self._format_json(content, metadata)
        elif self.format == ResponseFormat.MARKDOWN:
            return self._format_markdown(content, metadata)
        elif self.format == ResponseFormat.HTML:
            return self._format_html(content, metadata)
        elif self.format == ResponseFormat.PLAIN_TEXT:
            return self._format_plain_text(content, metadata)
        elif self.format == ResponseFormat.CONSOLE:
            return self._format_console(content, metadata)
        return content
    
    def _format_json(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format as JSON."""
        data = {
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        return json.dumps(data, indent=2)
    
    def _format_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format as Markdown."""
        md = content
        
        # Add metadata as collapsible section if present
        if metadata:
            md += "\n\n<details>\n<summary>Details</summary>\n\n"
            for key, value in metadata.items():
                md += f"- **{key}**: {value}\n"
            md += "\n</details>"
        
        return md
    
    def _format_html(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format as HTML."""
        # Simple markdown to HTML conversion
        html = content.replace("**", "<strong>").replace("*", "<em>")
        html = html.replace("\n\n", "</p><p>")
        html = f"<p>{html}</p>"
        
        if metadata:
            html += "\n<details><summary>Details</summary>\n<ul>"
            for key, value in metadata.items():
                html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul>\n</details>"
        
        return html
    
    def _format_plain_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format as plain text."""
        text = content
        
        if metadata:
            text += "\n\n--- Details ---"
            for key, value in metadata.items():
                text += f"\n{key}: {value}"
        
        return text
    
    def _format_console(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format for console output with colors."""
        # Simple ANSI color codes
        RESET = "\033[0m"
        BOLD = "\033[1m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        
        text = f"{BLUE}{BOLD}{content}{RESET}"
        
        if metadata:
            text += f"\n{GREEN}--- Details ---{RESET}"
            for key, value in metadata.items():
                text += f"\n  {key}: {value}"
        
        return text


class ConversationContext:
    """
    Manages conversation context across multiple turns.
    
    Tracks user preferences, history, and session state.
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.messages: List[Dict[str, str]] = []
        self.preferences: Dict[str, Any] = {}
        self.agents_used: List[str] = []
        self.tools_used: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_agent(self, agent_name: str):
        """Record agent usage."""
        if agent_name not in self.agents_used:
            self.agents_used.append(agent_name)
    
    def add_tool(self, tool_name: str):
        """Record tool usage."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.preferences.get(key, default)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "agents_used": self.agents_used,
            "tools_used": self.tools_used,
            "preferences": self.preferences
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for agent responses.
    
    Helps users understand how reliable the response is.
    """
    
    def __init__(self):
        self.weights = {
            "source_quality": 0.3,
            "fact_verification": 0.3,
            "completeness": 0.2,
            "recency": 0.2
        }
    
    def calculate_confidence(
        self,
        sources: List[Dict[str, Any]] = None,
        verified_facts: int = 0,
        total_claims: int = 0,
        has_comprehensive_answer: bool = True,
        source_date: str = None
    ) -> ConfidenceScore:
        """Calculate confidence score based on various factors."""
        factors = []
        score = 0.0
        
        # Source quality factor
        if sources:
            score += self.weights["source_quality"]
            factors.append(f"Source quality: {len(sources)} sources found")
        else:
            factors.append("No sources available")
        
        # Fact verification factor
        if total_claims > 0:
            verification_ratio = verified_facts / total_claims
            verification_score = verification_ratio * self.weights["fact_verification"]
            score += verification_score
            factors.append(
                f"Fact verification: {verified_facts}/{total_claims} claims verified"
            )
        else:
            factors.append("No claims to verify")
        
        # Completeness factor
        if has_comprehensive_answer:
            score += self.weights["completeness"]
            factors.append("Answer appears comprehensive")
        else:
            factors.append("Answer may be incomplete")
        
        # Recency factor (simple heuristic)
        if source_date:
            factors.append(f"Source date: {source_date}")
        else:
            score += self.weights["recency"] * 0.5
            factors.append("Source recency unknown")
        
        # Normalize score
        score = min(score, 1.0)
        
        return ConfidenceScore(
            score=score,
            confidence_level="high" if score >= 0.8 else "medium" if score >= 0.5 else "low",
            factors=factors
        )


# UX Manager - combines all UX features
class UXManager:
    """
    Central manager for user experience features.
    
    Coordinates streaming, formatting, context, and confidence scoring.
    """
    
    def __init__(
        self,
        response_format: ResponseFormat = ResponseFormat.MARKDOWN,
        enable_streaming: bool = False
    ):
        self.formatter = ResponseFormatter(response_format)
        self.streaming_enabled = enable_streaming
        self.confidence_scorer = ConfidenceScorer()
        self._contexts: Dict[str, ConversationContext] = {}
    
    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """Get or create a conversation context."""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(session_id)
        return self._contexts[session_id]
    
    def format_response(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Format a response."""
        return self.formatter.format_response(content, metadata)
    
    def create_streaming_response(self, content: str) -> StreamingResponse:
        """Create a streaming response from content."""
        # Split content into chunks
        chunks = self._split_into_chunks(content, chunk_size=50)
        return StreamingResponse(chunks)
    
    def _split_into_chunks(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split text into chunks for streaming."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(word)
            current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
    
    def calculate_confidence(
        self,
        sources: List[Dict[str, Any]] = None,
        verified_facts: int = 0,
        total_claims: int = 0,
        has_comprehensive_answer: bool = True,
        source_date: str = None
    ) -> ConfidenceScore:
        """Calculate confidence score."""
        return self.confidence_scorer.calculate_confidence(
            sources, verified_facts, total_claims,
            has_comprehensive_answer, source_date
        )


# Import uuid for session IDs
import uuid

# Global UX manager
ux_manager = UXManager()


def get_ux_manager() -> UXManager:
    """Get the global UX manager instance."""
    return ux_manager
