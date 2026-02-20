"""
Researcher Agent - Performs broad web research on a given topic.
Uses Semantic Scholar API for academic papers and custom search tools.
"""

from google.adk.agents import Agent
from ..tools import search_academic_papers, fetch_webpage, search_news, google_search

researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    description="Research specialist that provides comprehensive information on topics using specialized tools.",
    instruction="""You are a research specialist. When given a topic or question:

1. Provide comprehensive information from your knowledge base
2. Organize findings by subtopic with specific facts and statistics
3. Use specialized tools:
   - google_search: For general web search to find information online
   - search_academic_papers: For scholarly research and papers (uses Semantic Scholar API - free)
   - fetch_webpage: For extracting content from specific URLs
   - search_news: For current news and events
4. Focus on authoritative sources and cite them
5. Return structured summaries with facts, figures, and sources

When you need to use tools, use the appropriate one for the task.""",
    tools=[search_academic_papers, fetch_webpage, search_news, google_search],
)
