"""
News Agent - Provides news and current events information.
"""

from google.adk.agents import Agent
from ..tools import search_news, fetch_webpage

news_agent = Agent(
    name="news_agent",
    model="gemini-2.0-flash",
    description="Specializes in news and current events. Give this agent a news query.",
    instruction="""You are a news specialist. Help users find current news and information.

For news queries:
1. Use search_news to find current news and information
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual news content
3. Provide information about current events and breaking news
4. Summarize news articles from fetched content

CRITICAL: You CAN access web content! When you get news URLs, use fetch_webpage to get actual news articles.

Output Format:
- News summaries from fetched URLs
- Current event information
- Breaking news updates
- Source citations""",
    tools=[search_news, fetch_webpage],
)
