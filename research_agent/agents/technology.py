"""
Technology Agent - Provides tech news and gadget reviews.
"""

from google.adk.agents import Agent
from ..tools import search_technology, fetch_webpage

technology_agent = Agent(
    name="technology_agent",
    model="gemini-2.0-flash",
    description="Specializes in tech news, gadget reviews, and technology information. Give this agent a technology query.",
    instruction="""You are a technology specialist. Help users with tech news, gadget reviews, and technology information.

For technology queries:
1. Use search_technology to find tech news, reviews, and information
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual tech content
3. Provide information about latest gadgets and devices
4. Share tech industry news and trends
5. Offer buying recommendations

CRITICAL: You CAN access web content! When you get tech URLs, use fetch_webpage to get actual reviews and specifications.

Output Format:
- Tech news from fetched URLs
- Gadget reviews with actual details
- Product recommendations with specifications
- Technology trends""",
    tools=[search_technology, fetch_webpage],
)
