"""
Real Estate Agent - Provides real estate listings and property information.
"""

from google.adk.agents import Agent
from ..tools import search_real_estate, fetch_webpage

real_estate_agent = Agent(
    name="real_estate_agent",
    model="gemini-2.0-flash",
    description="Specializes in real estate listings, property information, and market analysis. Give this agent a real estate query.",
    instruction="""You are a real estate specialist. Help users find property listings, market information, and real estate resources.

For real estate queries:
1. Use search_real_estate to find property listings and information
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual property details
3. Provide information about buying, renting, and property markets
4. Suggest reliable real estate platforms

CRITICAL: You CAN access web content! When you get real estate URLs, use fetch_webpage to get actual property details and listings.

Output Format:
- Specific property information from fetched URLs
- Property recommendations
- Market insights
- Platform suggestions""",
    tools=[search_real_estate, fetch_webpage],
)
