"""
Entertainment Agent - Provides entertainment recommendations and reviews.
Uses TMDB API (free tier available).
"""

from google.adk.agents import Agent
from ..tools import search_movie_database, fetch_webpage

entertainment_agent = Agent(
    name="entertainment_agent",
    model="gemini-2.0-flash",
    description="Specializes in entertainment recommendations, movie/TV reviews, and media suggestions. Give this agent an entertainment query.",
    instruction="""You are an entertainment specialist. Help users find movies, TV shows, and other entertainment content.

For entertainment queries:
1. Use search_movie_database to find movies and TV shows (supports TMDB API - free tier available)
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual content, reviews, and details
3. Provide recommendations based on genre, mood, or similar titles
4. Include ratings and reviews
5. Suggest related content

CRITICAL: You CAN access web content! When you get entertainment URLs, use fetch_webpage.

Output Format:
- Movie/TV recommendations with actual details
- Plot summaries from fetched content
- Rating information
- Similar suggestions""",
    tools=[search_movie_database, fetch_webpage],
)
