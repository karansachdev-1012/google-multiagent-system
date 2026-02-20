"""
Career Agent - Provides career planning and job search assistance.
"""

from google.adk.agents import Agent
from ..tools import search_jobs, fetch_webpage

career_agent = Agent(
    name="career_agent",
    model="gemini-2.0-flash",
    description="Specializes in career planning, job search assistance, and professional development. Give this agent a career query.",
    instruction="""You are a career specialist. Help users with career planning, job search, and professional development.

For career queries:
1. Use search_jobs to find job opportunities
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual job details
3. Provide career planning advice
4. Suggest professional development resources
5. Help with resume and interview tips

CRITICAL: You CAN access web content! When you get job URLs, use fetch_webpage to get actual job descriptions and requirements.

Output Format:
- Job details from fetched URLs
- Job recommendations
- Career advice
- Professional development tips
- Job search strategies""",
    tools=[search_jobs, fetch_webpage],
)
