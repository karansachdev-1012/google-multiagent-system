"""
Fact Checker Agent - Verifies claims and information accuracy.
Uses academic papers and web fetching for verification.
"""

from google.adk.agents import Agent
from ..tools import search_academic_papers, fetch_webpage

fact_checker = Agent(
    name="fact_checker",
    model="gemini-2.0-flash",
    description="Specializes in verifying claims and checking information accuracy.",
    instruction="""You are a fact-checking specialist. Verify claims and check information accuracy.

For fact-checking:
1. Use search_academic_papers to find scholarly sources for verification (uses Semantic Scholar API)
2. Use fetch_webpage to extract content from specific URLs for verification
3. Provide evidence-based verification
4. Cite reliable sources

Output Format:
- Claim verification (True/False/Mixed)
- Supporting evidence
- Source citations
- Confidence level""",
    tools=[search_academic_papers, fetch_webpage],
)
