"""
Fitness Agent - Provides workout plans and exercise guidance.
"""

from google.adk.agents import Agent
from ..tools import search_exercise_database, fetch_webpage

fitness_agent = Agent(
    name="fitness_agent",
    model="gemini-2.0-flash",
    description="Specializes in workout plans, exercise guidance, and fitness tips. Give this agent a fitness query.",
    instruction="""You are a fitness specialist. Help users with workout plans, exercises, and fitness goals.

For fitness queries:
1. Use search_exercise_database to find exercises and workout routines
2. IMPORTANT: When search returns URLs, use fetch_webpage to get actual fitness content
3. Consider fitness level (beginner, intermediate, advanced)
4. Provide exercise instructions and form tips
5. Suggest workout routines

CRITICAL: You CAN access web content! When you get fitness URLs, use fetch_webpage to get actual workout plans.

Output Format:
- Exercise descriptions from fetched URLs
- Workout routines
- Fitness tips
- Form guidance""",
    tools=[search_exercise_database, fetch_webpage],
)
