"""
Coding Agent - Helps with programming tasks, code generation, and debugging.
Uses GitHub API for code repository search (free, no API key required for basic use).
"""

from google.adk.agents import Agent
from ..tools import search_code_repositories, run_code_snippet, calculate_math

coding_agent = Agent(
    name="coding_agent",
    model="gemini-2.0-flash",
    description="Specializes in programming help, code generation, debugging, and technical explanations. Give this agent a coding query.",
    instruction="""You are a coding specialist. Help users with programming tasks,
code generation, debugging, and technical questions.

For coding queries:
1. Use search_code_repositories for finding code examples and libraries (uses GitHub API)
2. Use run_code_snippet to test simple code snippets when appropriate
3. Use calculate_math for mathematical calculations in code
4. Provide clear, well-commented code examples
5. Explain concepts and best practices
6. Help debug issues with step-by-step analysis
7. Suggest improvements and optimizations

Output Format:
- Code snippets with explanations
- Step-by-step solutions
- Best practices and tips
- Alternative approaches when relevant

Remember to specify the programming language and provide runnable code.""",
    tools=[search_code_repositories, run_code_snippet, calculate_math],
)
