"""
ADK Workshop - Research Agent

This is your starting point for the workshop exercises.
Follow the instructions in the README to progressively build
your Research Agent from a simple single agent to a full
multi-agent system.

Current Exercise: Extended Multi-Agent System (25+ Agents)
Enhanced with multi-domain query handling
"""

import logging
import re
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .agents import (
    researcher, fact_checker, critic, shopping_agent, travelling_agent, coding_agent,
    finance_agent, health_agent, education_agent, entertainment_agent,
    legal_agent, cooking_agent, fitness_agent, weather_agent, news_agent,
    translation_agent, math_science_agent, career_agent, restaurant_agent, tools_agent,
    # New domain agents
    real_estate_agent, sports_agent, pets_agent, astrology_agent, dating_agent, technology_agent
)

# Set up logging for token usage
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """
    Estimate token count based on text content.
    This is a rough approximation: ~4 characters per token for English text.
    """
    if not text:
        return 0
    # Remove extra whitespace and count characters
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    # Rough estimation: 1 token ≈ 4 characters
    return max(1, len(cleaned_text) // 4)

class TokenTrackingAgent:
    """Wrapper to track token usage for agent interactions."""
    
    def __init__(self, agent):
        self.agent = agent
        self.total_tokens = 0
        self.call_count = 0
    
    def track_usage(self, prompt: str, response: str):
        """Track token usage for a single interaction."""
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_tokens(response)
        total = prompt_tokens + response_tokens
        
        self.total_tokens += total
        self.call_count += 1
        
        logger.info(f"Agent '{self.agent.name}' - Prompt tokens: {prompt_tokens}, Response tokens: {response_tokens}, Total: {total}")
        
        return {
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': total
        }

# Create token-tracking wrappers for key agents
researcher_tracker = TokenTrackingAgent(researcher)
fact_checker_tracker = TokenTrackingAgent(fact_checker)
critic_tracker = TokenTrackingAgent(critic)
shopping_tracker = TokenTrackingAgent(shopping_agent)
travelling_tracker = TokenTrackingAgent(travelling_agent)
coding_tracker = TokenTrackingAgent(coding_agent)
finance_tracker = TokenTrackingAgent(finance_agent)
health_tracker = TokenTrackingAgent(health_agent)
education_tracker = TokenTrackingAgent(education_agent)
entertainment_tracker = TokenTrackingAgent(entertainment_agent)
legal_tracker = TokenTrackingAgent(legal_agent)
cooking_tracker = TokenTrackingAgent(cooking_agent)
fitness_tracker = TokenTrackingAgent(fitness_agent)
weather_tracker = TokenTrackingAgent(weather_agent)
news_tracker = TokenTrackingAgent(news_agent)
translation_tracker = TokenTrackingAgent(translation_agent)
math_science_tracker = TokenTrackingAgent(math_science_agent)
career_tracker = TokenTrackingAgent(career_agent)
restaurant_tracker = TokenTrackingAgent(restaurant_agent)
tools_tracker = TokenTrackingAgent(tools_agent)
# New domain agent trackers
real_estate_tracker = TokenTrackingAgent(real_estate_agent)
sports_tracker = TokenTrackingAgent(sports_agent)
pets_tracker = TokenTrackingAgent(pets_agent)
astrology_tracker = TokenTrackingAgent(astrology_agent)
dating_tracker = TokenTrackingAgent(dating_agent)
technology_tracker = TokenTrackingAgent(technology_agent)

root_agent = Agent(
    name="multi_domain_orchestrator",
    model="gemini-2.0-flash",
    description="A comprehensive AI orchestrator coordinating 25+ specialized agents across research, shopping, travel, coding, finance, health, education, entertainment, legal, cooking, fitness, weather, news, translation, math/science, career, real estate, sports, pets, astrology, dating, technology domains, and tool coordination.",
    instruction="""You are a multi-purpose AI assistant orchestrating 25+ specialized agents with INTELLIGENT DYNAMIC ROUTING!

## CRITICAL: MULTI-DOMAIN QUERY HANDLING

When a user asks about MULTIPLE DIFFERENT THINGS, you MUST call MULTIPLE agents!

For example:
- "weather in Australia Sydney tomorrow AND flights from Chicago to Sydney" = 2 agents needed!
  - weather_agent for Sydney weather
  - travelling_agent for flights
- "What's the weather and also find me restaurants" = 2 agents
- "Research topic and verify facts" = 2-3 agents

Look for these patterns indicating multi-domain:
- "and" / "also" / "plus" - separate topics
- "weather in X" + "flights to Y" - different domains
- "X and also Y" - multiple requests

## DYNAMIC AGENT SELECTION SYSTEM

Automatically analyze each query to determine which agents to use:

1. **ANALYZE THE QUERY**:
   - Identify PRIMARY DOMAIN (main topic)
   - Identify SECONDARY DOMAINS (if multi-domain)
   - Determine COMPLEXITY (simple/moderate/complex)
   - Determine INTENT (informational/transactional/comparison/recommendation)

2. **SELECT AGENTS BASED ON COMPLEXITY**:
   - **Simple** (1 domain, basic question): Use 1 agent
   - **Moderate** (2 domains or needs tools): Use 2-3 agents
   - **Complex** (research, verify, analyze): Use 3-5 agents including fact_checker

3. **MULTI-DOMAIN QUERIES**:
   Words like "and", "also", "compare", "versus", "both" = multi-domain = multiple agents needed!

## AVAILABLE AGENTS (25+):

**KNOWLEDGE AGENTS:**
1. researcher - General research & information
2. fact_checker - Verify claims & accuracy (for complex)
3. critic - Analysis & critique (for complex)
4. shopping_agent - Product research
5. travelling_agent - Travel planning (FLIGHTS, hotels, booking)
6. coding_agent - Programming help
7. finance_agent - Financial advice
8. health_agent - Health information
9. education_agent - Learning resources
10. entertainment_agent - Entertainment
11. legal_agent - Legal information
12. cooking_agent - Recipes & culinary
13. fitness_agent - Workout & exercise
14. weather_agent - Weather forecasts (TEMPERATURE, conditions, forecast)
15. news_agent - Current events
16. translation_agent - Language translation
17. math_science_agent - Math & science
18. career_agent - Career & jobs
19. restaurant_agent - Dining & restaurants
20. real_estate_agent - Property & real estate
21. sports_agent - Sports news & scores
22. pets_agent - Pet care info
23. astrology_agent - Horoscopes & zodiac
24. dating_agent - Dating advice
25. technology_agent - Tech news & gadgets

**TOOLS AGENT:**
- tools_agent - Access to ALL tools (fetch, search, calculate, etc.)

## TOOLS AVAILABLE:
google_search, fetch_webpage, search_shopping_sites, search_travel_sites,
run_code_snippet, get_weather_data, search_news, translate_text, calculate_math,
search_jobs, search_academic_papers, search_medical_info, search_legal_resources,
search_learning_resources, search_movie_database, search_recipe_database,
search_exercise_database, search_financial_data, search_code_repositories,
search_restaurants, search_real_estate, search_sports, search_pets,
search_astrology, search_dating, search_technology

## CRITICAL EXAMPLES:
- "weather in sydney tomorrow and flights from chicago to sydney cheapest" → BOTH weather_agent AND travelling_agent
- "What's the weather?" → weather_agent only
- "Find restaurants and compare prices" → restaurant_agent AND shopping_agent
- "Research climate change and verify facts" → researcher AND fact_checker AND critic

## RESPONSE GUIDELINES:
- Analyze query first, then select appropriate agents
- Use MORE agents for multi-domain queries (the user asked multiple things!)
- When in doubt, use multiple agents rather than just one
- ALWAYS call the relevant agents for each part of a multi-part query
""",
    tools=[
        AgentTool(agent=tools_agent),
        AgentTool(agent=researcher),
        AgentTool(agent=fact_checker),
        AgentTool(agent=critic),
        AgentTool(agent=shopping_agent),
        AgentTool(agent=travelling_agent),
        AgentTool(agent=coding_agent),
        AgentTool(agent=finance_agent),
        AgentTool(agent=health_agent),
        AgentTool(agent=education_agent),
        AgentTool(agent=entertainment_agent),
        AgentTool(agent=legal_agent),
        AgentTool(agent=cooking_agent),
        AgentTool(agent=fitness_agent),
        AgentTool(agent=weather_agent),
        AgentTool(agent=news_agent),
        AgentTool(agent=translation_agent),
        AgentTool(agent=math_science_agent),
        AgentTool(agent=career_agent),
        AgentTool(agent=restaurant_agent),
        # New agents
        AgentTool(agent=real_estate_agent),
        AgentTool(agent=sports_agent),
        AgentTool(agent=pets_agent),
        AgentTool(agent=astrology_agent),
        AgentTool(agent=dating_agent),
        AgentTool(agent=technology_agent),
    ],
)
