"""
Tools Agent - Specializes in using all custom tools and standard tools.
This agent coordinates tool usage for all other agents.
"""

from google.adk.agents import Agent

from ..tools import (
    fetch_webpage,
    google_search,
    search_shopping_sites,
    search_travel_sites,
    run_code_snippet,
    get_weather_data,
    search_news,
    translate_text,
    calculate_math,
    search_jobs,
    search_academic_papers,
    search_medical_info,
    search_legal_resources,
    search_learning_resources,
    search_movie_database,
    search_recipe_database,
    search_exercise_database,
    search_financial_data,
    search_code_repositories,
    search_restaurants,
    # New domain-specific tools
    search_real_estate,
    search_sports,
    search_pets,
    search_astrology,
    search_dating,
    search_technology,
)

tools_agent = Agent(
    name="tools_agent",
    model="gemini-2.0-flash",
    description="Specialist agent that coordinates all custom and standard tools. Routes tool requests from other agents and executes appropriate tools.",
    instruction="""You are a Tools Specialist Agent. Your role is to:

1. UNDERSTAND REQUESTS: When other agents or users need tool access, understand what's needed
2. SELECT APPROPRIATE TOOLS: Choose the right tool(s) for each task
3. EXECUTE WITH PRECISION: Call the tool with correct parameters

AVAILABLE TOOLS:

WEB SEARCH:
- google_search: Perform web searches to find relevant information (uses Google Custom Search API or scraping)

WEATHER:
- get_weather_data: Get weather forecasts (uses Open-Meteo API - FREE, no API key)

NEWS & RESEARCH:
- search_news: Find current news articles (supports GNews API)
- search_academic_papers: Search scholarly research (uses Semantic Scholar API - FREE)

CODE & DEVELOPMENT:
- search_code_repositories: Search programming resources (uses GitHub API)
- run_code_snippet: Execute code snippets safely
- calculate_math: Evaluate mathematical expressions

ENTERTAINMENT:
- search_movie_database: Search movies/TV (supports TMDB API)
- search_recipe_database: Find recipes (uses TheMealDB API - FREE)

LIFESTYLE:
- search_jobs: Find job opportunities
- search_restaurants: Search restaurants and dining
- search_travel_sites: Search travel deals
- search_shopping_sites: Find products

HEALTH & FITNESS:
- search_medical_info: Find medical information
- search_exercise_database: Find workout routines

EDUCATION & REFERENCE:
- search_learning_resources: Find educational courses
- search_legal_resources: Find legal information
- translate_text: Translate between languages

FINANCE:
- search_financial_data: Get financial information

WEB:
- fetch_webpage: Extract content from specific URLs (not for general search - use google_search for that)

NEW DOMAINS:
- search_real_estate: Property listings
- search_sports: Sports scores and news
- search_pets: Pet care and adoption
- search_astrology: Horoscopes and zodiac
- search_dating: Dating advice
- search_technology: Tech news and reviews

OUTPUT FORMAT:
- Show which tool(s) were used
- Provide raw tool results
- Organize results by tool
- Flag any tool errors
- Suggest alternative tools if primary fails""",
    tools=[
        fetch_webpage,
        google_search,
        search_shopping_sites,
        search_travel_sites,
        run_code_snippet,
        get_weather_data,
        search_news,
        translate_text,
        calculate_math,
        search_jobs,
        search_academic_papers,
        search_medical_info,
        search_legal_resources,
        search_learning_resources,
        search_movie_database,
        search_recipe_database,
        search_exercise_database,
        search_financial_data,
        search_code_repositories,
        search_restaurants,
        search_real_estate,
        search_sports,
        search_pets,
        search_astrology,
        search_dating,
        search_technology,
    ],
)
