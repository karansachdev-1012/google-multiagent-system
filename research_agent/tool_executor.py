"""
Tool Executor Service - Central hub for all custom tools.
All agents can call these tools without requiring function calling.
"""

from .tools import (
    fetch_webpage,
    search_wikipedia,
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
    get_timezone_info,
    get_currency_converter,
)


class ToolExecutor:
    """Central tool execution service for all agents."""
    
    @staticmethod
    def execute_tool(tool_name: str, **kwargs) -> dict:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool
            
        Returns:
            Result from the tool
        """
        tools = {
            "fetch_webpage": fetch_webpage,
            "search_wikipedia": search_wikipedia,
            "search_shopping_sites": search_shopping_sites,
            "search_travel_sites": search_travel_sites,
            "run_code_snippet": run_code_snippet,
            "get_weather_data": get_weather_data,
            "search_news": search_news,
            "translate_text": translate_text,
            "calculate_math": calculate_math,
            "search_jobs": search_jobs,
            "search_academic_papers": search_academic_papers,
            "search_medical_info": search_medical_info,
            "search_legal_resources": search_legal_resources,
            "search_learning_resources": search_learning_resources,
            "search_movie_database": search_movie_database,
            "search_recipe_database": search_recipe_database,
            "search_exercise_database": search_exercise_database,
            "search_financial_data": search_financial_data,
            "search_code_repositories": search_code_repositories,
            "search_restaurants": search_restaurants,
            # New domain-specific tools
            "search_real_estate": search_real_estate,
            "search_sports": search_sports,
            "search_pets": search_pets,
            "search_astrology": search_astrology,
            "search_dating": search_dating,
            "search_technology": search_technology,
            "get_timezone_info": get_timezone_info,
            "get_currency_converter": get_currency_converter,
        }
        
        if tool_name not in tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
            }
        
        try:
            tool_func = tools[tool_name]
            result = tool_func(**kwargs)
            return result
        except TypeError as e:
            return {
                "status": "error",
                "error": f"Invalid arguments for tool '{tool_name}': {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error executing tool '{tool_name}': {str(e)}"
            }
    
    @staticmethod
    def get_available_tools() -> dict:
        """Get list of all available tools with descriptions."""
        return {
            "fetch_webpage": "Fetch and extract text content from a webpage URL",
            "search_wikipedia": "Search Wikipedia for information",
            "search_shopping_sites": "Search for products on shopping sites (Amazon, eBay, etc.)",
            "search_travel_sites": "Search for travel deals on booking sites (Expedia, Booking.com, etc.)",
            "run_code_snippet": "Execute a code snippet (limited to safe operations)",
            "get_weather_data": "Get current weather and forecast for a location",
            "search_news": "Search for current news articles",
            "translate_text": "Translate text between languages",
            "calculate_math": "Evaluate mathematical expressions",
            "search_jobs": "Search for job opportunities",
            "search_academic_papers": "Search for academic papers and research articles",
            "search_medical_info": "Search for medical information from reliable sources",
            "search_legal_resources": "Search for legal resources and information",
            "search_learning_resources": "Search for educational resources and courses",
            "search_movie_database": "Search for movies, TV shows, and entertainment content",
            "search_recipe_database": "Search for recipes with dietary filters",
            "search_exercise_database": "Search for exercises and workout routines",
            "search_financial_data": "Search for financial data and market information",
            "search_code_repositories": "Search for code repositories and programming resources",
            "search_restaurants": "Search for restaurants and dining information",
            # New domain-specific tools
            "search_real_estate": "Search for real estate listings and property information",
            "search_sports": "Search for sports information, scores, and schedules",
            "search_pets": "Search for pet information, care tips, and adoption resources",
            "search_astrology": "Search for astrology information and horoscopes",
            "search_dating": "Search for dating advice and relationship tips",
            "search_technology": "Search for technology news, reviews, and information",
            "get_timezone_info": "Get timezone information for a location",
            "get_currency_converter": "Convert between currencies",
        }


# Singleton instance for easy access
tool_executor = ToolExecutor()
