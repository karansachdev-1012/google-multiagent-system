"""
Agents module for the research agent system.

Contains 25 specialized agents:
- 17 original knowledge-based agents
- 6 new domain agents (real estate, sports, pets, astrology, dating, technology)
- 1 tools coordinator agent
- 1 critic agent
- 1 fact checker agent
"""

from .researcher import researcher
from .fact_checker import fact_checker
from .critic import critic
from .shopping import shopping_agent
from .travelling import travelling_agent
from .coding import coding_agent
from .finance import finance_agent
from .health import health_agent
from .education import education_agent
from .entertainment import entertainment_agent
from .legal import legal_agent
from .cooking import cooking_agent
from .fitness import fitness_agent
from .weather import weather_agent
from .news import news_agent
from .translation import translation_agent
from .math_science import math_science_agent
from .career import career_agent
from .restaurant import restaurant_agent
from .tools_coordinator import tools_agent
# New domain agents
from .real_estate import real_estate_agent
from .sports import sports_agent
from .pets import pets_agent
from .astrology import astrology_agent
from .dating import dating_agent
from .technology import technology_agent

__all__ = [
    # Original agents
    "researcher", "fact_checker", "critic", "shopping_agent", "travelling_agent", "coding_agent",
    "finance_agent", "health_agent", "education_agent", "entertainment_agent",
    "legal_agent", "cooking_agent", "fitness_agent", "weather_agent", "news_agent",
    "translation_agent", "math_science_agent", "career_agent", "restaurant_agent", "tools_agent",
    # New domain agents
    "real_estate_agent", "sports_agent", "pets_agent", "astrology_agent", "dating_agent", "technology_agent"
]
