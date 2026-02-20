# Advanced Multi-Agent System with Google ADK

A comprehensive AI agent system featuring 25+ specialized agents, advanced security measures, intelligent multi-domain query handling, and robust tool integrations. This system demonstrates enterprise-grade AI agent development with Google ADK.

## System Overview

| Component | Description |
|-----------|-------------|
| **25+ Specialized Agents** | Domain-specific agents covering research, shopping, travel, coding, finance, health, education, entertainment, legal, cooking, fitness, weather, news, translation, math/science, career, real estate, sports, pets, astrology, dating, technology |
| **Multi-Domain Routing** | Intelligent agent selection for queries asking about multiple topics (e.g., "weather AND flights") |
| **Advanced Security** | Rate limiting, content filtering, input validation, and error handling |
| **Token Tracking** | Real-time token usage monitoring and optimization |
| **Robust Tool Integrations** | Free APIs (Open-Meteo, Semantic Scholar, TheMealDB) with multiple fallbacks |
| **Enhanced Error Handling** | Never fail silently - always provide useful fallback data or search URLs |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                    │
│  "weather in Sydney tomorrow and flights from Chicago to Sydney"     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ROOT AGENT (Orchestrator)                         │
│  • Analyzes query for multiple topics                               │
│  • Routes to MULTIPLE agents for multi-domain queries               │
│  • weather_agent + travelling_agent (for the example above)        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
              ▼                                             ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│     WEATHER AGENT         │           │    TRAVELLING AGENT       │
│ • Uses Open-Meteo API    │           │ • Parses "from X to Y"   │
│ • Free, no API key       │           │ • Returns search URLs    │
│ • Falls back to wttr.in  │           │ • Skyscanner, Kiwi.com   │
└─────────────────────────────┘           └─────────────────────────────┘
              │                                             │
              ▼                                             ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│   get_weather_data tool   │           │  search_travel_sites tool │
│ • Nominatim geocoding     │           │ • Parses query patterns   │
│ • Open-Meteo forecast     │           │ • Returns direct URLs     │
│ • City coordinates fallback│           │ • Travel tips included   │
└─────────────────────────────┘           └─────────────────────────────┘
```

## Key Features

### 🔥 Multi-Domain Query Handling
When a user asks about MULTIPLE DIFFERENT THINGS, the system NOW calls MULTIPLE agents:

- "weather in Sydney AND flights to London" → weather_agent + travelling_agent
- "restaurants AND price comparison" → restaurant_agent + shopping_agent
- "research topic AND verify facts" → researcher + fact_checker + critic

### 🌤️ Weather Tool (Enhanced)
- Uses **Open-Meteo API** (FREE, no API key required)
- Falls back to **wttr.in** if Open-Meteo fails
- Falls back to **hardcoded city coordinates** as last resort
- Returns useful search URLs if all APIs fail

### ✈️ Travel Tool (Enhanced)
- Parses "from X to Y" patterns from queries
- Extracts travel dates (march, april, etc.)
- Extracts trip duration ("atleast 20 days", "20+ days")
- Returns direct search URLs:
  - Skyscanner
  - Kiwi.com
  - Expedia
  - Google Flights

## Agent Capabilities

### Core Research & Analysis (4 agents)
- **Researcher**: Broad information gathering and synthesis
- **Fact Checker**: Claim verification and source validation
- **Critic**: Analysis and identification of weaknesses
- **News Agent**: Current events and journalistic analysis

### Commerce & Lifestyle (8 agents)
- **Shopping Agent**: Product research and price comparison
- **Travel Agent**: Trip planning, flight search, hotel booking
- **Cooking Agent**: Recipes and culinary guidance
- **Entertainment Agent**: Media recommendations and reviews
- **Restaurant Agent**: Dining recommendations and reservations
- **Real Estate Agent**: Property listings and market analysis
- **Sports Agent**: Sports news and scores
- **Pets Agent**: Pet care information

### Professional & Personal Development (4 agents)
- **Finance Agent**: Investment advice and financial planning
- **Career Agent**: Job search and professional development
- **Education Agent**: Learning resources and study strategies
- **Fitness Agent**: Workout planning and exercise guidance

### Specialized Services (9 agents)
- **Coding Agent**: Programming help and code generation
- **Legal Agent**: Legal information and resource provision
- **Health Agent**: Wellness guidance and health information
- **Math/Science Agent**: Calculations and scientific explanations
- **Weather Agent**: Forecasts and climate data (Open-Meteo API)
- **Translation Agent**: Language translation and cultural adaptation
- **Astrology Agent**: Horoscopes and zodiac readings
- **Dating Agent**: Dating advice and relationship tips
- **Technology Agent**: Tech news and gadget reviews

### Utility Agent (1 agent)
- **Tools Agent**: Access to all custom tools (fetch, search, calculate, etc.)

## Custom Tools

| Tool | Purpose | API/Integration |
|------|---------|-----------------|
| `fetch_webpage` | Extract and summarize content from URLs | BeautifulSoup web scraping - **Agents can now fetch actual content from URLs instead of just providing links** |
| `get_weather_data` | Weather information | **Open-Meteo API** (FREE, no key) |
| `search_travel_sites` | Flight/hotel search | Skyscanner, Kiwi.com, Expedia URLs |
| `search_restaurants` | Restaurant search | Yelp API / scraping |
| `search_shopping_sites` | Product search | eBay API / Google Shopping |
| `search_news` | News aggregation | GNews API / Google News RSS |
| `search_academic_papers` | Academic research | Semantic Scholar API |
| `search_recipe_database` | Recipe search | TheMealDB API |
| `search_code_repositories` | Code search | GitHub API |
| `search_financial_data` | Stock/crypto data | Yahoo Finance |
| `translate_text` | Language translation | LibreTranslate |
| `calculate_math` | Mathematical computation | Safe eval |
| `search_jobs` | Job opportunities | Adzuna API / Indeed |
| `search_medical_info` | Medical research | PubMed |
| `search_legal_resources` | Legal information | Justia |

## Prerequisites

- Python 3.10+
- Google API Key (from [AI Studio](https://aistudio.google.com/apikey))
- Git

## Quick Start

```
bash
# Clone repository
git clone <repository-url>
cd google-multiagent-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

Access the web interface at `http://localhost:8080`

## Example Queries

### Single Domain (1 agent)
- "What's the weather in Sydney?" → weather_agent
- "Find me restaurants in Chicago" → restaurant_agent

### Multi-Domain (2+ agents!)
- "What's the weather in Sydney and flights to London?" → weather_agent + travelling_agent
- "Find restaurants and compare prices" → restaurant_agent + shopping_agent

### Complex Research (3+ agents)
- "Research climate change and verify the facts" → researcher + fact_checker + critic

## Security & Robustness Features

### 🔒 Security Measures
- **Rate Limiting**: Configurable request limits per user
- **Content Filtering**: Automatic detection of harmful content
- **Input Validation**: Length limits and sanitization
- **Error Handling**: Graceful failure recovery with user-friendly messages

### 🛡️ Robustness Features
- **Multiple Fallbacks**: Weather API fails → try wttr.in → try coordinates
- **Useful Errors**: Never just say "error" - provide search URLs as fallback
- **Agent Fallbacks**: If one agent fails, system tries alternatives
- **Query Parsing**: Extracts meaningful parts from complex queries

### 📊 Token Tracking
- **Real-time Monitoring**: Live token usage per agent and query
- **Cost Optimization**: Automatic selection of most efficient agent combinations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Weather returns error | Check network; tool has fallbacks but may need internet |
| Flight search returns URLs only | Direct flight APIs require authentication; URLs provided instead |
| Agent returns generic error | Check that agent tools are properly configured |
| Multi-domain query uses only 1 agent | Ensure root_agent has updated instructions for multi-domain |

## System Flow for User Query

For query: **"weather in australia sydney tomorrow and can you let me know flights from chicago to sydney the cheapest one between march to april to and fro for atleast 20 days travel"**

```
1. Root Agent analyzes query
   → Identifies MULTIPLE topics: weather + flights
   
2. Root Agent calls BOTH:
   - weather_agent → get_weather_data("Sydney, Australia")
   - travelling_agent → search_travel_sites(query)
   
3. Weather Agent result:
   - Location: Sydney, Australia
   - Temperature: XX°C (from Open-Meteo)
   - Condition: sunny/cloudy/rainy
   - Forecast: next 7 days
   
4. Travelling Agent result:
   - From: Chicago
   - To: Sydney
   - Dates: March-April
   - Duration: 20+ days
   - Search URLs: Skyscanner, Kiwi.com, etc.
   
5. Final response combines both results!
```

## License

This project is part of the Google ADK workshop series.

## Contributing

1. Follow the existing agent pattern for new agents
2. Add appropriate security measures
3. Include comprehensive error handling with fallbacks
4. Update documentation and tests
5. Ensure token efficiency

---

**Note**: This system has been enhanced with:
- Better multi-domain query handling
- Improved weather API with multiple fallbacks
- Enhanced travel tool with direct search URLs
- Never-fail error handling with useful fallbacks
- **Web scraping capability** - Agents now fetch ACTUAL content from URLs instead of just providing links!

## 🔧 Web Scraping Feature (Important!)

When agents return search results with URLs, they now use the `fetch_webpage` tool to:
1. Fetch the actual content from those URLs
2. Extract and summarize the relevant information
3. Provide specific answers rather than just links

### Example:
**Before**: Agent would say "Here's some legal information: [list of URLs]"

**Now**: Agent will:
1. Search for legal information
2. Fetch content from the most relevant URLs
3. Provide the **actual visa requirements** or legal information

This applies to all research agents including legal, travelling, shopping, restaurant, cooking, fitness, and more!
