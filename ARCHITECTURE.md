# Multi-Agent System Architecture

## Overview
This is a hierarchical multi-agent system with specialized agents and a centralized Tool Executor service.

## Architecture Pattern

```
User Query
    ↓
Main Orchestrator Agent (gemini-pro)
    ↓ (routes to appropriate agents)
Specialized Knowledge-Based Agents (18 agents)
    ↓ (suggest tool usage when needed)
Tool Executor Service (centralized tool access)
    ↓
Custom Tools (19 tools)
    ↓
Results returned and synthesized
```

## Key Components

### 1. Main Orchestrator (`research_agent/agent.py`)
- **Model**: gemini-pro (no function calling needed)
- **Role**: Route queries to appropriate specialized agents
- **Access**: All 18 specialized agents available for consultation
- **Tools**: Empty (delegates to Tool Executor when tools needed)

### 2. Specialized Agents (18 agents in `research_agent/agents/`)
Each agent covers a specific domain:
- **Researcher**: General research and information gathering
- **Fact Checker**: Verification of claims
- **Critic**: Analysis and critique
- **Shopping Agent**: Product research
- **Travelling Agent**: Travel planning
- **Coding Agent**: Programming help
- **Finance Agent**: Financial planning
- **Health Agent**: Health information
- **Education Agent**: Learning resources
- **Entertainment Agent**: Entertainment recommendations
- **Legal Agent**: Legal information
- **Cooking Agent**: Recipes and culinary advice
- **Fitness Agent**: Workout plans
- **Weather Agent**: Weather forecasts
- **News Agent**: Current events
- **Translation Agent**: Language translation
- **Math/Science Agent**: Calculations and explanations
- **Career Agent**: Career planning

**Each agent**:
- Uses `gemini-pro` model (no function calling)
- Has **no tools list** (knowledge-based only)
- Is instructed to **suggest tool usage** when appropriate
- Responds based on training knowledge
- Returns structured, domain-specific answers

### 3. Tool Executor Service (`research_agent/tool_executor.py`)
Centralized service providing access to all custom tools:
```python
from tool_executor import tool_executor

# Execute a tool
result = tool_executor.execute_tool("fetch_webpage", url="https://example.com")

# Get all available tools
tools = tool_executor.get_available_tools()
```

**Available Tools** (19 total):
1. `fetch_webpage` - Fetch webpage content
2. `search_shopping_sites` - Search products
3. `search_travel_sites` - Search travel deals
4. `run_code_snippet` - Execute code
5. `get_weather_data` - Get weather
6. `search_news` - Search news
7. `translate_text` - Translate languages
8. `calculate_math` - Math calculations
9. `search_jobs` - Find jobs
10. `search_academic_papers` - Search scholarly articles
11. `search_medical_info` - Medical information
12. `search_legal_resources` - Legal resources
13. `search_learning_resources` - Educational resources
14. `search_movie_database` - Search entertainment
15. `search_recipe_database` - Search recipes
16. `search_exercise_database` - Find exercises
17. `search_financial_data` - Financial information
18. `search_code_repositories` - Search code repos

## Workflow

### Single-Domain Query Example
1. User: "Tell me about quantum computing"
2. Orchestrator routes to `researcher` agent
3. Researcher provides comprehensive answer from knowledge base
4. If researcher suggests fetching specific resources, Tool Executor can be used

### Multi-Domain Query Example
1. User: "I'm planning a vegan travel trip to Japan. What restaurants and exercises should I try?"
2. Orchestrator coordinates:
   - `travelling_agent` for travel planning
   - `cooking_agent` for vegan restaurant recommendations
   - `fitness_agent` for exercise suggestions
3. Results synthesized into comprehensive response

## Why This Architecture Works

### Problem It Solves
- **Function Calling Incompatibility**: `gemini-pro` doesn't support function calling, so agents can't execute tools directly
- **Complexity Management**: Specialized agents handle domain expertise
- **Tool Access**: Tool Executor provides centralized, consistent tool access

### Benefits
1. **No Function Calling Required**: All agents use knowledge + suggestions
2. **Scalability**: Easy to add new agents or tools
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Agents can work independently or coordinate
5. **Reliability**: No asynchronous tool calling complexity

## Agent Instructions Pattern

Each agent is instructed to:
1. **Answer from knowledge base**: Provide comprehensive answers based on training
2. **Suggest tool usage**: When appropriate, recommend which tools would help
3. **Structured output**: Format responses clearly for the domain
4. **Cross-domain awareness**: Know when to defer to other agents

Example instruction pattern:
```python
instruction="""You are a [DOMAIN] specialist.

When answering questions:
1. Provide comprehensive answers from your knowledge
2. When current information is needed, suggest tools:
   - 'fetch_webpage' for specific URLs
   - '[search_*]' tools for domain data
3. Format responses with clear organization
4. Cite sources when possible

Output: [Domain-specific format]"""
```

## Tool Executor Usage

For framework/application code that needs tool execution:
```python
from research_agent.tool_executor import tool_executor

# Execute any tool
result = tool_executor.execute_tool("search_medical_info", query="diabetes")

# List all tools
available = tool_executor.get_available_tools()
```

## Token Usage

Estimated via character counting (~4 chars per token):
- Tracked per agent
- Logged to INFO level
- Summarized in responses

## Model Selection

- **gemini-pro**: Primary model
  - ✅ Stable and widely available
  - ✅ No function calling requirements
  - ✅ Good for knowledge-based responses
  - ✅ Suitable for multi-agent coordination

- Models NOT used:
  - ❌ gemini-2.0-flash: Doesn't support function calling
  - ❌ gemini-1.5-flash: Not found in API v1beta
  - ❌ gemini-1.5-pro: Not found in API v1beta

## Future Enhancements

1. **Real API Integration**: Replace placeholder tools with actual APIs
2. **Async Tool Execution**: Background tool execution with callbacks
3. **Caching**: Cache tool results for repeated queries
4. **Rate Limiting**: Tool-level rate limiting
5. **Agent Learning**: Track which agents handle which types best
6. **Custom Agents**: Easy framework for adding domain-specific agents
