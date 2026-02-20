"""
Advanced Multi-Agent Coordination and Robustness Features

This module provides advanced coordination between multiple sub-agents,
fallback mechanisms, and sophisticated routing strategies.
Now includes an intelligent QueryClassifier for dynamic agent selection.

Enhancements include:
- Fuzzy matching and synonyms for better domain detection
- N-gram/phrase matching
- Weighted keyword scoring
- Edge case handling
- LLM classification integration
- Classification caching
- Pre-compiled regex patterns
- Dynamic agent selection based on complexity
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re
from functools import lru_cache
import hashlib
import time

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Roles that agents can play in multi-agent coordination."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    VERIFIER = "verifier"
    CRITIC = "critic"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    domains: List[str]
    tools: List[str]
    confidence_score: float  # 0.0 to 1.0
    specialties: List[str]
    limitations: List[str]


@dataclass
class ClassificationResult:
    """Result of query classification."""
    primary_domain: str
    confidence: float
    secondary_domains: List[str]
    required_tools: List[str]
    is_multi_domain: bool
    complexity: str  # simple, moderate, complex
    reasoning: str
    ambiguous: bool = False  # New: Flag for ambiguous queries
    intent: str = "informational"  # New: Query intent


class QueryCache:
    """Simple in-memory cache for classification results."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl
    
    def _get_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[ClassificationResult]:
        """Get cached result if not expired."""
        key = self._get_key(query)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                del self._cache[key]
        return None
    
    def set(self, query: str, result: ClassificationResult):
        """Cache a classification result."""
        # Evict oldest if cache is full
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        key = self._get_key(query)
        self._cache[key] = (result, time.time())
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


class SynonymMatcher:
    """Fuzzy matching and synonym-based domain detection."""
    
    # Synonyms and related terms for each domain
    SYNONYMS = {
        "research": ["investigate", "explore", "discover", "find information", "look up", "search for", "gather info", "inquiry", "data", "analysis of"],
        "verification": ["verify", "check", "confirm", "validate", "accuracy", "fact check", "true", "false", "reliable", "source check"],
        "shopping": ["buy", "purchase", "shop", "product", "price", "deal", "discount", "order", "store", "e-commerce", "recommend product", "best product"],
        "travel": ["trip", "vacation", "holiday", "flight", "hotel", "booking", "destination", "itinerary", "tour", "visit", "accommodation", "airline"],
        "coding": ["program", "code", "developer", "software", "debug", "function", "class", "script", "algorithm", "repository", "github", "programming language"],
        "finance": ["money", "investment", "stock", "trading", "bank", "budget", "financial", "crypto", "portfolio", "wealth", "income", "expense", "saving"],
        "health": ["medical", "doctor", "health", "symptom", "treatment", "disease", "wellness", "healthcare", "medicine", "prescription", "hospital", "clinic", "diagnosis"],
        "education": ["learn", "study", "course", "tutorial", "school", "college", "university", "education", "teach", "training", "lesson", "homework", "exam"],
        "entertainment": ["movie", "film", "music", "song", "game", "book", "show", "series", "tv", "streaming", "netflix", "concert", "theater", "hobby"],
        "legal": ["law", "legal", "attorney", "lawyer", "court", "contract", "rights", "compliance", "regulation", "legal advice", "law firm", "litigation"],
        "cooking": ["recipe", "cook", "food", "meal", "kitchen", "chef", "ingredient", "dish", "cuisine", "baking", "preparation", "restaurant"],
        "fitness": ["exercise", "workout", "gym", "training", "fitness", "muscle", "cardio", "strength", "yoga", "diet", "weight loss", "health"],
        "weather": ["weather", "forecast", "temperature", "climate", "rain", "sunny", "snow", "storm", "humidity", "weather report", "meteorology"],
        "news": ["news", "current events", "headline", "breaking", "update", "journalism", "report", "media", "latest", "today", "happening"],
        "translation": ["translate", "language", "speak", "foreign", "english", "spanish", "french", "german", "chinese", "japanese", "interpretation", "conversion"],
        "math": ["calculate", "math", "equation", "formula", "solve", "computation", "arithmetic", "algebra", "geometry", "calculus", "statistics"],
        "career": ["job", "career", "employment", "work", "resume", "interview", "hiring", "job search", "profession", "salary", "promotion", "workplace"],
        "restaurant": ["restaurant", "dining", "cafe", "bistro", "eatery", "food", "eat", "lunch", "dinner", "breakfast", "reservation", "table booking", "opentable"],
        "real_estate": ["house", "apartment", "property", "rent", "buy home", "real estate", "mortgage", "listing", "realtor", "housing", "condo", "flat"],
        "sports": ["sport", "game", "match", "score", "team", "player", "championship", "league", "nfl", "nba", "mlb", "soccer", "football", "basketball"],
        "pets": ["pet", "dog", "cat", "animal", "veterinary", "vet", "puppy", "kitten", "pet care", "adopt", "breed", "pet food", "grooming"],
        "astrology": ["horoscope", "zodiac", "sign", "astrology", "star", "birth chart", "planets", "compatibility", "tarot", "psychic", "spiritual"],
        "dating": ["dating", "relationship", "romance", "partner", "single", "match", "love", "date", "compatibility", "relationship advice"],
        "science": ["science", "scientific", "research", "discovery", "experiment", "physics", "chemistry", "biology", "space", "nasa", "technology"],
        "technology": ["tech", "technology", "gadget", "device", "computer", "phone", "laptop", "software", "app", "ai", "robot", "innovation"],
    }
    
    # Multi-word phrases that indicate specific domains
    PHRASES = {
        "research": ["find information about", "look up information", "search for", "tell me about", "what is", "how does"],
        "shopping": ["best price", "buy online", "product review", "compare prices", "where to buy", "recommend"],
        "travel": ["book flight", "hotel reservation", "cheap flights", "vacation package", "travel guide", "things to do"],
        "coding": ["write code", "debug program", "how to code", "programming help", "code snippet", "software development"],
        "finance": ["stock price", "investment advice", "financial planning", "how to invest", "market analysis", "portfolio"],
        "health": ["health advice", "medical help", "symptoms of", "treatment for", "how to treat", "health tips"],
        "legal": ["legal advice", "lawyer near", "legal rights", "contract law", "how to file", "legal help"],
    }
    
    def __init__(self):
        # Pre-compile regex patterns for phrases
        self._phrase_patterns: Dict[str, re.Pattern] = {}
        for domain, phrases in self.PHRASES.items():
            pattern = r'\b(' + '|'.join(re.escape(p) for p in phrases) + r')\b'
            self._phrase_patterns[domain] = re.compile(pattern, re.IGNORECASE)
    
    def find_matches(self, query: str) -> Dict[str, float]:
        """
        Find domain matches with confidence scores based on synonyms.
        
        Returns:
            Dictionary of domain -> match score
        """
        query_lower = query.lower()
        scores = {}
        
        # Check for phrase matches (higher weight)
        for domain, pattern in self._phrase_patterns.items():
            matches = pattern.findall(query_lower)
            if matches:
                scores[domain] = scores.get(domain, 0) + len(matches) * 2
        
        # Check for synonym matches
        for domain, synonyms in self.SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    # Exact match gets higher score
                    if synonym in query_lower.split():
                        scores[domain] = scores.get(domain, 0) + 2
                    else:
                        scores[domain] = scores.get(domain, 0) + 1
        
        return scores


@dataclass
class AgentAllocation:
    """Represents allocated agents for a query."""
    agents: List[str]
    roles: Dict[str, AgentRole]
    sequence: List[str]
    reasoning: str
    confidence_scores: Dict[str, float]


@dataclass
class MultiAgentWorkflow:
    """Defines a workflow involving multiple agents."""
    name: str
    description: str
    agents: List[str]
    roles: Dict[str, AgentRole]
    sequence: List[str]  # Order of execution
    fallback_agents: Dict[str, List[str]]  # Agent -> fallback agents
    success_criteria: Dict[str, Any]


class QueryClassifier:
    """
    Intelligent classifier that uses LLM-based classification to determine
    which agents are needed for a given query.
    
    Enhanced with:
    - Fuzzy matching and synonyms
    - N-gram/phrase matching
    - Weighted keyword scoring
    - Edge case handling
    - Classification caching
    - Pre-compiled patterns
    """

    # Domain to agent mapping - EXPANDED
    DOMAIN_TO_AGENTS = {
        "research": ["researcher"],
        "verification": ["fact_checker"],
        "analysis": ["critic"],
        "shopping": ["shopping_agent"],
        "travel": ["travelling_agent"],
        "coding": ["coding_agent"],
        "programming": ["coding_agent"],
        "software": ["coding_agent"],
        "finance": ["finance_agent"],
        "investment": ["finance_agent"],
        "health": ["health_agent"],
        "medical": ["health_agent"],
        "wellness": ["health_agent"],
        "education": ["education_agent"],
        "learning": ["education_agent"],
        "entertainment": ["entertainment_agent"],
        "movies": ["entertainment_agent"],
        "music": ["entertainment_agent"],
        "games": ["entertainment_agent"],
        "legal": ["legal_agent"],
        "law": ["legal_agent"],
        "cooking": ["cooking_agent"],
        "recipes": ["cooking_agent"],
        "fitness": ["fitness_agent"],
        "exercise": ["fitness_agent"],
        "weather": ["weather_agent"],
        "forecast": ["weather_agent"],
        "news": ["news_agent"],
        "current_events": ["news_agent"],
        "translation": ["translation_agent"],
        "language": ["translation_agent"],
        "math": ["math_science_agent"],
        "science": ["math_science_agent"],
        "calculation": ["math_science_agent"],
        "career": ["career_agent"],
        "jobs": ["career_agent"],
        "employment": ["career_agent"],
        "restaurant": ["restaurant_agent"],
        "dining": ["restaurant_agent"],
        "food": ["restaurant_agent"],
        # New domains
        "real_estate": ["real_estate_agent"],
        "housing": ["real_estate_agent"],
        "property": ["real_estate_agent"],
        "sports": ["sports_agent"],
        "game_scores": ["sports_agent"],
        "pets": ["pets_agent"],
        "pet_care": ["pets_agent"],
        "veterinary": ["pets_agent"],
        "astrology": ["astrology_agent"],
        "horoscope": ["astrology_agent"],
        "dating": ["dating_agent"],
        "relationship": ["dating_agent"],
        "technology": ["technology_agent"],
        "tech": ["technology_agent"],
    }

    # Tools that may be required based on query type - EXPANDED
    DOMAIN_TOOLS = {
        "shopping": ["search_shopping_sites", "fetch_webpage"],
        "travel": ["search_travel_sites", "get_weather_data", "fetch_webpage"],
        "coding": ["run_code_snippet", "search_code_repositories"],
        "math": ["calculate_math"],
        "weather": ["get_weather_data"],
        "news": ["search_news"],
        "translation": ["translate_text"],
        "jobs": ["search_jobs"],
        "academic": ["search_academic_papers"],
        "medical": ["search_medical_info"],
        "legal": ["search_legal_resources"],
        "education": ["search_learning_resources"],
        "entertainment": ["search_movie_database"],
        "cooking": ["search_recipe_database"],
        "fitness": ["search_exercise_database"],
        "finance": ["search_financial_data"],
        "research": ["google_search"],
        "restaurant": ["search_restaurants", "fetch_webpage"],
        "real_estate": ["fetch_webpage", "google_search"],
        "sports": ["google_search"],
        "pets": ["google_search"],
        "astrology": ["google_search"],
        "dating": ["google_search"],
        "technology": ["google_search", "fetch_webpage"],
    }

    # Multi-domain query indicators - EXPANDED with more patterns
    MULTI_DOMAIN_INDICATORS = [
        # Direct conjunction words
        "and", "also", "plus", "both", "additionally", "furthermore",
        # Comparison words
        "compare", "versus", "vs", "difference between", "differ from",
        # Multi-word patterns
        "as well as", "not only", "while", "but also", "combined with",
        "together with", "in addition to", "simultaneously", "at the same time",
        "along with", "plus also", "add to that", "on top of that",
        # Additional multi-domain patterns
        "on the other hand", "however", "moreover", "add to this",
        "what about", "how about", "another thing", "other than",
        "apart from", "besides", "except", "including",
    ]

    # Domain connection patterns for detecting multi-domain queries
    DOMAIN_CONNECTORS = {
        "and": ["research", "shopping", "travel", "coding", "finance", "health", "education", 
                "entertainment", "legal", "cooking", "fitness", "weather", "news", "translation",
                "math", "career", "restaurant", "real_estate", "sports", "pets", "astrology", "dating", "technology"],
        "compare": ["shopping", "travel", "coding", "finance", "health", "education", 
                   "entertainment", "real_estate", "technology", "career"],
        "vs": ["shopping", "travel", "coding", "finance", "health", "education",
               "entertainment", "real_estate", "technology", "sports"],
    }

    # Complexity indicators - EXPANDED with better keyword matching
    COMPLEXITY_KEYWORDS = {
        "simple": [
            "what is", "how to", "find", "show", "tell me", "get", "look", "search", 
            "give me", "list", "what's", "whats", "do you know", "can you tell",
            "is there", "where is", "when is", "who is", "simple", "quick"
        ],
        "moderate": [
            "explain", "analyze", "compare", "recommend", "best", "difference between", 
            "pros and cons", "review", "suggestion", "opinion", "thoughts", "advice",
            "tips", "how does", "why does", "can i", "should i", "which one",
            "vs", "versus", "better", "worse", "alternative", "options"
        ],
        "complex": [
            "research", "investigate", "comprehensive", "detailed analysis",
            "evaluate", "synthesize", "systematic", "in-depth", "thoroughly",
            "deep dive", "complete guide", "everything about", "full details",
            "research paper", "thorough analysis", "detailed explanation", "systematically",
            "in detail", "comprehensive overview", "full analysis", "extensive research",
            "critical analysis", "evaluate the", "assess the", "examine the",
            "investigate the", "analyze thoroughly", "provide a comprehensive"
        ]
    }

    # Query intent patterns
    INTENT_PATTERNS = {
        "informational": ["what is", "how does", "why is", "when did", "where is", "who is", "tell me about", "explain"],
        "transactional": ["buy", "purchase", "order", "book", "reserve", "get me", "find me"],
        "navigation": ["go to", "open", "navigate to", "take me to"],
        "computational": ["calculate", "compute", "solve", "what is", "how much", "convert"],
        "recommendation": ["recommend", "suggest", "best", "top", "what should", "which one"],
        "comparison": ["compare", "versus", "vs", "difference between", "better", "worse"],
    }

    def __init__(self, model=None, enable_cache: bool = True):
        """
        Initialize the classifier.
        
        Args:
            model: Optional LLM model for classification. If not provided,
                  falls back to rule-based classification.
            enable_cache: Whether to enable classification caching
        """
        self.model = model
        self._cache = QueryCache() if enable_cache else None
        self._synonym_matcher = SynonymMatcher()
        
        # Pre-compile complexity patterns
        self._complexity_patterns = {}
        for level, keywords in self.COMPLEXITY_KEYWORDS.items():
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self._complexity_patterns[level] = re.compile(pattern, re.IGNORECASE)
        
        # Pre-compile multi-domain patterns
        self._multi_domain_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(ind) for ind in self.MULTI_DOMAIN_INDICATORS) + r')\b',
            re.IGNORECASE
        )

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query to determine which agents are needed.
        
        Args:
            query: The user's query string
            
        Returns:
            ClassificationResult with domain, tools, and complexity info
        """
        # Handle edge cases
        if not query or not query.strip():
            return self._create_default_result("research", "Empty query - defaulting to research")
        
        query = query.strip()
        
        # Check cache first
        if self._cache:
            cached_result = self._cache.get(query)
            if cached_result:
                return cached_result
        
        # Handle very short queries
        if len(query) < 3:
            result = self._classify_short_query(query)
            if self._cache:
                self._cache.set(query, result)
            return result
        
        # Use LLM if available and properly configured
        if self.model:
            llm_result = self._llm_classify(query)
            if llm_result:
                if self._cache:
                    self._cache.set(query, llm_result)
                return llm_result
        
        # Fallback to enhanced rule-based classification
        result = self._rule_based_classify(query)
        
        # Cache the result
        if self._cache:
            self._cache.set(query, result)
        
        return result

    def _create_default_result(self, domain: str, reasoning: str) -> ClassificationResult:
        """Create a default classification result for edge cases."""
        return ClassificationResult(
            primary_domain=domain,
            confidence=0.5,
            secondary_domains=[],
            required_tools=self.DOMAIN_TOOLS.get(domain, []),
            is_multi_domain=False,
            complexity="simple",
            reasoning=reasoning,
            ambiguous=True,
            intent="informational"
        )

    def _classify_short_query(self, query: str) -> ClassificationResult:
        """Classify very short queries."""
        query_lower = query.lower()
        
        # Single word queries
        if len(query_lower.split()) == 1:
            # Check if it's a known domain keyword
            for domain in self.DOMAIN_TO_AGENTS:
                if domain in query_lower or query_lower in domain:
                    return ClassificationResult(
                        primary_domain=domain,
                        confidence=0.6,
                        secondary_domains=[],
                        required_tools=self.DOMAIN_TOOLS.get(domain, []),
                        is_multi_domain=False,
                        complexity="simple",
                        reasoning=f"Single word query matched to {domain}",
                        ambiguous=False,
                        intent="informational"
                    )
        
        # Default for short queries
        return self._create_default_result("research", f"Short query '{query}' - defaulting to research")

    def _llm_classify(self, query: str) -> Optional[ClassificationResult]:
        """Use LLM for intelligent classification."""
        classification_prompt = f"""Classify the following query to determine:
1. Primary domain (main topic)
2. Secondary domains (if any)
3. Required tools (if any)
4. Whether it's multi-domain
5. Complexity level (simple/moderate/complex)
6. Query intent (informational/transactional/navigation/computational/recommendation/comparison)
7. Is the query ambiguous?

Available domains: research, verification, shopping, travel, coding, finance, 
health, education, entertainment, legal, cooking, fitness, weather, 
news, translation, math_science, career, restaurant, real_estate, 
sports, pets, astrology, dating, technology

Query: {query}

Respond in JSON format:
{{
    "primary_domain": "domain_name",
    "confidence": 0.0-1.0,
    "secondary_domains": ["domain1", "domain2"],
    "required_tools": ["tool1", "tool2"],
    "is_multi_domain": true/false,
    "complexity": "simple/moderate/complex",
    "intent": "informational/transactional/navigation/computational/recommendation/comparison",
    "ambiguous": true/false,
    "reasoning": "explanation"
}}
"""
        try:
            # Try to use the model for classification
            # In practice, this would call self.model.generate(classification_prompt)
            # For now, we'll return None to fall back to rule-based
            # TODO: Implement actual LLM integration
            return None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to rule-based")
            return None

    def _rule_based_classify(self, query: str) -> ClassificationResult:
        """Enhanced rule-based classification with fuzzy matching."""
        query_lower = query.lower()
        
        # 1. Detect query intent
        intent = self._detect_intent(query_lower)
        
        # 2. Detect complexity
        complexity = self._detect_complexity(query_lower)
        
        # 3. Detect domains using multiple methods
        domain_scores = self._detect_domains(query_lower)
        
        # 4. Detect multi-domain
        is_multi_domain = self._detect_multi_domain(query_lower, domain_scores)
        
        # 5. Determine primary and secondary domains
        if domain_scores:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            primary_domain = sorted_domains[0][0]
            
            # Calculate confidence based on score and number of matches
            max_score = sorted_domains[0][1]
            confidence = min(1.0, max_score / 5)  # Normalize with higher threshold
            
            # Boost confidence for clear matches
            if max_score >= 3:
                confidence = min(1.0, confidence + 0.2)
            
            # Get secondary domains (skip the first one)
            secondary_domains = [d for d, _ in sorted_domains[1:3]]
            
            # Check for ambiguity
            ambiguous = max_score < 2 or len(sorted_domains) == 0
        else:
            primary_domain = "research"  # Default
            confidence = 0.5
            secondary_domains = []
            ambiguous = True
        
        # 6. Determine required tools
        required_tools = self._get_required_tools(primary_domain, secondary_domains)
        
        # 7. Generate reasoning
        reasoning = self._generate_reasoning(
            primary_domain, domain_scores, is_multi_domain, complexity, intent, ambiguous
        )
        
        return ClassificationResult(
            primary_domain=primary_domain,
            confidence=confidence,
            secondary_domains=secondary_domains,
            required_tools=required_tools,
            is_multi_domain=is_multi_domain,
            complexity=complexity,
            reasoning=reasoning,
            ambiguous=ambiguous,
            intent=intent
        )

    def _detect_intent(self, query_lower: str) -> str:
        """Detect the intent of the query."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent
        return "informational"

    def _detect_complexity(self, query_lower: str) -> str:
        """Detect query complexity using pre-compiled patterns."""
        complex_matches = self._complexity_patterns["complex"].findall(query_lower)
        moderate_matches = self._complexity_patterns["moderate"].findall(query_lower)
        simple_matches = self._complexity_patterns["simple"].findall(query_lower)
        
        complex_count = len(complex_matches)
        moderate_count = len(moderate_matches)
        simple_count = len(simple_matches)
        
        # More nuanced complexity detection
        if complex_count > 0:
            return "complex"
        elif moderate_count > simple_count:
            return "moderate"
        elif simple_count > 0:
            return "simple"
        
        # Default based on query length
        if len(query_lower) > 200:
            return "complex"
        elif len(query_lower) > 50:
            return "moderate"
        return "simple"

    def _detect_domains(self, query_lower: str) -> Dict[str, float]:
        """Detect domains using keyword matching and synonym matching."""
        domain_scores = {}
        
        # Method 1: Direct keyword matching with weights
        for domain, keywords in self._get_domain_keywords().items():
            score = 0
            for kw in keywords:
                # Check for exact word match (higher weight)
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                    score += 2
                # Check for substring match (lower weight)
                elif kw in query_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = domain_scores.get(domain, 0) + score
        
        # Method 2: Synonym matching
        synonym_scores = self._synonym_matcher.find_matches(query_lower)
        for domain, score in synonym_scores.items():
            domain_scores[domain] = domain_scores.get(domain, 0) + score
        
        # Method 3: N-gram matching (2-3 word sequences)
        words = query_lower.split()
        if len(words) >= 2:
            # Check 2-grams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                for domain, keywords in self._get_domain_keywords().items():
                    if bigram in keywords:
                        domain_scores[domain] = domain_scores.get(domain, 0) + 3
        
        if len(words) >= 3:
            # Check 3-grams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                for domain, keywords in self._get_domain_keywords().items():
                    if trigram in keywords:
                        domain_scores[domain] = domain_scores.get(domain, 0) + 4
        
        return domain_scores

    def _detect_multi_domain(self, query_lower: str, domain_scores: Dict[str, float]) -> bool:
        """Detect if query spans multiple domains."""
        # Check for multi-domain indicators
        if self._multi_domain_pattern.search(query_lower):
            return True
        
        # Check if multiple domains have significant scores
        if len(domain_scores) >= 2:
            # Get top scores
            sorted_scores = sorted(domain_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                # If second domain has significant score, it's multi-domain
                if sorted_scores[1] >= sorted_scores[0] * 0.5:
                    return True
        
        return False

    def _get_domain_keywords(self) -> Dict[str, List[str]]:
        """Get comprehensive keyword mappings for domain detection."""
        return {
            "research": ["research", "information", "find", "learn", "study", "investigate", "explore", "discover", "inquiry"],
            "verification": ["verify", "check", "validate", "confirm", "fact", "accurate", "reliable", "true", "false"],
            "shopping": ["buy", "purchase", "price", "product", "shop", "store", "recommend", "deal", "discount", "order", "e-commerce", "best product"],
            "travel": ["travel", "trip", "flight", "hotel", "vacation", "booking", "destination", "itinerary", "tour", "visit", "accommodation", "airline", "cheap flights", "hotel reservation"],
            "coding": ["code", "program", "debug", "function", "class", "script", "developer", "software", "algorithm", "repository", "github", "programming", "code snippet"],
            "finance": ["money", "investment", "budget", "finance", "stock", "bank", "trading", "crypto", "portfolio", "wealth", "income", "expense", "saving", "financial", "market"],
            "health": ["health", "medical", "doctor", "symptom", "treatment", "wellness", "disease", "healthcare", "medicine", "prescription", "hospital", "clinic", "diagnosis", "health tips"],
            "education": ["learn", "study", "course", "school", "education", "teach", "tutorial", "lesson", "homework", "exam", "college", "university", "training"],
            "entertainment": ["movie", "music", "book", "game", "watch", "film", "series", "tv", "entertainment", "concert", "streaming", "netflix", "hobby", "show"],
            "legal": ["legal", "law", "contract", "rights", "court", "attorney", "lawyer", "compliance", "regulation", "legal advice"],
            "cooking": ["recipe", "cook", "food", "ingredient", "meal", "kitchen", "chef", "dish", "cuisine", "baking", "preparation", "restaurant"],
            "fitness": ["exercise", "workout", "fitness", "gym", "training", "muscle", "cardio", "strength", "yoga", "diet", "weight loss", "health"],
            "weather": ["weather", "forecast", "temperature", "rain", "climate", "sunny", "snow", "storm", "humidity", "meteorology", "weather report"],
            "news": ["news", "current", "event", "breaking", "update", "headline", "journalism", "report", "media", "latest", "happening"],
            "translation": ["translate", "language", "speak", "foreign", "english", "spanish", "french", "german", "chinese", "japanese", "interpretation", "conversion"],
            "math": ["calculate", "math", "equation", "formula", "number", "compute", "solve", "arithmetic", "algebra", "geometry", "calculus", "statistics"],
            "career": ["job", "career", "work", "employment", "resume", "interview", "hiring", "profession", "salary", "promotion", "workplace", "job search"],
            "restaurant": ["restaurant", "dining", "food", "eat", "lunch", "dinner", "breakfast", "cafe", "bistro", "eatery", "reservation", "table booking", "opentable"],
            "real_estate": ["house", "apartment", "property", "rent", "buy home", "real estate", "mortgage", "listing", "realtor", "housing", "condo", "flat", "home"],
            "sports": ["sport", "game", "match", "score", "team", "player", "championship", "league", "nfl", "nba", "mlb", "soccer", "football", "basketball", "baseball"],
            "pets": ["pet", "dog", "cat", "animal", "veterinary", "vet", "puppy", "kitten", "pet care", "adopt", "breed", "pet food", "grooming"],
            "astrology": ["horoscope", "zodiac", "sign", "astrology", "star", "birth chart", "planets", "compatibility", "tarot", "psychic", "spiritual"],
            "dating": ["dating", "relationship", "romance", "partner", "single", "match", "love", "date", "compatibility", "relationship advice"],
            "technology": ["tech", "technology", "gadget", "device", "computer", "phone", "laptop", "software", "app", "ai", "robot", "innovation", "digital"],
        }

    def _get_required_tools(self, primary: str, secondary: List[str]) -> List[str]:
        """Determine required tools based on domains."""
        tools = set(self.DOMAIN_TOOLS.get(primary, []))
        for domain in secondary:
            tools.update(self.DOMAIN_TOOLS.get(domain, []))
        return list(tools)

    def _generate_reasoning(
        self,
        primary_domain: str,
        domain_scores: Dict[str, float],
        is_multi_domain: bool,
        complexity: str,
        intent: str,
        ambiguous: bool
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        reasoning = f"Query classified as {primary_domain}"
        
        if domain_scores:
            matched = [d for d, score in domain_scores.items() if score > 0]
            reasoning += f" ({len(matched)} domain(s) matched)"
        
        if is_multi_domain:
            reasoning += ". Multi-domain query detected"
        
        if ambiguous:
            reasoning += ". Ambiguous query - using default routing"
        
        reasoning += f". Complexity: {complexity}"
        reasoning += f". Intent: {intent}"
        
        return reasoning

    def get_agents_for_domain(self, domain: str) -> List[str]:
        """Get agents mapped to a domain."""
        return self.DOMAIN_TO_AGENTS.get(domain, ["researcher"])

    def needs_tools_agent(self, required_tools: List[str]) -> bool:
        """Determine if the tools agent is needed."""
        return len(required_tools) > 0
    
    def clear_cache(self):
        """Clear the classification cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Classification cache cleared")


class AdvancedCoordinator:
    """Advanced coordination system for multiple agents with intelligent classification."""

    def __init__(self, model=None, enable_cache: bool = True):
        self.agent_capabilities = self._load_agent_capabilities()
        self.workflows = self._load_workflows()
        # Initialize the classifier with caching
        self.classifier = QueryClassifier(model=model, enable_cache=enable_cache)
        logger.info("AdvancedCoordinator initialized with enhanced QueryClassifier")

    def _load_agent_capabilities(self) -> Dict[str, AgentCapability]:
        """Load capabilities for all agents - EXPANDED."""
        return {
            "researcher": AgentCapability(
                domains=["research", "information", "analysis"],
                tools=["google_search"],
                confidence_score=0.9,
                specialties=["broad research", "information synthesis"],
                limitations=["real-time data", "specialized domains"]
            ),
            "fact_checker": AgentCapability(
                domains=["verification", "validation", "accuracy"],
                tools=["google_search"],
                confidence_score=0.95,
                specialties=["claim verification", "source validation"],
                limitations=["subjective topics", "future predictions"]
            ),
            "shopping_agent": AgentCapability(
                domains=["shopping", "products", "commerce"],
                tools=["google_search", "search_shopping_sites"],
                confidence_score=0.85,
                specialties=["product comparison", "price analysis"],
                limitations=["real-time inventory", "local availability"]
            ),
            "travelling_agent": AgentCapability(
                domains=["travel", "booking", "itinerary"],
                tools=["google_search", "search_travel_sites", "get_weather_data"],
                confidence_score=0.85,
                specialties=["travel planning", "booking assistance"],
                limitations=["real-time pricing", "personal preferences"]
            ),
            "coding_agent": AgentCapability(
                domains=["programming", "code", "development"],
                tools=["google_search", "search_code_repositories", "run_code_snippet"],
                confidence_score=0.9,
                specialties=["code generation", "debugging", "explanation"],
                limitations=["system-specific code", "security testing"]
            ),
            "finance_agent": AgentCapability(
                domains=["finance", "investment", "money"],
                tools=["google_search", "search_financial_data"],
                confidence_score=0.85,
                specialties=["financial analysis", "investment advice"],
                limitations=["personal financial advice", "real-time trading"]
            ),
            "health_agent": AgentCapability(
                domains=["health", "medical", "wellness"],
                tools=["google_search", "search_medical_info"],
                confidence_score=0.75,
                specialties=["general health info", "wellness tips"],
                limitations=["medical diagnosis", "personal treatment"]
            ),
            "education_agent": AgentCapability(
                domains=["education", "learning", "teaching"],
                tools=["google_search", "search_learning_resources", "search_academic_papers"],
                confidence_score=0.85,
                specialties=["learning resources", "study strategies"],
                limitations=["personalized curriculum", "assessment"]
            ),
            "entertainment_agent": AgentCapability(
                domains=["entertainment", "media", "leisure"],
                tools=["google_search", "search_movie_database"],
                confidence_score=0.85,
                specialties=["recommendations", "reviews"],
                limitations=["personal taste", "current availability"]
            ),
            "legal_agent": AgentCapability(
                domains=["legal", "law", "rights"],
                tools=["google_search", "search_legal_resources"],
                confidence_score=0.7,
                specialties=["general legal info", "resource provision"],
                limitations=["legal advice", "case-specific guidance"]
            ),
            "cooking_agent": AgentCapability(
                domains=["cooking", "recipes", "culinary"],
                tools=["google_search", "search_recipe_database"],
                confidence_score=0.9,
                specialties=["recipe provision", "cooking techniques"],
                limitations=["personal dietary needs", "ingredient availability"]
            ),
            "fitness_agent": AgentCapability(
                domains=["fitness", "exercise", "health"],
                tools=["google_search", "search_exercise_database"],
                confidence_score=0.85,
                specialties=["workout planning", "exercise guidance"],
                limitations=["medical conditions", "personal fitness levels"]
            ),
            "weather_agent": AgentCapability(
                domains=["weather", "climate", "forecast"],
                tools=["google_search", "get_weather_data"],
                confidence_score=0.95,
                specialties=["weather forecasting", "climate data"],
                limitations=["extreme precision", "long-term predictions"]
            ),
            "news_agent": AgentCapability(
                domains=["news", "current events", "journalism"],
                tools=["google_search", "search_news"],
                confidence_score=0.85,
                specialties=["news aggregation", "event analysis"],
                limitations=["breaking news speed", "source bias"]
            ),
            "translation_agent": AgentCapability(
                domains=["translation", "languages", "communication"],
                tools=["google_search", "translate_text"],
                confidence_score=0.9,
                specialties=["language translation", "cultural adaptation"],
                limitations=["idiomatic expressions", "context-specific terms"]
            ),
            "math_science_agent": AgentCapability(
                domains=["mathematics", "science", "calculations"],
                tools=["google_search", "calculate_math"],
                confidence_score=0.95,
                specialties=["mathematical computation", "scientific explanation"],
                limitations=["complex proofs", "experimental validation"]
            ),
            "career_agent": AgentCapability(
                domains=["career", "jobs", "professional development"],
                tools=["google_search", "search_jobs"],
                confidence_score=0.85,
                specialties=["career guidance", "job search"],
                limitations=["personal career assessment", "market predictions"]
            ),
            "restaurant_agent": AgentCapability(
                domains=["restaurant", "dining", "food"],
                tools=["google_search", "search_restaurants", "fetch_webpage"],
                confidence_score=0.9,
                specialties=["restaurant recommendations", "dining information", "reservations"],
                limitations=["real-time availability", "personal preferences"]
            ),
            "real_estate_agent": AgentCapability(
                domains=["real_estate", "housing", "property"],
                tools=["google_search", "fetch_webpage"],
                confidence_score=0.8,
                specialties=["property listings", "market analysis"],
                limitations=["real-time listings", "personal financing"]
            ),
            "sports_agent": AgentCapability(
                domains=["sports", "game_scores", "athletics"],
                tools=["google_search"],
                confidence_score=0.85,
                specialties=["sports news", "score updates", "team information"],
                limitations=["live game data", "personal betting"]
            ),
            "pets_agent": AgentCapability(
                domains=["pets", "pet_care", "veterinary"],
                tools=["google_search"],
                confidence_score=0.8,
                specialties=["pet care tips", "breed information", "vet recommendations"],
                limitations=["medical diagnosis", "emergency care"]
            ),
            "astrology_agent": AgentCapability(
                domains=["astrology", "horoscope", "spiritual"],
                tools=["google_search"],
                confidence_score=0.7,
                specialties=["horoscopes", "zodiac readings", "compatibility"],
                limitations=["scientific validation", "personal predictions"]
            ),
            "dating_agent": AgentCapability(
                domains=["dating", "relationship", "romance"],
                tools=["google_search"],
                confidence_score=0.75,
                specialties=["dating advice", "relationship tips"],
                limitations=["personal matching", "guaranteed results"]
            ),
            "technology_agent": AgentCapability(
                domains=["technology", "tech", "gadgets"],
                tools=["google_search", "fetch_webpage"],
                confidence_score=0.85,
                specialties=["tech reviews", "product comparisons"],
                limitations=["personal preferences", "pricing"]
            ),
            "tools_coordinator": AgentCapability(
                domains=["tool coordination", "workflow management"],
                tools=["fetch_webpage", "search_shopping_sites", "search_travel_sites", "run_code_snippet", "get_weather_data", "search_news", "translate_text", "calculate_math", "search_jobs", "search_academic_papers", "search_medical_info", "search_legal_resources", "search_learning_resources", "search_movie_database", "search_recipe_database", "search_exercise_database", "search_financial_data", "search_code_repositories", "search_restaurants"],
                confidence_score=0.9,
                specialties=["tool orchestration", "workflow optimization"],
                limitations=["domain-specific expertise", "real-time tool performance"]
            )
        }

    def _load_workflows(self) -> Dict[str, MultiAgentWorkflow]:
        """Load predefined multi-agent workflows - EXPANDED."""
        return {
            "research_verification": MultiAgentWorkflow(
                name="Research with Verification",
                description="Complete research workflow with verification and critique",
                agents=["researcher", "fact_checker", "critic", "tools_coordinator"],
                roles={
                    "researcher": AgentRole.PRIMARY,
                    "fact_checker": AgentRole.VERIFIER,
                    "critic": AgentRole.CRITIC,
                    "tools_coordinator": AgentRole.TERTIARY,
                },
                sequence=["researcher", "fact_checker", "critic", "tools_coordinator"],
                fallback_agents={
                    "researcher": ["education_agent"],
                    "fact_checker": ["researcher"],
                    "critic": ["researcher"],
                    "tools_coordinator": ["researcher"]
                },
                success_criteria={
                    "min_verified_claims": 3,
                    "max_criticisms": 5
                }
            ),
            "shopping_assistance": MultiAgentWorkflow(
                name="Comprehensive Shopping",
                description="Multi-faceted shopping assistance",
                agents=["shopping_agent", "finance_agent", "researcher", "tools_coordinator"],
                roles={
                    "shopping_agent": AgentRole.PRIMARY,
                    "finance_agent": AgentRole.SPECIALIST,
                    "researcher": AgentRole.SECONDARY,
                    "tools_coordinator": AgentRole.TERTIARY
                },
                sequence=["shopping_agent", "finance_agent", "researcher", "tools_coordinator"],
                fallback_agents={
                    "shopping_agent": ["researcher"],
                    "finance_agent": ["shopping_agent"],
                    "researcher": ["shopping_agent"],
                    "tools_coordinator": ["shopping_agent"]
                },
                success_criteria={
                    "min_options": 3,
                    "price_comparison": True
                }
            ),
            "travel_planning": MultiAgentWorkflow(
                name="Complete Travel Planning",
                description="End-to-end travel planning with multiple specialists",
                agents=["travelling_agent", "weather_agent", "finance_agent", "health_agent", "tools_coordinator"],
                roles={
                    "tools_coordinator": AgentRole.TERTIARY,
                    "travelling_agent": AgentRole.PRIMARY,
                    "weather_agent": AgentRole.SPECIALIST,
                    "finance_agent": AgentRole.SPECIALIST,
                    "health_agent": AgentRole.SPECIALIST
                },
                sequence=["travelling_agent", "weather_agent", "finance_agent", "health_agent", "tools_coordinator"],
                fallback_agents={
                    "travelling_agent": ["researcher"],
                    "weather_agent": ["travelling_agent"],
                    "finance_agent": ["travelling_agent"],
                    "health_agent": ["travelling_agent"],
                    "tools_coordinator": ["travelling_agent"]
                },
                success_criteria={
                    "itinerary_complete": True,
                    "safety_considered": True
                }
            ),
            "restaurant_booking": MultiAgentWorkflow(
                name="Restaurant Search and Booking",
                description="Find restaurants and get dining information",
                agents=["restaurant_agent", "tools_coordinator"],
                roles={
                    "restaurant_agent": AgentRole.PRIMARY,
                    "tools_coordinator": AgentRole.COORDINATOR
                },
                sequence=["restaurant_agent", "tools_coordinator"],
                fallback_agents={
                    "restaurant_agent": ["cooking_agent", "researcher"],
                    "tools_coordinator": ["researcher"]
                },
                success_criteria={
                    "min_options": 3,
                    "location_specified": True
                }
            ),
            "complex_research": MultiAgentWorkflow(
                name="Complex Research with Multiple Agents",
                description="Deep research requiring multiple specialized agents",
                agents=["researcher", "fact_checker", "critic", "tools_coordinator"],
                roles={
                    "researcher": AgentRole.PRIMARY,
                    "fact_checker": AgentRole.VERIFIER,
                    "critic": AgentRole.CRITIC,
                    "tools_coordinator": AgentRole.COORDINATOR
                },
                sequence=["researcher", "tools_coordinator", "fact_checker", "critic"],
                fallback_agents={
                    "researcher": ["education_agent"],
                    "fact_checker": ["researcher"],
                    "critic": ["researcher"],
                    "tools_coordinator": ["researcher"]
                },
                success_criteria={
                    "comprehensive_coverage": True,
                    "verified_information": True
                }
            )
        }

    def classify_query(self, query: str) -> ClassificationResult:
        """
        Classify a query using the intelligent classifier.
        
        Args:
            query: The user's query string
            
        Returns:
            ClassificationResult with domain, tools, and complexity info
        """
        return self.classifier.classify(query)

    def select_agents(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Intelligently select the best agents for a query using classification.
        
        Args:
            query: The user's query string
            context: Optional context information
            
        Returns:
            List of selected agent names
        """
        if context is None:
            context = {}

        # Use classifier to get classification result
        classification = self.classifier.classify(query)
        
        logger.info(f"Classification result: {classification.primary_domain} "
                   f"(confidence: {classification.confidence:.2f}, complexity: {classification.complexity})")

        # Get agents for the primary domain
        agents = self.classifier.get_agents_for_domain(classification.primary_domain)
        
        # Add agents for secondary domains
        for domain in classification.secondary_domains:
            domain_agents = self.classifier.get_agents_for_domain(domain)
            for agent in domain_agents:
                if agent not in agents:
                    agents.append(agent)
        
        # Add tools agent if needed
        if self.classifier.needs_tools_agent(classification.required_tools):
            if "tools_coordinator" not in agents:
                agents.append("tools_coordinator")
        
        # Add verification for complex queries
        if classification.complexity == "complex" and "fact_checker" not in agents:
            agents.append("fact_checker")
        if classification.complexity == "complex" and "critic" not in agents:
            agents.append("critic")
        
        # Dynamic agent limit based on complexity and ambiguity
        max_agents = self._get_max_agents(classification)
        
        # Limit agents based on complexity
        selected_agents = agents[:max_agents]
        
        # Ensure we have at least one agent
        if not selected_agents:
            selected_agents = ["researcher"]
            logger.warning(f"No agents selected for query, defaulting to researcher: {query[:50]}")

        # For ambiguous queries, add researcher as fallback
        if classification.ambiguous and "researcher" not in selected_agents:
            selected_agents.insert(0, "researcher")
            selected_agents = selected_agents[:max_agents]

        logger.info(f"Selected agents for query: {selected_agents}")
        return selected_agents
    
    def _get_max_agents(self, classification: ClassificationResult) -> int:
        """Determine maximum number of agents based on query complexity."""
        if classification.complexity == "complex":
            return 5
        elif classification.complexity == "moderate":
            return 4
        return 3

    def classify_and_route(self, query: str, context: Dict[str, Any] = None) -> AgentAllocation:
        """
        Complete classification and agent allocation pipeline.
        
        This is the main entry point for intelligent agent selection.
        
        Args:
            query: The user's query string
            context: Optional context information
            
        Returns:
            AgentAllocation with selected agents, roles, and sequence
        """
        if context is None:
            context = {}
        
        # Step 1: Classify the query
        classification = self.classifier.classify(query)
        
        logger.info(f"Query classification: {classification.reasoning}")
        
        # Step 2: Select agents based on classification
        selected_agents = self.select_agents(query, context)
        
        # Step 3: Assign roles based on complexity and domain
        roles = self._assign_roles(selected_agents, classification)
        
        # Step 4: Determine execution sequence
        sequence = self._determine_sequence(selected_agents, classification)
        
        # Step 5: Calculate confidence scores
        confidence_scores = self._calculate_confidence(selected_agents, classification)
        
        # Generate reasoning
        reasoning = f"Allocated {len(selected_agents)} agents based on {classification.primary_domain} domain. "
        reasoning += f"Query complexity: {classification.complexity}. "
        if classification.is_multi_domain:
            reasoning += f"Multi-domain query with secondary domains: {', '.join(classification.secondary_domains)}."
        if classification.ambiguous:
            reasoning += " Ambiguous query - researcher added as fallback."
        
        return AgentAllocation(
            agents=selected_agents,
            roles=roles,
            sequence=sequence,
            reasoning=reasoning,
            confidence_scores=confidence_scores
        )

    def _assign_roles(self, agents: List[str], classification: ClassificationResult) -> Dict[str, AgentRole]:
        """Assign roles to selected agents based on classification."""
        roles = {}
        
        if not agents:
            return roles
        
        # Primary domain agent gets PRIMARY role
        primary_agent = self.classifier.get_agents_for_domain(classification.primary_domain)[0]
        if primary_agent in agents:
            roles[primary_agent] = AgentRole.PRIMARY
        
        # Tools agent gets COORDINATOR role if present
        if "tools_coordinator" in agents:
            roles["tools_coordinator"] = AgentRole.COORDINATOR
        
        # Fact-checker gets VERIFIER for complex queries
        if classification.complexity == "complex" and "fact_checker" in agents:
            roles["fact_checker"] = AgentRole.VERIFIER
        
        # Critic gets CRITIC role for complex queries
        if classification.complexity == "complex" and "critic" in agents:
            roles["critic"] = AgentRole.CRITIC
        
        # Assign remaining roles
        remaining = [a for a in agents if a not in roles]
        role_order = [AgentRole.SECONDARY, AgentRole.TERTIARY, AgentRole.SPECIALIST]
        for i, agent in enumerate(remaining):
            roles[agent] = role_order[i % len(role_order)]
        
        return roles

    def _determine_sequence(self, agents: List[str], classification: ClassificationResult) -> List[str]:
        """Determine the execution sequence for agents."""
        if not agents:
            return []
        
        # Start with primary domain agent
        sequence = []
        primary_agent = self.classifier.get_agents_for_domain(classification.primary_domain)[0]
        
        if primary_agent in agents:
            sequence.append(primary_agent)
        
        # Add other agents
        for agent in agents:
            if agent not in sequence:
                sequence.append(agent)
        
        return sequence

    def _calculate_confidence(self, agents: List[str], classification: ClassificationResult) -> Dict[str, float]:
        """Calculate confidence scores for each agent."""
        scores = {}
        
        for agent in agents:
            capability = self.agent_capabilities.get(agent)
            if capability:
                # Base confidence from capability
                base_confidence = capability.confidence_score
                
                # Boost if agent matches primary domain
                if agent in self.classifier.get_agents_for_domain(classification.primary_domain):
                    boost = 0.1
                else:
                    boost = 0
                
                # Adjust by classification confidence
                adjusted = min(1.0, base_confidence + boost)
                
                # Penalize for ambiguous queries
                if classification.ambiguous:
                    adjusted -= 0.1
                
                scores[agent] = max(0.1, adjusted)
            else:
                scores[agent] = 0.5  # Default confidence
        
        return scores

    def _analyze_query_domains(self, query: str) -> List[str]:
        """Analyze query to extract relevant domains (legacy method)."""
        query_lower = query.lower()
        domains = []

        domain_keywords = {
            "research": ["research", "information", "find", "learn", "study"],
            "shopping": ["buy", "purchase", "price", "product", "shop", "store"],
            "travel": ["travel", "trip", "flight", "hotel", "vacation", "booking"],
            "coding": ["code", "program", "debug", "function", "class", "script"],
            "finance": ["money", "investment", "budget", "finance", "stock", "bank"],
            "health": ["health", "medical", "doctor", "symptom", "treatment", "wellness"],
            "education": ["learn", "study", "course", "school", "education", "teach"],
            "entertainment": ["movie", "music", "book", "game", "watch", "entertainment"],
            "legal": ["legal", "law", "contract", "rights", "court", "legal"],
            "cooking": ["recipe", "cook", "food", "ingredient", "meal", "kitchen"],
            "fitness": ["exercise", "workout", "fitness", "gym", "training", "health"],
            "weather": ["weather", "forecast", "temperature", "rain", "climate"],
            "news": ["news", "current", "event", "breaking", "update", "journalism"],
            "translation": ["translate", "language", "speak", "communication", "foreign"],
            "math": ["calculate", "math", "equation", "formula", "number", "compute"],
            "career": ["job", "career", "work", "employment", "resume", "interview"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)

        return domains if domains else ["research"]

    def get_workflow(self, query: str) -> Optional[MultiAgentWorkflow]:
        """Get appropriate workflow for a query."""
        query_lower = query.lower()

        # Enhanced workflow selection logic
        if any(word in query_lower for word in ["research", "investigate", "explore"]):
            if any(word in query_lower for word in ["verify", "check", "validate", "confirm"]):
                return self.workflows.get("research_verification")
            return self.workflows.get("complex_research")
        elif any(word in query_lower for word in ["buy", "purchase", "shopping", "price"]):
            return self.workflows.get("shopping_assistance")
        elif any(word in query_lower for word in ["travel", "trip", "vacation", "booking", "flight", "hotel"]):
            return self.workflows.get("travel_planning")
        elif any(word in query_lower for word in ["restaurant", "dining", "food", "eat", "lunch", "dinner"]):
            return self.workflows.get("restaurant_booking")

        return None

    def optimize_agent_order(self, agents: List[str], query: str) -> List[str]:
        """Optimize the order of agent execution for efficiency."""
        # Use classification for smarter ordering
        classification = self.classifier.classify(query)
        
        if not agents:
            return agents

        # Get primary agent
        primary_agent = self.classifier.get_agents_for_domain(classification.primary_domain)[0]
        
        # Reorder: primary first, then others
        ordered = [primary_agent] if primary_agent in agents else []
        for agent in agents:
            if agent not in ordered:
                ordered.append(agent)
        
        return ordered
    
    def get_agent_capability(self, agent_name: str) -> Optional[AgentCapability]:
        """Get capability information for a specific agent."""
        return self.agent_capabilities.get(agent_name)
    
    def list_available_agents(self) -> List[str]:
        """List all available agent names."""
        return list(self.agent_capabilities.keys())
    
    def clear_cache(self):
        """Clear the classification cache."""
        self.classifier.clear_cache()
        logger.info("Coordinator cache cleared")


# Global coordinator instance
coordinator = AdvancedCoordinator()


def dynamic_agent_router(query: str) -> dict:
    """
    Dynamic Agent Router - Intelligently selects agents based on query analysis.
    
    This tool analyzes the user's query and returns:
    - Selected agents (1, 2, or multiple depending on query)
    - Reasoning for agent selection
    - Execution order
    - Required tools
    
    Use this tool first when handling ANY user query to determine which 
    specialized agents should be engaged.
    
    Args:
        query: The user's query string
        
    Returns:
        Dictionary with selected agents, roles, sequence, and reasoning
    """
    try:
        # Use the coordinator to classify and route
        allocation = coordinator.classify_and_route(query)
        
        # Format the response
        return {
            "status": "success",
            "query": query,
            "selected_agents": allocation.agents,
            "roles": {agent: role.value for agent, role in allocation.roles.items()},
            "sequence": allocation.sequence,
            "reasoning": allocation.reasoning,
            "confidence_scores": allocation.confidence_scores,
            "agent_count": len(allocation.agents),
            "message": f"Routed to {len(allocation.agents)} agent(s): {', '.join(allocation.agents)}"
        }
    except Exception as e:
        logger.error(f"Error in dynamic_agent_router: {e}")
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "selected_agents": ["researcher"],
            "reasoning": "Error in routing, defaulting to researcher",
            "agent_count": 1
        }


def test_classifier(query: str) -> dict:
    """
    Test the query classifier to see how it analyzes a query.
    
    Use this to understand how the system classifies different types of queries.
    
    Args:
        query: The query to test
        
    Returns:
        Classification details including domains, complexity, intent
    """
    try:
        result = coordinator.classify_query(query)
        
        return {
            "status": "success",
            "query": query,
            "primary_domain": result.primary_domain,
            "confidence": result.confidence,
            "secondary_domains": result.secondary_domains,
            "is_multi_domain": result.is_multi_domain,
            "complexity": result.complexity,
            "intent": result.intent,
            "ambiguous": result.ambiguous,
            "required_tools": result.required_tools,
            "reasoning": result.reasoning
        }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e)
        }
