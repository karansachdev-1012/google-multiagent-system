"""
Testing and Validation for QueryClassifier and Dynamic Agent Router

This module provides test cases to validate:
1. Single domain queries (should route to 1 agent)
2. Multi-domain queries (should route to 2+ agents)
3. Complex queries (should include fact_checker, critic)
4. Ambiguous queries (should have fallback)
"""

from coordinator import dynamic_agent_router, test_classifier, coordinator


def test_single_domain_queries():
    """Test that single domain queries route to 1 agent."""
    test_cases = [
        "What's the weather today?",
        "Find me a good restaurant",
        "How do I learn Python?",
        "What's the stock price of Apple?",
    ]
    
    print("=" * 60)
    print("TEST: Single Domain Queries (should route to 1 agent)")
    print("=" * 60)
    
    for query in test_cases:
        result = dynamic_agent_router(query)
        print(f"\nQuery: {query}")
        print(f"  Selected agents: {result.get('selected_agents', [])}")
        print(f"  Agent count: {result.get('agent_count', 0)}")
        print(f"  Primary domain: {test_classifier(query).get('primary_domain', 'N/A')}")


def test_multi_domain_queries():
    """Test that multi-domain queries route to 2+ agents."""
    test_cases = [
        "Find me restaurants and compare prices",
        "What's the weather in Paris and London?",
        "I want to buy a laptop and also book a flight",
        "Compare iPhone vs Samsung phones",
        "Tell me about cooking recipes and also find restaurants",
    ]
    
    print("\n" + "=" * 60)
    print("TEST: Multi-Domain Queries (should route to 2+ agents)")
    print("=" * 60)
    
    for query in test_cases:
        result = dynamic_agent_router(query)
        print(f"\nQuery: {query}")
        print(f"  Selected agents: {result.get('selected_agents', [])}")
        print(f"  Agent count: {result.get('agent_count', 0)}")
        print(f"  Is multi-domain: {test_classifier(query).get('is_multi_domain', False)}")


def test_complex_queries():
    """Test that complex queries include fact_checker and critic."""
    test_cases = [
        "Research climate change and verify the facts",
        "Provide a comprehensive analysis of AI trends",
        "Investigate the pros and cons of cryptocurrency",
        "Thoroughly research and evaluate renewable energy",
        "Give me a detailed explanation of quantum computing",
    ]
    
    print("\n" + "=" * 60)
    print("TEST: Complex Queries (should include fact_checker, critic)")
    print("=" * 60)
    
    for query in test_cases:
        result = dynamic_agent_router(query)
        print(f"\nQuery: {query}")
        print(f"  Selected agents: {result.get('selected_agents', [])}")
        print(f"  Agent count: {result.get('agent_count', 0)}")
        print(f"  Complexity: {test_classifier(query).get('complexity', 'N/A')}")
        has_fact_checker = 'fact_checker' in result.get('selected_agents', [])
        has_critic = 'critic' in result.get('selected_agents', [])
        print(f"  Has fact_checker: {has_fact_checker}")
        print(f"  Has critic: {has_critic}")


def test_ambiguous_queries():
    """Test that ambiguous queries have fallback."""
    test_cases = [
        "Hello",
        "Tell me something",
        "What about?",
        "Info",
    ]
    
    print("\n" + "=" * 60)
    print("TEST: Ambiguous Queries (should have fallback)")
    print("=" * 60)
    
    for query in test_cases:
        result = dynamic_agent_router(query)
        print(f"\nQuery: {query}")
        print(f"  Selected agents: {result.get('selected_agents', [])}")
        print(f"  Agent count: {result.get('agent_count', 0)}")
        print(f"  Is ambiguous: {test_classifier(query).get('ambiguous', False)}")
        has_researcher = 'researcher' in result.get('selected_agents', [])
        print(f"  Has researcher fallback: {has_researcher}")


def test_intent_detection():
    """Test that intent detection works correctly."""
    test_cases = [
        ("What is Python?", "informational"),
        ("Buy me a phone", "transactional"),
        ("Recommend a restaurant", "recommendation"),
        ("Compare iPhone vs Samsung", "comparison"),
    ]
    
    print("\n" + "=" * 60)
    print("TEST: Intent Detection")
    print("=" * 60)
    
    for query, expected_intent in test_cases:
        result = test_classifier(query)
        detected_intent = result.get('intent', 'N/A')
        print(f"\nQuery: {query}")
        print(f"  Expected intent: {expected_intent}")
        print(f"  Detected intent: {detected_intent}")
        print(f"  Match: {detected_intent == expected_intent}")


def test_complexity_scoring():
    """Test that complexity scoring works correctly."""
    test_cases = [
        ("What is Python?", "simple"),
        ("Compare Python vs Java", "moderate"),
        ("Provide a comprehensive analysis of Python", "complex"),
    ]
    
    print("\n" + "=" * 60)
    print("TEST: Complexity Scoring")
    print("=" * 60)
    
    for query, expected_complexity in test_cases:
        result = test_classifier(query)
        detected_complexity = result.get('complexity', 'N/A')
        print(f"\nQuery: {query}")
        print(f"  Expected complexity: {expected_complexity}")
        print(f"  Detected complexity: {detected_complexity}")
        print(f"  Match: {detected_complexity == expected_complexity}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR CLASSIFIER AND ROUTER")
    print("=" * 60)
    
    test_single_domain_queries()
    test_multi_domain_queries()
    test_complex_queries()
    test_ambiguous_queries()
    test_intent_detection()
    test_complexity_scoring()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
