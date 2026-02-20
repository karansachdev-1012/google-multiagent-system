"""
Translation Agent - Provides language translation and cultural adaptation.
"""

from google.adk.agents import Agent
from ..tools import translate_text

translation_agent = Agent(
    name="translation_agent",
    model="gemini-2.0-flash",
    description="Specializes in language translation and cultural adaptation. Give this agent a translation query.",
    instruction="""You are a translation specialist. Help users with language translation and cultural adaptation.

For translation queries:
1. Use translate_text to translate text between languages
2. Support multiple language pairs
3. Provide cultural context when helpful
4. Suggest alternative translations

Note: Configure LibreTranslate or DeepL API for actual translations.

Output Format:
- Translated text
- Language pair information
- Cultural notes (when relevant)""",
    tools=[translate_text],
)
