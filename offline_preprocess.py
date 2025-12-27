"""
Offline LLM Preprocessing Script

This script implements the OFFLINE components from the paper architecture:
1. Entity-Aware Translator - Pre-generates translations for product catalog
2. Ambiguity Resolver - Pre-resolves ambiguous words with context
3. Contextual Rule Creator - Generates translation rules using LLM

Run this BEFORE starting the server to pre-populate the Translation Memory.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
os.chdir(script_dir)

from openai import OpenAI
from app.config import settings
from app.translation.tier1_cache import translation_memory

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OfflinePreprocessor:
    """
    Offline LLM-powered preprocessing to pre-populate Translation Memory.
    
    This aligns with the paper's architecture where Entity-Aware Translator
    and Ambiguity Resolver are OFFLINE components (blue boxes) that populate
    the Translation Memory BEFORE runtime.
    """
    
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._rules_cache: list[dict] = []
        
    def _get_client(self) -> OpenAI:
        """Get OpenAI client."""
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set. Please add it to .env file.")
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client
    
    # =========================================================================
    # 1. Entity-Aware Translation Pre-processing
    # =========================================================================
    
    def preprocess_entity_translations(self):
        """
        Pre-generate translations for product catalog entries.
        Uses LLM to create high-quality translations that preserve brand names.
        """
        logger.info("=" * 60)
        logger.info("ENTITY-AWARE TRANSLATION PREPROCESSING")
        logger.info("=" * 60)
        
        # Load brand catalog
        catalog_path = settings.DATA_DIR / "brand_catalog.json"
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
        
        # Generate translations for common product queries with brands
        brand_queries = self._generate_brand_queries(catalog)
        
        logger.info(f"Processing {len(brand_queries)} brand-related queries...")
        
        for query_data in brand_queries:
            try:
                translation = self._translate_with_entity_awareness(
                    query_data["french"],
                    query_data["brand"],
                    query_data["category"]
                )
                
                # Store in cache
                translation_memory.store(
                    query=query_data["french"],
                    translation=translation,
                    source="llm_generated",
                    confidence=0.95
                )
                
                logger.info(f"  ✓ '{query_data['french']}' → '{translation}'")
                
            except Exception as e:
                logger.error(f"  ✗ Failed for '{query_data['french']}': {e}")
        
        logger.info(f"Entity preprocessing complete. Cache size: {translation_memory.get_stats().total_entries}")
    
    def _generate_brand_queries(self, catalog: dict) -> list[dict]:
        """Generate common French queries containing brand names."""
        queries = []
        
        query_templates = [
            "{brand} {category_fr}",
            "acheter {brand}",
            "prix {brand}",
            "{brand} en promotion",
            "nouveau {brand}",
        ]
        
        category_fr_map = {
            "dairy": "yogurt",
            "household": "papier",
            "personal_care": "savon",
            "cosmetics": "parfum",
            "food": "céréales",
            "beverages": "boisson",
            "snacks": "chips",
            "baby": "couches"
        }
        
        for brand in catalog.get("brands", [])[:20]:  # Limit for demo
            brand_name = brand["name"]
            category = brand.get("category", "general")
            category_fr = category_fr_map.get(category, "produit")
            
            for template in query_templates[:3]:  # Limit templates for demo
                french_query = template.format(
                    brand=brand_name.lower(),
                    category_fr=category_fr
                )
                queries.append({
                    "french": french_query,
                    "brand": brand_name,
                    "category": category
                })
        
        return queries
    
    def _translate_with_entity_awareness(
        self, 
        french_query: str, 
        brand_name: str,
        category: str
    ) -> str:
        """Use LLM to translate while preserving brand name."""
        client = self._get_client()
        
        prompt = f"""Translate this French e-commerce search query to English.
IMPORTANT: Keep the brand name "{brand_name}" EXACTLY as-is (do not translate it).

French query: "{french_query}"
Category: {category}

Return ONLY the English translation, nothing else."""

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert French-to-English translator for e-commerce. Preserve brand names exactly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip().strip('"')
    
    # =========================================================================
    # 2. Ambiguity Resolution Pre-processing
    # =========================================================================
    
    def preprocess_ambiguous_translations(self):
        """
        Pre-generate translations for ambiguous words in different contexts.
        Creates multiple cache entries for the same word with different contexts.
        """
        logger.info("\n" + "=" * 60)
        logger.info("AMBIGUITY RESOLUTION PREPROCESSING")
        logger.info("=" * 60)
        
        # Load ambiguity dictionary
        dict_path = settings.DATA_DIR / "ambiguity_dict.json"
        with open(dict_path, 'r') as f:
            data = json.load(f)
        
        ambiguous_words = data.get("ambiguous_words", {})
        
        # Generate context-specific translations
        context_queries = self._generate_context_queries(ambiguous_words)
        
        logger.info(f"Processing {len(context_queries)} context-specific queries...")
        
        for query_data in context_queries:
            try:
                translation = self._translate_with_context(
                    query_data["french"],
                    query_data["context"],
                    query_data["expected_meaning"]
                )
                
                # Store with context-specific key
                cache_key = f"{query_data['french']}|context:{query_data['context']}"
                translation_memory.store(
                    query=cache_key,
                    translation=translation,
                    source="llm_generated",
                    confidence=0.90
                )
                
                # Also store the base query
                translation_memory.store(
                    query=query_data["french"],
                    translation=translation,
                    source="llm_generated",
                    confidence=0.85
                )
                
                logger.info(f"  ✓ '{query_data['french']}' ({query_data['context']}) → '{translation}'")
                
            except Exception as e:
                logger.error(f"  ✗ Failed for '{query_data['french']}': {e}")
        
        logger.info(f"Ambiguity preprocessing complete. Cache size: {translation_memory.get_stats().total_entries}")
    
    def _generate_context_queries(self, ambiguous_words: dict) -> list[dict]:
        """Generate queries with ambiguous words in different contexts."""
        queries = []
        
        query_templates = {
            "school_supplies": "acheter {word} pour l'école",
            "candy": "je veux {word} sucré",
            "fruit": "{word} fraîche du marché",
            "sports": "équipement de {word}",
            "cosmetics": "{word} beauté",
            "dairy": "{word} au lait",
            "bakery": "{word} pain frais",
            "jewelry": "{word} en or",
        }
        
        for word, info in list(ambiguous_words.items())[:10]:  # Limit for demo
            for meaning in info.get("meanings", []):
                category = meaning.get("category", "general")
                template = query_templates.get(category)
                
                if template:
                    french_query = template.format(word=word)
                    queries.append({
                        "french": french_query,
                        "context": category,
                        "expected_meaning": meaning.get("translation")
                    })
        
        return queries
    
    def _translate_with_context(
        self, 
        french_query: str, 
        context: str,
        expected_meaning: str
    ) -> str:
        """Use LLM to translate with specific context."""
        client = self._get_client()
        
        prompt = f"""Translate this French e-commerce search query to English.
Context: The user is shopping in the "{context}" category.
The word should be translated as "{expected_meaning}" in this context.

French query: "{french_query}"

Return ONLY the English translation, nothing else."""

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert French-to-English translator for e-commerce."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip().strip('"')
    
    # =========================================================================
    # 3. Contextual Rule Creator
    # =========================================================================
    
    def create_contextual_rules(self):
        """
        Use LLM to generate translation rules for the NMT post-processor.
        These rules are stored and used at runtime to fix common NMT errors.
        """
        logger.info("\n" + "=" * 60)
        logger.info("CONTEXTUAL RULE CREATION")
        logger.info("=" * 60)
        
        client = self._get_client()
        
        prompt = """Generate 10 translation rules for French-to-English e-commerce search queries.
Each rule should handle a common translation error or pattern.

Format each rule as JSON with:
- "pattern": the French pattern to match
- "wrong_translation": what NMT might incorrectly produce
- "correct_translation": the correct translation
- "explanation": why this rule matters

Example:
{"pattern": "liberté", "wrong_translation": "liberty", "correct_translation": "liberté", "explanation": "Liberté is a yogurt brand, not the word liberty"}

Return a JSON array of 10 rules."""

        try:
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in French-English translation for e-commerce."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse rules
            rules_text = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if "```json" in rules_text:
                rules_text = rules_text.split("```json")[1].split("```")[0]
            elif "```" in rules_text:
                rules_text = rules_text.split("```")[1].split("```")[0]
            
            self._rules_cache = json.loads(rules_text)
            
            # Save rules to file
            rules_path = settings.DATA_DIR / "contextual_rules.json"
            with open(rules_path, 'w') as f:
                json.dump({"rules": self._rules_cache}, f, indent=2)
            
            logger.info(f"Created {len(self._rules_cache)} contextual rules:")
            for rule in self._rules_cache[:5]:  # Show first 5
                logger.info(f"  • {rule.get('pattern')}: {rule.get('wrong_translation')} → {rule.get('correct_translation')}")
            
            logger.info(f"Rules saved to: {rules_path}")
            
        except Exception as e:
            logger.error(f"Failed to create contextual rules: {e}")
    
    # =========================================================================
    # Main Run Method
    # =========================================================================
    
    def run_all(self):
        """Run all offline preprocessing steps."""
        logger.info("\n" + "=" * 60)
        logger.info("OFFLINE LLM PREPROCESSING")
        logger.info("Populating Translation Memory with LLM-generated translations")
        logger.info("=" * 60 + "\n")
        
        # Step 1: Entity-aware translations
        self.preprocess_entity_translations()
        
        # Step 2: Ambiguity resolutions
        self.preprocess_ambiguous_translations()
        
        # Step 3: Create contextual rules
        self.create_contextual_rules()
        
        # Summary
        stats = translation_memory.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total cache entries: {stats.total_entries}")
        logger.info(f"Memory usage: {stats.memory_usage_mb:.2f} MB")
        logger.info("\nYou can now start the server with pre-populated translations!")
        
        # Save cache to file for persistence
        self._save_cache_to_file()
    
    def _save_cache_to_file(self):
        """Save cache entries to file for persistence across restarts."""
        cache_path = settings.DATA_DIR / "preprocessed_cache.json"
        
        entries = []
        for query, entry in translation_memory._cache.items():
            entries.append({
                "query": query,
                "translation": entry.translation,
                "source": entry.source,
                "confidence": entry.confidence
            })
        
        with open(cache_path, 'w') as f:
            json.dump(entries, f, indent=2)
        
        logger.info(f"Cache saved to: {cache_path}")


def main():
    """Run offline preprocessing."""
    preprocessor = OfflinePreprocessor()
    preprocessor.run_all()


if __name__ == "__main__":
    main()
