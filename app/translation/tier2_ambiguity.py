"""
Tier 2B: Ambiguity Resolver

Resolves words with multiple meanings using session context and LLM.
- Session context tracking
- Context-weighted scoring
- LLM fallback for complex cases
- Examples: "gomme" → eraser/gum, "peche" → peach/fishing
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, Tuple
from openai import OpenAI, APIError, RateLimitError, Timeout

from ..config import settings
from ..models import SessionContext, SessionEvent

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user session context for ambiguity resolution."""
    
    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}
    
    def get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id)
        return self._sessions[session_id]
    
    def add_event(self, session_id: str, event: SessionEvent):
        """Add event to session and update context."""
        session = self.get_or_create_session(session_id)
        
        # Update categories viewed
        if event.category:
            current_count = session.categories_viewed.get(event.category, 0)
            session.categories_viewed[event.category] = current_count + 1
            
            # Update top category
            session.top_category = max(
                session.categories_viewed.items(),
                key=lambda x: x[1]
            )[0]
        
        # Update recent brands
        if event.brand:
            if event.brand not in session.recent_brands:
                session.recent_brands.insert(0, event.brand)
                # Keep only last 10 brands
                session.recent_brands = session.recent_brands[:10]
        
        # Update recent searches
        if event.query_text:
            session.recent_searches.insert(0, event.query_text)
            session.recent_searches = session.recent_searches[:10]
        
        session.updated_at = datetime.utcnow()
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self._sessions:
            del self._sessions[session_id]


class AmbiguityResolver:
    """
    Resolves ambiguous French words using context.
    
    Implements Tier 2B with:
    - Ambiguity dictionary lookup
    - Session context scoring
    - LLM fallback for complex cases
    """
    
    def __init__(self):
        self._ambiguity_dict: dict = {}
        self._session_manager = SessionManager()
        self._openai_client: Optional[OpenAI] = None
        self._api_failures = 0
        self._load_ambiguity_dict()
    
    def _load_ambiguity_dict(self):
        """Load ambiguity dictionary from JSON file."""
        dict_path = settings.DATA_DIR / "ambiguity_dict.json"
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._ambiguity_dict = data.get("ambiguous_words", {})
            
            logger.info(f"Loaded {len(self._ambiguity_dict)} ambiguous words")
            
        except FileNotFoundError:
            logger.warning(f"Ambiguity dictionary not found at {dict_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing ambiguity dictionary: {e}")
    
    def _get_openai_client(self) -> Optional[OpenAI]:
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None and settings.OPENAI_API_KEY:
            self._openai_client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_TIMEOUT
            )
        return self._openai_client
    
    @property
    def session_manager(self) -> SessionManager:
        """Get session manager."""
        return self._session_manager
    
    @property
    def api_failures(self) -> int:
        """Get count of API failures."""
        return self._api_failures
    
    def detect_ambiguous_words(self, query: str) -> list[str]:
        """Detect which words in query are ambiguous."""
        words = query.lower().split()
        ambiguous = []
        
        for word in words:
            # Normalize accents for matching
            normalized = self._normalize_accents(word)
            if normalized in self._ambiguity_dict or word in self._ambiguity_dict:
                ambiguous.append(word)
        
        return ambiguous
    
    def _normalize_accents(self, text: str) -> str:
        """Normalize French accents for matching."""
        replacements = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ô': 'o', 'ö': 'o',
            'î': 'i', 'ï': 'i',
            'ç': 'c'
        }
        for accented, plain in replacements.items():
            text = text.replace(accented, plain)
        return text
    
    def resolve(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> Tuple[dict[str, str], dict]:
        """
        Resolve ambiguous words in query.
        
        Args:
            query: French query
            session_id: Optional session ID for context
            
        Returns:
            Tuple of:
            - dict mapping ambiguous words to resolved translations
            - metadata dict with resolution details
        """
        start_time = time.perf_counter()
        resolutions = {}
        metadata = {
            "ambiguous_words": [],
            "resolution_method": None,
            "confidence_scores": {},
            "latency_ms": 0
        }
        
        # Detect ambiguous words
        ambiguous_words = self.detect_ambiguous_words(query)
        metadata["ambiguous_words"] = ambiguous_words
        
        if not ambiguous_words:
            return resolutions, metadata
        
        # Get session context if available
        session_context = None
        if session_id:
            session_context = self._session_manager.get_session(session_id)
        
        # Resolve each ambiguous word
        for word in ambiguous_words:
            resolution, confidence, method = self._resolve_word(
                word, 
                query, 
                session_context
            )
            resolutions[word] = resolution
            metadata["confidence_scores"][word] = confidence
        
        metadata["resolution_method"] = method
        metadata["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return resolutions, metadata
    
    def _resolve_word(
        self, 
        word: str, 
        query: str,
        session_context: Optional[SessionContext]
    ) -> Tuple[str, float, str]:
        """
        Resolve a single ambiguous word.
        
        Returns:
            Tuple of (translation, confidence, method)
        """
        # Get word entry from dictionary
        normalized = self._normalize_accents(word.lower())
        word_entry = self._ambiguity_dict.get(
            word.lower(), 
            self._ambiguity_dict.get(normalized, {})
        )
        
        if not word_entry:
            return word, 0.5, "unknown"
        
        meanings = word_entry.get("meanings", [])
        if not meanings:
            return word_entry.get("default", word), 0.5, "default"
        
        # If we have session context, use it for scoring
        if session_context and session_context.top_category:
            scored_meanings = self._score_meanings_with_context(
                meanings, 
                session_context
            )
            
            best = max(scored_meanings, key=lambda x: x["score"])
            if best["score"] >= 0.60:
                return best["translation"], best["score"], "context"
        
        # If no strong context signal, try LLM
        if settings.OPENAI_API_KEY:
            try:
                llm_result = self._resolve_with_llm(word, query, meanings)
                if llm_result:
                    return llm_result, 0.85, "llm"
            except Exception as e:
                logger.warning(f"LLM resolution failed: {e}")
                self._api_failures += 1
        
        # Fallback to highest frequency default
        default_meaning = max(meanings, key=lambda x: x.get("frequency", 0))
        return default_meaning["translation"], default_meaning.get("frequency", 0.5), "frequency"
    
    def _score_meanings_with_context(
        self, 
        meanings: list[dict], 
        context: SessionContext
    ) -> list[dict]:
        """Score meanings based on session context."""
        scored = []
        
        for meaning in meanings:
            category = meaning.get("category", "general")
            base_frequency = meaning.get("frequency", 0.5)
            
            # Calculate context match score
            category_views = context.categories_viewed.get(category, 0)
            total_views = sum(context.categories_viewed.values()) or 1
            category_match = category_views / total_views
            
            # Calculate final score
            # Weights: 40% category match, 25% frequency, 20% recency, 15% brand context
            score = (
                0.40 * category_match +
                0.25 * base_frequency +
                0.20 * (1.0 if category == context.top_category else 0.0) +
                0.15 * self._calculate_brand_context(meaning, context)
            )
            
            scored.append({
                **meaning,
                "score": score
            })
        
        return scored
    
    def _calculate_brand_context(
        self, 
        meaning: dict, 
        context: SessionContext
    ) -> float:
        """Calculate brand context score."""
        if meaning.get("is_brand"):
            # Check if this brand was recently viewed
            brand_name = meaning.get("translation", "")
            if brand_name in context.recent_brands:
                return 1.0
        return 0.0
    
    def _resolve_with_llm(
        self, 
        word: str, 
        query: str, 
        meanings: list[dict]
    ) -> Optional[str]:
        """Use LLM to resolve ambiguity."""
        client = self._get_openai_client()
        if not client:
            return None
        
        meanings_str = "\n".join([
            f"- {m['translation']} (category: {m.get('category', 'general')})"
            for m in meanings
        ])
        
        prompt = f"""Given a French e-commerce search query, determine the most likely meaning of an ambiguous word.

Query: "{query}"
Ambiguous word: "{word}"

Possible translations:
{meanings_str}

Based on the query context, which translation is most appropriate for an e-commerce search?
Respond with ONLY the translation word, nothing else."""

        try:
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert French-English translator specializing in e-commerce product searches."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate result is one of our meanings
            valid_translations = [m["translation"] for m in meanings]
            if result.lower() in [t.lower() for t in valid_translations]:
                return result
            
            return None
            
        except (APIError, RateLimitError, Timeout) as e:
            logger.warning(f"OpenAI API error: {e}")
            self._api_failures += 1
            return None
    
    def apply_resolutions(self, query: str, resolutions: dict[str, str]) -> str:
        """Apply resolved translations to query (for debugging/display)."""
        result = query
        for original, translation in resolutions.items():
            # Mark resolved words
            result = result.replace(original, f"[{original}→{translation}]")
        return result


# Global ambiguity resolver instance
ambiguity_resolver = AmbiguityResolver()
