"""
Translation Pipeline Orchestrator

Main orchestration layer that coordinates all translation tiers.
Aligned with paper architecture (Figure 1):
- Language Detection ("Is English?" check)
- Translation Memory Lookup (Tier 1)
- Entity-Aware Translation (Tier 2A) 
- Ambiguity Resolution (Tier 2B)
- Neural Machine Translation (Tier 3)
- Contextual Rules post-processing

Offline components (Entity-Aware Translator, Ambiguity Resolver, Contextual Rule Creator)
pre-populate the Translation Memory via offline_preprocess.py
"""

import time
import logging
import re
import json
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models import (
    TranslationRequest, 
    TranslationResponse, 
    BatchTranslationRequest,
    BatchTranslationResponse,
    SessionEvent
)
from .tier1_cache import translation_memory
from .tier2_entity import entity_extractor
from .tier2_ambiguity import ambiguity_resolver
from .tier3_nmt import nmt_translator
from .metrics import metrics_collector

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    Simple language detection for French vs English.
    Checks for common French patterns and characters.
    
    Returns: 'fr' for French, 'en' for English
    """
    text_lower = text.lower()
    
    # French-specific patterns
    french_patterns = [
        r'\bje\b', r'\bveux\b', r'\bacheter\b', r'\bpour\b', r'\bdu\b', 
        r'\bdes\b', r'\ble\b', r'\bla\b', r'\bles\b', r'\bun\b', r'\bune\b',
        r'\bde\b', r'\bet\b', r'\best\b', r'\bsont\b', r'\bavec\b',
        r'\bfrais\b', r'\bfraîche\b', r'\bnouveau\b', r'\bnouvelle\b',
        r'[éèêëàâäùûüôöîïç]',  # French accents
    ]
    
    # English-specific patterns
    english_patterns = [
        r'\bthe\b', r'\band\b', r'\bwith\b', r'\bfor\b', r'\bwant\b',
        r'\bbuy\b', r'\bto\b', r'\bis\b', r'\bare\b', r'\bthis\b',
        r'\bthat\b', r'\bnew\b', r'\bfresh\b',
    ]
    
    french_score = sum(1 for p in french_patterns if re.search(p, text_lower))
    english_score = sum(1 for p in english_patterns if re.search(p, text_lower))
    
    # If text has French accents, strongly prefer French
    if re.search(r'[éèêëàâäùûüôöîïç]', text_lower):
        french_score += 3
    
    return 'fr' if french_score > english_score else 'en'


class TranslationPipeline:
    """
    Orchestrates the 3-tier translation pipeline.
    
    Flow:
    1. Check Translation Memory (Tier 1)
    2. If miss, extract entities (Tier 2A)
    3. Resolve ambiguities with context (Tier 2B)
    4. Translate with NMT (Tier 3)
    5. Apply entity preservation
    6. Cache result for future use
    """
    
    def __init__(self):
        self._initialized = False
    
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Main translation entry point.
        
        Args:
            request: Translation request with query and optional session
            
        Returns:
            TranslationResponse with translation and metadata
        """
        metrics_collector.start_request()
        start_time = time.perf_counter()
        
        query = request.query.strip()
        session_id = request.session_id
        
        # Initialize response fields
        tier_used = "cache"
        entities_preserved = []
        ambiguity_info = None
        confidence = 0.0
        translation = query  # Default fallback
        fallback = False
        error = None
        detected_language = None
        
        try:
            # ============================================================
            # LANGUAGE DETECTION: "Is English?" check (per paper Figure 1)
            # ============================================================
            detected_language = detect_language(query)
            
            if detected_language == 'en':
                # Query is already in English, no translation needed
                logger.debug(f"Query detected as English, returning as-is: '{query}'")
                total_latency = (time.perf_counter() - start_time) * 1000
                metrics_collector.end_request("passthrough")
                
                return TranslationResponse(
                    original=query,
                    translation=query,
                    tier_used="passthrough",
                    entities_preserved=[],
                    ambiguity_resolved=None,
                    confidence=1.0,
                    latency_ms=total_latency,
                    fallback=False,
                    error=None
                )
            
            # ============================================================
            # TIER 1: Translation Memory (Cache)
            tier_start = time.perf_counter()
            cached_translation, cache_confidence, match_type = translation_memory.lookup(query)
            tier_latency = (time.perf_counter() - tier_start) * 1000
            metrics_collector.record_tier_latency("cache", tier_latency)
            
            if cached_translation and match_type != "miss":
                metrics_collector.record_cache_hit()
                translation = cached_translation
                confidence = cache_confidence
                tier_used = "cache"
                
                logger.debug(f"Cache hit ({match_type}): '{query}' -> '{translation}'")
                
            else:
                metrics_collector.record_cache_miss()
                
                # ============================================================
                # TIER 2A: Entity Extraction
                # ============================================================
                tier_start = time.perf_counter()
                entities = entity_extractor.extract_entities(query)
                tier_latency = (time.perf_counter() - tier_start) * 1000
                metrics_collector.record_tier_latency("entity", tier_latency)
                
                entities_preserved = [e.text for e in entities if e.should_preserve]
                if entities_preserved:
                    metrics_collector.record_entity_preservation(len(entities_preserved))
                    tier_used = "entity"
                
                # ============================================================
                # TIER 2B: Ambiguity Resolution
                # ============================================================
                tier_start = time.perf_counter()
                resolutions, ambiguity_metadata = ambiguity_resolver.resolve(
                    query, 
                    session_id
                )
                tier_latency = (time.perf_counter() - tier_start) * 1000
                metrics_collector.record_tier_latency("ambiguity", tier_latency)
                
                if resolutions:
                    metrics_collector.record_ambiguity_resolution(len(resolutions))
                    ambiguity_info = {
                        "resolved_words": resolutions,
                        "method": ambiguity_metadata.get("resolution_method"),
                        "confidence": ambiguity_metadata.get("confidence_scores", {})
                    }
                    tier_used = "ambiguity"
                
                # ============================================================
                # TIER 3: NMT Translation
                # ============================================================
                tier_start = time.perf_counter()
                try:
                    translation, nmt_latency = nmt_translator.translate(
                        query,
                        entities=entities,
                        context_hints=resolutions
                    )
                    tier_latency = (time.perf_counter() - tier_start) * 1000
                    metrics_collector.record_tier_latency("nmt", tier_latency)
                    
                    # Apply entity restoration
                    translation = entity_extractor.restore_entities(
                        translation, 
                        query, 
                        entities
                    )
                    
                    # Apply ambiguity resolutions if needed
                    if resolutions:
                        translation = self._apply_ambiguity_resolutions(
                            translation, 
                            resolutions
                        )
                    
                    confidence = 0.85
                    tier_used = "nmt"
                    
                except Exception as e:
                    logger.error(f"NMT translation failed: {e}")
                    metrics_collector.record_nmt_failure()
                    translation = query  # Fallback to original
                    confidence = 0.0
                    fallback = True
                    error = "nmt_failure"
                
                # ============================================================
                # Cache the result for future queries
                # ============================================================
                if not fallback and translation != query:
                    translation_memory.store(
                        query=query,
                        translation=translation,
                        source="nmt_cached",
                        confidence=confidence
                    )
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            translation = query
            confidence = 0.0
            fallback = True
            error = str(e)
        
        # Calculate total latency
        total_latency = (time.perf_counter() - start_time) * 1000
        metrics_collector.end_request(tier_used)
        
        return TranslationResponse(
            original=query,
            translation=translation,
            tier_used=tier_used,
            entities_preserved=entities_preserved,
            ambiguity_resolved=ambiguity_info,
            confidence=confidence,
            latency_ms=total_latency,
            fallback=fallback,
            error=error
        )
    
    def translate_batch(self, request: BatchTranslationRequest) -> BatchTranslationResponse:
        """
        Batch translation for multiple queries.
        
        Args:
            request: Batch request with list of queries
            
        Returns:
            BatchTranslationResponse with all translations
        """
        start_time = time.perf_counter()
        
        translations = []
        for query in request.queries:
            single_request = TranslationRequest(
                query=query,
                session_id=request.session_id,
                include_metrics=False
            )
            response = self.translate(single_request)
            translations.append(response)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        avg_latency = total_latency / len(translations) if translations else 0
        
        return BatchTranslationResponse(
            translations=translations,
            total_latency_ms=total_latency,
            average_latency_ms=avg_latency
        )
    
    def _apply_ambiguity_resolutions(
        self, 
        translation: str, 
        resolutions: dict[str, str]
    ) -> str:
        """
        Apply ambiguity resolutions to the translation.
        
        This fixes NMT mistakes by replacing incorrect translations
        with the context-resolved meanings.
        """
        result = translation
        
        # Known NMT mistranslations for ambiguous words
        # Maps French word -> possible wrong English translations
        mistranslation_map = {
            "gomme": ["range", "gum", "eraser", "rubber"],  # NMT often gets this wrong
            "pêche": ["fishing", "peach", "sin"],
            "peche": ["fishing", "peach", "sin"],
            "carte": ["card", "map", "menu"],
            "glace": ["ice", "mirror", "ice cream"],
            "mousse": ["foam", "mousse", "moss"],
            "livre": ["book", "pound", "deliver"],
            "couronne": ["crown", "wreath", "corona"],
            "trésor": ["treasure", "treasury"],
            "tresor": ["treasure", "treasury"],
        }
        
        for french_word, english_meaning in resolutions.items():
            french_lower = french_word.lower()
            
            if french_lower in mistranslation_map:
                wrong_translations = mistranslation_map[french_lower]
                
                for wrong in wrong_translations:
                    # Skip if the wrong translation IS the correct meaning
                    if wrong.lower() == english_meaning.lower():
                        continue
                    
                    # Replace wrong translation with correct one
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                    if pattern.search(result):
                        result = pattern.sub(english_meaning, result)
                        logger.debug(f"Fixed '{wrong}' → '{english_meaning}' in translation")
        
        return result
    
    def add_session_event(self, session_id: str, event: SessionEvent):
        """Add a session event for context tracking."""
        ambiguity_resolver.session_manager.add_event(session_id, event)
    
    def get_session_context(self, session_id: str) -> Optional[dict]:
        """Get session context for debugging."""
        session = ambiguity_resolver.session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "categories_viewed": session.categories_viewed,
                "recent_brands": session.recent_brands,
                "recent_searches": session.recent_searches,
                "top_category": session.top_category,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
        return None
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = translation_memory.get_stats()
        return {
            "total_entries": stats.total_entries,
            "hit_count": stats.hit_count,
            "miss_count": stats.miss_count,
            "hit_rate": f"{stats.hit_rate:.1%}",
            "fuzzy_hits": stats.fuzzy_hits,
            "memory_usage_mb": f"{stats.memory_usage_mb:.2f}"
        }
    
    def preload_cache(self, entries: list[dict]):
        """Preload cache with known translations."""
        translation_memory.preload(entries)
    
    def warmup(self):
        """Warm up the pipeline by loading models."""
        logger.info("Warming up translation pipeline...")
        
        # Warm up NMT model
        try:
            nmt_translator.translate("test", None, None)
            logger.info("NMT model warmed up")
        except Exception as e:
            logger.warning(f"NMT warmup failed: {e}")
        
        self._initialized = True
        logger.info("Pipeline warmup complete")


# Global pipeline instance
translation_pipeline = TranslationPipeline()
