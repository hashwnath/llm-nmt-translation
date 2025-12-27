"""
Tier 3: Neural Machine Translation (NMT) + Contextual Rules

Fallback translation using MarianMT with optimization.
- 8-bit quantization for efficiency
- Beam search decoding
- Contextual rule engine for post-processing
- Batch translation support
- Target: 150ms latency, 3x throughput
"""

import time
import logging
import re
from typing import Optional
import torch
from transformers import MarianMTModel, MarianTokenizer

from ..config import settings
from ..models import Entity

logger = logging.getLogger(__name__)


class NMTTranslator:
    """
    Neural Machine Translation using MarianMT.
    
    Implements Tier 3 with:
    - MarianMT (Helsinki-NLP/opus-mt-fr-en)
    - Optional 8-bit quantization
    - Beam search decoding
    - Contextual rule engine for refinement
    - Batch translation support
    """
    
    def __init__(self):
        self._model: Optional[MarianMTModel] = None
        self._tokenizer: Optional[MarianTokenizer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_quantized = False
        self._model_loaded = False
        self._load_errors = 0
        
    def _load_model(self):
        """Lazy load the NMT model."""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"Loading NMT model: {settings.NMT_MODEL_NAME}")
            start_time = time.perf_counter()
            
            # Load tokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(
                settings.NMT_MODEL_NAME
            )
            
            # Load model
            self._model = MarianMTModel.from_pretrained(
                settings.NMT_MODEL_NAME
            )
            
            # Apply quantization if enabled and on CPU
            if settings.NMT_USE_QUANTIZATION and self._device == "cpu":
                self._apply_quantization()
            
            # Move to device
            self._model.to(self._device)
            self._model.eval()
            
            self._model_loaded = True
            load_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"NMT model loaded in {load_time:.0f}ms "
                       f"(device: {self._device}, quantized: {self._is_quantized})")
            
        except Exception as e:
            logger.error(f"Failed to load NMT model: {e}")
            self._load_errors += 1
            raise
    
    def _apply_quantization(self):
        """Apply 8-bit dynamic quantization to the model."""
        try:
            logger.info("Applying 8-bit quantization...")
            
            self._model = torch.quantization.quantize_dynamic(
                self._model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            self._is_quantized = True
            logger.info("Quantization applied successfully")
            
        except Exception as e:
            logger.warning(f"Quantization failed, using FP32: {e}")
            self._is_quantized = False
    
    def translate(
        self, 
        query: str, 
        entities: Optional[list[Entity]] = None,
        context_hints: Optional[dict] = None
    ) -> tuple[str, float]:
        """
        Translate French query to English.
        
        Args:
            query: French query text
            entities: Optional list of entities to preserve
            context_hints: Optional context for rule engine
            
        Returns:
            Tuple of (translation, latency_ms)
        """
        self._load_model()
        start_time = time.perf_counter()
        
        try:
            # Prepare input (mark entities if present)
            prepared_query = self._prepare_input(query, entities)
            
            # Tokenize
            inputs = self._tokenizer(
                prepared_query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.NMT_MAX_LENGTH
            ).to(self._device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    num_beams=settings.NMT_BEAM_SIZE,
                    max_length=settings.NMT_MAX_LENGTH,
                    early_stopping=True,
                    length_penalty=1.0
                )
            
            # Decode
            translation = self._tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Apply contextual rules
            translation = self._apply_rules(translation, query, entities, context_hints)
            
            latency = (time.perf_counter() - start_time) * 1000
            logger.debug(f"NMT translation: '{query}' -> '{translation}' in {latency:.1f}ms")
            
            return translation, latency
            
        except Exception as e:
            logger.error(f"NMT translation failed: {e}")
            latency = (time.perf_counter() - start_time) * 1000
            # Return original query as fallback
            return query, latency
    
    def translate_batch(
        self, 
        queries: list[str], 
        batch_size: int = 8
    ) -> tuple[list[str], float]:
        """
        Batch translation for higher throughput.
        
        Args:
            queries: List of French queries
            batch_size: Number of queries per batch
            
        Returns:
            Tuple of (list of translations, total_latency_ms)
        """
        self._load_model()
        start_time = time.perf_counter()
        
        translations = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=settings.NMT_MAX_LENGTH
                ).to(self._device)
                
                # Generate translations
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        num_beams=settings.NMT_BEAM_SIZE,
                        max_length=settings.NMT_MAX_LENGTH,
                        early_stopping=True
                    )
                
                # Decode batch
                batch_translations = self._tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                
                # Apply rules to each translation
                for j, translation in enumerate(batch_translations):
                    refined = self._apply_rules(translation, batch[j], None, None)
                    translations.append(refined)
                    
            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # Fallback: return original queries for failed batch
                translations.extend(batch)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        return translations, total_latency
    
    def _prepare_input(
        self, 
        query: str, 
        entities: Optional[list[Entity]]
    ) -> str:
        """Prepare input query for translation."""
        if not entities:
            return query
        
        # For now, just return the query
        # Entity preservation is handled in post-processing
        return query
    
    def _apply_rules(
        self, 
        translation: str, 
        original_query: str,
        entities: Optional[list[Entity]],
        context_hints: Optional[dict]
    ) -> str:
        """
        Apply contextual rules to refine translation.
        
        Rule categories:
        1. Entity preservation
        2. Grammar refinement
        3. Semantic coherence
        """
        refined = translation
        
        # Rule 1: Entity preservation
        refined = self._apply_entity_preservation(refined, original_query, entities)
        
        # Rule 2: Grammar fixes
        refined = self._apply_grammar_rules(refined)
        
        # Rule 3: Semantic coherence
        refined = self._apply_semantic_rules(refined, original_query)
        
        return refined.strip()
    
    def _apply_entity_preservation(
        self, 
        translation: str, 
        original: str,
        entities: Optional[list[Entity]]
    ) -> str:
        """Restore entities that may have been incorrectly translated."""
        if not entities:
            return translation
        
        result = translation
        
        for entity in entities:
            if not entity.should_preserve:
                continue
            
            original_text = entity.text
            
            # Known mistranslations to fix
            mistranslation_map = {
                "liberté": ["liberty", "freedom"],
                "royale": ["royal", "royalty"],
                "trésor": ["treasure"],
                "tresor": ["treasure"],
            }
            
            entity_lower = original_text.lower()
            if entity_lower in mistranslation_map:
                for wrong in mistranslation_map[entity_lower]:
                    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                    result = pattern.sub(original_text, result)
        
        return result
    
    def _apply_grammar_rules(self, translation: str) -> str:
        """Apply grammar refinement rules."""
        result = translation
        
        # Fix common grammar issues
        grammar_fixes = [
            # "want buy" -> "want to buy"
            (r'\bwant\s+buy\b', 'want to buy'),
            # "wish buy" -> "wish to buy"
            (r'\bwish\s+buy\b', 'wish to buy'),
            # "like buy" -> "like to buy"
            (r'\blike\s+buy\b', 'like to buy'),
            # Double spaces
            (r'\s+', ' '),
        ]
        
        for pattern, replacement in grammar_fixes:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _apply_semantic_rules(self, translation: str, original: str) -> str:
        """Apply semantic coherence rules."""
        # Currently a placeholder for more complex semantic rules
        # Could include:
        # - Checking if translated terms exist in product catalog
        # - Ensuring query makes sense in e-commerce context
        return translation
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": settings.NMT_MODEL_NAME,
            "loaded": self._model_loaded,
            "device": self._device,
            "quantized": self._is_quantized,
            "beam_size": settings.NMT_BEAM_SIZE,
            "max_length": settings.NMT_MAX_LENGTH,
            "load_errors": self._load_errors
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._model_loaded = False
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("NMT model unloaded")


# Global NMT translator instance
nmt_translator = NMTTranslator()
