"""
Tier 2A: Entity-Aware Translator

Identifies and preserves non-translatable entities (brands, product names).
- Entity extraction with regex patterns
- Confidence scoring
- Entity preservation using special tokens
- Target: +28% BLEU score improvement on entity-heavy queries
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..config import settings
from ..models import Entity

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Represents a matched entity in the query."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    canonical_name: str  # The correct casing/spelling


class EntityExtractor:
    """
    Entity-Aware component for preserving brands and product names.
    
    Implements the Tier 2A entity extraction with:
    - Brand name recognition from catalog
    - Product line detection
    - Franchise/symbol identification
    - Confidence scoring based on match quality
    """
    
    def __init__(self):
        self._brands: dict[str, dict] = {}
        self._product_lines: dict[str, dict] = {}
        self._franchises: dict[str, dict] = {}
        self._alias_to_brand: dict[str, str] = {}
        self._load_catalog()
    
    def _load_catalog(self):
        """Load brand and product catalog from JSON files."""
        catalog_path = settings.DATA_DIR / "brand_catalog.json"
        
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            
            # Load brands
            for brand in catalog.get("brands", []):
                name = brand["name"]
                self._brands[name.lower()] = brand
                
                # Build alias index
                for alias in brand.get("aliases", []):
                    self._alias_to_brand[alias.lower()] = name
            
            # Load product lines
            for product in catalog.get("product_lines", []):
                name = product["name"]
                self._product_lines[name.lower()] = product
            
            # Load franchises
            for franchise in catalog.get("franchises", []):
                name = franchise["name"]
                self._franchises[name.lower()] = franchise
            
            logger.info(f"Loaded {len(self._brands)} brands, "
                       f"{len(self._product_lines)} product lines, "
                       f"{len(self._franchises)} franchises")
                       
        except FileNotFoundError:
            logger.warning(f"Brand catalog not found at {catalog_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing brand catalog: {e}")
    
    def extract_entities(self, query: str) -> list[Entity]:
        """
        Extract entities from query.
        
        Args:
            query: French query text
            
        Returns:
            List of Entity objects with positions and confidence
        """
        entities = []
        query_lower = query.lower()
        
        # Step 1: Check for exact brand matches (highest priority)
        brand_entities = self._extract_brands(query, query_lower)
        entities.extend(brand_entities)
        
        # Step 2: Check for product line matches
        product_entities = self._extract_product_lines(query, query_lower)
        # Filter out overlapping entities (keep brand matches)
        for pe in product_entities:
            if not self._overlaps_with_existing(pe, entities):
                entities.append(pe)
        
        # Step 3: Check for franchise/symbol matches
        franchise_entities = self._extract_franchises(query, query_lower)
        for fe in franchise_entities:
            if not self._overlaps_with_existing(fe, entities):
                entities.append(fe)
        
        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        
        logger.debug(f"Extracted {len(entities)} entities from '{query}'")
        return entities
    
    def _extract_brands(self, query: str, query_lower: str) -> list[Entity]:
        """Extract brand name entities."""
        entities = []
        
        # Check aliases first (more comprehensive)
        for alias, canonical_name in self._alias_to_brand.items():
            # Use word boundary matching
            pattern = r'\b' + re.escape(alias) + r'\b'
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                brand_info = self._brands.get(canonical_name.lower(), {})
                
                # Calculate confidence
                confidence = self._calculate_brand_confidence(
                    alias, 
                    canonical_name, 
                    brand_info
                )
                
                entity = Entity(
                    text=query[match.start():match.end()],
                    entity_type="brand",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    should_preserve=confidence >= 0.80
                )
                entities.append(entity)
        
        return entities
    
    def _extract_product_lines(self, query: str, query_lower: str) -> list[Entity]:
        """Extract product line entities."""
        entities = []
        
        for product_name, product_info in self._product_lines.items():
            pattern = r'\b' + re.escape(product_name) + r'\b'
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                entity = Entity(
                    text=query[match.start():match.end()],
                    entity_type="product_line",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.90,
                    should_preserve=True
                )
                entities.append(entity)
        
        return entities
    
    def _extract_franchises(self, query: str, query_lower: str) -> list[Entity]:
        """Extract franchise/symbol entities."""
        entities = []
        
        for franchise_name, franchise_info in self._franchises.items():
            pattern = r'\b' + re.escape(franchise_name) + r'\b'
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                entity = Entity(
                    text=query[match.start():match.end()],
                    entity_type="franchise",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.85,
                    should_preserve=True
                )
                entities.append(entity)
        
        return entities
    
    def _calculate_brand_confidence(
        self, 
        matched_alias: str, 
        canonical_name: str, 
        brand_info: dict
    ) -> float:
        """
        Calculate confidence score for brand match.
        
        Confidence factors:
        - Exact canonical match: 0.95
        - Exact alias match: 0.90
        - Fuzzy match: 0.75-0.85
        """
        base_confidence = 0.95
        
        # Check if exact canonical match
        if matched_alias.lower() == canonical_name.lower():
            return base_confidence
        
        # Check if alias match
        aliases = brand_info.get("aliases", [])
        if matched_alias.lower() in [a.lower() for a in aliases]:
            return 0.90
        
        # Otherwise it's a fuzzy match
        return 0.80
    
    def _overlaps_with_existing(self, entity: Entity, existing: list[Entity]) -> bool:
        """Check if entity overlaps with any existing entity."""
        for e in existing:
            if (entity.start_pos < e.end_pos and entity.end_pos > e.start_pos):
                return True
        return False
    
    def mark_entities(self, query: str, entities: list[Entity]) -> str:
        """
        Mark entities in query with special tokens for preservation.
        
        Example:
            Input: "yogurt liberté"
            Output: "yogurt <BRAND>liberté</BRAND>"
        """
        if not entities:
            return query
        
        # Sort by position (reverse order for safe replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start_pos, reverse=True)
        
        marked_query = query
        for entity in sorted_entities:
            if entity.should_preserve:
                tag = entity.entity_type.upper()
                marked_text = f"<{tag}>{entity.text}</{tag}>"
                marked_query = (
                    marked_query[:entity.start_pos] + 
                    marked_text + 
                    marked_query[entity.end_pos:]
                )
        
        return marked_query
    
    def restore_entities(
        self, 
        translation: str, 
        original_query: str, 
        entities: list[Entity]
    ) -> str:
        """
        Restore preserved entities in translation.
        
        This handles cases where NMT incorrectly translated entity names.
        
        Example:
            Translation: "i want yogurt liberty"
            Original entities: [Entity(text="liberté", type="brand")]
            Output: "i want yogurt liberté"
        """
        if not entities:
            return translation
        
        restored = translation
        
        for entity in entities:
            if not entity.should_preserve:
                continue
            
            original_text = entity.text
            
            # Common mistranslations to fix
            mistranslations = self._get_mistranslations(original_text)
            
            for wrong_text in mistranslations:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(wrong_text), re.IGNORECASE)
                restored = pattern.sub(original_text, restored)
        
        # Also remove any remaining entity markers
        restored = re.sub(r'</?(?:BRAND|PRODUCT_LINE|FRANCHISE)>', '', restored)
        
        return restored
    
    def _get_mistranslations(self, entity_text: str) -> list[str]:
        """Get common mistranslations for an entity."""
        mistranslations = []
        
        # Known French -> English mistranslations for brands
        translation_map = {
            "liberté": ["liberty", "freedom"],
            "royale": ["royal", "royalty"],
            "trésor": ["treasure"],
            "tresor": ["treasure"],
        }
        
        entity_lower = entity_text.lower()
        if entity_lower in translation_map:
            mistranslations.extend(translation_map[entity_lower])
        
        return mistranslations
    
    def get_canonical_name(self, text: str) -> Optional[str]:
        """Get the canonical (correct) spelling of a brand name."""
        text_lower = text.lower()
        
        # Check brands directly
        if text_lower in self._brands:
            return self._brands[text_lower]["name"]
        
        # Check aliases
        if text_lower in self._alias_to_brand:
            return self._alias_to_brand[text_lower]
        
        return None


# Global entity extractor instance
entity_extractor = EntityExtractor()
