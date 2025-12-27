"""
Tier 1: Translation Memory (Cache Layer)

Fast, deterministic lookups for frequent queries.
- Exact match lookup with MD5 hashing
- Fuzzy matching with Levenshtein distance
- TTL management and cache population
- Target: 80-90% hit rate, <1ms latency
"""

import hashlib
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from rapidfuzz import fuzz
import logging

from ..config import settings
from ..models import CacheEntry, CacheStats

logger = logging.getLogger(__name__)


class TranslationMemory:
    """
    Translation Memory cache for fast lookups.
    
    Implements the Tier 1 caching layer with:
    - O(1) exact match lookup via hash
    - O(n) fuzzy matching when needed
    - TTL-based expiration
    - Multi-source population (LLM, NMT, user-corrected)
    """
    
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._hash_index: dict[str, str] = {}  # hash -> original query
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._fuzzy_hits: int = 0
        
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent matching."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        # Lowercase for matching
        normalized = normalized.lower()
        # Remove special punctuation (keep accents)
        normalized = re.sub(r'[^\w\s\u00C0-\u017F]', '', normalized)
        return normalized
    
    def _compute_hash(self, query: str) -> str:
        """Compute MD5 hash for query."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def lookup(self, query: str) -> Tuple[Optional[str], float, str]:
        """
        Look up translation in cache.
        
        Returns:
            Tuple of (translation, confidence, match_type)
            - translation: The cached translation or None
            - confidence: Confidence score (0-1)
            - match_type: "exact", "fuzzy", or "miss"
        """
        start_time = time.perf_counter()
        
        # Step 1: Exact lookup
        query_hash = self._compute_hash(query)
        
        if query_hash in self._hash_index:
            original_query = self._hash_index[query_hash]
            entry = self._cache.get(original_query)
            
            if entry:
                # Check TTL
                if self._is_valid(entry):
                    self._hit_count += 1
                    entry.frequency += 1
                    latency = (time.perf_counter() - start_time) * 1000
                    logger.debug(f"Cache exact hit for '{query}' in {latency:.2f}ms")
                    return entry.translation, entry.confidence, "exact"
                else:
                    # Expired, remove from cache
                    self._remove_entry(original_query)
        
        # Step 2: Fuzzy matching (if enabled and cache not too large)
        if len(self._cache) <= 10000:  # Only fuzzy search for smaller caches
            result = self._fuzzy_lookup(query)
            if result:
                self._fuzzy_hits += 1
                self._hit_count += 1
                return result
        
        # Step 3: Cache miss
        self._miss_count += 1
        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Cache miss for '{query}' in {latency:.2f}ms")
        return None, 0.0, "miss"
    
    def _fuzzy_lookup(self, query: str) -> Optional[Tuple[str, float, str]]:
        """Perform fuzzy matching using Levenshtein distance."""
        normalized_query = self._normalize_query(query)
        best_match = None
        best_ratio = 0.0
        
        for cached_query, entry in self._cache.items():
            if not self._is_valid(entry):
                continue
                
            normalized_cached = self._normalize_query(cached_query)
            ratio = fuzz.ratio(normalized_query, normalized_cached) / 100.0
            
            if ratio >= settings.CACHE_FUZZY_THRESHOLD and ratio > best_ratio:
                best_match = entry
                best_ratio = ratio
        
        if best_match:
            # Adjust confidence based on fuzzy match quality
            adjusted_confidence = best_match.confidence * best_ratio
            logger.debug(f"Cache fuzzy hit for '{query}' with ratio {best_ratio:.2f}")
            return best_match.translation, adjusted_confidence, "fuzzy"
        
        return None
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)."""
        expiry_time = entry.timestamp + timedelta(seconds=entry.ttl_seconds)
        return datetime.utcnow() < expiry_time
    
    def _remove_entry(self, query: str):
        """Remove entry from cache and index."""
        if query in self._cache:
            query_hash = self._compute_hash(query)
            del self._cache[query]
            if query_hash in self._hash_index:
                del self._hash_index[query_hash]
    
    def store(
        self,
        query: str,
        translation: str,
        source: str = "nmt_cached",
        confidence: float = 0.8,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store translation in cache.
        
        Args:
            query: Original French query
            translation: English translation
            source: Source of translation (llm_generated, nmt_cached, user_corrected)
            confidence: Confidence score (0-1)
            ttl_seconds: Time to live in seconds (uses default if None)
            
        Returns:
            True if stored successfully
        """
        # Check cache size limit
        if len(self._cache) >= settings.CACHE_MAX_SIZE:
            self._evict_oldest()
        
        entry = CacheEntry(
            query=query,
            translation=translation,
            source=source,
            confidence=confidence,
            ttl_seconds=ttl_seconds or settings.CACHE_TTL_SECONDS,
            timestamp=datetime.utcnow()
        )
        
        self._cache[query] = entry
        query_hash = self._compute_hash(query)
        self._hash_index[query_hash] = query
        
        logger.debug(f"Cached translation: '{query}' -> '{translation}' (source: {source})")
        return True
    
    def _evict_oldest(self):
        """Evict oldest/least frequently used entries."""
        if not self._cache:
            return
        
        # Sort by frequency (ascending) and timestamp (ascending)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].frequency, x[1].timestamp)
        )
        
        # Remove bottom 10%
        num_to_remove = max(1, len(sorted_entries) // 10)
        for query, _ in sorted_entries[:num_to_remove]:
            self._remove_entry(query)
            
        logger.info(f"Evicted {num_to_remove} cache entries")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
        
        # Estimate memory usage (rough approximation)
        import sys
        memory_bytes = sum(
            sys.getsizeof(k) + sys.getsizeof(v.query) + sys.getsizeof(v.translation)
            for k, v in self._cache.items()
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        return CacheStats(
            total_entries=len(self._cache),
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            hit_rate=hit_rate,
            fuzzy_hits=self._fuzzy_hits,
            memory_usage_mb=memory_mb
        )
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._hash_index.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._fuzzy_hits = 0
        logger.info("Cache cleared")
    
    def preload(self, entries: list[dict]):
        """Preload cache with known translations."""
        for entry_data in entries:
            self.store(
                query=entry_data["query"],
                translation=entry_data["translation"],
                source=entry_data.get("source", "nmt_cached"),
                confidence=entry_data.get("confidence", 0.9)
            )
        logger.info(f"Preloaded {len(entries)} cache entries")


# Global cache instance
translation_memory = TranslationMemory()
