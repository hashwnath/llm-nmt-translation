"""
Metrics and Monitoring Module

Tracks translation performance, quality metrics, and system health.
- Latency tracking per tier
- BLEU score calculation
- Cache hit rate monitoring
- Error tracking
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from statistics import mean, median

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    
from ..models import MetricsSummary

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Statistics for latency tracking."""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float):
        self.values.append(value)
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def mean(self) -> float:
        return mean(self.values) if self.values else 0.0
    
    @property
    def median(self) -> float:
        return median(self.values) if self.values else 0.0
    
    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class MetricsCollector:
    """
    Collects and aggregates translation metrics.
    
    Tracked metrics:
    - Translation latency (end-to-end and per-tier)
    - Cache hit rate
    - Tier distribution
    - Entity preservation count
    - Ambiguity resolutions
    - Error counts
    - BLEU scores (when references available)
    """
    
    def __init__(self):
        # Counters
        self._total_translations = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._entity_preservations = 0
        self._ambiguity_resolutions = 0
        self._openai_failures = 0
        self._nmt_failures = 0
        
        # Tier distribution
        self._tier_counts = {
            "passthrough": 0,  # English queries that skip translation
            "cache": 0,
            "entity": 0,
            "ambiguity": 0,
            "nmt": 0
        }
        
        # Latency tracking
        self._total_latency = LatencyStats()
        self._tier_latencies = {
            "cache": LatencyStats(),
            "entity": LatencyStats(),
            "ambiguity": LatencyStats(),
            "nmt": LatencyStats()
        }
        
        # BLEU scores
        self._bleu_scores: deque = deque(maxlen=100)
        
        # Timing context
        self._current_request_start: Optional[float] = None
    
    def start_request(self):
        """Mark the start of a translation request."""
        self._current_request_start = time.perf_counter()
    
    def end_request(self, tier_used: str):
        """Mark the end of a translation request."""
        if self._current_request_start:
            latency = (time.perf_counter() - self._current_request_start) * 1000
            self._total_latency.add(latency)
            self._current_request_start = None
        
        self._total_translations += 1
        self._tier_counts[tier_used] = self._tier_counts.get(tier_used, 0) + 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self._cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self._cache_misses += 1
    
    def record_tier_latency(self, tier: str, latency_ms: float):
        """Record latency for a specific tier."""
        if tier in self._tier_latencies:
            self._tier_latencies[tier].add(latency_ms)
    
    def record_entity_preservation(self, count: int = 1):
        """Record entity preservation."""
        self._entity_preservations += count
    
    def record_ambiguity_resolution(self, count: int = 1):
        """Record ambiguity resolution."""
        self._ambiguity_resolutions += count
    
    def record_openai_failure(self):
        """Record an OpenAI API failure."""
        self._openai_failures += 1
    
    def record_nmt_failure(self):
        """Record an NMT failure."""
        self._nmt_failures += 1
    
    def record_bleu_score(self, hypothesis: str, reference: str):
        """Calculate and record BLEU score."""
        if not SACREBLEU_AVAILABLE:
            return
        
        try:
            bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
            self._bleu_scores.append(bleu.score)
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
    
    def get_summary(self) -> MetricsSummary:
        """Get aggregated metrics summary."""
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_cache_requests 
            if total_cache_requests > 0 else 0.0
        )
        
        return MetricsSummary(
            total_translations=self._total_translations,
            cache_hit_rate=cache_hit_rate,
            average_latency_ms=self._total_latency.mean,
            tier_distribution=self._tier_counts.copy(),
            entity_preservation_count=self._entity_preservations,
            ambiguity_resolutions=self._ambiguity_resolutions,
            openai_api_failures=self._openai_failures,
            nmt_failures=self._nmt_failures
        )
    
    def get_detailed_stats(self) -> dict:
        """Get detailed statistics for dashboard."""
        total_cache = self._cache_hits + self._cache_misses
        
        return {
            "totals": {
                "translations": self._total_translations,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "entity_preservations": self._entity_preservations,
                "ambiguity_resolutions": self._ambiguity_resolutions
            },
            "rates": {
                "cache_hit_rate": self._cache_hits / total_cache if total_cache > 0 else 0.0
            },
            "latency": {
                "total": {
                    "mean_ms": self._total_latency.mean,
                    "median_ms": self._total_latency.median,
                    "p95_ms": self._total_latency.p95,
                    "count": self._total_latency.count
                },
                "per_tier": {
                    tier: {
                        "mean_ms": stats.mean,
                        "median_ms": stats.median,
                        "p95_ms": stats.p95,
                        "count": stats.count
                    }
                    for tier, stats in self._tier_latencies.items()
                }
            },
            "tier_distribution": self._tier_counts,
            "errors": {
                "openai_failures": self._openai_failures,
                "nmt_failures": self._nmt_failures
            },
            "quality": {
                "bleu_scores": list(self._bleu_scores),
                "average_bleu": mean(self._bleu_scores) if self._bleu_scores else None
            }
        }
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        lines = []
        
        # Total translations
        lines.append(f"# HELP translation_total Total number of translations")
        lines.append(f"# TYPE translation_total counter")
        lines.append(f"translation_total {self._total_translations}")
        
        # Cache metrics
        lines.append(f"# HELP cache_hits_total Total cache hits")
        lines.append(f"# TYPE cache_hits_total counter")
        lines.append(f"cache_hits_total {self._cache_hits}")
        
        lines.append(f"# HELP cache_misses_total Total cache misses")
        lines.append(f"# TYPE cache_misses_total counter")
        lines.append(f"cache_misses_total {self._cache_misses}")
        
        # Latency
        lines.append(f"# HELP translation_latency_ms Translation latency in milliseconds")
        lines.append(f"# TYPE translation_latency_ms gauge")
        lines.append(f"translation_latency_ms{{quantile=\"0.5\"}} {self._total_latency.median}")
        lines.append(f"translation_latency_ms{{quantile=\"0.95\"}} {self._total_latency.p95}")
        
        # Tier distribution
        lines.append(f"# HELP tier_translations_total Translations per tier")
        lines.append(f"# TYPE tier_translations_total counter")
        for tier, count in self._tier_counts.items():
            lines.append(f'tier_translations_total{{tier="{tier}"}} {count}')
        
        # Errors
        lines.append(f"# HELP openai_failures_total OpenAI API failures")
        lines.append(f"# TYPE openai_failures_total counter")
        lines.append(f"openai_failures_total {self._openai_failures}")
        
        lines.append(f"# HELP nmt_failures_total NMT translation failures")
        lines.append(f"# TYPE nmt_failures_total counter")
        lines.append(f"nmt_failures_total {self._nmt_failures}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        self._total_translations = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._entity_preservations = 0
        self._ambiguity_resolutions = 0
        self._openai_failures = 0
        self._nmt_failures = 0
        
        for tier in self._tier_counts:
            self._tier_counts[tier] = 0
        
        self._total_latency = LatencyStats()
        for tier in self._tier_latencies:
            self._tier_latencies[tier] = LatencyStats()
        
        self._bleu_scores.clear()
        
        logger.info("Metrics reset")


def calculate_bleu(hypothesis: str, reference: str) -> Optional[float]:
    """
    Calculate BLEU score for a translation.
    
    Args:
        hypothesis: The generated translation
        reference: The reference (correct) translation
        
    Returns:
        BLEU score (0-100) or None if calculation fails
    """
    if not SACREBLEU_AVAILABLE:
        logger.warning("sacrebleu not installed, BLEU calculation unavailable")
        return None
    
    try:
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score
    except Exception as e:
        logger.error(f"BLEU calculation error: {e}")
        return None


# Global metrics collector instance
metrics_collector = MetricsCollector()
