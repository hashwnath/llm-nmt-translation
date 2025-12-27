"""
Pydantic models for the LLM-NMT Translation System.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Translation Request/Response Models
# ============================================================================

class TranslationRequest(BaseModel):
    """Request model for translation endpoint."""
    query: str = Field(..., description="French query to translate")
    session_id: Optional[str] = Field(None, description="Session ID for context tracking")
    include_metrics: bool = Field(True, description="Include performance metrics in response")


class TranslationResponse(BaseModel):
    """Response model for translation endpoint."""
    original: str = Field(..., description="Original French query")
    translation: str = Field(..., description="Translated English query")
    tier_used: Literal["passthrough", "cache", "entity", "ambiguity", "nmt"] = Field(..., description="Which tier handled the translation")
    entities_preserved: list[str] = Field(default_factory=list, description="Entities that were preserved")
    ambiguity_resolved: Optional[dict] = Field(None, description="Ambiguity resolution details if applicable")
    confidence: float = Field(..., description="Translation confidence score")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    fallback: bool = Field(False, description="Whether a fallback was used")
    error: Optional[str] = Field(None, description="Error message if any")


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation endpoint."""
    queries: list[str] = Field(..., description="List of French queries to translate", max_length=32)
    session_id: Optional[str] = Field(None, description="Session ID for context tracking")


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation endpoint."""
    translations: list[TranslationResponse] = Field(..., description="List of translation responses")
    total_latency_ms: float = Field(..., description="Total batch latency in milliseconds")
    average_latency_ms: float = Field(..., description="Average per-query latency")


# ============================================================================
# Session Context Models
# ============================================================================

class SessionEvent(BaseModel):
    """Model for session events (user activity tracking)."""
    event_type: Literal["view_product", "add_to_cart", "search_query", "filter_applied"] = Field(...)
    category: Optional[str] = Field(None, description="Product category if applicable")
    brand: Optional[str] = Field(None, description="Brand if applicable")
    product_id: Optional[str] = Field(None, description="Product ID if applicable")
    query_text: Optional[str] = Field(None, description="Search query text if applicable")
    filter_type: Optional[str] = Field(None, description="Filter type if applicable")
    filter_value: Optional[str] = Field(None, description="Filter value if applicable")


class SessionContext(BaseModel):
    """Model for session context data."""
    session_id: str = Field(...)
    categories_viewed: dict[str, int] = Field(default_factory=dict)
    recent_brands: list[str] = Field(default_factory=list)
    recent_searches: list[str] = Field(default_factory=list)
    top_category: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Entity Models
# ============================================================================

class Entity(BaseModel):
    """Model for extracted entities."""
    text: str = Field(..., description="Entity text")
    entity_type: Literal["brand", "product_line", "franchise", "not_entity"] = Field(...)
    start_pos: int = Field(..., description="Start position in query")
    end_pos: int = Field(..., description="End position in query")
    confidence: float = Field(..., description="Entity confidence score")
    should_preserve: bool = Field(True, description="Whether to preserve during translation")


# ============================================================================
# Cache Models
# ============================================================================

class CacheEntry(BaseModel):
    """Model for translation cache entries."""
    query: str = Field(...)
    translation: str = Field(...)
    source: Literal["llm_generated", "nmt_cached", "user_corrected"] = Field(...)
    confidence: float = Field(...)
    language_pair: str = Field(default="fr-en")
    locale: str = Field(default="ca_FR")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frequency: int = Field(default=1)
    ttl_seconds: int = Field(default=86400)


class CacheStats(BaseModel):
    """Model for cache statistics."""
    total_entries: int = Field(...)
    hit_count: int = Field(...)
    miss_count: int = Field(...)
    hit_rate: float = Field(...)
    fuzzy_hits: int = Field(...)
    memory_usage_mb: float = Field(...)


# ============================================================================
# Metrics Models
# ============================================================================

class MetricsSummary(BaseModel):
    """Model for metrics summary."""
    total_translations: int = Field(default=0)
    cache_hit_rate: float = Field(default=0.0)
    average_latency_ms: float = Field(default=0.0)
    tier_distribution: dict[str, int] = Field(default_factory=dict)
    entity_preservation_count: int = Field(default=0)
    ambiguity_resolutions: int = Field(default=0)
    openai_api_failures: int = Field(default=0)
    nmt_failures: int = Field(default=0)


class BenchmarkResult(BaseModel):
    """Model for quantization benchmark results."""
    model_type: Literal["fp32", "int8"] = Field(...)
    single_query_latency_ms: float = Field(...)
    batch_latency_ms: float = Field(...)
    throughput_qps: float = Field(...)
    memory_usage_mb: float = Field(...)
