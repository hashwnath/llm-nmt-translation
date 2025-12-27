"""
FastAPI Application for LLM-NMT Translation System

Main API endpoints for translation, session management, and metrics.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
    SessionEvent,
    CacheStats,
    MetricsSummary
)
from .translation.pipeline import translation_pipeline
from .translation.tier1_cache import translation_memory
from .translation.metrics import metrics_collector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM-NMT Translation System...")
    
    # Preload some common translations
    common_translations = [
        {"query": "bonjour", "translation": "hello", "confidence": 0.99},
        {"query": "merci", "translation": "thank you", "confidence": 0.99},
        {"query": "rechercher", "translation": "search", "confidence": 0.95},
        {"query": "acheter", "translation": "buy", "confidence": 0.95},
        {"query": "produit", "translation": "product", "confidence": 0.95},
    ]
    translation_pipeline.preload_cache(common_translations)
    
    logger.info("Translation system ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down translation system...")


# Create FastAPI app
app = FastAPI(
    title="LLM-NMT Translation System",
    description="Three-tier cascading translation system for cross-lingual e-commerce search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")


# ============================================================================
# Health & Root Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>LLM-NMT Translation System</h1><p>API is running. UI not found.</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "llm-nmt-translation",
        "version": "1.0.0"
    }


# ============================================================================
# Translation Endpoints
# ============================================================================

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate a French query to English.
    
    Uses the 3-tier cascading system:
    1. Translation Memory (cache)
    2. Entity-Aware Translation + Ambiguity Resolution
    3. Neural Machine Translation
    """
    try:
        response = translation_pipeline.translate(request)
        return response
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """
    Batch translate multiple French queries to English.
    
    Max batch size: 32 queries
    """
    if len(request.queries) > settings.BATCH_MAX_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum of {settings.BATCH_MAX_SIZE}"
        )
    
    try:
        response = translation_pipeline.translate_batch(request)
        return response
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Session Endpoints
# ============================================================================

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session context for debugging."""
    context = translation_pipeline.get_session_context(session_id)
    if context:
        return context
    return {"session_id": session_id, "message": "No session data found"}


@app.post("/session/{session_id}/event")
async def add_session_event(session_id: str, event: SessionEvent):
    """
    Log a session event for context tracking.
    
    Event types:
    - view_product: User viewed a product
    - add_to_cart: User added item to cart
    - search_query: User performed a search
    - filter_applied: User applied a filter
    """
    try:
        translation_pipeline.add_session_event(session_id, event)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.error(f"Session event error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Cache Endpoints
# ============================================================================

@app.get("/cache/stats")
async def get_cache_stats():
    """Get translation cache statistics."""
    return translation_pipeline.get_cache_stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear the translation cache."""
    translation_memory.clear()
    return {"status": "success", "message": "Cache cleared"}


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get Prometheus-compatible metrics."""
    return metrics_collector.get_prometheus_metrics()


@app.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """Get human-readable metrics summary."""
    return metrics_collector.get_summary()


@app.get("/metrics/detailed")
async def get_detailed_metrics():
    """Get detailed metrics for dashboard."""
    return metrics_collector.get_detailed_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics."""
    metrics_collector.reset()
    return {"status": "success", "message": "Metrics reset"}


# ============================================================================
# Model Info Endpoints
# ============================================================================

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded NMT model."""
    from .translation.tier3_nmt import nmt_translator
    return nmt_translator.get_model_info()


@app.post("/model/warmup")
async def warmup_model():
    """Warm up the NMT model."""
    try:
        translation_pipeline.warmup()
        return {"status": "success", "message": "Model warmed up"}
    except Exception as e:
        logger.error(f"Warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Demo Endpoints
# ============================================================================

@app.get("/demo/queries")
async def get_demo_queries():
    """Get sample queries for demo."""
    return {
        "queries": [
            {
                "french": "yogurt liberté logo",
                "expected": "yogurt liberté logo",
                "feature": "Entity preservation (brand name)"
            },
            {
                "french": "je veux acheter gomme",
                "expected": "i want to buy eraser",
                "feature": "Ambiguity resolution (context: school)"
            },
            {
                "french": "papier royale",
                "expected": "paper royale",
                "feature": "Brand preservation"
            },
            {
                "french": "pêche fraîche",
                "expected": "fresh peach",
                "feature": "Context-aware translation"
            },
            {
                "french": "tresor parfum",
                "expected": "tresor perfume",
                "feature": "Brand detection (cosmetics)"
            },
            {
                "french": "lait danone",
                "expected": "danone milk",
                "feature": "Brand preservation (dairy)"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
