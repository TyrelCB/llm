"""FastAPI application for the LLM agent."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting LLM Agent API")

    # Check Ollama availability
    from src.llm.local import LocalLLM

    if LocalLLM.check_availability():
        logger.info("Ollama is available")
    else:
        logger.warning(
            "Ollama is not available. Local LLM features will not work. "
            f"Ensure Ollama is running at {settings.ollama_base_url}"
        )

    yield

    # Shutdown
    logger.info("Shutting down LLM Agent API")


app = FastAPI(
    title="LLM Agent API",
    description="Local-first LLM agent with RAG-based knowledge base",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Agent API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from src.llm.local import LocalLLM

    ollama_available = LocalLLM.check_availability()

    return {
        "status": "healthy" if ollama_available else "degraded",
        "ollama_available": ollama_available,
    }
