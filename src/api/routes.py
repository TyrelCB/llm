"""API routes for the LLM agent."""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from config.settings import settings
from src.agent import Agent
from src.ingestion import IngestionPipeline
from src.knowledge import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared instances
_agent: Agent | None = None
_pipeline: IngestionPipeline | None = None
_vectorstore: VectorStore | None = None


def get_agent() -> Agent:
    """Get or create agent instance."""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def get_pipeline() -> IngestionPipeline:
    """Get or create ingestion pipeline instance."""
    global _pipeline, _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    if _pipeline is None:
        _pipeline = IngestionPipeline(_vectorstore)
    return _pipeline


def get_vectorstore() -> VectorStore:
    """Get or create vectorstore instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore


# Request/Response models


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The user's question")
    include_history: bool = Field(
        default=True, description="Include conversation history"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    response: str = Field(..., description="The agent's response")
    provider: str = Field(..., description="LLM provider used")
    documents_used: int = Field(..., description="Number of documents used")
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results from tool executions"
    )


class IngestTextRequest(BaseModel):
    """Request model for text ingestion."""

    text: str = Field(..., description="Text content to ingest")
    source: str = Field(..., description="Source identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoints."""

    files_processed: int
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    errors: list[str]


class KBStatsResponse(BaseModel):
    """Response model for knowledge base stats."""

    collection_name: str
    document_count: int
    path: str


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str = Field(..., description="Search query")
    k: int = Field(default=4, description="Number of results")
    min_score: float = Field(default=0.5, description="Minimum similarity score")


class SearchResult(BaseModel):
    """Single search result."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: list[SearchResult]
    query: str


# Endpoints


@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """
    Query the agent with a question.

    The agent will:
    1. Route the query appropriately
    2. Retrieve relevant documents if needed
    3. Generate a response using local or fallback LLM
    """
    try:
        agent = get_agent()
        result = await agent.aquery(
            query=request.query,
            include_history=request.include_history,
        )

        return QueryResponse(
            response=result["response"],
            provider=result["provider"],
            documents_used=result["documents_used"],
            tool_results=result["tool_results"],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/sync", response_model=QueryResponse)
def query_agent_sync(request: QueryRequest) -> QueryResponse:
    """Synchronous version of query endpoint."""
    try:
        agent = get_agent()
        result = agent.query(
            query=request.query,
            include_history=request.include_history,
        )

        return QueryResponse(
            response=result["response"],
            provider=result["provider"],
            documents_used=result["documents_used"],
            tool_results=result["tool_results"],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
def clear_history():
    """Clear conversation history."""
    agent = get_agent()
    agent.clear_history()
    return {"status": "cleared"}


@router.get("/history/length")
def get_history_length():
    """Get conversation history length."""
    agent = get_agent()
    return {"length": agent.get_history_length()}


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Ingest raw text into the knowledge base."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.ingest_text(
            text=request.text,
            source=request.source,
            metadata=request.metadata,
        )

        return IngestResponse(
            files_processed=stats.files_processed,
            documents_loaded=stats.documents_loaded,
            chunks_created=stats.chunks_created,
            chunks_stored=stats.chunks_stored,
            errors=stats.errors,
        )

    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest a file into the knowledge base."""
    try:
        # Save uploaded file temporarily
        temp_path = settings.data_dir / "temp" / file.filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        temp_path.write_bytes(content)

        try:
            pipeline = get_pipeline()
            stats = pipeline.ingest_file(temp_path)

            return IngestResponse(
                files_processed=stats.files_processed,
                documents_loaded=stats.documents_loaded,
                chunks_created=stats.chunks_created,
                chunks_stored=stats.chunks_stored,
                errors=stats.errors,
            )
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/directory", response_model=IngestResponse)
def ingest_directory(
    path: str,
    recursive: bool = True,
) -> IngestResponse:
    """Ingest all documents from a directory."""
    try:
        directory = Path(path)
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

        pipeline = get_pipeline()
        stats = pipeline.ingest_directory(directory, recursive=recursive)

        return IngestResponse(
            files_processed=stats.files_processed,
            documents_loaded=stats.documents_loaded,
            chunks_created=stats.chunks_created,
            chunks_stored=stats.chunks_stored,
            errors=stats.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directory ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kb/stats", response_model=KBStatsResponse)
def get_kb_stats() -> KBStatsResponse:
    """Get knowledge base statistics."""
    vectorstore = get_vectorstore()
    stats = vectorstore.get_collection_stats()

    return KBStatsResponse(
        collection_name=stats["collection_name"],
        document_count=stats["document_count"],
        path=stats["path"],
    )


@router.post("/kb/search", response_model=SearchResponse)
def search_kb(request: SearchRequest) -> SearchResponse:
    """Search the knowledge base."""
    try:
        vectorstore = get_vectorstore()
        documents = vectorstore.similarity_search(
            query=request.query,
            k=request.k,
        )

        results = []
        for doc in documents:
            score = doc.metadata.get("similarity_score", 0)
            if score >= request.min_score:
                results.append(
                    SearchResult(
                        content=doc.page_content[:500],  # Truncate for API
                        source=doc.metadata.get("source", "unknown"),
                        score=score,
                        metadata={
                            k: v
                            for k, v in doc.metadata.items()
                            if k not in ("similarity_score", "source")
                        },
                    )
                )

        return SearchResponse(results=results, query=request.query)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/kb/clear")
def clear_kb():
    """Clear all documents from the knowledge base."""
    try:
        vectorstore = get_vectorstore()
        vectorstore.clear()
        return {"status": "cleared"}

    except Exception as e:
        logger.error(f"KB clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/kb/source/{source}")
def delete_source(source: str):
    """Delete all documents from a specific source."""
    try:
        vectorstore = get_vectorstore()
        count = vectorstore.delete_by_source(source)
        return {"status": "deleted", "count": count}

    except Exception as e:
        logger.error(f"Source deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
