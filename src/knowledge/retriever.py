"""RAG retrieval logic."""

import logging
from typing import Any

from langchain_core.documents import Document

from config.settings import settings
from src.knowledge.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Document retriever for RAG operations."""

    def __init__(self, vectorstore: VectorStore | None = None) -> None:
        """Initialize the retriever with a vector store."""
        self._vectorstore = vectorstore or VectorStore()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            min_score: Minimum similarity score threshold

        Returns:
            List of relevant documents sorted by relevance
        """
        k = k or settings.retriever_k
        min_score = min_score or settings.relevance_threshold

        documents = self._vectorstore.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum score
        filtered_docs = [
            doc for doc in documents
            if doc.metadata.get("similarity_score", 0) >= min_score
        ]

        logger.debug(
            f"Retrieved {len(filtered_docs)} documents "
            f"(filtered from {len(documents)}) for query: {query[:50]}..."
        )

        return filtered_docs

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with their similarity scores.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        documents = self._vectorstore.similarity_search(query=query, k=k)
        return [
            (doc, doc.metadata.get("similarity_score", 0.0))
            for doc in documents
        ]

    def multi_query_retrieve(
        self,
        queries: list[str],
        k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve documents using multiple query variations.

        This helps improve recall by searching with different phrasings.

        Args:
            queries: List of query variations
            k: Number of documents per query

        Returns:
            Deduplicated list of relevant documents
        """
        k = k or settings.retriever_k
        seen_hashes: set[str] = set()
        unique_docs: list[Document] = []

        for query in queries:
            docs = self._vectorstore.similarity_search(query=query, k=k)
            for doc in docs:
                content_hash = doc.metadata.get("content_hash", "")
                if content_hash and content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)

        # Sort by similarity score
        unique_docs.sort(
            key=lambda d: d.metadata.get("similarity_score", 0),
            reverse=True,
        )

        logger.debug(
            f"Multi-query retrieval found {len(unique_docs)} unique documents "
            f"from {len(queries)} queries"
        )

        return unique_docs[:k * 2]  # Return up to 2x k documents

    def format_context(self, documents: list[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.

        Args:
            documents: List of documents to format

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            score = doc.metadata.get("similarity_score", 0)
            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)
