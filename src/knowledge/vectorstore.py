"""ChromaDB vector store operations."""

import hashlib
import logging
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for document storage and retrieval."""

    def __init__(self) -> None:
        """Initialize the vector store with ChromaDB and Ollama embeddings."""
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process at once

        Returns:
            List of document IDs that were added
        """
        added_ids: list[str] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.page_content for doc in batch]
            metadatas = []
            ids = []

            for doc in batch:
                content_hash = self._content_hash(doc.page_content)
                doc_id = f"doc_{content_hash}"

                # Check for duplicates
                existing = self._collection.get(ids=[doc_id])
                if existing["ids"]:
                    logger.debug(f"Skipping duplicate document: {doc_id}")
                    continue

                metadata = {
                    **doc.metadata,
                    "content_hash": content_hash,
                    "added_at": datetime.now().isoformat(),
                }
                metadatas.append(metadata)
                ids.append(doc_id)

            if not ids:
                continue

            # Generate embeddings
            embeddings = self._embeddings.embed_documents(texts[: len(ids)])

            # Add to ChromaDB
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts[: len(ids)],
                metadatas=metadatas,
            )
            added_ids.extend(ids)
            logger.info(f"Added {len(ids)} documents to vector store")

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant Document objects with scores
        """
        k = k or settings.retriever_k
        query_embedding = self._embeddings.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i, doc_content in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            # Convert distance to similarity score (cosine distance to similarity)
            distance = results["distances"][0][i] if results["distances"] else 0
            metadata["similarity_score"] = 1 - distance

            documents.append(
                Document(
                    page_content=doc_content,
                    metadata=metadata,
                )
            )

        return documents

    def get_by_id(self, doc_id: str) -> Document | None:
        """Get a document by its ID."""
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None

        return Document(
            page_content=result["documents"][0],
            metadata=result["metadatas"][0] if result["metadatas"] else {},
        )

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from vector store")

    def delete_by_source(self, source: str) -> int:
        """Delete all documents from a specific source."""
        # Get all documents with matching source
        results = self._collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} documents from source: {source}")
            return len(results["ids"])
        return 0

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        count = self._collection.count()
        return {
            "collection_name": settings.chroma_collection_name,
            "document_count": count,
            "path": str(settings.chroma_db_path),
        }

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all documents from vector store")
