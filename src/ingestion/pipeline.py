"""Document ingestion orchestrator."""

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from config.settings import settings
from src.ingestion.loaders import DocumentLoader
from src.knowledge.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    files_processed: int
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    errors: list[str]


class IngestionPipeline:
    """Orchestrates document ingestion into the knowledge base."""

    LANGUAGE_MAP = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".jsx": Language.JS,
        ".tsx": Language.TS,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".c": Language.C,
        ".cpp": Language.CPP,
        ".h": Language.C,
        ".hpp": Language.CPP,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".sh": Language.MARKDOWN,  # Fallback
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
    }

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        loader: DocumentLoader | None = None,
    ) -> None:
        """Initialize the ingestion pipeline."""
        self._vectorstore = vectorstore or VectorStore()
        self._loader = loader or DocumentLoader()
        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _get_splitter(self, extension: str) -> RecursiveCharacterTextSplitter:
        """Get appropriate text splitter based on file extension."""
        language = self.LANGUAGE_MAP.get(extension)

        if language:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        return self._default_splitter

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        chunks = []

        for doc in documents:
            ext = doc.metadata.get("extension", "")
            splitter = self._get_splitter(ext)

            doc_chunks = splitter.split_documents([doc])

            # Add chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(doc_chunks)

            chunks.extend(doc_chunks)

        return chunks

    def ingest_file(self, file_path: Path) -> IngestionStats:
        """
        Ingest a single file into the knowledge base.

        Args:
            file_path: Path to the file

        Returns:
            IngestionStats with processing details
        """
        errors: list[str] = []

        documents = self._loader.load_file(file_path)
        if not documents:
            return IngestionStats(
                files_processed=1,
                documents_loaded=0,
                chunks_created=0,
                chunks_stored=0,
                errors=[f"Failed to load: {file_path}"],
            )

        chunks = self._split_documents(documents)
        stored_ids = self._vectorstore.add_documents(chunks)

        logger.info(
            f"Ingested {file_path}: {len(documents)} docs -> "
            f"{len(chunks)} chunks -> {len(stored_ids)} stored"
        )

        return IngestionStats(
            files_processed=1,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(stored_ids),
            errors=errors,
        )

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> IngestionStats:
        """
        Ingest all files from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            exclude_patterns: Patterns to exclude

        Returns:
            IngestionStats with processing details
        """
        documents = self._loader.load_directory(
            directory,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
        )

        if not documents:
            return IngestionStats(
                files_processed=0,
                documents_loaded=0,
                chunks_created=0,
                chunks_stored=0,
                errors=[f"No documents found in: {directory}"],
            )

        chunks = self._split_documents(documents)
        stored_ids = self._vectorstore.add_documents(chunks)

        # Count unique files
        unique_files = {doc.metadata.get("source") for doc in documents}

        logger.info(
            f"Ingested directory {directory}: {len(unique_files)} files -> "
            f"{len(documents)} docs -> {len(chunks)} chunks -> {len(stored_ids)} stored"
        )

        return IngestionStats(
            files_processed=len(unique_files),
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(stored_ids),
            errors=[],
        )

    def ingest_text(
        self,
        text: str,
        source: str,
        metadata: dict | None = None,
    ) -> IngestionStats:
        """
        Ingest raw text into the knowledge base.

        Args:
            text: The text content
            source: Source identifier
            metadata: Additional metadata

        Returns:
            IngestionStats with processing details
        """
        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "file_type": "text",
                **(metadata or {}),
            },
        )

        chunks = self._default_splitter.split_documents([doc])
        stored_ids = self._vectorstore.add_documents(chunks)

        return IngestionStats(
            files_processed=1,
            documents_loaded=1,
            chunks_created=len(chunks),
            chunks_stored=len(stored_ids),
            errors=[],
        )

    def clear_source(self, source: str) -> int:
        """
        Remove all documents from a specific source.

        Args:
            source: The source to clear

        Returns:
            Number of documents removed
        """
        return self._vectorstore.delete_by_source(source)

    def get_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        return self._vectorstore.get_collection_stats()
