"""File type loaders for document ingestion."""

import logging
import mimetypes
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Unified document loader supporting multiple file types."""

    SUPPORTED_EXTENSIONS = {
        ".md": "markdown",
        ".txt": "text",
        ".pdf": "pdf",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".jsx": "code",
        ".tsx": "code",
        ".java": "code",
        ".go": "code",
        ".rs": "code",
        ".c": "code",
        ".cpp": "code",
        ".h": "code",
        ".hpp": "code",
        ".rb": "code",
        ".php": "code",
        ".swift": "code",
        ".kt": "code",
        ".scala": "code",
        ".sh": "code",
        ".bash": "code",
        ".zsh": "code",
        ".yaml": "config",
        ".yml": "config",
        ".json": "config",
        ".toml": "config",
        ".ini": "config",
        ".cfg": "config",
        ".html": "web",
        ".css": "web",
        ".xml": "config",
    }

    def __init__(self) -> None:
        """Initialize the document loader."""
        mimetypes.init()

    def load_file(self, file_path: Path) -> list[Document]:
        """
        Load a single file and return Document objects.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects (may be multiple for large files)
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []

        ext = file_path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)

        if file_type is None:
            logger.debug(f"Unsupported file type: {ext}")
            return []

        try:
            if file_type == "pdf":
                return self._load_pdf(file_path)
            else:
                return self._load_text(file_path, file_type)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def _load_text(self, file_path: Path, file_type: str) -> list[Document]:
        """Load a text-based file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception:
                logger.warning(f"Could not decode {file_path}")
                return []

        metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "extension": file_path.suffix,
        }

        return [Document(page_content=content, metadata=metadata)]

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """Load a PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            documents = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    metadata = {
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "pdf",
                        "extension": ".pdf",
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                    }
                    documents.append(Document(page_content=text, metadata=metadata))

            return documents

        except ImportError:
            logger.error("pypdf not installed. Install with: pip install pypdf")
            return []

    def load_directory(
        self,
        directory: Path,
        recursive: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> list[Document]:
        """
        Load all supported files from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", ".git"])

        Returns:
            List of Document objects
        """
        exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "dist",
            "build",
            ".egg-info",
        ]

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check exclude patterns
            if any(excl in str(file_path) for excl in exclude_patterns):
                continue

            docs = self.load_file(file_path)
            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return list(self.SUPPORTED_EXTENSIONS.keys())
