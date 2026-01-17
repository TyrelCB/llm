"""Knowledge base update from external sources."""

import logging
import re
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import settings
from src.knowledge.vectorstore import VectorStore

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a fact extraction assistant. Extract factual claims from the given text that would be useful to store in a knowledge base.

For each fact:
1. Extract standalone, self-contained statements
2. Include relevant context so the fact makes sense on its own
3. Avoid opinions or speculative statements
4. Keep facts concise but complete

Format your response as a numbered list of facts. If there are no extractable facts, respond with "NO_FACTS"."""


class KnowledgeBaseUpdater:
    """Updates knowledge base with information from external providers."""

    def __init__(self, vectorstore: VectorStore | None = None) -> None:
        """Initialize the updater."""
        self._vectorstore = vectorstore or VectorStore()
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,
        )

    def extract_facts(self, response: str, query: str) -> list[str]:
        """
        Extract factual claims from an LLM response.

        Args:
            response: The LLM response text
            query: The original query for context

        Returns:
            List of extracted fact strings
        """
        user_prompt = f"""Original question: {query}

Response to extract facts from:
{response}

Extract the key factual claims from this response."""

        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            result = self._llm.invoke(messages)
            content = result.content.strip()

            if "NO_FACTS" in content:
                return []

            # Parse numbered list
            facts = []
            lines = content.split("\n")
            for line in lines:
                # Remove numbering and clean up
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if cleaned and len(cleaned) > 20:  # Skip very short lines
                    facts.append(cleaned)

            logger.debug(f"Extracted {len(facts)} facts from response")
            return facts

        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []

    def update_from_response(
        self,
        query: str,
        response: str,
        provider: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Update the knowledge base with information from an external response.

        Args:
            query: The original query
            response: The response from the external provider
            provider: Name of the provider (e.g., "claude", "gpt4o")
            metadata: Additional metadata to store

        Returns:
            Number of facts added to the knowledge base
        """
        facts = self.extract_facts(response, query)

        if not facts:
            logger.info("No facts extracted from response")
            return 0

        documents = []
        for fact in facts:
            doc_metadata = {
                "source": f"external_{provider}",
                "source_type": "llm_response",
                "provider": provider,
                "original_query": query[:500],  # Truncate long queries
                "extracted_at": datetime.now().isoformat(),
                **(metadata or {}),
            }
            documents.append(
                Document(page_content=fact, metadata=doc_metadata)
            )

        added_ids = self._vectorstore.add_documents(documents)
        logger.info(
            f"Added {len(added_ids)} facts from {provider} to knowledge base"
        )
        return len(added_ids)

    def update_from_document(
        self,
        content: str,
        source: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Update the knowledge base with content directly (no fact extraction).

        Args:
            content: The document content
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Number of documents added
        """
        doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "added_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
        )
        added_ids = self._vectorstore.add_documents([doc])
        return len(added_ids)

    def should_update_kb(
        self,
        query: str,
        response: str,
        confidence: float,
    ) -> bool:
        """
        Determine if a response should be added to the knowledge base.

        Args:
            query: The original query
            response: The response content
            confidence: Confidence score of the response

        Returns:
            True if the response should be stored
        """
        # Don't store low confidence responses
        if confidence < 0.6:
            return False

        # Don't store very short responses
        if len(response) < 100:
            return False

        # Don't store responses that indicate uncertainty
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i can't",
            "no information",
            "unable to",
            "not available",
        ]
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            return False

        return True

    def get_external_sources_count(self) -> int:
        """Get the count of documents from external sources."""
        stats = self._vectorstore.get_collection_stats()
        # This is a simplified count - in production you'd query by source_type
        return stats.get("document_count", 0)
