"""Document relevance grading using LLM."""

import logging
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import settings

logger = logging.getLogger(__name__)


class RelevanceGrade(Enum):
    """Document relevance grades."""

    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"


@dataclass
class GradingResult:
    """Result of document grading."""

    document: Document
    grade: RelevanceGrade
    confidence: float
    reasoning: str


GRADING_SYSTEM_PROMPT = """You are a document relevance grader. Your task is to assess whether a document is relevant to answering a user's question.

Evaluate the document based on:
1. Does it contain information directly related to the question?
2. Would it help provide an accurate, complete answer?
3. Is the information current and applicable?

Respond with ONLY one of these exact words:
- "relevant" - The document directly helps answer the question
- "partially_relevant" - The document has some useful context but doesn't directly answer
- "not_relevant" - The document is not useful for this question

Do not include any other text in your response."""


class DocumentGrader:
    """Grades document relevance using local LLM."""

    def __init__(self) -> None:
        """Initialize the grader with Ollama."""
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,  # Deterministic grading
        )

    def grade_document(
        self,
        document: Document,
        query: str,
    ) -> GradingResult:
        """
        Grade a single document's relevance to a query.

        Args:
            document: The document to grade
            query: The user's query

        Returns:
            GradingResult with grade, confidence, and reasoning
        """
        user_prompt = f"""Question: {query}

Document content:
{document.page_content[:2000]}

Is this document relevant to answering the question?"""

        messages = [
            SystemMessage(content=GRADING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            response_text = response.content.strip().lower()

            # Parse the grade
            if "not_relevant" in response_text or "not relevant" in response_text:
                grade = RelevanceGrade.NOT_RELEVANT
                confidence = 0.9
            elif "partially" in response_text:
                grade = RelevanceGrade.PARTIALLY_RELEVANT
                confidence = 0.7
            elif "relevant" in response_text:
                grade = RelevanceGrade.RELEVANT
                confidence = 0.9
            else:
                # Default to partially relevant if unclear
                grade = RelevanceGrade.PARTIALLY_RELEVANT
                confidence = 0.5
                logger.warning(f"Unclear grading response: {response_text}")

            return GradingResult(
                document=document,
                grade=grade,
                confidence=confidence,
                reasoning=response_text,
            )

        except Exception as e:
            logger.error(f"Error grading document: {e}")
            # Fall back to using similarity score
            score = document.metadata.get("similarity_score", 0.5)
            if score >= settings.relevance_threshold:
                grade = RelevanceGrade.RELEVANT
            elif score >= settings.relevance_threshold * 0.7:
                grade = RelevanceGrade.PARTIALLY_RELEVANT
            else:
                grade = RelevanceGrade.NOT_RELEVANT

            return GradingResult(
                document=document,
                grade=grade,
                confidence=score,
                reasoning=f"Fallback: similarity score {score:.2f}",
            )

    def grade_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> list[GradingResult]:
        """
        Grade multiple documents.

        Args:
            documents: Documents to grade
            query: The user's query

        Returns:
            List of GradingResults
        """
        results = []
        for doc in documents:
            result = self.grade_document(doc, query)
            results.append(result)
            logger.debug(
                f"Document graded as {result.grade.value} "
                f"(confidence: {result.confidence:.2f})"
            )
        return results

    def filter_relevant(
        self,
        documents: list[Document],
        query: str,
        include_partial: bool = True,
    ) -> list[Document]:
        """
        Filter documents to only relevant ones.

        Args:
            documents: Documents to filter
            query: The user's query
            include_partial: Whether to include partially relevant docs

        Returns:
            List of relevant documents
        """
        results = self.grade_documents(documents, query)

        relevant_docs = []
        for result in results:
            if result.grade == RelevanceGrade.RELEVANT:
                relevant_docs.append(result.document)
            elif include_partial and result.grade == RelevanceGrade.PARTIALLY_RELEVANT:
                relevant_docs.append(result.document)

        logger.info(
            f"Filtered to {len(relevant_docs)} relevant documents "
            f"from {len(documents)} total"
        )
        return relevant_docs

    def should_fallback(
        self,
        grading_results: list[GradingResult],
    ) -> bool:
        """
        Determine if we should fall back to external provider.

        Returns True if no relevant documents were found.

        Args:
            grading_results: Results from grading documents

        Returns:
            True if fallback is recommended
        """
        relevant_count = sum(
            1 for r in grading_results
            if r.grade in (RelevanceGrade.RELEVANT, RelevanceGrade.PARTIALLY_RELEVANT)
        )
        return relevant_count == 0
