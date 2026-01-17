"""Local LLM integration via Ollama."""

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    content: str
    provider: str
    model: str
    confidence: float
    tokens_used: int | None = None


class LocalLLM:
    """Ollama-based local LLM interface."""

    UNCERTAINTY_PHRASES = [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "i can't",
        "i do not have",
        "i'm unable to",
        "no information",
        "not available",
        "beyond my knowledge",
        "outside my training",
        "i apologize",
        "i don't have access",
    ]

    def __init__(self, model: str | None = None) -> None:
        """Initialize the local LLM.

        Args:
            model: Model name to use. Defaults to settings.ollama_model.
        """
        self._model = model or settings.ollama_model
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=self._model,
            temperature=0.7,
        )

    def set_model(self, model_name: str) -> None:
        """Switch to a different Ollama model.

        Args:
            model_name: Name of the model to use (e.g., 'mistral:7b', 'deepseek-r1:7b')
        """
        logger.info(f"Switching model from {self._model} to {model_name}")
        self._model = model_name
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_name,
            temperature=0.7,
        )

    def get_model(self) -> str:
        """Get the current model name."""
        return self._model

    @staticmethod
    def list_available_models() -> list[str]:
        """List all available Ollama models.

        Returns:
            List of model names installed in Ollama.
        """
        try:
            response = httpx.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code != 200:
                logger.warning(f"Failed to list models: HTTP {response.status_code}")
                return []

            data = response.json()
            return [m["name"] for m in data.get("models", [])]

        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    @staticmethod
    def check_availability() -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            response = httpx.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            # Check if configured model exists
            model_name = settings.ollama_model.split(":")[0]
            return any(model_name in m for m in models)

        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False

    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence in the response.

        Returns a score between 0 and 1.
        """
        response_lower = response.lower()

        # Check for uncertainty phrases
        uncertainty_count = sum(
            1 for phrase in self.UNCERTAINTY_PHRASES
            if phrase in response_lower
        )

        if uncertainty_count >= 2:
            return 0.3
        elif uncertainty_count == 1:
            return 0.5

        # Very short responses might indicate uncertainty
        if len(response) < 50:
            return 0.6

        # Longer, confident responses
        return 0.85

    def generate(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
    ) -> GenerationResult:
        """
        Generate a response from the local LLM.

        Args:
            messages: List of messages (conversation history)
            temperature: Optional temperature override

        Returns:
            GenerationResult with the response and metadata
        """
        if temperature is not None:
            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=self._model,
                temperature=temperature,
            )
        else:
            llm = self._llm

        try:
            response = llm.invoke(messages)
            content = response.content

            confidence = self._estimate_confidence(content)

            logger.debug(
                f"Local LLM generated response (confidence: {confidence:.2f})"
            )

            return GenerationResult(
                content=content,
                provider="ollama",
                model=self._model,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """
        Generate a response with RAG context.

        Args:
            query: The user's question
            context: Retrieved context from knowledge base
            system_prompt: Optional system prompt override

        Returns:
            GenerationResult with the response
        """
        default_system = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context doesn't contain enough information to answer fully, say so clearly.
Be concise but thorough in your answers."""

        messages = [
            SystemMessage(content=system_prompt or default_system),
            HumanMessage(
                content=f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
            ),
        ]

        return self.generate(messages)

    def stream(
        self,
        messages: list[BaseMessage],
    ) -> Iterator[str]:
        """
        Stream a response from the local LLM.

        Args:
            messages: List of messages

        Yields:
            Response chunks as strings
        """
        try:
            for chunk in self._llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Local LLM streaming failed: {e}")
            raise

    async def agenerate(
        self,
        messages: list[BaseMessage],
    ) -> GenerationResult:
        """
        Async generate a response from the local LLM.

        Args:
            messages: List of messages

        Returns:
            GenerationResult with the response
        """
        try:
            response = await self._llm.ainvoke(messages)
            content = response.content
            confidence = self._estimate_confidence(content)

            return GenerationResult(
                content=content,
                provider="ollama",
                model=self._model,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Async local LLM generation failed: {e}")
            raise

    async def astream(
        self,
        messages: list[BaseMessage],
    ) -> AsyncIterator[str]:
        """
        Async stream a response from the local LLM.

        Args:
            messages: List of messages

        Yields:
            Response chunks as strings
        """
        try:
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Async local LLM streaming failed: {e}")
            raise
