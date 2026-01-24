"""Local LLM integration via Ollama."""

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
    input_tokens: int | None = None
    output_tokens: int | None = None


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
        self._llm: Any = None

    def _build_llm(self, temperature: float | None = None) -> "ChatOllama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=self._model,
            temperature=temperature if temperature is not None else 0.7,
            num_ctx=settings.ollama_num_ctx,
        )

    def _get_llm(self) -> "ChatOllama":
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def set_model(self, model_name: str) -> None:
        """Switch to a different Ollama model.

        Args:
            model_name: Name of the model to use (e.g., 'mistral:7b', 'deepseek-r1:7b')
        """
        logger.info(f"Switching model from {self._model} to {model_name}")
        self._model = model_name
        self._llm = None

    def get_model(self) -> str:
        """Get the current model name."""
        return self._model

    def set_context_window(self, num_ctx: int) -> None:
        """Set the context window size for the model.

        Args:
            num_ctx: Context window size in tokens (e.g., 4096, 8192, 32768)
        """
        if num_ctx < 512:
            raise ValueError("Context window must be at least 512 tokens")
        if num_ctx > 128000:
            raise ValueError("Context window cannot exceed 128,000 tokens")

        logger.info(f"Setting context window to {num_ctx} tokens")
        settings.ollama_num_ctx = num_ctx
        # Force rebuild of LLM with new context size
        self._llm = None

    def get_context_window(self) -> int:
        """Get the current context window size.

        Returns:
            Context window size in tokens
        """
        return settings.ollama_num_ctx

    @staticmethod
    def list_available_models() -> list[str]:
        """List all available Ollama models.

        Returns:
            List of model names installed in Ollama.
        """
        try:
            import httpx

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
            import httpx

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

    def get_model_info(self) -> dict[str, Any]:
        """Get model information including context window size.

        Returns:
            Dict with model details including context_length
        """
        try:
            import httpx

            # Get model details from Ollama
            response = httpx.post(
                f"{settings.ollama_base_url}/api/show",
                json={"name": self._model},
                timeout=10.0,
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get model info: HTTP {response.status_code}")
                return {"context_length": settings.ollama_num_ctx}

            data = response.json()

            # Extract context length from model parameters
            model_info = data.get("model_info", {})

            # Try to get context length from different possible locations
            context_length = settings.ollama_num_ctx  # Default fallback

            # Check modelfile for num_ctx parameter
            modelfile = data.get("modelfile", "")
            if "num_ctx" in modelfile:
                import re
                match = re.search(r"num_ctx\s+(\d+)", modelfile)
                if match:
                    context_length = int(match.group(1))

            # Check model_info for context length
            if isinstance(model_info, dict):
                for key in ["context_length", "max_position_embeddings", "n_ctx"]:
                    if key in model_info:
                        context_length = int(model_info[key])
                        break

            return {
                "name": self._model,
                "context_length": context_length,
                "size": data.get("size", 0),
                "parameters": data.get("details", {}).get("parameter_size", "unknown"),
            }

        except Exception as e:
            logger.debug(f"Failed to get model info: {e}")
            return {
                "name": self._model,
                "context_length": settings.ollama_num_ctx,
            }

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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4

    def generate(
        self,
        messages: list["BaseMessage"],
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
            llm = self._build_llm(temperature=temperature)
        else:
            llm = self._get_llm()

        try:
            response = llm.invoke(messages)
            content = response.content

            confidence = self._estimate_confidence(content)

            # Estimate token usage
            input_tokens = sum(self._estimate_tokens(msg.content) for msg in messages)
            output_tokens = self._estimate_tokens(content)
            total_tokens = input_tokens + output_tokens

            logger.debug(
                f"Local LLM generated response (confidence: {confidence:.2f}, "
                f"tokens: {total_tokens} [in:{input_tokens}, out:{output_tokens}])"
            )

            return GenerationResult(
                content=content,
                provider="ollama",
                model=self._model,
                confidence=confidence,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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

        from langchain_core.messages import HumanMessage, SystemMessage

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
        messages: list["BaseMessage"],
    ) -> Iterator[str]:
        """
        Stream a response from the local LLM.

        Args:
            messages: List of messages

        Yields:
            Response chunks as strings
        """
        try:
            for chunk in self._get_llm().stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Local LLM streaming failed: {e}")
            raise

    async def agenerate(
        self,
        messages: list["BaseMessage"],
    ) -> GenerationResult:
        """
        Async generate a response from the local LLM.

        Args:
            messages: List of messages

        Returns:
            GenerationResult with the response
        """
        try:
            response = await self._get_llm().ainvoke(messages)
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
        messages: list["BaseMessage"],
    ) -> AsyncIterator[str]:
        """
        Async stream a response from the local LLM.

        Args:
            messages: List of messages

        Yields:
            Response chunks as strings
        """
        try:
            async for chunk in self._get_llm().astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Async local LLM streaming failed: {e}")
            raise
