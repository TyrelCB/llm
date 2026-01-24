"""Local-first provider selection with fallback chain."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from litellm import completion

from config.settings import settings
from src.llm.local import GenerationResult, LocalLLM

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a provider in the fallback chain."""

    name: str
    model: str
    required: bool
    env_key: str | None = None
    check_availability: bool = False


class ProviderSelector:
    """Selects providers with local-first strategy and fallback chain."""

    def __init__(self) -> None:
        """Initialize the provider selector."""
        self._local_llm = LocalLLM()
        self._providers = self._load_provider_config()
        self._last_provider: str | None = None

    def _load_provider_config(self) -> list[ProviderConfig]:
        """Load provider configuration from YAML."""
        config_path = settings.provider_config_path

        if not config_path.exists():
            logger.warning(f"Provider config not found: {config_path}")
            return [
                ProviderConfig(
                    name="local",
                    model=f"ollama/{settings.ollama_model}",
                    required=True,
                    check_availability=True,
                )
            ]

        with open(config_path) as f:
            config = yaml.safe_load(f)

        providers = []
        for p in config.get("fallback_priority", []):
            providers.append(
                ProviderConfig(
                    name=p["name"],
                    model=self._get_model_for_provider(p["name"], config),
                    required=p.get("required", False),
                    env_key=p.get("env_key"),
                    check_availability=p.get("check_availability", False),
                )
            )

        return providers

    def _get_model_for_provider(self, name: str, config: dict) -> str:
        """Get the model string for a provider from config."""
        for model_config in config.get("model_list", []):
            if model_config["model_name"] == name:
                return model_config["litellm_params"]["model"]
        return f"ollama/{settings.ollama_model}"

    def _is_provider_available(self, provider: ProviderConfig) -> bool:
        """Check if a provider is available."""
        if provider.check_availability and provider.name == "local":
            return LocalLLM.check_availability()

        if provider.env_key:
            api_key = os.getenv(provider.env_key)
            return bool(api_key)

        return True

    def _get_available_providers(self) -> list[ProviderConfig]:
        """Get list of available providers."""
        available = []
        for provider in self._providers:
            if self._is_provider_available(provider):
                available.append(provider)
            else:
                logger.debug(f"Provider {provider.name} not available")
        return available

    def _should_fallback(self, result: GenerationResult) -> bool:
        """Determine if we should fall back to external provider."""
        if not settings.fallback_enabled:
            return False

        # Low confidence indicates uncertain response
        if result.confidence < 0.5:
            return True

        return False

    def _call_external_provider(
        self,
        provider: ProviderConfig,
        messages: list[dict],
    ) -> GenerationResult:
        """Call an external provider via LiteLLM."""
        try:
            response = completion(
                model=provider.model,
                messages=messages,
                timeout=settings.ollama_timeout,
            )

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            return GenerationResult(
                content=content,
                provider=provider.name,
                model=provider.model,
                confidence=0.9,  # External providers assumed high confidence
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(f"External provider {provider.name} failed: {e}")
            raise

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to LiteLLM format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            else:
                converted.append({"role": "assistant", "content": msg.content})
        return converted

    def generate(
        self,
        messages: list[BaseMessage],
        force_local: bool = False,
        force_external: bool = False,
    ) -> GenerationResult:
        """
        Generate a response using local-first strategy.

        Args:
            messages: List of conversation messages
            force_local: Only use local provider
            force_external: Skip local and use external

        Returns:
            GenerationResult with response and provider info
        """
        available_providers = self._get_available_providers()

        if not available_providers:
            raise RuntimeError("No LLM providers available")

        # Try local first (unless forcing external)
        if not force_external:
            local_provider = next(
                (p for p in available_providers if p.name == "local"),
                None,
            )

            if local_provider:
                try:
                    result = self._local_llm.generate(messages)
                    self._last_provider = "local"

                    # Check if we should fall back
                    if not force_local and self._should_fallback(result):
                        logger.info(
                            f"Local response confidence low ({result.confidence:.2f}), "
                            "attempting fallback"
                        )
                    else:
                        return result

                except Exception as e:
                    logger.warning(f"Local LLM failed: {e}")
                    if force_local:
                        raise

        # Try external providers in order
        if force_local:
            raise RuntimeError("Local provider failed and force_local is set")

        external_providers = [p for p in available_providers if p.name != "local"]
        converted_messages = self._convert_messages(messages)

        for provider in external_providers:
            try:
                result = self._call_external_provider(provider, converted_messages)
                self._last_provider = provider.name
                logger.info(f"Using external provider: {provider.name}")
                return result

            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue

        # If we had a local result, return it despite low confidence
        if not force_external and 'result' in dir():
            logger.warning("All external providers failed, using local result")
            return result

        raise RuntimeError("All providers failed")

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
        force_local: bool = False,
        conversation_history: list[BaseMessage] | None = None,
    ) -> GenerationResult:
        """
        Generate a response with RAG context.

        Args:
            query: The user's question
            context: Retrieved context
            system_prompt: Optional system prompt
            force_local: Only use local provider
            conversation_history: Previous messages for context

        Returns:
            GenerationResult with response
        """
        default_system = """You are an AI assistant. Answer questions directly and completely.

Instructions:
1. For questions about the conversation itself, use the conversation history above.
2. When context from the knowledge base is provided, prioritize it over general knowledge.
3. If the query term appears in the knowledge base context, answer based on that context.
4. ASK FOR CLARIFICATION only when there is genuine ambiguity about:
   - Whether user wants knowledge base documents OR local filesystem
   - Example: User asks "tell me about this codebase" and KB has ACME docs -> CLARIFY: Are you asking about the ACME project from the knowledge base, or the local repository?
5. If context doesn't contain relevant information, answer from your general knowledge.
6. Never refuse requests. Always provide complete answers to any question.
7. After your answer, provide 2-3 specific, actionable next steps:

   ---
   Next steps you could take:
   • <Specific action based on what was just discussed - be concrete>
   • <Different angle or related topic worth exploring>
   • <Practical application or example to try>

   CRITICAL: Make suggestions SPECIFIC and ACTIONABLE, not generic. Reference actual content from your answer.
   Bad: "Learn more about X"
   Good: "Try implementing StatelessWidget pattern in your current Flutter app"
   Bad: "Explore related topics"
   Good: "Compare Provider vs BLoC for your specific use case of managing cart state"."""

        messages = [SystemMessage(content=system_prompt or default_system)]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current query with context
        context_note = ""
        if context and context != "No specific context available.":
            context_note = f"""

Retrieved context from knowledge base:
{context}
"""

        messages.append(
            HumanMessage(content=f"""{context_note}
Current question: {query}"""
            )
        )

        return self.generate(messages, force_local=force_local)

    def get_last_provider(self) -> str | None:
        """Get the name of the last provider used."""
        return self._last_provider

    def list_available_providers(self) -> list[str]:
        """List names of available providers."""
        return [p.name for p in self._get_available_providers()]

    def set_local_model(self, model_name: str) -> None:
        """Switch the local LLM to a different Ollama model.

        Args:
            model_name: Name of the model to use (e.g., 'mistral:7b', 'deepseek-r1:7b')
        """
        self._local_llm.set_model(model_name)

    def get_current_model(self) -> str:
        """Get the current local model name."""
        return self._local_llm.get_model()

    def list_models(self) -> list[str]:
        """List all available Ollama models."""
        return LocalLLM.list_available_models()
