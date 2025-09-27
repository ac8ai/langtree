from enum import Enum

from attrs import frozen
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

api_calls_per_second = 1
_default_rate_limiter = InMemoryRateLimiter(
    requests_per_second=api_calls_per_second,
    check_every_n_seconds=max(0.1, 1.0 / api_calls_per_second),
    max_bucket_size=1,
)


class Provider(Enum):
    anthropic = "anthropic"
    google = "google"
    openai = "OpenAI"  # TODO: switch to "openai" after bumping a version of library


@frozen
class ModelParam:
    provider: Provider
    model: str
    temperature: float | None = None


embed_model_classes = {Provider.openai: OpenAIEmbeddings}
EmbeddingModel = OpenAIEmbeddings

chat_models_classes = {
    Provider.openai: ChatOpenAI,
    Provider.google: ChatGoogleGenerativeAI,
    Provider.anthropic: ChatAnthropic,
}


class LLMProvider:
    """Lightweight registry/factory for chat & embedding model instances.

    Responsibilities:
      - Maintain a mutable mapping of model name -> `ModelParam` config.
      - Instantiate provider specific LangChain classes on demand.
      - Offer simple update/set helpers without global side effects.

    Notes:
      - Rate limiter may be injected per-call; a default is defined but not applied
        automatically to avoid unintended throttling in batch contexts (TODO: evaluate
        central limiter strategy if concurrent usage increases).
      - Does not cache instantiated models; callers decide lifecycle management.
    """

    def __init__(self, default_model_params: dict[str, ModelParam] | None = None):
        self._model_params = (default_model_params or {}).copy()

    def get_embed_model(
        self, name: str, rate_limiter: BaseRateLimiter | None = None
    ) -> EmbeddingModel:
        """Get an embedding model by its registered name.

        Params:
            name: Logical model key registered in provider configuration.
            rate_limiter: Optional rate limiter (currently unused for embeddings; placeholder for parity).

        Returns:
            Instantiated embedding model (`EmbeddingModel`).

        Raises:
            KeyError: If the model name is not registered.
        """
        if name not in self._model_params:
            raise KeyError(
                f"Model {name} is not defined in the provider. Available models: {self.list_models()}"
            )
        model_params = self._model_params[name]
        model_class = embed_model_classes[model_params.provider]
        embed_model = model_class(
            model=model_params.model,
        )
        return embed_model

    def get_llm(
        self, name: str, rate_limiter: BaseRateLimiter | None = None
    ) -> BaseChatModel:
        """Get a chat (LLM) model by its registered name.

        Params:
            name: Logical model key registered in provider configuration.
            rate_limiter: Optional rate limiter to throttle API calls (defaults to caller-provided or None).

        Returns:
            Instantiated chat model (`BaseChatModel`).

        Raises:
            KeyError: If the model name is not registered.
        """
        if name not in self._model_params:
            raise KeyError(
                f"Model {name} is not defined in the provider. Available models: {self.list_models()}"
            )
        model_params = self._model_params[name]
        model_class = chat_models_classes[model_params.provider]
        llm = model_class(
            model=model_params.model,
            temperature=model_params.temperature,
            rate_limiter=rate_limiter,
        )
        return llm

    def list_models(self) -> list[str]:
        """List all registered model keys.

        Returns:
            List of model names.
        """
        return list(self._model_params.keys())

    def update_model(self, name: str, model_params: ModelParam) -> None:
        """Update configuration for an existing model.

        Params:
            name: Existing model key to update.
            model_params: New parameter set.

        Raises:
            KeyError: If the model name is not registered.
        """
        if name not in self._model_params:
            raise KeyError(
                f"Model {name} is not defined in the provider. Available models: {self.list_models()}"
            )
        self.set_model(name, model_params)

    def set_model(self, name: str, model_params: ModelParam) -> None:
        """Insert a new model configuration or overwrite an existing one.

        Params:
            name: Model key.
            model_params: Parameter object to store.
        """
        self._model_params[name] = model_params
