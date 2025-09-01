import os
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncAzureOpenAI
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini as GeminiLlamaIndex
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ---------------------------------------------------------------------------
# Azure OpenAI configuration
# ---------------------------------------------------------------------------

# Chat model configuration
chat_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
chat_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
chat_azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or ""
chat_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

# Embedding model configuration
embedding_api_key = os.environ.get("EMBEDDING__KEY")
embedding_api_version = os.environ.get("EMBEDDING__API_VERSION")
embedding_azure_endpoint = os.environ.get("EMBEDDING__ENDPOINT") or ""
embedding_deployment_name = os.environ.get("EMBEDDING__DEPLOYMENT_NAME")

# ---------------------------------------------------------------------------
# Low-level Azure OpenAI clients
# ---------------------------------------------------------------------------

azure_chat_client = None
if chat_api_key:
    azure_chat_client = AsyncAzureOpenAI(
        api_key=chat_api_key,
        api_version=chat_api_version,
        azure_endpoint=chat_azure_endpoint,
    )

azure_embedding_client = None
if embedding_api_key:
    azure_embedding_client = AsyncAzureOpenAI(
        api_key=embedding_api_key,
        api_version=embedding_api_version,
        azure_endpoint=embedding_azure_endpoint,
    )

# ---------------------------------------------------------------------------
# High-level helper wrappers expected by Graphiti
# ---------------------------------------------------------------------------

# class AzureOpenAILLMClient(OpenAIClient):
#     """Wrapper around the Azure OpenAI chat completion endpoint compatible with Graphiti."""

#     def __init__(self):
#         # We intentionally do *not* call super().__init__ to avoid the normal OpenAI initialisation.
#         self.client = azure_chat_client
#         self.model = chat_deployment_name

#         # Configuration options – fall back to sensible defaults if env vars not set
#         self.max_tokens = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "4096"))
#         self.temperature = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0"))
#         self.seed = int(os.environ.get("AZURE_OPENAI_SEED", "0"))
#         self.max_token_context = int(os.environ.get("AZURE_OPENAI_MAX_TOKEN_CONTEXT", "100000"))
#         self.tiktoken_model = os.environ.get("AZURE_OPENAI_TIKTOKEN_MODEL", "cl100k_base")

#         # Graphiti expects these attributes to exist even if they point to the same deployment
#         self.small_model = chat_deployment_name
#         self.large_model = chat_deployment_name

#         # Retry behaviour
#         self.max_retries = 3
#         self.backoff_factor = 2


# class AzureOpenAIEmbedder(OpenAIEmbedder):
#     """Wrapper around the Azure OpenAI embedding endpoint compatible with Graphiti."""

#     def __init__(self):
#         # Skip parent initialisation to avoid non-Azure paths
#         self.client = azure_embedding_client
#         self.model = embedding_deployment_name

#         # Build embedder configuration – Graphiti uses this object internally
#         self.config = OpenAIEmbedderConfig(
#             embedding_model=embedding_deployment_name or "text-embedding-3-small",
#         )
#         self.dimensions = int(os.environ.get("EMBEDDING__DIMENSIONS", "1536"))


# class AzureOpenAIReranker(OpenAIRerankerClient):
#     """Wrapper around the Azure OpenAI reranking endpoint compatible with Graphiti."""

#     def __init__(self):
#         # Again, skip the normal OpenAI initialisation
#         self.client = azure_chat_client
#         self.model = chat_deployment_name

#         self.max_tokens = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "4096"))
#         self.temperature = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0"))


# ---------------------------------------------------------------------------
# Gemini / Google Generative AI helpers
# ---------------------------------------------------------------------------

def create_gemini_embedding(api_key: str, model_name: str = "gemini-embedding-001", **kwargs) -> GeminiEmbedding:
    """Return a LlamaIndex GeminiEmbedding instance with standard naming convention."""
    return GeminiEmbedding(api_key=api_key, model_name=f"models/{model_name}", **kwargs)


def create_gemini_llamaindex_llm(api_key: str, model_name: str, **kwargs) -> GeminiLlamaIndex:
    """Return a LlamaIndex Gemini LLM instance."""
    return GeminiLlamaIndex(api_key=api_key, model_name=f"models/{model_name}", **kwargs)


def create_gemini_langchain_llm(api_key: str, model_name: str, **kwargs) -> ChatGoogleGenerativeAI:
    """Return a LangChain ChatGoogleGenerativeAI instance using Gemini models."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        **kwargs,
    )


# Exported names – helps linters/IDE completion
__all__ = [
    "AzureOpenAILLMClient",
    "AzureOpenAIEmbedder",
    "AzureOpenAIReranker",
]

__all__ += [
    "create_gemini_embedding",
    "create_gemini_llamaindex_llm",
    "create_gemini_langchain_llm",
] 