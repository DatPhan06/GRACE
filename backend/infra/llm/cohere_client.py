import logging
from typing import Optional
from cohere import BedrockClientV2

_COHERE_CLIENT: Optional[BedrockClientV2] = None
_INITIALIZED = False


def get_cohere_client(aws_region: str = None) -> Optional[BedrockClientV2]:
    """Get or create the singleton Cohere client using Bedrock."""
    global _COHERE_CLIENT, _INITIALIZED

    if _INITIALIZED:
        return _COHERE_CLIENT

    from shared.settings.config import settings
    
    aws_region = aws_region or settings.llm.AWS_LLM_REGION
    
    _INITIALIZED = True

    if not settings.llm.AWS_LLM_ACCESS_KEY_ID or not settings.llm.AWS_LLM_SECRET_ACCESS_KEY:
        logging.warning("Cohere Bedrock client skipped: AWS credentials not configured. Falling back to LLM reranker.")
        return None

    try:
        _COHERE_CLIENT = BedrockClientV2(
            aws_region=aws_region,
            aws_access_key=settings.llm.AWS_LLM_ACCESS_KEY_ID,
            aws_secret_key=settings.llm.AWS_LLM_SECRET_ACCESS_KEY,
        )
        logging.info(f"Cohere Bedrock client initialized (region: {aws_region})")
    except Exception as e:
        logging.error(f"Cohere Bedrock client initialization failed: {e}. Reranking will be skipped.")

    return _COHERE_CLIENT
