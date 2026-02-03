import os
import logging
from typing import Optional
from cohere import BedrockClientV2

# Global singleton for Cohere client
_COHERE_CLIENT: Optional[BedrockClientV2] = None

def get_cohere_client(aws_region: str = None) -> Optional[BedrockClientV2]:
    """Get or create the singleton Cohere client using Bedrock."""
    global _COHERE_CLIENT
    
    if _COHERE_CLIENT:
        return _COHERE_CLIENT

    from shared.settings.config import settings
    
    aws_region = aws_region or settings.llm.AWS_LLM_REGION
    
    try:
        _COHERE_CLIENT = BedrockClientV2(
            aws_region=aws_region,
            aws_access_key=settings.llm.AWS_LLM_ACCESS_KEY_ID,
            aws_secret_key=settings.llm.AWS_LLM_SECRET_ACCESS_KEY,
        )
        logging.info(f"Cohere Bedrock client initialized (region: {aws_region})")
    except Exception as e:
        _COHERE_CLIENT = None
        logging.error(f"Cohere Bedrock client initialization failed: {e}. Reranking will be skipped.")
    
    return _COHERE_CLIENT
