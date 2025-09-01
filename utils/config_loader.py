"""
Configuration loader utility for loading API keys from environment variables.
This module provides functions to load API keys from .env file and make them available
to the rest of the application.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_together_api_keys() -> List[str]:
    """
    Load Together AI API keys from environment variables.
    
    Returns:
        List of Together AI API keys
    """
    together_keys = []
    for i in range(13):  # 0-12 based on the original config
        key = os.getenv(f"TOGETHER_AI_API_KEY_{i}")
        if key:
            together_keys.append(key)
    return together_keys


def load_google_api_keys() -> List[str]:
    """
    Load Google API keys from environment variables.
    
    Returns:
        List of Google API keys
    """
    google_keys = []
    for i in range(27):  # 0-26 based on the original config
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            google_keys.append(key)
    return google_keys


def load_omdb_api_key() -> str:
    """
    Load OMDB API key from environment variables.
    
    Returns:
        OMDB API key string
    """
    return os.getenv("OMDB_API_KEY", "")


def get_api_keys() -> Dict[str, Any]:
    """
    Get all API keys as a dictionary similar to the original config structure.
    
    Returns:
        Dictionary containing all API keys organized by service
    """
    return {
        "TOGETHER_AI_API_KEYS": load_together_api_keys(),
        "GOOGLE_API_KEYS": load_google_api_keys(),
        "OMDB_API_KEY": load_omdb_api_key()
    }


def get_google_api_key_by_index(index: int) -> str:
    """
    Get a specific Google API key by index.
    
    Args:
        index: Index of the API key (0-26)
        
    Returns:
        API key string or empty string if not found
    """
    return os.getenv(f"GOOGLE_API_KEY_{index}", "")


def get_together_api_key_by_index(index: int) -> str:
    """
    Get a specific Together AI API key by index.
    
    Args:
        index: Index of the API key (0-12)
        
    Returns:
        API key string or empty string if not found
    """
    return os.getenv(f"TOGETHER_AI_API_KEY_{index}", "")


# For backward compatibility, create the same structure as the original config
def get_api_key_config() -> Dict[str, Dict[str, str]]:
    """
    Get API keys in the same format as the original config.yaml structure.
    This maintains backward compatibility with existing code.
    
    Returns:
        Dictionary with the same structure as the original APIKey section
    """
    api_keys = {}
    
    # Add Together AI keys
    for i in range(13):
        key = os.getenv(f"TOGETHER_AI_API_KEY_{i}")
        if key:
            api_keys[f"TOGETHER_AI_API_KEY_{i}"] = key
    
    # Add Google API keys
    for i in range(27):
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            api_keys[f"GOOGLE_API_KEY_{i}"] = key
    
    # Add OMDB API key
    omdb_key = os.getenv("OMDB_API_KEY")
    if omdb_key:
        api_keys["OMDB_API_KEY"] = omdb_key
    
    return api_keys


if __name__ == "__main__":
    # Test the configuration loader
    print("Testing configuration loader...")
    
    together_keys = load_together_api_keys()
    print(f"Loaded {len(together_keys)} Together AI API keys")
    
    google_keys = load_google_api_keys()
    print(f"Loaded {len(google_keys)} Google API keys")
    
    omdb_key = load_omdb_api_key()
    print(f"OMDB API key loaded: {'Yes' if omdb_key else 'No'}")
    
    # Test backward compatibility
    api_config = get_api_key_config()
    print(f"Total API keys loaded: {len(api_config)}")
