import sys
import os

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.settings.config import settings

print(f"AWS_LLM_ACCESS_KEY_ID: '{settings.llm.AWS_LLM_ACCESS_KEY_ID}'")
print(f"AWS_LLM_SECRET_ACCESS_KEY: '{settings.llm.AWS_LLM_SECRET_ACCESS_KEY}'")
print(f"AWS_LLM_REGION: '{settings.llm.AWS_LLM_REGION}'")
