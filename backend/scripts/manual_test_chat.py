import sys
import os
import asyncio
import json

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chat.service import ChatService

async def main():
    print("Initializing ChatService...")
    try:
        service = ChatService()
        message = "Hi, I really enjoyed seeing The Matrix. Do you have any similar movies?"
        print(f"User Message: {message}")
        
        print("Calling service.chat()...")
        result = await service.chat(message)
        
        print("\n--- Result ---")
        print(f"Response: >>>{result['response']}<<<")
        print(f"Recommendations count: {len(result['recommendations'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
