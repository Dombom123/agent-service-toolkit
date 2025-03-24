import asyncio
import sys
import os

# Add the src directory to the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agents.character_agent import test_character_agent

if __name__ == "__main__":
    print("Testing character agent...")
    result = asyncio.run(test_character_agent())
    print("Test completed!") 