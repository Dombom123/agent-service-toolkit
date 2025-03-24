#!/usr/bin/env python
"""
Simple test script for the LLM functionality.
"""

import asyncio
from langchain.schema import HumanMessage, SystemMessage
from core import get_model, settings

async def run_test():
    """Run a simple test of the LLM functionality."""
    print("Loading model...")
    model = get_model(settings.DEFAULT_MODEL)
    
    # Simple messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me a short joke.")
    ]
    
    print("\nSending test prompt...")
    print("-" * 50)
    
    # Invoke the model
    try:
        response = model.invoke(messages)
        print("\nResponse from model:")
        print("-" * 50)
        print(response.content)
    except Exception as e:
        print(f"Error: {str(e)}")
        
    return response

if __name__ == "__main__":
    asyncio.run(run_test()) 