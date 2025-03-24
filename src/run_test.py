#!/usr/bin/env python
"""
Test script for the character agent.
Run this directly to test the character agent with a simple interaction.
"""

import asyncio
import sys
import random
import string
import time
from typing import Optional
from langchain.schema import HumanMessage
from core import settings
from agents.character_agent import build_character_agent, test_character_agent

async def run_test(model_name: Optional[str] = None):
    """Run a simple test of both character agents.
    
    Args:
        model_name: Optional model name to use for the test
    """
    if model_name:
        settings.DEFAULT_MODEL = model_name
    
    # First test Frank
    print("\n=== Testing Frank Character Agent ===")
    await test_character_agent("frank")
    
    # Then test Lisa
    print("\n=== Testing Lisa Character Agent ===")
    await test_character_agent("lisa")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test character agents")
    parser.add_argument("--model", help="Model to use for testing", default=None)
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(run_test(args.model)) 