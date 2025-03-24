from typing import List, Dict, Any, Literal, Annotated, Set
from datetime import datetime
import uuid
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.messages import ToolMessage
from langchain.schema.runnable import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.schema import OutputParserException

from core import get_model, settings
from agents.character_prompts import (
    get_basic_self_prompt,
    get_emotional_self_prompt,
    get_social_self_prompt,
    get_system_prompt,
    set_character
)

# Define our state
class CharacterState(MessagesState, total=False):
    """State for character roleplay agent."""
    remaining_steps: int = 10

# Create character-specific tool functions
def create_character_tools(character_key: str):
    """Create a set of tools specific to a character.
    
    Args:
        character_key: The key of the character to use ("frank" or "lisa")
        
    Returns:
        A list of tools for the specified character
    """
    # Set the character context first
    set_character(character_key)
    
    # Create the basic_self tool for this character
    @tool
    def basic_self(message: str) -> str:
        """
        Consult your basic self about core needs and practical concerns.
        The basic self represents core needs, survival instincts, and practical thinking.
        """
        try:
            model = get_model(settings.DEFAULT_MODEL)
            messages = [
                SystemMessage(content=get_basic_self_prompt()),
                HumanMessage(content=f"Respond to this situation: {message}")
            ]
            response = model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error in basic_self tool: {str(e)}")
            # Use the character name from the prompt context
            character_name = character_key.capitalize()
            return f"As a basic self, I'm thinking about {character_name}'s practical concerns like comfort, safety, and immediate needs."

    # Create the emotional_self tool for this character
    @tool
    def emotional_self(message: str) -> str:
        """
        Consult your emotional self about feelings and desires.
        The emotional self represents feelings, desires, and emotional reactions.
        """
        try:
            model = get_model(settings.DEFAULT_MODEL)
            messages = [
                SystemMessage(content=get_emotional_self_prompt()),
                HumanMessage(content=f"Respond to this situation: {message}")
            ]
            response = model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error in emotional_self tool: {str(e)}")
            # Use the character name from the prompt context
            character_name = character_key.capitalize()
            return f"As an emotional self, I'm feeling a mix of curiosity and caution about this situation, typical of {character_name}'s nature."

    # Create the social_self tool for this character
    @tool
    def social_self(message: str) -> str:
        """
        Consult your social self about social dynamics and relationships.
        The social self represents social awareness, relationship dynamics, and public persona.
        """
        try:
            model = get_model(settings.DEFAULT_MODEL)
            messages = [
                SystemMessage(content=get_social_self_prompt()),
                HumanMessage(content=f"Respond to this situation: {message}")
            ]
            response = model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error in social_self tool: {str(e)}")
            # Use the character name from the prompt context
            character_name = character_key.capitalize()
            return f"As a social self, I'm considering how this might affect {character_name}'s relationships and social standing."

    # Return the tools for this character
    return [basic_self, emotional_self, social_self]

def wrap_model(model, character_key: str) -> RunnableSerializable[CharacterState, AIMessage]:
    """Wrap the model with system prompt and tools.
    
    Args:
        model: The LLM to wrap
        character_key: The character key to use for tools and prompts
        
    Returns:
        A runnable that processes the state and returns an AIMessage
    """
    # Make sure the character context is set
    set_character(character_key)
    
    # Get the character-specific tools
    character_tools = create_character_tools(character_key)
    
    # Make sure the model is configured for JSON mode if using functions
    model_kwargs = {}
    
    # Some models need specific formats for function calling
    if hasattr(model, 'model_name') and 'gpt-' in getattr(model, 'model_name', ''):
        model_kwargs = {"response_format": {"type": "text"}}
    
    # Configure the model with tools and kwargs
    model_with_tools = model.bind_tools(
        character_tools, 
        tool_choice="auto",
        **model_kwargs
    )
    
    # Create a preprocessor to inject the system prompt
    def prepare_messages(state):
        """Prepare messages with system prompt."""
        # Make sure the character context is set before getting the system prompt
        set_character(character_key)
        if not state.get("messages"):
            # Default message if none exists
            return [SystemMessage(content=get_system_prompt())]
        return [SystemMessage(content=get_system_prompt())] + state["messages"]
    
    preprocessor = RunnableLambda(
        prepare_messages,
        name="StateModifier",
    )
    
    # Chain the preprocessor and model
    return preprocessor | model_with_tools

async def acall_model(state: CharacterState, config: RunnableConfig) -> CharacterState:
    """Call the model with the current state."""
    # Get the appropriate model based on configuration
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    character_key = config["configurable"].get("character", "frank")
    
    # Set the character context
    set_character(character_key)
    
    # Get the model
    m = get_model(model_name)
    model_runnable = wrap_model(m, character_key)
    
    try:
        # Ensure state has the minimum required structure
        if "messages" not in state:
            state = {"messages": []}
        
        # Check remaining steps before processing to prevent partial tool usage
        if state.get("remaining_steps", 10) < 4:
            return {
                "messages": [
                    AIMessage(
                        content="I need more processing capacity to respond thoughtfully to this request.",
                    )
                ]
            }
        
        # Invoke the model with the current state
        try:
            response = await model_runnable.ainvoke(state, config)
        except Exception as model_error:
            print(f"Error during model invocation: {str(model_error)}")
            # Fall back to a simpler model call without tools
            fallback_model = get_model(model_name)
            # Get character name (capitalized)
            character_name = character_key.capitalize()
            basic_prompt = f"You are roleplaying as {character_name}. Respond to this message: {state['messages'][-1].content if state['messages'] else 'Hello'}"
            fallback_messages = [SystemMessage(content=basic_prompt)]
            response = await fallback_model.ainvoke(fallback_messages)
        
        # Handle empty responses by prompting the model again
        if not hasattr(response, "tool_calls") or (hasattr(response, "content") and not response.content) or (
            isinstance(response.content, list) and not response.content[0].get("text")):
            # Add a guiding message to help the model generate a proper response
            # Get character name (capitalized)
            character_name = character_key.capitalize()
            messages = state["messages"] + [HumanMessage(content=f"Please respond naturally as {character_name}, considering your inner perspectives.")]
            state_with_guide = {**state, "messages": messages}
            try:
                response = await model_runnable.ainvoke(state_with_guide, config)
            except Exception as retry_error:
                print(f"Error during retry: {str(retry_error)}")
                # If retry fails, provide a simple response
                character_name = character_key.capitalize()
                response = AIMessage(content=f"Hi there, I'm {character_name}. What's on your mind today?")
        
        return {"messages": [response]}
    
    except Exception as e:
        # Print the full error for debugging
        print(f"Outer exception in acall_model: {type(e).__name__}: {str(e)}")
        
        # Provide a helpful error message that maintains character
        character_name = character_key.capitalize()
        error_message = f"Hi, I'm {character_name}. I'm having a moment collecting my thoughts. Let's try again with something specific about my life."
        return {"messages": [AIMessage(content=error_message)]}

# Define the graph
def build_character_agent(character_key: str = "frank", model_name=None, checkpointer=None):
    """Build and return the character agent graph.
    
    Args:
        character_key: The character to use ("frank" or "lisa")
        model_name: Optional model name to use, otherwise uses default
        checkpointer: Optional checkpointer to use, defaults to MemorySaver
    
    Returns:
        Compiled agent graph with character_config property
    """
    agent = StateGraph(CharacterState)
    
    # Set the character context
    set_character(character_key)
    
    # Get character-specific tools
    character_tools = create_character_tools(character_key)
    
    # Add nodes
    agent.add_node("model", acall_model)
    agent.add_node("tools", ToolNode(character_tools))
    
    # Set entry point
    agent.set_entry_point("model")
    
    # Define conditional edges
    def should_use_tools(state: CharacterState) -> Literal["tools", "done"]:
        """Determine if we should use tools or end."""
        try:
            # Get the last message
            last_message = state["messages"][-1]
            
            # If not an AI message, we're done
            if not isinstance(last_message, AIMessage):
                return "done"
                
            # If there are tool calls, use tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
                
            # Otherwise, we're done
            return "done"
        except Exception as e:
            print(f"Error in should_use_tools: {str(e)}")
            # Default to done if there's an error
            return "done"
    
    # Add conditional edges
    agent.add_conditional_edges(
        "model",
        should_use_tools,
        {"tools": "tools", "done": END}
    )
    
    # Always run model after tools
    agent.add_edge("tools", "model")
    
    # Use provided checkpointer or create memory saver
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Create config with character information
    config = {"configurable": {"character": character_key}}
    if model_name:
        config["configurable"]["model"] = model_name
        
    # Compile the agent
    compiled_agent = agent.compile(checkpointer=checkpointer)
    
    # Attach the character config as a property on the compiled agent
    compiled_agent.character_config = config
    
    return compiled_agent

# Create the character agent
def create_character_agent(character_key: str):
    """
    Create a character agent for the specified character.
    
    Args:
        character_key: The key of the character to use ("frank" or "lisa")
        
    Returns:
        The character agent with character_config property
    """
    # Create a dedicated memory saver for this character
    checkpointer = MemorySaver()
    
    # Build and return the agent with the specified character
    return build_character_agent(character_key=character_key, checkpointer=checkpointer)

# Create Frank and Lisa agents
frank_agent = create_character_agent("frank")
lisa_agent = create_character_agent("lisa")

# Simple test function
async def test_character_agent(character_key: str = "frank"):
    """Test the character agent with a simple message."""
    from langgraph.checkpoint.memory import MemorySaver
    
    # Create test agent with the specified character
    test_agent = build_character_agent(character_key=character_key, checkpointer=MemorySaver())
    
    # Set the character for prompt generation
    set_character(character_key)
    
    # Get character name (capitalized)
    character_name = character_key.capitalize()
    
    # Create test message
    test_message = HumanMessage(content=f"Hello {character_name}, tell me about yourself and your work.")
    
    # Create config with required thread_id
    config = test_agent.character_config.copy()
    config["configurable"]["thread_id"] = str(uuid.uuid4())
    
    # Invoke agent with its character config
    result = await test_agent.ainvoke({"messages": [test_message]}, config)
    
    print(f"Test Result for {character_name}:", result)
    return result

# Uncomment to run test
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(test_character_agent("frank"))
#     asyncio.run(test_character_agent("lisa")) 