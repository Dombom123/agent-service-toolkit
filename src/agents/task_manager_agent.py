from typing import List, Dict, Any, Annotated, Literal
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import RemainingSteps

from core import get_model, settings

# Global tasks list (to be managed by the agent)
_TASKS = []

# Define our state
class TaskState(MessagesState, total=False):
    """State for task management agent."""
    remaining_steps: RemainingSteps

# Define our tools
def create_task(title: str, description: str) -> str:
    """Create a new task with title and description."""
    global _TASKS
    task = {"title": title, "description": description}
    _TASKS.append(task)
    return f"Created task: {title} - {description}"

def list_tasks() -> str:
    """List all current tasks."""
    global _TASKS
    if not _TASKS:
        return "No tasks found."
    return "Current tasks:\n" + "\n".join([f"{i}. {task['title']}: {task['description']}" for i, task in enumerate(_TASKS)])

def complete_task(task_index: int) -> str:
    """Complete a task by its index."""
    global _TASKS
    if 0 <= task_index < len(_TASKS):
        task = _TASKS.pop(task_index)
        return f"Completed task: {task['title']}"
    return f"Invalid task index. Please provide a number between 0 and {len(_TASKS)-1}."

# Initialize tools
tools = [
    create_task,
    list_tasks,
    complete_task
]

# System prompt
SYSTEM_PROMPT = """You are a helpful task management assistant. You can help users create, list, and complete tasks.
Use the provided tools to manage tasks effectively.

Available tools:
- create_task: Create a new task with title and description
- list_tasks: List all current tasks (no parameters needed)
- complete_task: Complete a task by its index (numbered starting from 0)

Always think step by step:
1. Understand the user's request
2. Choose the appropriate tool
3. Execute the tool
4. Provide a clear response

Remember to:
- Keep track of task indices when completing tasks (they start from 0)
- Provide clear feedback after each action
- Help users organize their tasks effectively
"""

def wrap_model(model) -> RunnableSerializable[TaskState, AIMessage]:
    """Wrap the model with system prompt and tools."""
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: TaskState, config: RunnableConfig) -> TaskState:
    """Call the model with the current state."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    
    try:
        response = await model_runnable.ainvoke(state, config)
        
        # Check remaining steps
        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        
        # If the LLM returns an empty response, we will re-prompt it
        if not response.tool_calls and (not response.content or isinstance(response.content, list) and not response.content[0].get("text")):
            messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
            state = {**state, "messages": messages}
            response = await model_runnable.ainvoke(state, config)
        
        return {"messages": [response]}
    except Exception as e:
        # Handle any errors gracefully
        error_message = f"Error in model call: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}

# Define the graph
agent = StateGraph(TaskState)

# Add nodes
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))

# Set entry point
agent.set_entry_point("model")

# Define conditional edges
def should_use_tools(state: TaskState) -> Literal["tools", "done"]:
    """Determine if we should use tools or end."""
    try:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return "done"
        if last_message.tool_calls:
            return "tools"
        return "done"
    except Exception as e:
        # Handle any errors gracefully
        return "done"

# Add edges
agent.add_conditional_edges(
    "model",
    should_use_tools,
    {"tools": "tools", "done": END}
)

# Always run model after tools
agent.add_edge("tools", "model")

# Compile the agent with memory saver
task_manager_agent = agent.compile(checkpointer=MemorySaver()) 