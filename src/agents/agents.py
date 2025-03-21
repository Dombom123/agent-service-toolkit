from dataclasses import dataclass
import importlib
import logging

from langgraph.graph.state import CompiledStateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.interrupt_agent import interrupt_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent, create_supervisor_agent
from agents.research_assistant import research_assistant
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "command-agent": Agent(description="A command agent.", graph=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="A langgraph supervisor agent", graph=langgraph_supervisor_agent
    ),
    "interrupt-agent": Agent(description="An agent the uses interrupts.", graph=interrupt_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]


def reload_agent(agent_id: str) -> bool:
    """Reload an agent to pick up prompt changes.
    
    Args:
        agent_id: The ID of the agent to reload
        
    Returns:
        bool: True if reload was successful
    """
    try:
        if agent_id not in agents:
            logger.warning(f"Agent {agent_id} not found for reload")
            return False
            
        # For langgraph-supervisor-agent, we have a special reload function
        if agent_id == "langgraph-supervisor-agent":
            agents[agent_id].graph = create_supervisor_agent()
            logger.info(f"Reloaded agent {agent_id} with new prompts")
            return True
            
        # For other agents, we would need to implement similar functionality
        # This is a placeholder for future implementation
        logger.warning(f"Reload not implemented for agent {agent_id}")
        return False
            
    except Exception as e:
        logger.error(f"Error reloading agent {agent_id}: {str(e)}")
        return False
