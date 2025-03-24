from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.character_agent import frank_agent, lisa_agent
from schema import AgentInfo

DEFAULT_AGENT = "frank-character"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "frank-character": Agent(
        description="Frank - Frank Schulz, 35, engineer with his own company and family. Motivated by wanting to be loved, even as an emotional man. Inner monologue: 'I am too much.'",
        graph=frank_agent
    ),
    "lisa-character": Agent(
        description="Lisa - Lisa Schulz, 33, dance teacher and passionate mother. Motivated by constant self-improvement. Inner monologue: 'I am not enough.'",
        graph=lisa_agent
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    """Get the agent graph. The agent's character_config is available as a property on the graph."""
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
