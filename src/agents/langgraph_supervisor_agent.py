from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from core import get_model, settings
from service.prompts import get_prompt

model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


def get_math_prompt():
    """Get the math agent prompt dynamically."""
    prompt = get_prompt("langgraph-supervisor-agent_prompt_1")
    if prompt:
        return prompt.content
    return "You are a math expert. Always use one tool at a time."


def get_research_prompt():
    """Get the research agent prompt dynamically."""
    prompt = get_prompt("langgraph-supervisor-agent_prompt_2")
    if prompt:
        return prompt.content
    return "You are a world class researcher with access to web search. Do not do any math."


def get_supervisor_prompt():
    """Get the supervisor prompt dynamically."""
    prompt = get_prompt("langgraph-supervisor-agent_prompt_3")
    if prompt:
        return prompt.content
    return (
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )


def create_supervisor_agent():
    """Create the supervisor agent with the latest prompts."""
    math_agent = create_react_agent(
        model=model,
        tools=[add, multiply],
        name="math_expert",
        prompt=get_math_prompt(),
    ).with_config(tags=["skip_stream"])

    research_agent = create_react_agent(
        model=model,
        tools=[web_search],
        name="research_expert",
        prompt=get_research_prompt(),
    ).with_config(tags=["skip_stream"])

    # Create supervisor workflow
    workflow = create_supervisor(
        [research_agent, math_agent],
        model=model,
        prompt=get_supervisor_prompt(),
        add_handoff_back_messages=False,
    )

    return workflow.compile(checkpointer=MemorySaver())


# Create the agent with the current prompts
langgraph_supervisor_agent = create_supervisor_agent()
