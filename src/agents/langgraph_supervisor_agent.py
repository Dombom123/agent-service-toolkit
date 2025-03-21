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

# Load prompts from store or use defaults
math_prompt = get_prompt("langgraph_supervisor_agent_prompt_1")
math_prompt_text = math_prompt.content if math_prompt else "You are a math expert. Always use one tool at a time."

research_prompt = get_prompt("langgraph_supervisor_agent_prompt_2")
research_prompt_text = research_prompt.content if research_prompt else "You are a world class researcher with access to web search. Do not do any math."

supervisor_prompt = get_prompt("langgraph_supervisor_agent_prompt_3")
supervisor_prompt_text = supervisor_prompt.content if supervisor_prompt else (
    "You are a team supervisor managing a research expert and a math expert. "
    "For current events, use research_agent. "
    "For math problems, use math_agent."
)

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt=math_prompt_text,
).with_config(tags=["skip_stream"])

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt=research_prompt_text,
).with_config(tags=["skip_stream"])

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=supervisor_prompt_text,
    add_handoff_back_messages=False,
)

langgraph_supervisor_agent = workflow.compile(checkpointer=MemorySaver())
