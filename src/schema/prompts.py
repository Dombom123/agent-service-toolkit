from pydantic import BaseModel


class Prompt(BaseModel):
    """Schema for a prompt template."""
    id: str
    name: str
    content: str
    description: str = ""
    agent_id: str


class PromptList(BaseModel):
    """Schema for a list of prompts."""
    prompts: list[Prompt]


class UpdatePromptRequest(BaseModel):
    """Schema for updating a prompt."""
    prompt_id: str
    content: str 