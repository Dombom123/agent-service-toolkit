"""Prompt management module."""
import importlib
import inspect
import json
import logging
import os
import re
from typing import Dict, List, Optional

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import PromptTemplate

from schema.prompts import Prompt

# Configure logging
logger = logging.getLogger(__name__)

# Path to the prompt store JSON file
PROMPT_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prompt_store.json")

# Dictionary to store prompt templates
_prompt_templates: Dict[str, str] = {}

def _ensure_data_dir():
    """Ensure the data directory exists."""
    data_dir = os.path.dirname(PROMPT_STORE_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def _load_from_store() -> Dict[str, Prompt]:
    """Load prompts from the store file."""
    _ensure_data_dir()
    
    if not os.path.exists(PROMPT_STORE_PATH):
        return {}
    
    try:
        with open(PROMPT_STORE_PATH, 'r') as f:
            data = json.load(f)
            prompts = {}
            for item in data:
                prompt = Prompt(**item)
                prompts[prompt.id] = prompt
            return prompts
    except Exception as e:
        logger.error(f"Error loading prompts from store: {str(e)}")
        return {}

def _save_to_store(prompts: List[Prompt]):
    """Save prompts to the store file."""
    _ensure_data_dir()
    
    try:
        data = [prompt.dict() for prompt in prompts]
        with open(PROMPT_STORE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(prompts)} prompts to store")
    except Exception as e:
        logger.error(f"Error saving prompts to store: {str(e)}")

def _initialize_store():
    """Initialize the prompt store with prompts from the codebase if it doesn't exist."""
    if not os.path.exists(PROMPT_STORE_PATH):
        logger.info("Initializing prompt store from codebase")
        prompts = scan_codebase_for_prompts()
        if prompts:
            _save_to_store(prompts)
            return True
    return False

def scan_codebase_for_prompts() -> List[Prompt]:
    """Scan the codebase for prompt templates and return them as Prompt objects."""
    prompts = []
    
    try:
        # Scan agent directory
        agent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents")
        
        if not os.path.exists(agent_dir):
            logger.warning(f"Agent directory not found: {agent_dir}")
            return prompts
            
        for filename in os.listdir(agent_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"agents.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    agent_id = filename[:-3]
                    
                    # Get the source code for regex matching
                    try:
                        module_source = inspect.getsource(module)
                    except Exception as e:
                        logger.warning(f"Could not get source for {module_name}: {str(e)}")
                        module_source = ""
                    
                    # Look for prompt templates in the module
                    for name, obj in inspect.getmembers(module):
                        try:
                            # Handle SystemMessagePromptTemplate objects
                            if isinstance(obj, SystemMessagePromptTemplate):
                                try:
                                    # Look for the assignment of this variable in the source code
                                    var_pattern = r'{} = SystemMessagePromptTemplate\.from_template\("""(.*?)"""\)'.format(re.escape(name))
                                    matches = re.search(var_pattern, module_source, re.DOTALL)
                                    if matches:
                                        template = matches.group(1)
                                    else:
                                        # Try alternative pattern with single quotes
                                        var_pattern = r"{} = SystemMessagePromptTemplate\.from_template\('''(.*?)'''\)".format(re.escape(name))
                                        matches = re.search(var_pattern, module_source, re.DOTALL)
                                        if matches:
                                            template = matches.group(1)
                                        else:
                                            # As a fallback, try to convert the object to a string
                                            # and extract relevant parts
                                            formatted = obj.format()
                                            if hasattr(formatted, "content"):
                                                template = formatted.content
                                            else:
                                                logger.warning(f"Could not extract template from {name}")
                                                continue
                                    
                                    prompt_id = f"{agent_id}_{name}"
                                    _prompt_templates[prompt_id] = template
                                    prompts.append(
                                        Prompt(
                                            id=prompt_id,
                                            name=name,
                                            content=template,
                                            description=f"System prompt for {name} in {agent_id}",
                                            agent_id=agent_id,
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(f"Cannot extract template from SystemMessagePromptTemplate object {name}: {str(e)}")
                            # Handle regular PromptTemplate objects
                            elif isinstance(obj, PromptTemplate):
                                try:
                                    template = obj.template
                                    prompt_id = f"{agent_id}_{name}"
                                    _prompt_templates[prompt_id] = template
                                    prompts.append(
                                        Prompt(
                                            id=prompt_id,
                                            name=name,
                                            content=template,
                                            description=f"Prompt template for {name} in {agent_id}",
                                            agent_id=agent_id,
                                        )
                                    )
                                except AttributeError:
                                    logger.warning(f"PromptTemplate object {name} has no template attribute")
                            # Look for template strings in the module
                            elif isinstance(obj, str) and "prompt" in name.lower() and "\n" in obj:
                                prompt_id = f"{agent_id}_{name}"
                                _prompt_templates[prompt_id] = obj
                                prompts.append(
                                    Prompt(
                                        id=prompt_id,
                                        name=name,
                                        content=obj,
                                        description=f"Prompt string for {name} in {agent_id}",
                                        agent_id=agent_id,
                                    )
                                )
                        except Exception as e:
                            logger.error(f"Error processing prompt {name} in {module_name}: {str(e)}")
                            continue
                    
                    # Look for prompts in function calls using regex
                    # Pattern for: prompt="text" or prompt='text'
                    prompt_kwargs = re.findall(r'prompt\s*=\s*"([^"]*)"', module_source)
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*'([^']*)'", module_source))
                    
                    # Pattern for: prompt=("""text""") or prompt=('''text''')
                    prompt_kwargs.extend(re.findall(r'prompt\s*=\s*\(\s*"""(.*?)"""', module_source, re.DOTALL))
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*\(\s*'''(.*?)'''", module_source, re.DOTALL))
                    
                    # Pattern for: prompt=("text") or prompt=('text')
                    prompt_kwargs.extend(re.findall(r'prompt\s*=\s*\(\s*"([^"]*)"', module_source))
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*\(\s*'([^']*)'", module_source))
                    
                    # For each found prompt, create an entry
                    for idx, prompt_content in enumerate(prompt_kwargs):
                        prompt_id = f"{agent_id}_prompt_{idx+1}"
                        prompt_name = f"prompt_{idx+1}"
                        _prompt_templates[prompt_id] = prompt_content
                        prompts.append(
                            Prompt(
                                id=prompt_id,
                                name=prompt_name,
                                content=prompt_content,
                                description=f"Function prompt in {agent_id}",
                                agent_id=agent_id,
                            )
                        )
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing module {module_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error scanning codebase for prompts: {str(e)}")
    
    return prompts


def get_prompts() -> List[Prompt]:
    """Get all prompts from the store or scan codebase if store is empty."""
    try:
        # First try to load from the store
        store_prompts = _load_from_store()
        
        if store_prompts:
            return list(store_prompts.values())
        
        # If store is empty, scan codebase and initialize store
        if not _prompt_templates:
            prompts = scan_codebase_for_prompts()
            if prompts:
                _save_to_store(prompts)
            return prompts
        
        # If we already have templates in memory, convert them to Prompt objects
        prompts = []
        for prompt_id, template in _prompt_templates.items():
            try:
                # Safely split the prompt_id, defaulting to 'unknown' if it fails
                parts = prompt_id.split("_", 1)
                if len(parts) == 2:
                    agent_id, name = parts
                else:
                    agent_id = "unknown"
                    name = prompt_id
                
                prompts.append(
                    Prompt(
                        id=prompt_id,
                        name=name,
                        content=template,
                        description=f"Prompt for {name} in {agent_id}",
                        agent_id=agent_id,
                    )
                )
            except Exception as e:
                logger.error(f"Error creating Prompt object for {prompt_id}: {str(e)}")
        
        return prompts
    except Exception as e:
        logger.error(f"Error in get_prompts: {str(e)}")
        # Return an empty list instead of raising an exception
        return []

def get_prompt(prompt_id: str) -> Optional[Prompt]:
    """Get a specific prompt by ID."""
    store_prompts = _load_from_store()
    
    if prompt_id in store_prompts:
        return store_prompts[prompt_id]
    
    # If not found in store, try to find it in memory
    prompts = get_prompts()
    for prompt in prompts:
        if prompt.id == prompt_id:
            return prompt
    
    return None

def update_prompt(prompt_id: str, content: str) -> Optional[Prompt]:
    """Update a prompt in the store.
    
    Args:
        prompt_id: The ID of the prompt to update
        content: The new content for the prompt
        
    Returns:
        Optional[Prompt]: The updated prompt or None if update failed
    """
    try:
        logger.info(f"Updating prompt with ID: {prompt_id}")
        
        # Load all prompts from the store
        store_prompts = _load_from_store()
        
        # If store is empty, initialize it first
        if not store_prompts:
            prompts = scan_codebase_for_prompts()
            if prompts:
                _save_to_store(prompts)
                store_prompts = {p.id: p for p in prompts}
        
        # Update the prompt in the store
        if prompt_id in store_prompts:
            prompt = store_prompts[prompt_id]
            prompt.content = content
            _save_to_store(list(store_prompts.values()))
            
            # Also update in memory
            _prompt_templates[prompt_id] = content
            
            logger.info(f"Updated prompt {prompt_id} in store")
            return prompt
        else:
            logger.warning(f"Prompt with ID {prompt_id} not found in store")
            
            # If not in store, also scan the codebase again to ensure it's not missing
            if not _prompt_templates:
                scan_codebase_for_prompts()
            
            # Still need to check if it exists in memory
            if prompt_id in _prompt_templates:
                # Create a new prompt object
                parts = prompt_id.split("_", 1)
                if len(parts) == 2:
                    agent_id, name = parts
                else:
                    agent_id = "unknown"
                    name = prompt_id
                
                prompt = Prompt(
                    id=prompt_id,
                    name=name,
                    content=content,
                    description=f"Prompt for {name} in {agent_id}",
                    agent_id=agent_id,
                )
                
                # Update in memory
                _prompt_templates[prompt_id] = content
                
                # Add to store
                store_prompts[prompt_id] = prompt
                _save_to_store(list(store_prompts.values()))
                
                logger.info(f"Added new prompt {prompt_id} to store")
                return prompt
            
            return None
            
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {str(e)}")
        return None

# Call initialize store at module load time
_initialize_store() 