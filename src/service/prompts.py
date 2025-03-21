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

def _normalize_agent_id(agent_id: str) -> str:
    """Normalize agent ID by converting underscores to hyphens."""
    return agent_id.replace('_', '-')

def _denormalize_agent_id(agent_id: str) -> str:
    """Denormalize agent ID by converting hyphens to underscores (for filenames)."""
    return agent_id.replace('-', '_')

def scan_codebase_for_prompts() -> List[Prompt]:
    """Scan the codebase for prompt templates and return them as Prompt objects."""
    prompts = []
    
    try:
        # Scan agent directory
        agent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents")
        
        if not os.path.exists(agent_dir):
            logger.warning(f"Agent directory not found: {agent_dir}")
            return prompts
            
        logger.info(f"Scanning agent directory: {agent_dir}")
        for filename in os.listdir(agent_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"agents.{filename[:-3]}"
                try:
                    logger.info(f"Processing module: {module_name}")
                    module = importlib.import_module(module_name)
                    # Extract the agent_id from the filename, but normalize it to match
                    # the format in agents.py (using hyphens instead of underscores)
                    agent_id = _normalize_agent_id(filename[:-3])
                    logger.info(f"Agent ID: {agent_id}")
                    
                    # Get the source code for regex matching
                    try:
                        module_source = inspect.getsource(module)
                    except Exception as e:
                        logger.warning(f"Could not get source for {module_name}: {str(e)}")
                        module_source = ""
                    
                    # Look for prompt templates in the module
                    logger.info(f"Looking for prompts in {module_name}")
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
                    logger.info(f"Looking for prompt kwargs in {module_name}")
                    prompt_kwargs = re.findall(r'prompt\s*=\s*"([^"]*)"', module_source)
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*'([^']*)'", module_source))
                    
                    # Pattern for: prompt=("""text""") or prompt=('''text''')
                    prompt_kwargs.extend(re.findall(r'prompt\s*=\s*\(\s*"""(.*?)"""', module_source, re.DOTALL))
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*\(\s*'''(.*?)'''", module_source, re.DOTALL))
                    
                    # Pattern for: prompt=("text") or prompt=('text')
                    prompt_kwargs.extend(re.findall(r'prompt\s*=\s*\(\s*"([^"]*)"', module_source))
                    prompt_kwargs.extend(re.findall(r"prompt\s*=\s*\(\s*'([^']*)'", module_source))
                    
                    # Look for hardcoded prompts in return statements
                    # Pattern for functions like get_math_prompt() that return a default prompt
                    hardcoded_prompts = re.findall(r'def get_(\w+)_prompt\(\).*?return\s+["\']([^"\']+)["\']', module_source, re.DOTALL)
                    hardcoded_prompts.extend(re.findall(r'def get_(\w+)_prompt\(\).*?return\s+\(\s*["\']([^"\']+)["\']', module_source, re.DOTALL))
                    
                    # For triple-quoted strings in returns
                    triple_quoted = re.findall(r'def get_(\w+)_prompt\(\).*?return\s+"""(.*?)"""', module_source, re.DOTALL)
                    triple_quoted.extend(re.findall(r"def get_(\w+)_prompt\(\).*?return\s+'''(.*?)'''", module_source, re.DOTALL))
                    
                    # For parenthesized triple-quoted strings in returns
                    parenthesized_triple = re.findall(r'def get_(\w+)_prompt\(\).*?return\s+\(\s*"""(.*?)"""', module_source, re.DOTALL)
                    parenthesized_triple.extend(re.findall(r"def get_(\w+)_prompt\(\).*?return\s+\(\s*'''(.*?)'''", module_source, re.DOTALL))
                    
                    # Process the hardcoded prompts
                    for prompt_type, prompt_content in hardcoded_prompts:
                        prompt_id = f"{agent_id}_prompt_{prompt_type}"
                        prompt_name = f"prompt_{prompt_type}"
                        logger.info(f"Creating hardcoded prompt: {prompt_id}")
                        _prompt_templates[prompt_id] = prompt_content
                        prompts.append(
                            Prompt(
                                id=prompt_id,
                                name=prompt_name,
                                content=prompt_content,
                                description=f"Default {prompt_type} prompt in {agent_id}",
                                agent_id=agent_id,
                            )
                        )
                    
                    # Process triple-quoted hardcoded prompts
                    for prompt_type, prompt_content in triple_quoted + parenthesized_triple:
                        prompt_id = f"{agent_id}_prompt_{prompt_type}"
                        prompt_name = f"prompt_{prompt_type}"
                        logger.info(f"Creating triple-quoted hardcoded prompt: {prompt_id}")
                        _prompt_templates[prompt_id] = prompt_content
                        prompts.append(
                            Prompt(
                                id=prompt_id,
                                name=prompt_name,
                                content=prompt_content,
                                description=f"Default {prompt_type} prompt in {agent_id}",
                                agent_id=agent_id,
                            )
                        )
                    
                    # Look for prompt IDs referenced in get_prompt() calls
                    logger.info(f"Looking for get_prompt() calls in {module_name}")
                    prompt_refs = re.findall(r'get_prompt\(["\']([^"\']+)["\']\)', module_source)
                    logger.info(f"Found {len(prompt_refs)} get_prompt() calls in {module_name}: {prompt_refs}")
                    
                    for prompt_id in prompt_refs:
                        # Check if this ID is already in our prompts
                        if not any(p.id == prompt_id for p in prompts):
                            logger.info(f"Found referenced prompt ID: {prompt_id}")
                            # Try to extract the prompt name and agent ID from the prompt ID
                            ref_agent_id = agent_id  # Default to current agent
                            if "_" in prompt_id:
                                parts = prompt_id.split("_", 1)
                                if len(parts) > 1 and "-" in parts[0]:  
                                    # Handle format like "langgraph-supervisor-agent_prompt_1"
                                    ref_agent_id = parts[0]
                                    ref_name = parts[1]
                                else:
                                    # Handle simple format like "agent_name"
                                    ref_agent_id = agent_id
                                    ref_name = prompt_id
                            else:
                                ref_name = prompt_id
                            
                            logger.info(f"Extracted agent_id: {ref_agent_id}, name: {ref_name} from {prompt_id}")
                            
                            # Look for default content in the same function where get_prompt is called
                            # This regex is more precise in capturing the content of a function that uses get_prompt
                            function_pattern = r'def\s+[^(]+\([^)]*\)[^{]*?get_prompt\(["\']' + re.escape(prompt_id) + r'["\']\)[^}]*?return\s+["\']([^"\']+)["\']'
                            function_matches = re.findall(function_pattern, module_source, re.DOTALL)
                            
                            # Try another pattern for return with parentheses
                            if not function_matches:
                                function_pattern = r'def\s+[^(]+\([^)]*\)[^{]*?get_prompt\(["\']' + re.escape(prompt_id) + r'["\']\)[^}]*?return\s+\(\s*["\']([^"\']+)["\']'
                                function_matches = re.findall(function_pattern, module_source, re.DOTALL)
                            
                            logger.info(f"Looking for default content for {prompt_id}. Found matches: {function_matches}")
                            
                            default_content = ""
                            if function_matches:
                                default_content = function_matches[0]
                            
                            # If we couldn't extract directly, try a simpler regex
                            if not default_content:
                                # Try to find any return statement with a string after checking for the prompt ID
                                simple_pattern = r'get_prompt\(["\']' + re.escape(prompt_id) + r'["\']\).*?return\s+["\']([^"\']+)["\']'
                                simple_matches = re.findall(simple_pattern, module_source, re.DOTALL)
                                if simple_matches:
                                    default_content = simple_matches[0]
                                    logger.info(f"Found default content with simple pattern for {prompt_id}: {default_content[:30]}...")
                            
                            # Create a placeholder prompt if we still don't have content
                            if not default_content:
                                logger.info(f"No default content found for {prompt_id}, using placeholder")
                                default_content = f"Default content for {prompt_id}"
                            
                            _prompt_templates[prompt_id] = default_content
                            prompts.append(
                                Prompt(
                                    id=prompt_id,
                                    name=ref_name,
                                    content=default_content,
                                    description=f"Referenced prompt in {agent_id}",
                                    agent_id=ref_agent_id,
                                )
                            )
                    
                    logger.info(f"Found {len(prompt_kwargs)} keyword prompts in {module_name}")
                    
                    # For each found prompt, create an entry
                    for idx, prompt_content in enumerate(prompt_kwargs):
                        prompt_name = f"prompt_{idx+1}"
                        prompt_id = f"{agent_id}_{prompt_name}"
                        logger.info(f"Creating prompt: {prompt_id}")
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
            
            # Update the source file
            if _update_prompt_in_source_file(prompt_id, content):
                logger.info(f"Updated prompt {prompt_id} in source file")
            else:
                logger.warning(f"Failed to update prompt {prompt_id} in source file")
            
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

def _update_prompt_in_source_file(prompt_id: str, content: str) -> bool:
    """Update the prompt directly in the source file.
    
    Args:
        prompt_id: The ID of the prompt to update
        content: The new content for the prompt
        
    Returns:
        bool: True if the update was successful, False otherwise
    """
    try:
        # Extract agent_id from prompt_id
        parts = prompt_id.split("_", 1)
        agent_id = parts[0] if len(parts) > 0 else None
        
        if not agent_id:
            logger.warning(f"Could not extract agent_id from prompt_id: {prompt_id}")
            return False
        
        # Handle special cases - we know that langgraph-supervisor-agent_prompt_1 corresponds
        # to the math prompt in langgraph_supervisor_agent.py
        if prompt_id == "langgraph-supervisor-agent_prompt_1":
            function_name = "get_math_prompt"
        elif prompt_id == "langgraph-supervisor-agent_prompt_2":
            function_name = "get_research_prompt"
        elif prompt_id == "langgraph-supervisor-agent_prompt_3":
            function_name = "get_supervisor_prompt"
        else:
            # For other prompts, try to find the function name or variable name
            function_name = None
        
        # Normalize agent_id to get the file name
        file_name = _denormalize_agent_id(agent_id)
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents", f"{file_name}.py")
        
        if not os.path.exists(file_path):
            # Try to find the file in a subdirectory
            for root, dirs, files in os.walk(os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents")):
                if f"{file_name}.py" in files:
                    file_path = os.path.join(root, f"{file_name}.py")
                    break
            
            if not os.path.exists(file_path):
                logger.warning(f"Agent file not found: {file_path}")
                return False
        
        logger.info(f"Updating prompt in file: {file_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Create a backup
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(source)
        
        updated = False
        
        # If we have a function name, look for that function and update its default return value
        if function_name:
            logger.info(f"Looking for function: {function_name}")
            
            # Pattern for function with double-quoted string return
            pattern = r'(def\s+' + re.escape(function_name) + r'\s*\(\).*?return\s+)["\']([^"\']+)["\'](.*?)(?=\ndef|\Z)'
            replacement = r'\1"{}"\3'.format(content.replace('\\', '\\\\').replace('"', '\\"'))
            
            new_source, count = re.subn(pattern, replacement, source, flags=re.DOTALL)
            if count > 0:
                logger.info(f"Updated double-quoted string in function {function_name}")
                updated = True
                source = new_source
            else:
                # Try single quotes
                pattern = r"(def\s+" + re.escape(function_name) + r"\s*\(\).*?return\s+)['\"]([^'\"]+)['\"](.*?)(?=\ndef|\Z)"
                replacement = r"\1'{}'\3".format(content.replace('\\', '\\\\').replace("'", "\\'"))
                
                new_source, count = re.subn(pattern, replacement, source, flags=re.DOTALL)
                if count > 0:
                    logger.info(f"Updated single-quoted string in function {function_name}")
                    updated = True
                    source = new_source
        
        # If we haven't updated yet, try to look for the prompt ID directly
        if not updated:
            logger.info(f"Looking for prompt ID: {prompt_id}")
            
            # Pattern for get_prompt call
            pattern = r'(get_prompt\(["\']' + re.escape(prompt_id) + r'["\']\).*?return\s+)["\']([^"\']+)["\'](.*?)(?=\ndef|\Z)'
            replacement = r'\1"{}"\3'.format(content.replace('\\', '\\\\').replace('"', '\\"'))
            
            new_source, count = re.subn(pattern, replacement, source, flags=re.DOTALL)
            if count > 0:
                logger.info(f"Updated double-quoted string with prompt ID {prompt_id}")
                updated = True
                source = new_source
            else:
                # Try single quotes
                pattern = r"(get_prompt\(['\"]" + re.escape(prompt_id) + r"['\"].*?return\s+)['\"](.*?)['\"](.*?)(?=\ndef|\Z)"
                replacement = r"\1'{}'\3".format(content.replace('\\', '\\\\').replace("'", "\\'"))
                
                new_source, count = re.subn(pattern, replacement, source, flags=re.DOTALL)
                if count > 0:
                    logger.info(f"Updated single-quoted string with prompt ID {prompt_id}")
                    updated = True
                    source = new_source
        
        if updated:
            # Write the updated file
            with open(file_path, 'w') as f:
                f.write(source)
            logger.info(f"Successfully updated prompt in file: {file_path}")
            return True
        else:
            logger.warning(f"Could not find a pattern to update in file: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating prompt in source file: {str(e)}")
        return False

# Call initialize store at module load time
_initialize_store() 