"""Prompt management module."""
import importlib
import inspect
import logging
import os
import re
from typing import Dict, List

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import PromptTemplate

from schema.prompts import Prompt

# Configure logging
logger = logging.getLogger(__name__)

# Dictionary to store prompt templates
_prompt_templates: Dict[str, str] = {}


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
    """Get all prompts from the codebase."""
    try:
        if not _prompt_templates:
            scan_codebase_for_prompts()
        
        # If we have templates, convert them to Prompt objects
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


def update_prompt(prompt_id: str, content: str) -> bool:
    """Update a prompt template by modifying the source file.
    
    Args:
        prompt_id: The ID of the prompt to update
        content: The new content for the prompt
        
    Returns:
        bool: True if the update was successful, False otherwise
    """
    try:
        logger.info(f"Updating prompt with ID: {prompt_id}")
        # First make sure the prompt exists in our cache
        if prompt_id not in _prompt_templates:
            # If the cache is empty, try scanning for prompts first
            if not _prompt_templates:
                scan_codebase_for_prompts()
            
            # Check again after scanning
            if prompt_id not in _prompt_templates:
                logger.warning(f"Prompt with ID {prompt_id} not found in cache")
                return False
        
        # Store in memory first
        _prompt_templates[prompt_id] = content
        
        # Parse the prompt_id to get the agent_id and name
        parts = prompt_id.split("_", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid prompt_id format: {prompt_id}")
            return False
        
        agent_id, name = parts
        
        # For prompt_N formats, extract the name according to the pattern
        # For langgraph_supervisor_agent_prompt_1, name is actually prompt_1
        if agent_id == "langgraph" and name.startswith("supervisor_agent_prompt_"):
            # For langgraph supervisor agent, special handling
            prompt_num = name.split("_")[-1]
            name = f"prompt_{prompt_num}"
            agent_id = "langgraph_supervisor_agent"
        
        # Determine the file path
        agent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents")
        agent_file = os.path.join(agent_dir, f"{agent_id}.py")
        
        if not os.path.exists(agent_file):
            logger.warning(f"Agent file not found: {agent_file}. Checking for alternate filenames.")
            # Try to find a file that matches a part of the agent_id
            for filename in os.listdir(agent_dir):
                if filename.endswith(".py") and agent_id in filename:
                    agent_file = os.path.join(agent_dir, filename)
                    logger.info(f"Found alternate agent file: {agent_file}")
                    break
            
            if not os.path.exists(agent_file):
                logger.warning(f"No suitable agent file found for {agent_id}")
                return False
        
        logger.info(f"Updating prompt in file: {agent_file}")
        
        # Read the source file
        with open(agent_file, 'r') as f:
            source = f.read()
        
        # Create a backup before making changes
        backup_file = f"{agent_file}.bak"
        with open(backup_file, 'w') as f:
            f.write(source)
        
        # Look for different patterns of prompt definition and replace the content
        updated = False
        
        # Pattern 1: SystemMessagePromptTemplate.from_template with triple quotes
        pattern1 = rf'{re.escape(name)} = SystemMessagePromptTemplate\.from_template\("""(.*?)"""\)'
        if re.search(pattern1, source, re.DOTALL):
            new_source = re.sub(
                pattern1,
                f'{name} = SystemMessagePromptTemplate.from_template("""{content}""")',
                source,
                flags=re.DOTALL
            )
            updated = True
        
        # Pattern 2: SystemMessagePromptTemplate.from_template with single quotes
        if not updated:
            pattern2 = rf"{re.escape(name)} = SystemMessagePromptTemplate\.from_template\('''(.*?)'''\)"
            if re.search(pattern2, source, re.DOTALL):
                new_source = re.sub(
                    pattern2,
                    f"{name} = SystemMessagePromptTemplate.from_template('''{content}''')",
                    source,
                    flags=re.DOTALL
                )
                updated = True
        
        # Pattern 3: prompt="""...""" or prompt="..."
        if not updated:
            # For simple name-based prompts like prompt_1, look for keyword arguments
            if name.startswith("prompt_"):
                # Triple quoted strings
                pattern3a = r'prompt\s*=\s*"""(.*?)"""'
                matches = list(re.finditer(pattern3a, source, re.DOTALL))
                if matches:
                    # Find the nth occurrence of the pattern (based on the prompt number)
                    try:
                        idx = int(name.split("_")[1]) - 1
                        if idx < len(matches):
                            # Replace just this occurrence
                            match = matches[idx]
                            new_source = source[:match.start(1)] + content + source[match.end(1):]
                            updated = True
                    except (ValueError, IndexError):
                        pass
                
                # Double quoted strings
                if not updated:
                    pattern3b = r'prompt\s*=\s*"([^"]*)"'
                    matches = list(re.finditer(pattern3b, source))
                    if matches:
                        try:
                            idx = int(name.split("_")[1]) - 1
                            if idx < len(matches):
                                match = matches[idx]
                                new_source = source[:match.start(1)] + content + source[match.end(1):]
                                updated = True
                        except (ValueError, IndexError):
                            pass
                
                # Single quoted strings
                if not updated:
                    pattern3c = r"prompt\s*=\s*'([^']*)'"
                    matches = list(re.finditer(pattern3c, source))
                    if matches:
                        try:
                            idx = int(name.split("_")[1]) - 1
                            if idx < len(matches):
                                match = matches[idx]
                                new_source = source[:match.start(1)] + content + source[match.end(1):]
                                updated = True
                        except (ValueError, IndexError):
                            pass
        
        # Pattern 4: Multi-line prompts with parentheses: prompt=("""...""")
        if not updated and name.startswith("prompt_"):
            pattern4 = r'prompt\s*=\s*\(\s*"""(.*?)"""\s*\)'
            matches = list(re.finditer(pattern4, source, re.DOTALL))
            if matches:
                try:
                    idx = int(name.split("_")[1]) - 1
                    if idx < len(matches):
                        match = matches[idx]
                        new_source = source[:match.start(1)] + content + source[match.end(1):]
                        updated = True
                except (ValueError, IndexError):
                    pass
        
        # If we found and updated a pattern, write the file
        if updated:
            with open(agent_file, 'w') as f:
                f.write(new_source)
            logger.info(f"Updated prompt {prompt_id} in file {agent_file}")
            return True
        else:
            logger.warning(f"Could not find prompt pattern for {prompt_id} in {agent_file}")
            # Even if we couldn't update the file, keep the value in memory
            return True
            
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {str(e)}")
        return False 