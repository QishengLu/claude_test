import yaml
import os

def get_system_prompt() -> str:
    yaml_path = os.path.join(os.path.dirname(__file__), 'system_prompt.yaml')
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
            
        return f"""
{prompt_data.get('role', '')}

{prompt_data.get('task_description', '')}

{prompt_data.get('tools_description', '')}

{prompt_data.get('protocol', '')}
"""
    except Exception as e:
        print(f"Error loading system prompt from yaml: {e}")
        return "Error loading system prompt."
