import json
import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path to import tools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import list_tables_in_directory, get_schema, query_parquet_files
from prompt.system_prompt import get_system_prompt

class RCAAgent:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.history: List[Dict[str, str]] = []

    def call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call your DeepResearch API or any LLM API here.
        
        Args:
            messages: A list of message dictionaries, e.g., 
                      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        
        Returns:
            The string content of the LLM's response.
        """
        print("\n[System] Calling LLM API...")
        
        api_key = os.getenv("DEEPRESEARCH_API_KEY")
        api_url = os.getenv("DEEPRESEARCH_API_URL")
        model = os.getenv("DEEPRESEARCH_MODEL", "deepresearch-v1")
        
        if not api_key or api_key == "your_api_key_here":
            raise NotImplementedError(
                "Please set your DEEPRESEARCH_API_KEY in the .env file."
            )
            
        if not api_url:
             raise NotImplementedError(
                "Please set your DEEPRESEARCH_API_URL in the .env file."
            )

        # Example integration using requests
        import requests
        max_retries = 3
        retry_delay = 2
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SOTA-agents/api_test", # Optional: For OpenRouter rankings
            "X-Title": "RCA Agent" # Optional: For OpenRouter rankings
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0
                    }
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"[System] Rate limit hit (429). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                raise Exception(f"API Call failed: {e}")
        
        raise Exception("Max retries exceeded for API call.")

    def save_history(self, output_path: str = "experiments/openai/output.json"):
        """Save the conversation history to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"\n[System] History saved to {output_path}")
        except Exception as e:
            print(f"\n[System] Error saving history: {e}")

    def run(self):
        print("Starting RCA Agent...")
        
        # Initial System Prompt
        system_prompt = get_system_prompt()
        self.history.append({"role": "system", "content": system_prompt})
        
        step = 0
        max_steps = 15
        
        import re
        
        while step < max_steps:
            step += 1
            try:
                print(f"\n--- Step {step} ---")
                
                # 1. Get response from LLM
                response = self.call_llm_api(self.history)
                print(f"LLM: {response}")
                
                self.history.append({"role": "assistant", "content": response})
                
                # 2. Check for final answer
                if "Root cause service:" in response:
                    print("\nAnalysis Complete.")
                    print(response)
                    break
                
                # 3. Parse response for tool calls
                # Look for ```json ... ``` blocks
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        tool_call = json.loads(json_str)
                        tool_name = tool_call.get("tool")
                        args = tool_call.get("args", {})
                        
                        print(f"Executing tool: {tool_name} with args: {args}")
                        
                        result = ""
                        if tool_name == "list_tables_in_directory":
                            result = list_tables_in_directory(**args)
                        elif tool_name == "get_schema":
                            result = get_schema(**args)
                        elif tool_name == "query_parquet_files":
                            result = query_parquet_files(**args)
                        else:
                            result = f"Error: Unknown tool {tool_name}"
                        
                        print(f"Tool Result: {result[:200]}... (truncated)") 
                        self.history.append({"role": "user", "content": f"Tool Output: {result}"})
                        
                    except json.JSONDecodeError:
                        print("Error: Failed to parse JSON tool call.")
                        self.history.append({"role": "user", "content": "Error: Invalid JSON format in tool call."})
                else:
                    # If no tool call, prompt the model to continue or use a tool if it hasn't finished
                    if "Root cause service:" not in response:
                         self.history.append({"role": "user", "content": "Please continue your analysis. If you need more data, use a tool."})
                    
            except NotImplementedError as e:
                print(f"\n[!] {e}")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # Save history at the end of the run
        self.save_history()

if __name__ == "__main__":
    agent = RCAAgent()
    agent.run()
