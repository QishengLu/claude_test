import json
import os
import sys
import time
import requests
from typing import List, Dict, Any, Optional
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
        self.history: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        
        # Three parquet tools - OpenAI function format for OpenRouter
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_tables_in_directory",
                    "description": "List all parquet files in the specified directory with metadata including filename, path, row_count, and column_count. Use this first to discover available data tables.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "The directory path to search for parquet files. Default is 'data'."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Get the schema (column names and types) of a parquet file. Returns column names, data types, and row count. Use this to understand table structure before querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parquet_file": {
                                "type": "string",
                                "description": "The path to the parquet file (e.g., 'data/abnormal_logs.parquet')."
                            }
                        },
                        "required": ["parquet_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_parquet_files",
                    "description": """Execute SQL queries on parquet files using DuckDB. 
The parquet files are registered as views using their base filename (without .parquet extension).
Use standard SQL syntax. If column names contain dots or special characters, enclose them in double quotes.
Example: SELECT service_name, "attr.status_code" FROM abnormal_logs WHERE level = 'ERROR' LIMIT 10""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parquet_files": {
                                "type": "string",
                                "description": "Path to the parquet file(s) to query. Can be a single path (e.g., 'data/logs.parquet') or comma-separated paths."
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute. Use the parquet filename without extension as the table name (e.g., 'abnormal_logs' for 'abnormal_logs.parquet')."
                            }
                        },
                        "required": ["parquet_files", "query"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool call locally."""
        try:
            print(f"\n[Tool Call] {tool_name}")
            print(f"[Input] {json.dumps(tool_args, ensure_ascii=False)}")
            
            if tool_name == "list_tables_in_directory":
                directory = tool_args.get("directory", self.data_dir)
                result = list_tables_in_directory(directory)
            
            elif tool_name == "get_schema":
                parquet_file = tool_args.get("parquet_file", "")
                if not parquet_file:
                    result = json.dumps({"error": "parquet_file parameter is required"})
                else:
                    result = get_schema(parquet_file)
            
            elif tool_name == "query_parquet_files":
                parquet_files = tool_args.get("parquet_files", "")
                query = tool_args.get("query", "")
                if not parquet_files or not query:
                    result = json.dumps({"error": "parquet_files and query parameters are required"})
                else:
                    # Handle comma-separated files
                    if isinstance(parquet_files, str) and "," in parquet_files:
                        parquet_files = [f.strip() for f in parquet_files.split(",")]
                    result = query_parquet_files(parquet_files, query)
            
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            
            # Truncate for display
            display_result = result[:2000] + "..." if len(result) > 2000 else result
            print(f"[Result] {display_result}")
            
            return result
                
        except Exception as e:
            error_result = json.dumps({"error": f"Tool execution failed: {str(e)}"})
            print(f"[Error] {error_result}")
            return error_result

    def call_claude_api(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call Claude API via OpenRouter with extended thinking mode.
        """
        print("\n[System] Calling Claude API via OpenRouter (Extended Thinking)...")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENROUTER_API_KEY in .env")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "RCA Agent"),
        }
        
        payload = {
            "model": "anthropic/claude-sonnet-4",
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",  # Required for thinking mode
            "max_tokens": 2000,  # Reduced for limited credits
            "thinking": {
                "type": "enabled",
                "budget_tokens": 500  # Reduced thinking budget
            }
        }
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"OpenRouter API call failed: {e}\nResponse: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API call failed: {e}")

    def process_response(self, response: Dict[str, Any]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Process OpenRouter/Claude response and extract content/tool calls.
        Returns (final_text, tool_calls)
        """
        if "error" in response:
            raise Exception(f"API Error: {response['error']}")
        
        choices = response.get("choices", [])
        if not choices:
            raise Exception("No choices in response")
        
        message = choices[0].get("message", {})
        
        # Debug: print raw message structure
        print(f"\n[Debug] Message keys: {message.keys()}")
        
        # Extract and display thinking content if present
        if "thinking" in message:
            thinking = message["thinking"]
            print(f"\n[Thinking] {thinking[:1500]}..." if len(thinking) > 1500 else f"\n[Thinking] {thinking}")
        
        # Check for tool calls - handle different possible formats
        tool_calls = message.get("tool_calls", [])
        
        # Debug: print tool_calls structure if present
        if tool_calls:
            print(f"[Debug] Tool calls structure: {json.dumps(tool_calls, indent=2)[:500]}")
        
        content = message.get("content", "") or ""
        
        if tool_calls:
            return content, tool_calls
        
        return content, []

    def save_history(self, output_path: str = "experiments/claude/output.json"):
        """Save the conversation history to a JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"\n[System] History saved to {output_path}")
        except Exception as e:
            print(f"\n[System] Error saving history: {e}")

    def run(self, max_iterations: int = 30, output_path: str = "experiments/claude/output.json"):
        """
        Run the RCA Agent with Claude's extended thinking mode via OpenRouter.
        """
        print("=" * 70)
        print("  RCA Agent - Claude Extended Thinking Mode (OpenRouter)")
        print("=" * 70)
        
        # Get system prompt from YAML config
        system_prompt = get_system_prompt()
        
        # Initialize conversation
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please begin the root cause analysis. Start by discovering available data tables using list_tables_in_directory."}
        ]
        
        self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": self.messages[1]["content"]})
        
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                print(f"\n{'=' * 60}")
                print(f"  Iteration {iteration}")
                print(f"{'=' * 60}")
                
                # Call Claude API
                response = self.call_claude_api(self.messages)
                
                # Process response
                content, tool_calls = self.process_response(response)
                
                # If we have content but no tool calls, we're done
                if content and not tool_calls:
                    print("\n" + "=" * 60)
                    print("  FINAL ANALYSIS")
                    print("=" * 60)
                    print(content)
                    
                    self.messages.append({"role": "assistant", "content": content})
                    self.history.append({"role": "assistant", "content": content})
                    break
                
                # Process tool calls
                if tool_calls:
                    # Build assistant message with tool calls
                    assistant_message = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    }
                    self.messages.append(assistant_message)
                    
                    # Log tool calls for history
                    tool_calls_log = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tool_calls_log.append({
                            "name": func.get("name", "unknown"),
                            "arguments": func.get("arguments", "{}")
                        })
                    
                    self.history.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls_log
                    })
                    
                    # Execute each tool call
                    for tool_call in tool_calls:
                        func = tool_call.get("function", {})
                        tool_name = func.get("name", "")
                        tool_args_str = func.get("arguments", "{}")
                        tool_call_id = tool_call.get("id", "")
                        
                        try:
                            tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                        except json.JSONDecodeError:
                            tool_args = {}
                        
                        # Execute tool
                        result = self.execute_tool(tool_name, tool_args)
                        
                        # Add tool result message
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result
                        })
                        
                        self.history.append({
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result
                        })
                
                time.sleep(0.5)
            
            if iteration >= max_iterations:
                print(f"\n[Warning] Reached maximum iterations ({max_iterations})")
                
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()
        
        # Save history
        self.save_history(output_path)
        print("\n[System] RCA Agent completed.")


if __name__ == "__main__":
    agent = RCAAgent()
    agent.run()
