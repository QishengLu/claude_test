import os
import sys
import subprocess
import argparse

def main():
    """
    Top-level execution script for the RCA Agent.
    Runs the agent logic located in src/rca_agent.py without importing internal packages directly.
    """
    parser = argparse.ArgumentParser(description="Run RCA Agent")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--output_path", type=str, default="experiments/claude/output.json", help="Path to save output JSON")
    args = parser.parse_args()

    # Get the absolute path of the current directory (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define potential paths to the agent script
    # 1. Same directory (standalone mode)
    # 2. Sibling directory 'claude_test' (when running from run_scripts)
    possible_paths = [
        os.path.join(project_root, "src", "rca_agent.py"),
        os.path.join(project_root, "..", "claude_test", "src", "rca_agent.py")
    ]
    
    agent_script = None
    for path in possible_paths:
        if os.path.exists(path):
            agent_script = os.path.abspath(path)
            break
    
    # Check if the script exists
    if not agent_script:
        print(f"Error: Agent script not found. Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
        
    print(f"Starting RCA Agent from: {agent_script}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output path: {args.output_path}")
    
    # Prepare the environment
    env = os.environ.copy()
    # Ensure PYTHONPATH includes the project root if needed
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    # Run the agent script as a subprocess
    try:
        subprocess.run([sys.executable, agent_script, "--data_dir", args.data_dir, "--output_path", args.output_path], cwd=project_root, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running agent: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nAgent execution interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
