from groq import Groq
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_code_quality",
            "description": "Analyzes code that was already fetched and returns metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path of the file to analyze (must have been fetched first)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_github_file",
            "description": "Fetches a file from a GitHub repository. Provide owner, repo name, and file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub username or org"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "file_path": {"type": "string", "description": "Path to file in repo"}
                },
                "required": ["owner", "repo", "file_path"]
            }
        }
    }
]

# Store fetched files
file_cache = {}

def fetch_github_file(owner, repo, file_path):
    """Fetches raw file content from GitHub"""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
    response = requests.get(url)
    if response.status_code == 200:
        file_cache[file_path] = response.text  # Cache it
        return {"success": True, "file_path": file_path, "size": len(response.text)}
    else:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
        response = requests.get(url)
        if response.status_code == 200:
            file_cache[file_path] = response.text  # Cache it
            return {"success": True, "file_path": file_path, "size": len(response.text)}
        return {"success": False, "error": "File not found"}

def analyze_code_quality(file_path):
    """Basic code metrics - uses cached file"""
    if file_path not in file_cache:
        return {"error": "File not fetched yet. Call fetch_github_file first."}
    
    code = file_cache[file_path]
    lines = code.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith(('//', '#', '/*', '*'))])
    function_count = code.count('function ') + code.count('def ') + code.count('const ') + code.count('let ')
    
    return {
        "file_path": file_path,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "estimated_functions": function_count,
        "note": "Metrics calculated from cached file."
    }

def process_tool_call(tool_name, tool_input):
    if tool_name == "fetch_github_file":
        return fetch_github_file(**tool_input)
    elif tool_name == "analyze_code_quality":
        return analyze_code_quality(**tool_input)

def run_agent(user_message):
    messages = [{"role": "user", "content": user_message},{"role": "system", "content": "You are a code quality analysis agent. Use the tools to fetch and analyze code from GitHub repositories. Call one tool at a time."}]
    print(messages)
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}\n")
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"ITERATION {iteration}")
        print("-" * 60)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.5,
            max_tokens=4096
        )
        print(response)
        assistant_message = response.choices[0].message
        print(assistant_message)
        
        # Check if agent wants to use tools
        if assistant_message.tool_calls:
            # ONLY EXECUTE THE FIRST TOOL (sequential execution)
            tool_call = assistant_message.tool_calls[0]
            
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"TOOL CALL: {tool_name}")
            print(f"Arguments: {tool_args}\n")
            
            result = process_tool_call(tool_name, tool_args)
            
            print(f"Result: {result}\n")
            
            # Add ONLY this tool call to history
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }]
            })
            
            print(messages)
            # Add tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            print(messages)
            
        else:
            # Agent is done
            final_response = assistant_message.content
            print(f"AGENT FINAL RESPONSE:\n{final_response}\n")
            print(f"Total iterations: {iteration}")
            return final_response
    
    return "Max iterations reached"

if __name__ == "__main__":
    print("Starting Code Quality Analyzer Agent with Groq...\n")
    
    run_agent(
        "Analyze the code quality of 'src/core.js' from the 'jquery/jquery' repository. "
        "Give me a quality score out of 10 and explain what makes this legacy code."
    )
    print("\nAgent finished!")
