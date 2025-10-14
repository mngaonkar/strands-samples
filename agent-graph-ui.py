import logging
from strands import Agent
from strands import tool as strands_tool
from strands_tools import calculator, current_time, use_aws, mcp_client, shell
from strands.multiagent import GraphBuilder
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from strands.agent.conversation_manager import SlidingWindowConversationManager
from dotenv import load_dotenv
import os, sys, json, yaml
from typing import List, Dict, Any

load_dotenv()

# GitHub token configuration - add GITHUB_PERSONAL_ACCESS_TOKEN to your .env file
# You can create a personal access token at: https://github.com/settings/tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load system prompt from YAML file
def load_system_prompt(file_path="system_prompt.yaml"):
    """Load system prompt and agent configuration from YAML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config["system_prompt"]
    except FileNotFoundError:
        logger.error(f"System prompt file '{file_path}' not found")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return None

# AWS documentation MCP client setup
mcp_client_aws_docs = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx", 
        args=["awslabs.aws-documentation-mcp-server@latest"],
        env={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        }
    )
))

# GitHub MCP client setup
mcp_client_github = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="npx", 
        args=["-y", "@modelcontextprotocol/server-github"],
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        }
    )
))

# Initialize tools list
all_tools = [calculator, current_time, use_aws, shell.shell]

# Try to add MCP tools - graceful handling if MCP clients fail
try:
    with mcp_client_aws_docs:
        aws_tools = mcp_client_aws_docs.list_tools_sync()
        for tool in aws_tools:
            all_tools.append(tool)
    print("✅ AWS MCP tools loaded successfully")
except Exception as e:
    print(f"⚠️ AWS MCP tools failed to load: {e}")

try:
    with mcp_client_github:
        github_tools = mcp_client_github.list_tools_sync()  
        for tool in github_tools:
            all_tools.append(tool)
    print("✅ GitHub MCP tools loaded successfully")
except Exception as e:
    print(f"⚠️ GitHub MCP tools failed to load: {e}")

print(f"📦 Total tools loaded: {len(all_tools)}")


task_decomposer = Agent(
    name = "task_decomposer",
    system_prompt='''You are a task decomposition agent. Break down complex tasks into simpler sub-tasks that can be executed using available tools.

Available tools for execution:
- AWS operations (use_aws tool)
- GitHub operations (GitHub MCP tools)  
- Shell/Bash commands (shell tool) - for kubectl, docker, git, file operations, etc.
- Mathematical calculations (calculator tool)
- Time operations (current_time tool)

The output should be a JSON array with content as {"index": <task index>, "task_name": <task name>, "task_description": <task description>}.

When decomposing tasks, consider which tools will be needed and structure sub-tasks accordingly.''',
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    tools=[]
)

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  # Maximum number of message pairs to keep
)

kubectl_command_agent = Agent(
    name="kubectl_command_agent",
    system_prompt=load_system_prompt("kubectl_command_agent.yaml"),
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    conversation_manager=conversation_manager,
    record_direct_tool_call=True,
    tools=[shell.shell]
)

task_executor = Agent(
    name="task_executor",
    system_prompt='''You are a task execution agent that processes tasks recursively.
    
    When you receive input:
    1. If it's a JSON array of tasks: Execute ALL tasks in the array sequentially
    2. If it's a single task: Execute that task directly
    
    For JSON array processing:
    - Process each task: {"index": <index>, "task_name": <name>, "task_description": <description>}
    - For each task, use appropriate tools to execute it:
      • AWS operations: use_aws tool
      • GitHub operations: GitHub MCP tools
      • Shell/Bash commands: shell tool (for kubectl, docker, git, etc.)
      • Mathematical calculations: calculator tool
      • Time operations: current_time tool
    - Collect all results and return as JSON array:
      [
        {"index": <index>, "task_name": <name>, "result": <result>, "status": "completed/failed"},
        ...
      ]
    
    Execute tasks thoroughly using the most appropriate tools based on task requirements.
    Log progress as you execute each task.''',
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    tools=all_tools
)

# Create a result aggregator agent
result_aggregator = Agent(
    name="result_aggregator",
    system_prompt='''You are a result aggregation agent. You receive the results from multiple executed tasks and need to:
    
    1. Analyze all task execution results
    2. Identify patterns, connections, and insights across the results
    3. Synthesize the information into a comprehensive final answer
    4. Highlight any errors or issues found
    5. Provide actionable recommendations
    
    Present your final analysis in a clear, structured format with:
    - Executive Summary
    - Key Findings
    - Issues/Errors (if any)
    - Recommendations
    ''',
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    tools=[]
)

# Build the simplified graph: decomposer -> executor -> aggregator
builder = GraphBuilder()
builder.add_node(task_decomposer, "task_decomposer")
builder.add_node(task_executor, "task_executor")
builder.add_node(kubectl_command_agent, "kubectl_command_agent")
builder.add_node(result_aggregator, "result_aggregator")

# Set up the linear workflow
builder.add_edge("task_decomposer", "kubectl_command_agent")
builder.add_edge("kubectl_command_agent", "result_aggregator")
builder.set_entry_point("task_decomposer")

# The graph is built above using the simplified linear workflow

# Keep MCP clients active during graph execution
# with mcp_client_aws_docs, mcp_client_github:
graph = builder.build()

# Execute the query
result = graph("find any issues with EKS cluster sliverblaze")

# Extract and format only the result_aggregator output
def format_markdown_output(result):
    """Format the result_aggregator output as proper markdown"""
    if hasattr(result, 'execution_order'):
        # Find the result_aggregator output in the execution order
        for step in result.execution_order:
            if step.get('agent_name') == 'result_aggregator' or 'result_aggregator' in str(step):
                output = step.get('output', step.get('result', str(step)))
                break
        else:
            # If not found in execution_order, use the final result
            output = str(result)
    else:
        output = str(result)
    
    # Clean up the output and ensure proper markdown formatting
    lines = output.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Format headers
        if any(header in line.lower() for header in ['executive summary', 'key findings', 'issues/errors', 'recommendations', 'analysis']):
            if not line.startswith('#'):
                formatted_lines.append(f"## {line}")
            else:
                formatted_lines.append(line)
        # Format bullet points
        elif line.startswith('-') or line.startswith('•'):
            if not line.startswith('- '):
                line = f"- {line[1:].strip()}"
            formatted_lines.append(line)
        # Format numbered lists
        elif len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']:
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

class StrandsAgentManager:
    def __init__(self):
        self.tools = self._load_tools()
        self.conversation_manager = SlidingWindowConversationManager()
        
    def _load_tools(self):
        """Load available tools"""
        all_tools = [calculator, current_time, use_aws, shell.shell]
        
        # Try to add AWS MCP tools
        try:
            with mcp_client_aws_docs:
                aws_tools = mcp_client_aws_docs.list_tools_sync()
                print("✅ AWS MCP tools loaded successfully")
        except Exception as e:
            print(f"⚠️ AWS MCP tools failed to load: {e}")
        
        # Try to add GitHub MCP tools
        try:
            with mcp_client_github:
                github_tools = mcp_client_github.list_tools_sync()
                print("✅ GitHub MCP tools loaded successfully")  
        except Exception as e:
            print(f"⚠️ GitHub MCP tools failed to load: {e}")
        
        print(f"📦 Total tools loaded: {len(all_tools)}")
        return all_tools
        
    def execute_task(self, user_input: str):
        """Execute a task using the multi-agent workflow"""
        
        # Task decomposer agent
        task_decomposer = Agent(
            model="claude-3-5-sonnet-20241022",
            system_prompt=load_system_prompt(),
            tools=self.tools,
            conversation_manager=self.conversation_manager
        )
        
        # Executor agent
        executor = Agent(
            model="claude-3-5-sonnet-20241022", 
            system_prompt=load_system_prompt("kubectl_command_agent.yaml"),
            tools=self.tools,
            conversation_manager=self.conversation_manager
        )
        
        # Result aggregator agent
        result_aggregator = Agent(
            model="claude-3-5-sonnet-20241022",
            system_prompt="""You are an expert result aggregator and report generator. Your job is to:
            1. Take the results from multiple task executions
            2. Analyze and synthesize the findings
            3. Create a comprehensive, well-formatted report
            4. Highlight key issues, patterns, and recommendations
            5. Present information in clear, actionable markdown format
            
            Format your response as proper markdown with:
            - Clear headers (##)
            - Bullet points for lists
            - Code blocks for commands/outputs when relevant
            - Bold text for important findings
            - Organized sections
            
            Focus on being concise but thorough.""",
            tools=self.tools,
            conversation_manager=self.conversation_manager
        )
        
        # Create the graph
        graph_builder = GraphBuilder()
        
        # Add nodes with correct parameter order
        graph_builder.add_node(task_decomposer, "task_decomposer")
        graph_builder.add_node(executor, "executor") 
        graph_builder.add_node(result_aggregator, "result_aggregator")
        
        # Add edges
        graph_builder.add_edge("task_decomposer", "executor")
        graph_builder.add_edge("executor", "result_aggregator")
        
        # Build and execute
        graph = graph_builder.build()
        result = graph(user_input)
        
        return result
    
    def format_markdown_output(self, result):
        """Extract and format the result_aggregator output as clean markdown"""
        output = ""
        
        # Try to extract result_aggregator output specifically
        if hasattr(result, 'execution_order') and result.execution_order:
            for node_name in reversed(result.execution_order):
                if 'result_aggregator' in node_name.lower():
                    if hasattr(result, node_name) and hasattr(getattr(result, node_name), 'messages'):
                        messages = getattr(result, node_name).messages
                        if messages and len(messages) > 0:
                            # Get the last message (response)
                            last_message = messages[-1]
                            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                                output = last_message.content
                                break
                            elif hasattr(last_message, 'text') and isinstance(last_message.text, str):
                                output = last_message.text
                                break
            
            # If not found in execution_order, use the final result
            if not output:
                output = str(result)
        else:
            output = str(result)
        
        # Clean up the output and ensure proper markdown formatting
        lines = output.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
                
            # Format headers
            if any(header in line.lower() for header in ['executive summary', 'key findings', 'issues/errors', 'recommendations', 'analysis']):
                if not line.startswith('#'):
                    formatted_lines.append(f"## {line}")
                else:
                    formatted_lines.append(line)
            # Format numbered lists
            elif len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']:
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

if __name__ == "__main__":
    import sys
    
    # Check if input is provided via command line
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        try:
            user_input = input("What would you like help with? ")
        except EOFError:
            user_input = "What is 2+2?"  # Default test case
    
    agent_manager = StrandsAgentManager()
    result = agent_manager.execute_task(user_input)
    
    # Format and print the result
    formatted_output = agent_manager.format_markdown_output(result)
    print(formatted_output)