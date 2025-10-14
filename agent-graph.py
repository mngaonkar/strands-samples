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

# Test Github MCP client directly
with mcp_client_github:
    result = mcp_client_github.call_tool_sync(
        tool_use_id="tool-123",
        name="search_repositories",
        arguments={"query": "mngaonkar"}
    )

all_tools = [calculator, current_time, use_aws, shell.shell]


# Add MCP tools to the tools list
with mcp_client_aws_docs:
    aws_tools = mcp_client_aws_docs.list_tools_sync()
    for tool in aws_tools:
        all_tools.append(tool)

with mcp_client_github:
    github_tools = mcp_client_github.list_tools_sync()  
    for tool in github_tools:
        all_tools.append(tool)


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
print("🚀 Starting multi-agent task execution workflow...")
print("="*80)

# result = graph("list logs for EKS cluster sliverblaze and find any errors")
# result = graph("find EKS cluster nodes")
# result = graph("list EKS clusters")
result = graph("find any issues with EKS cluster sliverblaze")

# result = graph("find any errors in pod logs in flix-finder namespace")

# print("="*80)
# print("🎯 FINAL WORKFLOW RESULT:")
# print(result.execution_order)
# print("="*80)