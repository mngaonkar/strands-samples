import logging
from strands import Agent, tool
from strands_tools import calculator, current_time, use_aws, mcp_client
# from strands.models.ollama import OllamaModel
# from strands.models.openai import OpenAIModel
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from dotenv import load_dotenv
import os, sys,json

load_dotenv()

# GitHub token configuration - add GITHUB_PERSONAL_ACCESS_TOKEN to your .env file
# You can create a personal access token at: https://github.com/settings/tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model = OllamaModel(model_id="qwen3:8b",
#                     host="http://localhost:11434")

# model = OpenAIModel(model_id="gpt-4o")

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

all_tools = [calculator, current_time, use_aws]

with mcp_client_aws_docs:
    tools = mcp_client_aws_docs.list_tools_sync()
    print("Available AWS Documentation MCP Tools:")
    for tool in tools:
        print(f"- {tool.tool_name}")
        all_tools.append(tool)

with mcp_client_github:
    tools = mcp_client_github.list_tools_sync()
    print("Available GitHub MCP Tools:")
    for tool in tools:
        print(f"- {tool.tool_name}")
        print(f"  Spec: {tool.tool_spec}")
        print()
        all_tools.append(tool)

# Test Github MCP client directly
with mcp_client_github:
    result = mcp_client_github.call_tool_sync(
        tool_use_id="tool-123",
        name="search_repositories",
        arguments={"query": "mngaonkar"}
    )

    # print(f"GitHub MCP search_repositories result: {result['content']}")

    for content in result['content']:
        for item in json.loads(content['text'])['items']:
            print(f"repo - {item['full_name']}")
            print("-----------------")

# Keep MCP clients active while using the agent
with mcp_client_aws_docs, mcp_client_github:
    agent = Agent(name="IntroAgent",
                model="openai.gpt-oss-20b-1:0",
                description="An agent that demonstrates basic functionalities with GitHub integration.",
                tools=all_tools)

    # Example queries - uncomment the one you want to run
    # result = agent("List my EKS clusters in us-east-1 region")
    # result = agent("Find issues with my EKS cluster in us-east1 region")
    # result = agent("what is product of 2.5 and 2.5?")

    # GitHub MCP examples (make sure to set GITHUB_PERSONAL_ACCESS_TOKEN in .env)
    result = agent("List GitHub repositories at mngaonkar")
    # result = agent("What is Lambda?")
    # result = agent("Get issues from my most recent repository")
    # result = agent("Create a new issue in my repository with title 'Test Issue' and body 'This is a test issue created by the agent'")
    # result = agent("Search for repositories with 'python' in the name")

    # print(result)
