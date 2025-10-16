import streamlit as st
import logging
from strands import Agent
from strands_tools import calculator, current_time, use_aws, mcp_client, shell
from strands.multiagent import GraphBuilder
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from strands.agent.conversation_manager import SlidingWindowConversationManager
from dotenv import load_dotenv
import os, sys, json, yaml, time
from typing import List, Dict, Any
import plotly.express as px
import pandas as pd
from dataclasses import dataclass

# Configure page
st.set_page_config(
    page_title="Strands Agent Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentEvent:
    timestamp: float
    event_type: str
    tool_name: str = ""
    tool_input: str = ""
    message: str = ""

class StrandsAgentManager:
    def __init__(self):
        self.graph = None
        self.mcp_client_aws_docs = None
        self.mcp_client_github = None
        self.all_tools = []
        self.agent_events = []
        self._initialize_tools()
        self._build_graph()
    
    def load_system_prompt(self, file_path="system_prompt.yaml"):
        """Load system prompt from YAML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config["system_prompt"]
        except FileNotFoundError:
            logger.error(f"System prompt file '{file_path}' not found")
            return "You are a helpful AI assistant that can break down complex tasks and execute them using various tools."
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            return "You are a helpful AI assistant that can break down complex tasks and execute them using various tools."
    
    def _initialize_tools(self):
        """Initialize MCP clients and tools"""
        # AWS documentation MCP client setup
        self.mcp_client_aws_docs = MCPClient(lambda: stdio_client(
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
        self.mcp_client_github = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="npx", 
                args=["-y", "@modelcontextprotocol/server-github"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
                }
            )
        ))
        
        # Basic tools
        self.all_tools = [calculator, current_time, use_aws, shell.shell]
        
        # Add MCP tools (optional - loaded only if environment is configured)
        self.mcp_tools_loaded = False
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            try:
                # Load AWS MCP tools if AWS is configured
                if os.getenv("AWS_ACCESS_KEY_ID"):
                    with self.mcp_client_aws_docs:
                        aws_tools = self.mcp_client_aws_docs.list_tools_sync()
                        for tool in aws_tools:
                            self.all_tools.append(tool)
                
                # Load GitHub MCP tools if GitHub is configured  
                if os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
                    with self.mcp_client_github:
                        github_tools = self.mcp_client_github.list_tools_sync()  
                        for tool in github_tools:
                            self.all_tools.append(tool)
                
                self.mcp_tools_loaded = True
                logger.info("MCP tools loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load MCP tools: {e}")
                self.mcp_tools_loaded = False

    def process_event(self, **kwargs):
        self.event_handler_func(kwargs)

    def event_handler_func(self, event):
        """Shared event processor for both async iterators and callback handlers"""

        logger.info(f"Event: {event}")
        timestamp = time.time()
        event_type = "n/a"
        tool_name = ""
        tool_input = ""
        message = ""

        if event.get("init_event_loop", False):
            print("🔄 Event loop initialized")
        elif event.get("start_event_loop", False):
            print("▶️ Event loop cycle starting")
        elif event.get("start", False):
            print("▶️ Starting")

        
        # Agent is invoking a tool
        if "current_tool_use" in event and event["current_tool_use"].get("name"):
            tool_name = event["current_tool_use"]["name"]
            tool_input = json.dumps(event['current_tool_use'].get('input', {}))
            print(f"🔧 Using tool: {tool_name}")
            print(f"   Input: {tool_input}")
            event_type = "tool_use"

        # Agent is producing text output
        if "data" in event:
            data_snippet = event["data"]
            print(f"📟 Text: {data_snippet}")
            event_type = "text"
            message = event["data"]

        if "event" in event:
            if "messageStart" in event["event"]:
                message = "Message generation started"
                event_type = "control"

            if "messageStop" in event["event"]:
                message = "Message generation ended"
                event_type = "control"

            elif "contentBlockStart" in event["event"]:
                if "start" in event["event"]["contentBlockStart"] and "toolUse" in event["event"]["contentBlockStart"]["start"]:
                    event_type = "tool_use"
                    tool_name = event["event"]["contentBlockStart"]["start"]["toolUse"]["name"]

                message = "Content block generation started"
                event_type = "control"

            elif "contentBlockStop" in event["event"]:
                message = "Content block generation ended"
                event_type = "control"

            elif "contentBlockDelta" in event["event"]:
                if "delta" in event["event"]["contentBlockDelta"] and "text" in event["event"]["contentBlockDelta"]["delta"]:
                    event_type = "text"
                    message = event["event"]["contentBlockDelta"]["delta"]["text"]
                elif "delta" in event["event"]["contentBlockDelta"] and "toolUse" in event["event"]["contentBlockDelta"]["delta"]:
                    event_type = "tool_use"
                    tool_input = event["event"]["contentBlockDelta"]["delta"]["toolUse"]["input"]

            elif "metadata" in event["event"]:
                message = f"Metadata: {json.dumps(event['event']['metadata'])}"
                event_type = "metadata"
    
        if "delta" in event:
            if "current_tool_use" in event["delta"]:
                tool_name = event["delta"]["current_tool_use"]["name"]
                tool_input = event['delta']['current_tool_use']["input"]
                event_type = "tool_use"
            
        if "message" in event and "role" in event["message"]:
            message = json.dumps(event['message'])
            event_type = "message"
            print(f"📧 Message: {message}")

        if event_type in ["message"]:
            self.agent_events.append(
                AgentEvent(timestamp=timestamp, 
                        event_type=event_type, 
                        tool_name=tool_name, 
                        tool_input=tool_input, 
                        message=message))

    def _build_graph(self):
        """Build the agent graph"""
        # Task decomposer agent
        task_decomposer = Agent(
            name="task_decomposer",
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
        
        # kubectl command agent
        kubectl_command_agent = Agent(
            name="kubectl_command_agent",
            system_prompt=self.load_system_prompt("kubectl_command_agent.yaml") or 
                         "You are a Kubernetes command execution agent. Execute kubectl commands safely using the shell tool.",
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            tools=[shell.shell],
            callback_handler=self.process_event
        )
        
        # Task executor agent  
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
            tools=self.all_tools
        )
        
        # Result aggregator agent
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
        
        # Build the graph
        builder = GraphBuilder()
        builder.add_node(task_decomposer, "task_decomposer")
        builder.add_node(task_executor, "task_executor")
        builder.add_node(kubectl_command_agent, "kubectl_command_agent")
        builder.add_node(result_aggregator, "result_aggregator")
        
        # Set up the workflow
        builder.add_edge("task_decomposer", "kubectl_command_agent")
        builder.add_edge("kubectl_command_agent", "result_aggregator")
        builder.set_entry_point("task_decomposer")
        
        self.graph = builder.build()
    
    def execute_query(self, query: str, progress_callback=None):
        """Execute a query using the agent graph"""
        try:
            if progress_callback:
                progress_callback("🚀 Starting task execution...")
            
            if self.graph is None:
                raise ValueError("Agent graph not initialized properly")
            
            result = self.graph(query)
            
            if progress_callback:
                progress_callback("✅ Task execution completed!")
            
            # Extract only the final result from the last agent (result_aggregator)
            final_result = self._extract_final_result(result)
            
            return {"success": True, "result": final_result, "error": None}
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "result": None, "error": error_msg}
    
    def _extract_final_result(self, graph_result):
        """Extract only the final result from the result_aggregator agent"""
        try:
            # If the result has execution_order, get the last agent's result
            if hasattr(graph_result, 'execution_order') and graph_result.execution_order:
                # Get the last execution result (should be from result_aggregator)
                last_execution = graph_result.execution_order[-1]
                if hasattr(last_execution, 'result') and hasattr(last_execution.result, 'result'):
                    # Access the AgentResult within NodeResult
                    agent_result = last_execution.result.result
                    if hasattr(agent_result, 'message') and 'content' in agent_result.message:
                        return str(agent_result.message['content'][-1]['text'])
            
            # Try to get result from results dictionary using the correct agent name
            if hasattr(graph_result, 'results') and 'result_aggregator' in graph_result.results:
                node_result = graph_result.results['result_aggregator']
                if hasattr(node_result, 'result') and hasattr(node_result.result, 'message'):
                    return str(node_result.result.message['content'][-1]['text'])
            
            # If the result is a string, return it directly
            if isinstance(graph_result, str):
                return graph_result
            
            # Fallback to string representation
            return str(graph_result)
            
        except Exception as e:
            logger.warning(f"Could not extract final result: {e}")
            # More detailed fallback - try to find any content in the result
            try:
                if hasattr(graph_result, 'results') and not isinstance(graph_result, str):
                    for agent_name, node_result in graph_result.results.items():
                        if hasattr(node_result, 'result') and hasattr(node_result.result, 'message'):
                            if 'content' in node_result.result.message:
                                logger.info(f"Using result from agent: {agent_name}")
                                return str(node_result.result.message['content'][-1]['text'])
            except Exception as fallback_error:
                logger.warning(f"Fallback extraction also failed: {fallback_error}")
            
            return str(graph_result)

def main():
    st.title("🤖 Strands Multi-Agent Interface")
    st.markdown("**Intelligent task decomposition and execution with specialized agents**")
    
    # Initialize session state
    if 'agent_manager' not in st.session_state:
        with st.spinner('Initializing agents...'):
            st.session_state.agent_manager = StrandsAgentManager()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection (display only for now)
        st.selectbox(
            "Model",
            ["global.anthropic.claude-sonnet-4-5-20250929-v1:0"],
            disabled=True,
            help="Model selection (currently fixed)"
        )
        
        # Available tools info
        st.subheader("🛠️ Available Tools")
        tools_info = {
            "AWS Operations": "✅ use_aws tool",
            "Shell Commands": "✅ shell tool (kubectl, docker, git)",
            "Mathematics": "✅ calculator tool", 
            "Time Operations": "✅ current_time tool",
            "GitHub MCP": "⚠️ Available when configured",
            "AWS Docs MCP": "⚠️ Available when configured"
        }
        
        for tool, status in tools_info.items():
            st.write(f"**{tool}**: {status}")
        
        # Environment status
        st.subheader("🌍 Environment")
        aws_configured = bool(os.getenv("AWS_ACCESS_KEY_ID"))
        github_configured = bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
        
        st.write(f"**AWS**: {'✅ Configured' if aws_configured else '❌ Not configured'}")
        st.write(f"**GitHub**: {'✅ Configured' if github_configured else '❌ Not configured'}")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (query, response) in enumerate(st.session_state.chat_history):
                with st.expander(f"Query {i+1}: {query[:60]}{'...' if len(query) > 60 else ''}", expanded=(i == len(st.session_state.chat_history)-1)):
                    st.write("**You:**", query)
                    st.write("**Agent:**")
                    if response.get("success"):
                        st.success("✅ Execution successful")
                        # Display the result as markdown for better formatting
                        st.markdown(response["result"])
                    else:
                        st.error("❌ Execution failed")
                        st.write(response["error"])
        
        # Query input
        query_input = st.text_area(
            "Enter your query:",
            placeholder="Examples:\n- Find issues with EKS cluster sliverblaze\n- List all AWS EC2 instances in us-east-1\n- Check kubectl cluster info and node status",
            height=100
        )
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Find issues with EKS cluster sliverblaze",
            "List all AWS EC2 instances in us-east-1", 
            "Check kubectl cluster info and node status",
            "Calculate the average of 10, 20, 30, 40, 50",
            "What time is it and list current directory contents"
        ]
        
        selected_example = st.selectbox("Choose an example:", [""] + example_queries)
        if selected_example and st.button("Use Example"):
            query_input = selected_example
    
    with col2:
        st.subheader("📊 Execution Status")

        # Add a scrollable box for events
        event_box = st.container()
        with event_box:
            event_box.markdown("### Agent Events")
            event_box.markdown("<div style='height:300px; overflow-y:auto; border:1px solid #ccc; padding:10px;'>", unsafe_allow_html=True)
            for event in st.session_state.agent_manager.agent_events:
                event_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
                if event.event_type == "message":
                    event_box.markdown(f"**[{event_time}] Message:** {event.message}")
                elif event.event_type == "tool_use":
                    event_box.markdown(f"**[{event_time}] Tool Use:** {event.tool_name} with input {event.tool_input}")
                elif event.event_type == "text":
                    event_box.markdown(f"**[{event_time}] Text:** {event.message}")
                elif event.event_type == "n/a":
                    continue
            event_box.markdown("</div>", unsafe_allow_html=True)
    
    # Execute button
    if st.button("🚀 Execute Query", type="primary") and query_input.strip():
        with st.spinner('Executing query...'):
            # Progress callback
            def update_progress(message):
                # status_placeholder.info(message)
                pass
            
            # Execute the query
            response = st.session_state.agent_manager.execute_query(
                query_input.strip(), 
                progress_callback=update_progress
            )
            
            # Add to chat history
            st.session_state.chat_history.append((query_input.strip(), response))
            
            # Clear status
            # status_placeholder.empty()
            
            # Rerun to update the display
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Strands Multi-Agent Framework*")

if __name__ == "__main__":
    main()