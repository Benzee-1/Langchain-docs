# Complete Course: Building AI Agents with LangGraph and LangChain

## Table of Contents

1. [Course Introduction](#course-introduction)
2. [Module 1: Foundations of Agent Development](#module-1-foundations-of-agent-development)
3. [Module 2: Building Your First Agent](#module-2-building-your-first-agent)
4. [Module 3: Advanced Agent Architectures](#module-3-advanced-agent-architectures)
5. [Module 4: Deep Agents Framework](#module-4-deep-agents-framework)
6. [Module 5: Integration and Production](#module-5-integration-and-production)
7. [Module 6: Best Practices and Troubleshooting](#module-6-best-practices-and-troubleshooting)

---

## Course Introduction

Welcome to the comprehensive course on building AI agents using LangGraph and LangChain. This course will teach you everything from basic concepts to advanced implementation techniques for creating powerful, production-ready AI agents.

### What You'll Learn

- Build intelligent agents using LangGraph's Graph API
- Implement tools and function calling
- Create complex agent workflows with state management
- Deploy deep agents with planning capabilities
- Integrate with various AI models and services
- Handle human-in-the-loop processes
- Build production-ready agent systems

### Prerequisites

- Basic Python programming knowledge
- Understanding of AI/ML concepts
- Familiarity with APIs and web services

---

## Module 1: Foundations of Agent Development

### Section 1.1: Introduction to LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs. It's built on top of LangChain and provides a framework for creating complex agent workflows.

#### Key Concepts:

1. **Graphs**: Define the structure of your agent workflow
2. **Nodes**: Individual functions or operations in your graph
3. **Edges**: Connections between nodes that define flow
4. **State**: Shared data that persists throughout execution

### Section 1.2: Setting Up Your Environment

```python
# Installation
pip install langchain langgraph langchain-community

# Basic imports
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage
```

### Section 1.3: Understanding Agent Architecture

Agents are systems that can:
- Reason about problems
- Use tools to gather information
- Make decisions based on context
- Interact with external systems
- Learn from interactions

---

## Module 2: Building Your First Agent

### Section 2.1: Basic Tool Definition

Tools are functions that agents can call to perform specific tasks:

```python
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a / b
```

### Section 2.2: Model Setup and Tool Binding

```python
# Initialize the model
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)

# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

### Section 2.3: Defining State

State management is crucial for maintaining context throughout agent execution:

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

### Section 2.4: Creating Agent Nodes

#### LLM Node
```python
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
```

#### Tool Node
```python
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### Section 2.5: Implementing Decision Logic

```python
from typing import Literal

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tool_node"
    return END
```

### Section 2.6: Building and Compiling the Agent

```python
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()
```

### Section 2.7: Running Your Agent

```python
# Invoke the agent
messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})

# Print results
for m in result["messages"]:
    m.pretty_print()
```

---

## Module 3: Advanced Agent Architectures

### Section 3.1: Custom SQL Agent

Building agents that can interact with databases requires specialized handling for safety and efficiency.

#### Database Setup
```python
import requests, pathlib
from langchain_community.utilities import SQLDatabase

# Download sample database
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")
response = requests.get(url)
local_path.write_bytes(response.content)

# Create database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
```

#### SQL Toolkit Integration
```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
```

### Section 3.2: Specialized Agent Nodes

#### List Tables Node
```python
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}
```

#### Schema Retrieval Node
```python
def call_get_schema(state: MessagesState):
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### Section 3.3: Human-in-the-Loop Implementation

```python
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

@tool
def run_query_tool_with_interrupt(config: RunnableConfig, **tool_input):
    request = {
        "action": "sql_db_query",
        "args": tool_input,
        "description": "Please review the tool call"
    }
    response = interrupt([request])
    
    if response["type"] == "accept":
        tool_response = run_query_tool.invoke(tool_input, config)
    elif response["type"] == "edit":
        tool_input = response["args"]["args"]
        tool_response = run_query_tool.invoke(tool_input, config)
    elif response["type"] == "response":
        tool_response = response["args"]
    else:
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")
    
    return tool_response
```

---

## Module 4: Deep Agents Framework

### Section 4.1: Introduction to Deep Agents

Deep agents are advanced AI systems with built-in capabilities for:
- Planning and task decomposition
- Context management through filesystems
- Subagent spawning for specialized tasks
- Long-term memory persistence

### Section 4.2: Quick Start with Deep Agents

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

# Setup search tool
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Create deep agent
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and write polished reports."""

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions
)

# Run the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
```

### Section 4.3: Agent Harness Capabilities

#### File System Access
Deep agents provide six tools for file operations:
- `ls`: List files in a directory
- `read_file`: Read file contents with line numbers
- `write_file`: Create new files
- `edit_file`: Perform exact string replacements
- `glob`: Find files matching patterns
- `grep`: Search file contents

#### Planning with To-Do Lists
```python
from deepagents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[
        TodoListMiddleware(
            system_prompt="Use the write_todos tool to track your progress"
        ),
    ],
)
```

### Section 4.4: Subagent Configuration

```python
research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-4o",
}

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    subagents=[research_subagent]
)
```

### Section 4.5: Long-term Memory Implementation

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),  # Ephemeral storage
        routes={
            "/memories/": StoreBackend(runtime)  # Persistent storage
        }
    )

agent = create_deep_agent(
    store=InMemoryStore(),
    backend=make_backend
)
```

### Section 4.6: Human-in-the-Loop Configuration

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # Require approval
        "read_file": False,   # No interrupts
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
    },
    checkpointer=checkpointer
)
```

---

## Module 5: Integration and Production

### Section 5.1: Model Provider Integration

#### OpenAI Integration
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4",
    api_key="your-openai-key"
)
```

#### Anthropic Integration
```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-key"
)
```

#### Azure Integration
```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-azure-key",
    api_version="2024-02-15-preview"
)
```

### Section 5.2: Vector Store Integration

#### Pinecone
```python
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(
    index=my_index,
    embedding=embeddings
)
```

#### Chroma
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings
)
```

### Section 5.3: Backend Configuration

#### State Backend (Ephemeral)
```python
from deepagents.backends import StateBackend

agent = create_deep_agent(
    backend=lambda rt: StateBackend(rt)
)
```

#### Filesystem Backend (Local Disk)
```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir=".", virtual_mode=True)
)
```

#### Store Backend (Persistent)
```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=lambda rt: StoreBackend(rt),
    store=InMemoryStore()
)
```

### Section 5.4: Monitoring and Observability

#### LangSmith Integration
```python
import os

# Set up LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-key"
```

---

## Module 6: Best Practices and Troubleshooting

### Section 6.1: Agent Design Best Practices

#### Tool Design Guidelines
1. **Clear Descriptions**: Write descriptive docstrings for tools
2. **Input Validation**: Validate inputs before processing
3. **Error Handling**: Provide meaningful error messages
4. **Performance**: Optimize for speed and reliability

#### State Management
1. **Minimal State**: Keep state as small as possible
2. **Immutable Updates**: Use proper state update patterns
3. **Clear Structure**: Organize state logically

### Section 6.2: Subagent Best Practices

#### Description Writing
```python
# ✅ Good: Specific and actionable
{
    "name": "data-analyzer",
    "description": "Analyzes financial data and generates investment insights with confidence scores"
}

# ❌ Bad: Vague and unclear
{
    "name": "helper",
    "description": "Does finance stuff"
}
```

#### System Prompts
```python
research_subagent = {
    "name": "research-agent",
    "description": "Conducts in-depth research using web search",
    "system_prompt": """You are a thorough researcher. Your job is to:
    1. Break down research questions into searchable queries
    2. Use internet_search to find relevant information
    3. Synthesize findings into comprehensive summaries
    4. Cite sources when making claims
    
    Keep responses under 500 words to maintain clean context.""",
    "tools": [internet_search],
}
```

### Section 6.3: Common Issues and Solutions

#### Subagent Not Being Called
**Problem**: Main agent doesn't delegate to subagents

**Solutions**:
1. Make descriptions more specific
2. Instruct main agent to delegate in system prompt
3. Ensure subagent names are descriptive

#### Context Bloating
**Problem**: Agent context becomes too large

**Solutions**:
1. Use filesystem for large data storage
2. Instruct subagents to return concise summaries
3. Implement proper context management

#### Tool Call Errors
**Problem**: Tools fail or return unexpected results

**Solutions**:
1. Add input validation
2. Implement error handling
3. Provide clear error messages

### Section 6.4: Performance Optimization

#### Prompt Caching (Anthropic)
Deep agents automatically enable prompt caching for Anthropic models to reduce latency and costs.

#### Conversation Summarization
Agents automatically compress conversation history when token limits are approached.

#### Large Tool Result Eviction
Large tool outputs are automatically saved to files to prevent context overflow.

### Section 6.5: Security Considerations

#### Database Access
- Always scope database permissions narrowly
- Use read-only connections when possible
- Validate SQL queries before execution

#### File System Access
- Use virtual mode for sandboxed access
- Implement path validation
- Set appropriate size limits

#### API Keys and Secrets
- Use environment variables for sensitive data
- Implement key rotation policies
- Monitor API usage and costs

---

## Conclusion

This course has covered the complete journey from basic agent concepts to advanced production deployments. You've learned:

1. **Foundation Skills**: Understanding LangGraph, state management, and tool integration
2. **Agent Building**: Creating your first agents with proper architecture
3. **Advanced Patterns**: Implementing SQL agents, human-in-the-loop processes
4. **Deep Agents**: Leveraging the powerful deep agents framework
5. **Production Integration**: Connecting with various models, databases, and services
6. **Best Practices**: Following industry standards for reliability and security

### Next Steps

1. **Practice**: Build your own agents using the patterns shown
2. **Experiment**: Try different model providers and tools
3. **Deploy**: Move your agents to production environments
4. **Monitor**: Set up proper observability and logging
5. **Iterate**: Continuously improve based on user feedback

### Additional Resources

- **LangGraph Documentation**: Official documentation and examples
- **LangChain Hub**: Pre-built prompts and chains
- **Community**: Join the LangChain Discord and GitHub discussions
- **LangSmith**: Use for production monitoring and debugging

Remember: Building great AI agents is an iterative process. Start simple, test thoroughly, and gradually add complexity as needed. Good luck with your agent-building journey!