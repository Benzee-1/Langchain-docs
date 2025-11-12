# LangChain Agent Engineering Course

## Course Overview

This comprehensive course covers LangChain agent engineering, from basic concepts to advanced multi-agent systems. You'll learn to build, customize, and deploy production-ready AI agents using LangChain's powerful abstractions.

---

## Table of Contents

1. [Getting Started](#section-1-getting-started)
2. [Core Components](#section-2-core-components)
3. [Agent Development](#section-3-agent-development)
4. [Middleware & Context Engineering](#section-4-middleware--context-engineering)
5. [Advanced Features](#section-5-advanced-features)
6. [Multi-Agent Systems](#section-6-multi-agent-systems)
7. [Production Deployment](#section-7-production-deployment)
8. [Tools & Integrations](#section-8-tools--integrations)

---

## Section 1: Getting Started

### 1.1 Introduction to LangChain

LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and more. LangChain provides a pre-built agent architecture and model integrations to help you get started quickly.

**Key Benefits:**
- Standard model interface across providers
- Easy to use, highly flexible agent framework
- Built on top of LangGraph for durability
- Deep debugging capabilities with LangSmith

### 1.2 Installation

```bash
# Install LangChain
pip install -U langchain

# Install provider integrations
pip install -U langchain-openai
pip install -U langchain-anthropic
```

### 1.3 Basic Agent Creation

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

### 1.4 Real-World Agent Example

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.
You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location."""

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer
)
```

---

## Section 2: Core Components

### 2.1 Models

Models are the reasoning engine of agents. They drive decision-making, tool selection, and response generation.

#### Static Model Configuration

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Using model identifier string
agent = create_agent("gpt-5", tools=tools)

# Using model instance for more control
model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)
agent = create_agent(model, tools=tools)
```

#### Dynamic Model Selection

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def dynamic_model_selection(request, handler):
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])
    if message_count > 10:
        request.model = advanced_model
    else:
        request.model = basic_model
    return handler(request)

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

### 2.2 Tools

Tools give agents the ability to take actions beyond text generation.

#### Creating Tools

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"
```

#### Tool Error Handling

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
```

### 2.3 Messages and Memory

Messages form the conversation history that agents maintain automatically.

#### Custom State Schema

```python
from langchain.agents import AgentState
from typing import Any

class CustomState(AgentState):
    user_preferences: dict
    session_data: Any

agent = create_agent(
    model,
    tools=tools,
    state_schema=CustomState
)
```

### 2.4 Structured Output

Ensure agents return data in specific, predictable formats.

#### Using Pydantic Models

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address")
    phone: str = Field(description="The phone number")

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ContactInfo)
)
```

---

## Section 3: Agent Development

### 3.1 Agent Architecture

Agents follow the ReAct ("Reasoning + Acting") pattern:
1. **Reasoning**: Brief analysis of the current situation
2. **Acting**: Targeted tool calls based on reasoning
3. **Observation**: Processing tool results
4. **Iteration**: Repeat until task completion

### 3.2 System Prompts

System prompts shape how agents approach tasks.

#### Static System Prompts

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

#### Dynamic System Prompts

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def user_role_prompt(request) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."
    
    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."
    return base_prompt
```

### 3.3 Invocation Patterns

#### Basic Invocation

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

#### Streaming

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news"}]
}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
```

---

## Section 4: Middleware & Context Engineering

### 4.1 Understanding Middleware

Middleware provides extensibility for customizing agent behavior at different execution stages.

#### Built-in Middleware Types

1. **Fallback Middleware** - Handle model failures gracefully
2. **PII Detection** - Detect and sanitize sensitive information
3. **Todo List** - Task planning and tracking
4. **Tool Retry** - Automatic retry with exponential backoff
5. **Context Editing** - Manage conversation context

### 4.2 Fallback Middleware

```python
from langchain.agents.middleware import FallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        FallbackMiddleware(
            fallback_models=["gpt-4o-mini", "claude-haiku"],
            exceptions=(Exception,)
        )
    ]
)
```

### 4.3 PII Detection

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block"
        ),
    ],
)
```

### 4.4 Custom Middleware

#### Decorator-based Middleware

```python
from langchain.agents.middleware import before_model, after_model

@before_model
def log_before_model(state, runtime):
    print(f"About to call model with {len(state['messages'])} messages")
    return None

@after_model
def validate_output(state, runtime):
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None
```

#### Class-based Middleware

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Model call with {len(state['messages'])} messages")
        return None
    
    def after_model(self, state, runtime):
        print(f"Model returned: {state['messages'][-1].content}")
        return None
```

### 4.5 Context Engineering

Context engineering involves providing the right information and tools at the right time.

#### Model Context

```python
@dynamic_prompt
def state_aware_prompt(request) -> str:
    message_count = len(request.messages)
    base = "You are a helpful assistant."
    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."
    return base
```

#### Tool Context

```python
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up user info from store."""
    store = runtime.store
    user_id = runtime.context.user_id
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"
```

---

## Section 5: Advanced Features

### 5.1 Human-in-the-Loop

Add human oversight to agent tool calls for sensitive operations.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[write_file_tool, execute_sql_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # All decisions allowed
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
            },
        ),
    ],
    checkpointer=InMemorySaver(),
)

# Handle interrupts
result = agent.invoke({"messages": [...]}, config=config)
if result.get('__interrupt__'):
    # Present to human for review
    # Resume with decisions
    agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
```

### 5.2 Memory Management

#### Short-term Memory (State)

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
        ),
    ],
)
```

#### Long-term Memory (Store)

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

@tool
def save_user_preference(preference: str, runtime: ToolRuntime) -> str:
    """Save user preference to long-term memory."""
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("preferences",), user_id, {"preference": preference})
    return "Preference saved successfully."

agent = create_agent(
    model="gpt-4o",
    tools=[save_user_preference],
    store=store
)
```

### 5.3 Retrieval-Augmented Generation (RAG)

Build semantic search engines over documents.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load and split documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(splits)

# Create retriever tool
@tool
def search_documents(query: str) -> str:
    """Search through uploaded documents."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

agent = create_agent(
    model="gpt-4o",
    tools=[search_documents],
    system_prompt="Use the search_documents tool to find relevant information before answering questions."
)
```

---

## Section 6: Multi-Agent Systems

### 6.1 Supervisor Pattern

The supervisor pattern uses a central coordinator to manage specialized worker agents.

#### Creating Specialized Sub-Agents

```python
# Calendar agent
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests into proper ISO datetime formats."
    )
)

# Email agent
email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on natural language requests."
    )
)
```

#### Wrapping Sub-Agents as Tools

```python
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language."""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
```

#### Creating the Supervisor

```python
supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "Break down user requests into appropriate tool calls and coordinate results."
    )
)
```

### 6.2 Multi-Agent Communication

#### Information Flow Control

```python
@tool
def schedule_event(request: str, runtime: ToolRuntime) -> str:
    """Schedule with full conversation context."""
    original_message = next(
        msg for msg in runtime.state["messages"] 
        if msg.type == "human"
    )
    
    prompt = f"""
    Original request: {original_message.text}
    Sub-task: {request}
    """
    
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return result["messages"][-1].text
```

---

## Section 7: Production Deployment

### 7.1 Local Development with Studio

#### Setting up Agent Server

```python
# agent.py
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """Send an email"""
    return f"Email sent to {to}"

agent = create_agent(
    "gpt-4o",
    tools=[send_email],
    system_prompt="You are a helpful email assistant.",
)
```

#### Configuration

```json
// langgraph.json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./src/agent.py:agent"
    },
    "env": ".env"
}
```

#### Starting the Server

```bash
# Install CLI
pip install --upgrade "langgraph-cli[inmem]"

# Start development server
langgraph dev
```

### 7.2 Model Context Protocol (MCP)

Integrate with MCP servers for standardized tool access.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],
    },
    "weather": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",
    }
})

tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

### 7.3 Observability

#### LangSmith Integration

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

# Automatic tracing of agent runs
agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

#### Custom Monitoring

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"callbacks": [callback]}
)

print(callback.usage_metadata)  # Token usage statistics
```

---

## Section 8: Tools & Integrations

### 8.1 Built-in Tool Categories

#### Search and Retrieval Tools
- Web search integration
- Document retrieval
- Database queries
- API calls

#### Communication Tools
- Email sending
- Slack integration
- SMS/messaging
- Webhook notifications

#### File and Data Tools
- File operations
- Data processing
- Image generation
- PDF manipulation

### 8.2 Custom Tool Development

#### Advanced Tool Patterns

```python
from langchain.tools import tool
from typing import Optional, List

@tool(parse_docstring=True)
def advanced_search(
    query: str,
    filters: Optional[List[str]] = None,
    limit: int = 10
) -> str:
    """Advanced search with filtering capabilities.
    
    Args:
        query: Search query string
        filters: Optional list of filter criteria
        limit: Maximum number of results to return
    """
    # Implementation
    return "Search results..."
```

#### Tool with Runtime Context

```python
@tool
def context_aware_tool(
    input_data: str,
    runtime: ToolRuntime
) -> str:
    """Tool that uses runtime context."""
    user_id = runtime.context.get("user_id")
    state_data = runtime.state.get("custom_field")
    store_data = runtime.store.get(("namespace",), "key")
    
    # Process with context
    return f"Processed {input_data} for user {user_id}"
```

### 8.3 Integration Examples

#### Database Integration

```python
import sqlite3

@tool
def query_database(sql: str) -> str:
    """Execute SQL query on database."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return str(results)
    finally:
        conn.close()
```

#### API Integration

```python
import requests

@tool
def call_external_api(endpoint: str, data: dict) -> str:
    """Call external REST API."""
    response = requests.post(
        f"https://api.example.com/{endpoint}",
        json=data,
        headers={"Authorization": "Bearer token"}
    )
    return response.json()
```

---

## Best Practices

### Development Guidelines

1. **Start Simple**: Begin with basic agents before adding complexity
2. **Test Incrementally**: Add one feature at a time
3. **Monitor Performance**: Track token usage, latency, and success rates
4. **Use Built-in Middleware**: Leverage existing solutions before building custom ones
5. **Document Context Strategy**: Make it clear what context is being passed and why

### Production Considerations

1. **Error Handling**: Implement robust error handling and fallbacks
2. **Rate Limiting**: Respect API rate limits and implement backoff strategies
3. **Security**: Sanitize inputs and protect sensitive information
4. **Monitoring**: Set up comprehensive logging and alerting
5. **Testing**: Implement thorough testing including edge cases

### Performance Optimization

1. **Model Selection**: Choose appropriate models for each task
2. **Context Management**: Implement efficient context trimming and summarization
3. **Tool Selection**: Dynamically select relevant tools to reduce overhead
4. **Caching**: Implement caching for expensive operations
5. **Parallel Processing**: Use batch operations where possible

---

## Conclusion

This course has covered the essential aspects of LangChain agent engineering, from basic concepts to advanced multi-agent systems. You now have the knowledge to:

- Build and customize LangChain agents
- Implement sophisticated middleware for context engineering
- Create multi-agent systems with supervisor patterns
- Deploy agents in production environments
- Integrate with external tools and services

Continue practicing with real-world projects and explore the extensive LangChain ecosystem to build powerful AI applications.

---

## Additional Resources

- [LangChain Documentation](https://docs.langchain.com)
- [LangSmith for Observability](https://smith.langchain.com)
- [LangChain Academy](https://academy.langchain.com)
- [Community Forum](https://community.langchain.com)
- [GitHub Repository](https://github.com/langchain-ai/langchain)

---

*Course completed. Continue building amazing AI agents with LangChain!*