# LangSmith API Deployment and Development Complete Course

## Course Overview
This comprehensive course covers the entire ecosystem of LangSmith API development, deployment, and management. From basic concepts to advanced implementation techniques, you'll learn everything needed to build, deploy, and manage production-ready AI applications using LangSmith and LangGraph.

## Table of Contents
1. [Foundation and Setup](#foundation-and-setup)
2. [API Architecture and Events](#api-architecture-and-events)
3. [Deployment Configuration](#deployment-configuration)
4. [Application Development](#application-development)
5. [Tracing and Observability](#tracing-and-observability)
6. [Evaluation and Testing](#evaluation-and-testing)
7. [Production Management](#production-management)
8. [Advanced Topics](#advanced-topics)

---

# Section 1: Foundation and Setup

## Lesson 1.1: Introduction to LangSmith Deployment Architecture

### Learning Objectives
- Understand the components of LangSmith Deployment
- Learn about Agent Server and its role
- Explore the control plane and data plane architecture

### Key Components

**Agent Server**: The core runtime that defines an opinionated API for deploying graphs and agents. It handles:
- Execution management
- State persistence
- Request routing
- Tool integration

**Control Plane**: The UI and APIs for creating, updating, and managing Agent Server deployments.

**Data Plane**: The runtime layer that executes graphs, including:
- Agent Servers
- Backing services (PostgreSQL, Redis)
- State reconciliation listeners

**Studio**: A specialized IDE for visualization, interaction, and debugging that connects to local Agent Server instances.

**LangGraph CLI**: Command-line interface for building, packaging, and interacting with graphs locally.

### Architecture Benefits
- **Scalability**: Built-in horizontal scaling capabilities
- **Reliability**: Robust state management and persistence
- **Observability**: Comprehensive tracing and monitoring
- **Developer Experience**: Streamlined deployment workflow

## Lesson 1.2: Environment Setup and Configuration

### Prerequisites
- Python 3.9+ or Node.js 20+
- LangSmith API account
- Docker (for local development)

### Initial Configuration
```bash
# Set up environment variables
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_PROJECT=<project-name>
export LANGSMITH_TRACING=true
```

### Project Structure Setup
```
my-app/
├── src/                    # Application code
├── langgraph.json         # Configuration file
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

---

# Section 2: API Architecture and Events

## Lesson 2.1: Understanding Stream Events

### Event System Overview
LangSmith uses a comprehensive event system to track execution flow and provide real-time updates. The v2 API provides structured events with consistent formatting.

### Core Event Structure
```python
{
    "event": "event_name",           # Type of event
    "name": "runnable_name",         # Name of the component
    "data": {...},                   # Event-specific data
    "tags": [...],                   # Associated tags
    "metadata": {...}                # Additional metadata
}
```

### Key Event Types

#### Chat Model Events
```python
# Model starts processing
{
    "event": "on_chat_model_start",
    "name": "[model name]",
    "data": {"input": {"messages": [[SystemMessage, HumanMessage]]}}
}

# Streaming response chunks
{
    "event": "on_chat_model_stream", 
    "name": "[model name]",
    "data": {"chunk": AIMessageChunk(content="hello")}
}

# Model completes processing
{
    "event": "on_chat_model_end",
    "name": "[model name]",
    "data": {"output": AIMessageChunk(content="hello world")}
}
```

#### Chain Events
```python
# Chain execution start
{
    "event": "on_chain_start",
    "name": "format_docs",
    "data": {"input": [Document(...)]}
}

# Chain streaming output
{
    "event": "on_chain_stream",
    "name": "format_docs", 
    "data": {"chunk": "hello world!, goodbye world!"}
}

# Chain execution complete
{
    "event": "on_chain_end",
    "name": "format_docs",
    "data": {"output": "hello world!, goodbye world!"}
}
```

#### Tool Events
```python
# Tool invocation start
{
    "event": "on_tool_start",
    "name": "some_tool",
    "data": {"input": {"x": 1, "y": "2"}}
}

# Tool execution complete
{
    "event": "on_tool_end", 
    "name": "some_tool",
    "data": {"output": {"x": 1, "y": "2"}}
}
```

### Custom Events
You can dispatch custom events for application-specific monitoring:

```python
from langsmith.callbacks.manager import adispatch_custom_event

async def slow_operation(input: str, config: RunnableConfig) -> str:
    await adispatch_custom_event(
        "progress_event",
        {"message": "Finished step 1 of 3"},
        config=config
    )
    # Continue processing...
    return "Done"
```

## Lesson 2.2: Stream Event Parameters and Configuration

### astream_events Parameters

#### Core Parameters
- `input`: The input to the Runnable (Any type)
- `config`: RunnableConfig for execution context
- `version`: API version ('v1' or 'v2' - use 'v2')

#### Filtering Parameters
- `include_names`: Only include events from specific Runnable names
- `include_types`: Filter by Runnable types
- `include_tags`: Filter by associated tags
- `exclude_names`: Exclude specific Runnable names
- `exclude_types`: Exclude specific types
- `exclude_tags`: Exclude specific tags

#### Advanced Configuration
```python
async for event in chain.astream_events(
    "hello", 
    version="v2",
    include_names=["my_chain", "my_tool"],
    exclude_types=["retriever"],
    include_tags=["production"]
):
    process_event(event)
```

---

# Section 3: Deployment Configuration

## Lesson 3.1: Application Structure and Dependencies

### Python Applications with pyproject.toml

#### Project Structure
```
my-app/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py              # Main graph definition
│   └── utils/
│       ├── __init__.py
│       ├── tools.py          # Tool definitions
│       ├── nodes.py          # Node functions
│       └── state.py          # State management
├── pyproject.toml            # Dependencies
├── .env                      # Environment variables
└── langgraph.json           # Configuration
```

#### Dependencies Configuration
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-agent"
version = "0.0.1"
dependencies = [
    "langgraph>=0.6.0",
    "langchain-fireworks>=0.1.3"
]

[tool.hatch.build.targets.wheel]
packages = ["my_agent"]
```

### JavaScript Applications

#### Package Configuration
```json
{
    "name": "langgraph-app",
    "dependencies": {
        "@langchain/core": "^0.3.42",
        "@langchain/langgraph": "^0.2.57",
        "@langchain/openai": "^0.2.8"
    }
}
```

#### TypeScript Graph Example
```typescript
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-4o" });

const workflow = new StateGraph(MessagesAnnotation)
    .addNode("callModel", async (state) => {
        const response = await model.invoke(state.messages);
        return { messages: [response] };
    })
    .addEdge("__start__", "callModel")
    .addEdge("callModel", "__end__");

export const graph = workflow.compile();
```

## Lesson 3.2: Configuration Files and Environment Management

### langgraph.json Configuration
```json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./my_agent/agent.py:graph"
    },
    "env": ".env",
    "dockerfile_lines": [],
    "python_version": "3.11"
}
```

### Environment Variables
```bash
# Core Configuration
LANGSMITH_API_KEY=your-api-key
LANGSMITH_PROJECT=your-project
LANGSMITH_TRACING=true

# Model APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Custom Variables
MY_ENV_VAR_1=value1
MY_ENV_VAR_2=value2
```

### Advanced Configuration Options
```json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./my_agent/agent.py:graph"
    },
    "env": ".env",
    "dockerfile_lines": [
        "RUN apt-get update && apt-get install -y curl"
    ],
    "pip_config_file": "pip.conf",
    "python_version": "3.11",
    "node_version": "20"
}
```

---

# Section 4: Application Development

## Lesson 4.1: Graph Definition and State Management

### Basic Graph Structure
```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    user_id: str

def call_model(state: AgentState):
    # Process messages and return new state
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
```

### Advanced State Management
```python
from typing import Annotated
from langgraph.graph.message import add_messages

class ComplexState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    metadata: dict[str, any]
    step_count: int
    
def increment_step(state: ComplexState):
    return {"step_count": state.get("step_count", 0) + 1}
```

## Lesson 4.2: Tool Integration and Function Calling

### Tool Definition
```python
from langchain.tools import tool
from typing import Dict, Any

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation details
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    # Safe evaluation logic
    return eval(expression)

tools = [web_search, calculate]
```

### Tool Node Integration
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

# Add to graph
workflow.add_node("tools", tool_node)

# Model binding
model = ChatOpenAI().bind_tools(tools)
```

### Custom Tool Execution
```python
def custom_tool_executor(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Custom tool execution logic
        if tool_name == "special_tool":
            result = handle_special_tool(**tool_args)
        else:
            result = execute_standard_tool(tool_name, **tool_args)
            
        results.append(result)
    
    return {"messages": [create_tool_message(results)]}
```

---

# Section 5: Tracing and Observability

## Lesson 5.1: LangSmith Tracing Integration

### Automatic Tracing Setup
```python
import os

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "your-project"
```

### OpenAI Wrapper Integration
```python
from langsmith.wrappers import wrap_openai
from openai import OpenAI

client = wrap_openai(OpenAI())

# Automatic tracing of OpenAI calls
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Custom Tracing with Decorators
```python
from langsmith import traceable

@traceable(name="Custom Function", run_type="chain")
def process_data(data: dict) -> dict:
    # Processing logic here
    return processed_data

@traceable(run_type="tool")
def external_api_call(endpoint: str, params: dict):
    # API call implementation
    return api_response
```

## Lesson 5.2: OpenTelemetry Integration

### Basic OTEL Setup
```python
from langsmith.integrations.otel import configure

# Configure LangSmith OTEL export
configure(project_name="my-project")

# Your application code runs normally
# All spans automatically sent to LangSmith
```

### Manual OTEL Configuration
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Set up tracer
tracer_provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(
    endpoint="https://api.smith.langchain.com/otel/v1/traces",
    headers={"x-api-key": "your-api-key"}
)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)
```

### Custom Span Attributes
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def my_function():
    with tracer.start_as_current_span("custom_operation") as span:
        span.set_attribute("langsmith.metadata.user_id", "user_123")
        span.set_attribute("langsmith.span.tags", "production,critical")
        span.set_attribute("custom_metric", 42)
        
        # Your logic here
        return result
```

---

# Section 6: Evaluation and Testing

## Lesson 6.1: Dataset Creation and Management

### Creating Datasets
```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset(
    dataset_name="weather_agent_test",
    description="Test cases for weather agent"
)

# Add examples
examples = [
    {
        "inputs": {"question": "What's the weather in SF?"},
        "outputs": {"answer": "It's 60 degrees and foggy."}
    },
    {
        "inputs": {"question": "Will it rain tomorrow?"},
        "outputs": {"answer": "There's a 30% chance of rain."}
    }
]

for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"]
    )
```

### Dataset from Traces
```python
# Create dataset from existing traces
traces = client.list_runs(
    project_name="production_logs",
    filter='eq(feedback_key, "thumbs_up")'
)

dataset_examples = []
for trace in traces:
    dataset_examples.append({
        "inputs": trace.inputs,
        "outputs": trace.outputs
    })

dataset = client.create_dataset(
    dataset_name="curated_examples",
    inputs=[ex["inputs"] for ex in dataset_examples],
    outputs=[ex["outputs"] for ex in dataset_examples]
)
```

## Lesson 6.2: Evaluation Implementation

### Basic Evaluators
```python
from langsmith import aevaluate
from langchain_openai import ChatOpenAI

judge_model = ChatOpenAI(model="gpt-4")

async def accuracy_evaluator(outputs: dict, reference_outputs: dict) -> bool:
    """Evaluate if the output is accurate."""
    prompt = f"""
    Compare the actual answer with the expected answer.
    Actual: {outputs.get('answer', '')}
    Expected: {reference_outputs.get('answer', '')}
    
    Are they equivalent? Answer YES or NO.
    """
    
    response = await judge_model.ainvoke([{"role": "user", "content": prompt}])
    return "YES" in response.content.upper()

async def relevance_evaluator(outputs: dict, inputs: dict) -> dict:
    """Evaluate relevance with scoring."""
    prompt = f"""
    Question: {inputs.get('question', '')}
    Answer: {outputs.get('answer', '')}
    
    Rate the relevance from 1-5 where:
    1 = Completely irrelevant
    5 = Perfectly relevant
    
    Return only the number.
    """
    
    response = await judge_model.ainvoke([{"role": "user", "content": prompt}])
    score = int(response.content.strip())
    
    return {
        "key": "relevance",
        "score": score,
        "value": score,
        "comment": f"Relevance score: {score}/5"
    }
```

### Graph Evaluation
```python
async def evaluate_graph():
    def prepare_input(example_input: dict) -> dict:
        return {"messages": [{"role": "user", "content": example_input["question"]}]}
    
    # Create evaluation target
    target = prepare_input | graph
    
    # Run evaluation
    results = await aevaluate(
        target,
        data="weather_agent_test",
        evaluators=[accuracy_evaluator, relevance_evaluator],
        experiment_prefix="graph_v1",
        max_concurrency=5
    )
    
    return results
```

### Advanced Evaluation Techniques
```python
from langsmith.schemas import Run, Example

def trace_evaluator(run: Run, example: Example) -> dict:
    """Evaluate based on trace information."""
    # Check if correct tools were called
    tool_calls = []
    for child_run in run.child_runs:
        if child_run.run_type == "tool":
            tool_calls.append(child_run.name)
    
    expected_tools = ["web_search", "calculator"]
    tools_used_correctly = all(tool in tool_calls for tool in expected_tools)
    
    # Check response time
    duration_ms = (run.end_time - run.start_time).total_seconds() * 1000
    fast_response = duration_ms < 5000
    
    return {
        "key": "execution_quality",
        "value": tools_used_correctly and fast_response,
        "comment": f"Tools: {tool_calls}, Duration: {duration_ms}ms"
    }
```

---

# Section 7: Production Management

## Lesson 7.1: Deployment Strategies

### Cloud Deployment
```yaml
# Deploy to LangSmith Cloud
langsmith deploy

# With custom configuration
langsmith deploy --config production.json --env production
```

### Self-Hosted Deployment
```docker
# Dockerfile for self-hosted deployment
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy Agent
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install langgraph-cli
      - name: Deploy
        run: langsmith deploy
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
```

## Lesson 7.2: Monitoring and Scaling

### Health Monitoring
```python
async def health_check():
    """Basic health check endpoint."""
    try:
        # Test graph compilation
        test_graph = workflow.compile()
        
        # Test model connectivity
        await model.ainvoke([{"role": "user", "content": "test"}])
        
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Performance Monitoring
```python
from langsmith import traceable
import time

@traceable(name="Performance Monitor")
def monitor_performance():
    start_time = time.time()
    
    # Execute operation
    result = expensive_operation()
    
    duration = time.time() - start_time
    
    # Log metrics
    logger.info(f"Operation completed in {duration:.2f}s")
    
    if duration > 10.0:  # Alert threshold
        alert_slow_operation(duration)
    
    return result
```

### Auto-scaling Configuration
```python
# Agent Server configuration for scaling
{
    "scaling": {
        "min_replicas": 2,
        "max_replicas": 10,
        "target_cpu_utilization": 70,
        "scale_up_threshold": 5,
        "scale_down_threshold": 2
    },
    "resources": {
        "requests": {
            "cpu": "100m",
            "memory": "256Mi"
        },
        "limits": {
            "cpu": "500m", 
            "memory": "512Mi"
        }
    }
}
```

---

# Section 8: Advanced Topics

## Lesson 8.1: Webhooks and Event-Driven Architecture

### Webhook Configuration
```python
import httpx

async def create_run_with_webhook():
    webhook_url = "https://my-app.com/webhook"
    
    # Create run with webhook notification
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Hello"}]},
        webhook=webhook_url
    )
    
    return run
```

### Webhook Handler
```python
from fastapi import FastAPI, Request
import hmac
import hashlib

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
    # Verify webhook signature
    signature = request.headers.get("x-signature")
    body = await request.body()
    
    if not verify_signature(body, signature):
        return {"error": "Invalid signature"}, 403
    
    # Process webhook payload
    payload = await request.json()
    
    if payload["status"] == "completed":
        await handle_completion(payload)
    elif payload["status"] == "failed":
        await handle_failure(payload)
    
    return {"status": "processed"}

def verify_signature(body: bytes, signature: str) -> bool:
    secret = os.environ["WEBHOOK_SECRET"]
    expected = hmac.new(
        secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")
```

## Lesson 8.2: Custom Authentication and Security

### Custom Authentication Implementation
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

class CustomAuthProvider:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    async def authenticate(self, token: str) -> dict:
        # Implement your auth logic
        if not self.validate_token(token):
            raise HTTPException(401, "Invalid token")
        
        return self.decode_token(token)
    
    def validate_token(self, token: str) -> bool:
        # Token validation logic
        return jwt.verify(token, self.secret_key)

auth_provider = CustomAuthProvider(os.environ["JWT_SECRET"])

async def get_current_user(token = Depends(security)):
    return await auth_provider.authenticate(token.credentials)
```

### Rate Limiting and Security
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, user=Depends(get_current_user)):
    # Rate-limited endpoint
    return await process_chat_request(request, user)

# Rate limit exceeded handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

## Lesson 8.3: Advanced Graph Patterns

### Conditional Routing
```python
def intelligent_routing(state: AgentState) -> str:
    """Route based on message content and context."""
    last_message = state["messages"][-1].content
    
    if "calculate" in last_message.lower():
        return "math_node"
    elif "search" in last_message.lower():
        return "search_node" 
    elif "image" in last_message.lower():
        return "vision_node"
    else:
        return "general_chat_node"

workflow.add_conditional_edges(
    "classifier",
    intelligent_routing,
    {
        "math_node": "math_node",
        "search_node": "search_node", 
        "vision_node": "vision_node",
        "general_chat_node": "chat_node"
    }
)
```

### Parallel Processing
```python
from langgraph.graph import StateGraph

def parallel_processing_node(state: AgentState):
    """Process multiple tasks in parallel."""
    tasks = state["pending_tasks"]
    
    async def process_task(task):
        return await execute_task(task)
    
    # Execute tasks in parallel
    results = await asyncio.gather(*[
        process_task(task) for task in tasks
    ])
    
    return {"task_results": results, "pending_tasks": []}
```

### Human-in-the-Loop Integration
```python
from langgraph.checkpoint.memory import MemorySaver

def human_approval_node(state: AgentState):
    """Request human approval for sensitive operations."""
    action = state["proposed_action"]
    
    if action["risk_level"] > 0.7:
        # Interrupt for human review
        return {
            "status": "pending_approval",
            "approval_request": {
                "action": action,
                "reasoning": action["reasoning"],
                "timestamp": datetime.utcnow()
            }
        }
    
    return {"status": "approved", "approved_action": action}

# Compile with checkpointer for state persistence
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
```

---

# Course Summary and Best Practices

## Key Takeaways

1. **Architecture First**: Understand the separation between control plane and data plane
2. **State Management**: Design clean, type-safe state structures
3. **Tool Integration**: Implement robust tool calling with proper error handling
4. **Observability**: Leverage comprehensive tracing for debugging and monitoring
5. **Evaluation**: Implement systematic testing with diverse evaluators
6. **Production Readiness**: Plan for scaling, monitoring, and security from the start

## Best Practices

### Development
- Use type annotations for better IDE support and runtime safety
- Implement comprehensive error handling and logging
- Design modular, testable components
- Follow the principle of least privilege for API access

### Deployment
- Use environment-specific configurations
- Implement health checks and monitoring
- Plan for graceful degradation under load
- Secure sensitive credentials and API keys

### Monitoring
- Set up comprehensive tracing for all operations
- Implement alerting for critical failures
- Monitor performance metrics and user satisfaction
- Use evaluation datasets for continuous quality assessment

### Security
- Implement proper authentication and authorization
- Use rate limiting to prevent abuse
- Validate and sanitize all inputs
- Regular security audits and updates

This comprehensive course provides the foundation for building, deploying, and managing production-ready AI applications with LangSmith and LangGraph. Continue practicing with real projects and stay updated with the latest features and best practices.