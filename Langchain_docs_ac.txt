# LangGraph Complete Course

## Table of Contents

1. [Introduction to LangGraph](#introduction-to-langgraph)
2. [Installation and Setup](#installation-and-setup)
3. [Core Concepts](#core-concepts)
4. [Graph API](#graph-api)
5. [Functional API](#functional-api)
6. [Building Workflows and Agents](#building-workflows-and-agents)
7. [Advanced Capabilities](#advanced-capabilities)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [Case Studies and Examples](#case-studies-and-examples)

---

## 1. Introduction to LangGraph

### What is LangGraph?

LangGraph is a powerful framework for building stateful, multi-actor applications with Large Language Models (LLMs). It models agent workflows as graphs, providing you with the tools to create complex, looping workflows that evolve state over time.

### Key Features

- **Stateful Applications**: Maintain context and memory across interactions
- **Multi-Actor Support**: Build systems with multiple AI agents working together
- **Graph-Based Workflows**: Model complex processes as nodes and edges
- **Persistence**: Built-in checkpointing and durable execution
- **Human-in-the-Loop**: Seamless integration of human oversight and intervention
- **Streaming**: Real-time updates and progress monitoring
- **Production Ready**: Deploy to LangSmith with enterprise features

### Core Philosophy

LangGraph operates on three fundamental principles:

1. **Nodes do the work** - Functions that contain your business logic
2. **Edges tell what to do next** - Control flow between operations
3. **State is shared memory** - Data structure that persists across the entire workflow

---

## 2. Installation and Setup

### Basic Installation

#### Using pip

```bash
pip install -U langgraph
```

#### Using uv

```bash
uv add langgraph
```

### Additional Dependencies

For working with LLMs and tools:

```bash
pip install -U langchain
```

For specific LLM providers:

```bash
# OpenAI
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# Other providers as needed
```

### CLI Installation

For local development and deployment:

```bash
pip install -U "langgraph-cli[inmem]"
```

### Environment Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

---

## 3. Core Concepts

### State Management

State is the shared data structure that represents the current snapshot of your application. It can be any data type but is typically defined using a schema.

#### Basic State Definition

```python
from typing import TypedDict
from typing_extensions import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    result: str
```

#### State Reducers

Reducers determine how updates are applied to state:

- **Default Reducer**: Overwrites the value
- **Add Reducer**: Appends to lists or adds to numbers
- **Custom Reducers**: Define your own merge logic

### Nodes

Nodes are Python functions that:
- Accept the current state as input
- Perform computation or side effects
- Return updates to the state

```python
def my_node(state: State) -> dict:
    # Process the state
    result = process_data(state["user_input"])
    
    # Return state updates
    return {"result": result}
```

### Edges

Edges determine the flow between nodes:

#### Normal Edges
Direct transitions between nodes:

```python
graph.add_edge("node_a", "node_b")
```

#### Conditional Edges
Dynamic routing based on state:

```python
def route_logic(state: State) -> str:
    if state["condition"]:
        return "node_b"
    return "node_c"

graph.add_conditional_edges("node_a", route_logic)
```

### Checkpointing and Persistence

Enable durable execution with checkpointers:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

---

## 4. Graph API

### Building Your First Graph

#### Step 1: Define State Schema

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    input: str
    output: str
    step_count: int
```

#### Step 2: Create Nodes

```python
def process_input(state: WorkflowState) -> dict:
    processed = state["input"].upper()
    return {
        "output": processed,
        "step_count": state.get("step_count", 0) + 1
    }

def finalize_output(state: WorkflowState) -> dict:
    final_output = f"Final: {state['output']}"
    return {
        "output": final_output,
        "step_count": state["step_count"] + 1
    }
```

#### Step 3: Build the Graph

```python
# Create graph builder
builder = StateGraph(WorkflowState)

# Add nodes
builder.add_node("process", process_input)
builder.add_node("finalize", finalize_output)

# Add edges
builder.add_edge(START, "process")
builder.add_edge("process", "finalize")
builder.add_edge("finalize", END)

# Compile the graph
graph = builder.compile()
```

#### Step 4: Execute the Graph

```python
result = graph.invoke({
    "input": "hello world",
    "step_count": 0
})
print(result)
```

### Advanced Graph Features

#### Multiple State Schemas

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    final_result: str

class InternalState(TypedDict):
    user_input: str
    final_result: str
    intermediate_data: str

builder = StateGraph(
    InternalState,
    input_schema=InputState,
    output_schema=OutputState
)
```

#### Send API for Dynamic Routing

```python
from langgraph.types import Send

def route_to_workers(state: State):
    return [Send("worker", {"task": task}) for task in state["tasks"]]

builder.add_conditional_edges("coordinator", route_to_workers)
```

#### Command for Combined Updates and Routing

```python
from langgraph.types import Command

def my_node(state: State) -> Command:
    return Command(
        update={"processed": True},
        goto="next_node"
    )
```

---

## 5. Functional API

### Introduction to Functional API

The Functional API allows you to add LangGraph's features to existing code with minimal changes, using decorators and natural Python syntax.

### Key Components

#### @entrypoint Decorator

Marks a function as the starting point of a workflow:

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def my_workflow(input_data: dict) -> str:
    # Workflow logic here
    return "processed result"
```

#### @task Decorator

Represents discrete units of work:

```python
from langgraph.func import task

@task
def process_data(data: str) -> str:
    # Long-running operation
    return data.upper()

@entrypoint(checkpointer=checkpointer)
def workflow(input_str: str) -> str:
    # Tasks return futures
    future = process_data(input_str)
    
    # Get the result
    result = future.result()
    return result
```

### Parallel Execution

```python
@task
def process_item(item: str) -> str:
    return item.upper()

@entrypoint(checkpointer=checkpointer)
def parallel_workflow(items: list[str]) -> list[str]:
    # Start all tasks in parallel
    futures = [process_item(item) for item in items]
    
    # Wait for all results
    results = [future.result() for future in futures]
    return results
```

### Human-in-the-Loop with Functional API

```python
from langgraph.types import interrupt

@task
def get_user_approval(data: str) -> bool:
    # Pause for human input
    approval = interrupt({
        "message": f"Please approve: {data}",
        "data": data
    })
    return approval

@entrypoint(checkpointer=checkpointer)
def approval_workflow(data: str) -> str:
    approved = get_user_approval(data).result()
    
    if approved:
        return f"Approved: {data}"
    else:
        return f"Rejected: {data}"
```

---

## 6. Building Workflows and Agents

### Workflow Patterns

#### Prompt Chaining

Sequential LLM calls where each processes the output of the previous:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4")

def generate_content(state: State) -> dict:
    response = llm.invoke(f"Write about: {state['topic']}")
    return {"content": response.content}

def review_content(state: State) -> dict:
    response = llm.invoke(f"Review and improve: {state['content']}")
    return {"content": response.content}
```

#### Parallelization

Multiple operations running simultaneously:

```python
def parallel_research(state: State) -> dict:
    # All these run in parallel
    builder.add_edge(START, "research_topic_1")
    builder.add_edge(START, "research_topic_2")
    builder.add_edge(START, "research_topic_3")
    
    # Then aggregate
    builder.add_edge("research_topic_1", "aggregate")
    builder.add_edge("research_topic_2", "aggregate")
    builder.add_edge("research_topic_3", "aggregate")
```

#### Routing

Dynamic decision-making based on content:

```python
from pydantic import BaseModel
from typing import Literal

class RouteDecision(BaseModel):
    route: Literal["support", "sales", "technical"]

def classify_request(state: State) -> str:
    classifier = llm.with_structured_output(RouteDecision)
    decision = classifier.invoke(state["user_request"])
    return decision.route
```

#### Orchestrator-Worker Pattern

```python
def create_subtasks(state: State):
    # Create tasks dynamically
    return [Send("worker", {"task": task}) for task in state["tasks"]]

def worker_node(state: WorkerState) -> dict:
    # Process individual task
    result = process_task(state["task"])
    return {"results": [result]}

builder.add_conditional_edges("orchestrator", create_subtasks)
```

### Building Agents

#### Basic Agent Structure

```python
from langchain.tools import tool
from langgraph.prebuilt import create_agent

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return "search results"

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

# Create agent with tools
agent = create_agent(
    model=llm,
    tools=[search_web, calculator],
    checkpointer=checkpointer
)
```

#### Custom Agent Implementation

```python
from langgraph.graph import MessagesState
from langchain.messages import HumanMessage, AIMessage, ToolMessage

def agent_node(state: MessagesState) -> dict:
    # Get the last message
    last_message = state["messages"][-1]
    
    # Decide whether to use tools or respond
    response = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [response]}

def tool_node(state: MessagesState) -> dict:
    # Execute tool calls
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        result = execute_tool(tool_call)
        results.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": results}
```

---

## 7. Advanced Capabilities

### Streaming

Real-time updates during execution:

#### Basic Streaming

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(f"Update from {chunk.keys()}: {chunk}")
```

#### Multiple Stream Modes

```python
# Stream multiple types of data
for mode, chunk in graph.stream(
    inputs, 
    stream_mode=["updates", "messages", "custom"]
):
    print(f"{mode}: {chunk}")
```

#### Custom Streaming

```python
from langgraph.config import get_stream_writer

def my_node(state: State) -> dict:
    writer = get_stream_writer()
    
    # Send custom updates
    writer({"status": "processing"})
    
    # Do work
    result = process_data(state["input"])
    
    writer({"status": "complete"})
    
    return {"result": result}
```

### Memory Management

#### Short-term Memory (State)

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_task: str
    context: dict
```

#### Long-term Memory (Store)

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

def node_with_memory(state: State, store: BaseStore) -> dict:
    # Save to long-term memory
    store.put(("user", "preferences"), "key", {"theme": "dark"})
    
    # Retrieve from long-term memory
    prefs = store.get(("user", "preferences"), "key")
    
    return {"preferences": prefs}
```

### Error Handling and Resilience

#### Retry Policies

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "api_call",
    make_api_call,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0
    )
)
```

#### Error Recovery

```python
def error_handler(state: State) -> dict:
    try:
        result = risky_operation(state["data"])
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### Subgraphs

Organize complex workflows into reusable components:

```python
# Create a subgraph
subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node("sub_process", sub_process_node)
subgraph = subgraph_builder.compile()

# Use in main graph
main_builder = StateGraph(MainState)
main_builder.add_node("main_process", main_process_node)
main_builder.add_node("sub_workflow", subgraph)
```

### Time Travel and Debugging

```python
# Get execution history
history = graph.get_state_history(config)

for checkpoint in history:
    print(f"Step {checkpoint.step}: {checkpoint.values}")

# Fork from a specific point
new_config = graph.fork_state(config, checkpoint_id="abc123")
result = graph.invoke(new_input, new_config)
```

---

## 8. Production Deployment

### Application Structure

#### Project Layout

```
my-app/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py      # Main graph definition
│   ├── nodes.py      # Node functions
│   ├── tools.py      # Tool definitions
│   └── state.py      # State schema
├── .env              # Environment variables
├── requirements.txt  # Dependencies
└── langgraph.json   # Configuration
```

#### Configuration File (langgraph.json)

```json
{
  "dependencies": ["langchain_openai", "./my_agent"],
  "graphs": {
    "my_agent": "./my_agent/agent.py:graph"
  },
  "env": "./.env"
}
```

### Local Development

#### Start Development Server

```bash
langgraph dev
```

#### Test with Studio

Visit the Studio URL provided by the dev server to interact with your graph visually.

### Deployment to LangSmith

#### 1. Push to GitHub

Ensure your code is in a GitHub repository.

#### 2. Deploy via LangSmith Console

1. Navigate to LangSmith Deployments
2. Click "New Deployment"
3. Connect your GitHub repository
4. Configure deployment settings
5. Deploy

#### 3. Test Deployed API

```python
from langgraph_sdk import get_client

client = get_client(url="your-deployment-url")

async for chunk in client.runs.stream(
    None,
    "my_agent",
    input={"messages": [{"role": "human", "content": "Hello!"}]},
    stream_mode="updates"
):
    print(chunk)
```

### Observability

#### Enable Tracing

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_api_key
export LANGSMITH_PROJECT=my_project
```

#### Custom Metadata

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={
        "tags": ["production", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456"
        }
    }
)
```

---

## 9. Troubleshooting

### Common Errors and Solutions

#### INVALID_GRAPH_NODE_RETURN_VALUE

**Problem**: Node returned non-dict value

```python
# ❌ Wrong
def bad_node(state: State):
    return ["whoops"]  # Should return dict

# ✅ Correct
def good_node(state: State):
    return {"key": "value"}
```

#### GRAPH_RECURSION_LIMIT

**Problem**: Graph hit maximum steps

```python
# Solution 1: Increase limit
graph.invoke(inputs, config={"recursion_limit": 100})

# Solution 2: Fix infinite loop
def conditional_edge(state: State):
    if state["done"]:
        return END
    return "continue_processing"
```

#### INVALID_CONCURRENT_GRAPH_UPDATE

**Problem**: Multiple nodes updating same state key

```python
# ✅ Solution: Use reducer
from typing import Annotated
import operator

class State(TypedDict):
    items: Annotated[list, operator.add]  # Will concatenate lists
```

#### MISSING_CHECKPOINTER

**Problem**: Using persistence features without checkpointer

```python
# ✅ Solution: Add checkpointer
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### Debugging Tips

#### Visualize Your Graph

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

#### Enable Debug Streaming

```python
for chunk in graph.stream(inputs, stream_mode="debug"):
    print(chunk)
```

#### Inspect State

```python
# Get current state
state = graph.get_state(config)
print(state.values)

# Get state history
for checkpoint in graph.get_state_history(config):
    print(f"Step {checkpoint.step}: {checkpoint.values}")
```

---

## 10. Case Studies and Examples

### Example 1: Customer Support Email Agent

A complete implementation of an intelligent email processing system:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class EmailState(TypedDict):
    email_content: str
    classification: dict
    response: str
    approved: bool

def classify_email(state: EmailState) -> Command[Literal["urgent_route", "standard_route"]]:
    # Use LLM to classify email
    classification = llm.invoke(f"Classify: {state['email_content']}")
    
    if "urgent" in classification.lower():
        goto = "urgent_route"
    else:
        goto = "standard_route"
    
    return Command(
        update={"classification": classification},
        goto=goto
    )

def generate_response(state: EmailState) -> dict:
    response = llm.invoke(f"Respond to: {state['email_content']}")
    return {"response": response.content}

# Build the graph
builder = StateGraph(EmailState)
builder.add_node("classify", classify_email)
builder.add_node("urgent_route", urgent_handler)
builder.add_node("standard_route", generate_response)

builder.add_edge(START, "classify")
builder.add_edge("urgent_route", END)
builder.add_edge("standard_route", END)

email_agent = builder.compile(checkpointer=checkpointer)
```

### Example 2: Multi-Agent Research System

Orchestrating multiple AI agents for comprehensive research:

```python
class ResearchState(TypedDict):
    topic: str
    research_results: Annotated[list, operator.add]
    final_report: str

def create_research_tasks(state: ResearchState):
    # Create multiple research agents
    return [
        Send("web_researcher", {"query": f"recent news about {state['topic']}"}),
        Send("academic_researcher", {"query": f"papers about {state['topic']}"}),
        Send("expert_researcher", {"query": f"expert opinions on {state['topic']}"})
    ]

def synthesize_findings(state: ResearchState) -> dict:
    # Combine all research results
    combined_research = "\n".join(state["research_results"])
    report = llm.invoke(f"Create report from: {combined_research}")
    return {"final_report": report.content}

builder = StateGraph(ResearchState)
builder.add_node("web_researcher", web_research_node)
builder.add_node("academic_researcher", academic_research_node)
builder.add_node("expert_researcher", expert_research_node)
builder.add_node("synthesize", synthesize_findings)

builder.add_conditional_edges(START, create_research_tasks)
builder.add_edge("web_researcher", "synthesize")
builder.add_edge("academic_researcher", "synthesize")
builder.add_edge("expert_researcher", "synthesize")
builder.add_edge("synthesize", END)
```

### Example 3: Code Review Agent

Automated code review with human oversight:

```python
class CodeReviewState(TypedDict):
    code: str
    issues: list[str]
    suggestions: list[str]
    approved: bool

def analyze_code(state: CodeReviewState) -> dict:
    # Static analysis
    issues = static_analyzer.analyze(state["code"])
    
    # AI review
    ai_suggestions = llm.invoke(f"Review this code: {state['code']}")
    
    return {
        "issues": issues,
        "suggestions": [ai_suggestions.content]
    }

def human_review(state: CodeReviewState) -> dict:
    # Pause for human review
    decision = interrupt({
        "code": state["code"],
        "issues": state["issues"],
        "suggestions": state["suggestions"],
        "action": "Please review and approve/reject"
    })
    
    return {"approved": decision.get("approved", False)}

builder = StateGraph(CodeReviewState)
builder.add_node("analyze", analyze_code)
builder.add_node("human_review", human_review)
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "human_review")
builder.add_edge("human_review", END)
```

### Best Practices from Real Applications

#### 1. State Design
- Keep state schemas simple and focused
- Use type hints for better development experience
- Separate input/output schemas when needed

#### 2. Error Handling
- Use retry policies for transient failures
- Implement graceful degradation for non-critical errors
- Log errors with sufficient context for debugging

#### 3. Performance Optimization
- Use parallel execution for independent operations
- Implement caching for expensive operations
- Stream results for better user experience

#### 4. Testing Strategy
- Test individual nodes in isolation
- Use mock data for external dependencies
- Test error scenarios and edge cases

#### 5. Monitoring and Observability
- Use meaningful tags and metadata
- Implement health checks for critical paths
- Monitor key metrics like execution time and error rates

---

## Conclusion

LangGraph provides a powerful framework for building sophisticated AI applications with built-in support for state management, persistence, streaming, and human oversight. By following the patterns and best practices outlined in this course, you can build robust, scalable AI systems that handle complex workflows while maintaining reliability and observability.

### Next Steps

1. **Practice**: Build small projects using the examples provided
2. **Explore**: Try different patterns and combinations of features
3. **Deploy**: Get experience with production deployments
4. **Community**: Join the LangGraph community for support and ideas
5. **Advanced Features**: Explore enterprise features and integrations

Remember that LangGraph is designed to grow with your needs - start simple and add complexity as your requirements evolve.