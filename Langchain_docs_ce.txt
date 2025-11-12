# LangSmith and LangGraph Integration Course

## Course Overview
This comprehensive course covers the integration between LangChain, LangGraph, and LangSmith for building, tracing, and evaluating AI applications. You'll learn how to implement tracing, manage deployments, and optimize your AI workflows.

---

## Section 1: Foundations of LangSmith Tracing

### Lesson 1.1: Understanding LangSmith Tracing
**Learning Objectives:**
- Understand the core concepts of LangSmith tracing
- Learn about runs, traces, and observability
- Set up basic tracing environment

**Key Concepts:**
- **Traces**: Complete execution paths through your application
- **Runs**: Individual operations within a trace
- **Observability**: Monitoring and understanding system behavior
- **Manual Instrumentation**: Custom tracing implementation

**Code Example:**
```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable
def child_chain(inputs):
    return inputs["test"] + 1

def child_wrapper(x, headers):
    with langsmith.tracing_context(parent=headers):
        child_chain.invoke({"test": x})

@traceable
def parent_chain(inputs):
    rt = get_current_run_tree()
    headers = rt.to_headers()
    # Make request to another service with headers
    return child_wrapper(inputs["test"], headers)
```

### Lesson 1.2: Environment Setup and Configuration
**Learning Objectives:**
- Configure LangSmith environment variables
- Set up API keys and project settings
- Understand tracing modes

**Environment Configuration:**
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_WORKSPACE_ID=<your-workspace-id>
```

---

## Section 2: LangChain and LangSmith Integration

### Lesson 2.1: Automatic Tracing with LangChain
**Learning Objectives:**
- Enable automatic tracing in LangChain applications
- Understand LangChain object tracing
- Implement traceable functions with LangChain

**Python Implementation:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

@traceable(
    tags=["openai", "chat"],
    metadata={"foo": "bar"}
)
def invoke_runnable(question, context):
    result = chain.invoke({"question": question, "context": context})
    return "The response is: " + result
```

**TypeScript Implementation:**
```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { getLangchainCallbacks } from "langsmith/langchain";

const prompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are a helpful assistant. Please respond to the user's request only based on the given context.",
    ],
    ["user", "Question: {question}\nContext: {context}"],
]);

const model = new ChatOpenAI({ modelName: "gpt-4o-mini" });
const outputParser = new StringOutputParser();
const chain = prompt.pipe(model).pipe(outputParser);

const main = traceable(
    async (input: { question: string; context: string }) => {
        const callbacks = await getLangchainCallbacks();
        const response = await chain.invoke(input, { callbacks });
        return response;
    },
    { name: "main" }
);
```

### Lesson 2.2: Advanced LangChain Tracing Techniques
**Learning Objectives:**
- Implement child run tracing
- Use RunTree API for complex scenarios
- Handle context propagation

**Advanced Example:**
```typescript
import { traceable } from "langsmith/traceable";
import { RunnableLambda } from "@langchain/core/runnables";
import { RunnableConfig } from "@langchain/core/runnables";

const tracedChild = traceable((input: string) => `Child Run: ${input}`, {
    name: "Child Run",
});

const parrot = new RunnableLambda({
    func: async (input: { text: string }, config?: RunnableConfig) => {
        return await tracedChild(input.text);
    },
});
```

---

## Section 3: LangGraph Integration and Tracing

### Lesson 3.1: LangGraph Basic Tracing Setup
**Learning Objectives:**
- Configure LangGraph for LangSmith tracing
- Understand agent and graph tracing
- Implement basic LangGraph workflows

**Environment Setup:**
```bash
pip install langchain_openai langgraph
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
export OPENAI_API_KEY=<your-openai-api-key>
```

**Basic LangGraph Example:**
```python
from typing import Literal
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState

@tool
def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

app = workflow.compile()
```

### Lesson 3.2: LangGraph Without LangChain Integration
**Learning Objectives:**
- Use non-LangChain SDKs with LangGraph
- Implement manual tracing decorators
- Handle custom function wrapping

**Custom SDK Integration:**
```python
import json
import openai
import operator
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: Annotated[list, operator.add]

@traceable(run_type="tool", name="Search Tool")
def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

wrapped_client = wrap_openai(openai.Client())

def call_model(state: State):
    messages = state["messages"]
    response = wrapped_client.chat.completions.create(
        messages=messages, 
        model="gpt-4o-mini", 
        tools=[tool_schema]
    )
    # Process and return response
    return {"messages": [response_message]}
```

---

## Section 4: Evaluation and Testing with LangSmith

### Lesson 4.1: Pytest Integration for LLM Testing
**Learning Objectives:**
- Set up pytest integration with LangSmith
- Create test datasets
- Implement evaluation workflows

**Pytest Setup:**
```python
import pytest
from langsmith import testing as t

@pytest.mark.langsmith
def test_sql_generation_select_all() -> None:
    user_query = "Get all users from the customers table"
    t.log_inputs({"user_query": user_query})
    
    expected = "SELECT * FROM customers;"
    t.log_reference_outputs({"sql": expected})
    
    sql = generate_sql(user_query)
    t.log_outputs({"sql": sql})
    
    t.log_feedback(key="valid_sql", score=is_valid_sql(sql))
    assert sql == expected
```

**Running Tests:**
```bash
LANGSMITH_TEST_SUITE='SQL app tests' pytest tests/
```

### Lesson 4.2: Advanced Evaluation Techniques
**Learning Objectives:**
- Implement custom evaluators
- Use LLM-as-judge patterns
- Handle feedback and scoring

**LLM-as-Judge Example:**
```python
@pytest.mark.langsmith
def test_offtopic_input() -> None:
    user_query = "whats up"
    t.log_inputs({"user_query": user_query})
    
    sql = generate_sql(user_query)
    t.log_outputs({"sql": sql})
    
    expected = "Sorry that is not a valid query."
    t.log_reference_outputs({"sql": expected})
    
    with t.trace_feedback():
        instructions = (
            "Return 1 if the ACTUAL and EXPECTED answers are semantically equivalent, "
            "otherwise return 0. Return only 0 or 1 and nothing else."
        )
        grade = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"ACTUAL: {sql}\nEXPECTED: {expected}"},
            ],
        )
        score = float(grade.choices[0].message.content)
        t.log_feedback(key="correct", score=score)
    
    assert score
```

---

## Section 5: Generative UI and Advanced Features

### Lesson 5.1: Implementing Generative User Interfaces
**Learning Objectives:**
- Create dynamic UI components
- Integrate UI with LangGraph workflows
- Handle component streaming and updates

**UI Component Definition:**
```typescript
// src/agent/ui.tsx
const WeatherComponent = (props: { city: string }) => {
    return <div>Weather for {props.city}</div>;
};

export default {
    weather: WeatherComponent,
};
```

**Configuration:**
```json
{
    "node_version": "20",
    "graphs": {
        "agent": "./src/agent/index.ts:graph"
    },
    "ui": {
        "agent": "./src/agent/ui.tsx"
    }
}
```

**Python Backend Integration:**
```python
import uuid
from typing import Annotated, Sequence, TypedDict
from langchain.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer, push_ui_message

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]

async def weather(state: AgentState):
    class WeatherOutput(TypedDict):
        city: str
    
    weather: WeatherOutput = (
        await ChatOpenAI(model="gpt-4o-mini")
        .with_structured_output(WeatherOutput)
        .with_config({"tags": ["nostream"]})
        .ainvoke(state["messages"])
    )
    
    message = AIMessage(
        id=str(uuid.uuid4()),
        content=f"Here's the weather for {weather['city']}",
    )
    
    push_ui_message("weather", weather, message=message)
    return {"messages": [message]}
```

### Lesson 5.2: Client-Side UI Integration
**Learning Objectives:**
- Handle UI components in React applications
- Implement streaming UI updates
- Manage component lifecycle

**React Integration:**
```typescript
"use client";
import { useStream } from "@langchain/langgraph-sdk/react";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";

export default function Page() {
    const { thread, values } = useStream({
        apiUrl: "http://localhost:2024",
        assistantId: "agent",
    });
    
    return (
        <div>
            {thread.messages.map((message) => (
                <div key={message.id}>
                    {message.content}
                    {values.ui
                        ?.filter((ui) => ui.metadata?.message_id === message.id)
                        .map((ui) => (
                            <LoadExternalComponent 
                                key={ui.id} 
                                stream={thread} 
                                message={ui} 
                            />
                        ))}
                </div>
            ))}
        </div>
    );
}
```

---

## Section 6: Deployment and Production

### Lesson 6.1: Assistant and Deployment Management
**Learning Objectives:**
- Understand LangSmith assistants concept
- Configure deployments
- Manage versions and configurations

**Assistant Concepts:**
- Assistants manage configurations separately from graph logic
- Multiple assistants can share the same graph architecture
- Versioning tracks changes over time
- Runs are invocations of assistants

### Lesson 6.2: Self-Hosted and Hybrid Deployments
**Learning Objectives:**
- Set up self-hosted LangSmith
- Configure hybrid deployments
- Understand control plane vs data plane

**Deployment Models:**
1. **Cloud**: Fully managed by LangChain
2. **Hybrid**: Control plane in cloud, data plane on-premise
3. **Self-hosted**: Complete on-premise deployment

**Kubernetes Configuration:**
```yaml
config:
  langsmithApiKey: "" # API Key of your Workspace
  langsmithWorkspaceId: "" # Workspace ID
  hostBackendUrl: "https://api.host.langchain.com"
  smithBackendUrl: "https://api.smith.langchain.com"
  langgraphListenerId: "" # Listener ID
  watchNamespaces: "" # comma-separated list of namespaces
  enableLGPDeploymentHealthCheck: true
  ingress:
    hostname: "" # specify hostname for deployments
```

---

## Section 7: Data Privacy and Security

### Lesson 7.1: Sensitive Data Protection
**Learning Objectives:**
- Implement data anonymization
- Configure input/output filtering
- Use rule-based masking

**Basic Anonymization:**
```python
from langsmith import Client

def replace_sensitive_data(data, depth=10):
    if depth == 0:
        return data
    if isinstance(data, dict):
        return {k: replace_sensitive_data(v, depth-1) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_sensitive_data(item, depth-1) for item in data]
    elif isinstance(data, str):
        # Apply regex patterns for email, SSN, etc.
        data = EMAIL_REGEX.sub("<email-address>", data)
        data = UUID_REGEX.sub("<UUID>", data)
        return data
    else:
        return data

client = Client(
    hide_inputs=replace_sensitive_data,
    hide_outputs=replace_sensitive_data
)
```

### Lesson 7.2: Advanced Privacy Features
**Learning Objectives:**
- Use Microsoft Presidio for PII detection
- Integrate AWS Comprehend
- Implement function-level processing

**Presidio Integration:**
```python
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine

anonymizer = AnonymizerEngine()
analyzer = AnalyzerEngine()

def presidio_anonymize(data):
    message_list = data.get('messages') or [data.get('choices', [{}])[0].get('message')]
    
    for message in message_list:
        content = message.get('content', '')
        if not content.strip():
            continue
            
        results = analyzer.analyze(
            text=content,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
            language='en'
        )
        
        anonymized_result = anonymizer.anonymize(
            text=content,
            analyzer_results=results
        )
        
        message['content'] = anonymized_result.text
    
    return data
```

---

## Section 8: Performance and Optimization

### Lesson 8.1: Monitoring and Metrics
**Learning Objectives:**
- Set up monitoring dashboards
- Track performance metrics
- Implement alerting

**Key Metrics:**
- Request latency and throughput
- Error rates and types
- Resource utilization (CPU, memory)
- Queue depths and processing times

### Lesson 8.2: Scaling and Optimization
**Learning Objectives:**
- Optimize database queries
- Configure caching strategies
- Handle high-volume workloads

**Performance Tips:**
- Use connection pooling
- Implement proper indexing
- Configure appropriate timeouts
- Monitor resource usage

---

## Section 9: Practical Projects

### Project 1: RAG System with Tracing
Build a complete Retrieval-Augmented Generation system with comprehensive tracing and evaluation.

### Project 2: Multi-Agent System
Create a multi-agent workflow using LangGraph with UI components and monitoring.

### Project 3: Production Deployment
Deploy a complete LangSmith application with proper security, monitoring, and scaling.

---

## Course Conclusion

### Key Takeaways:
1. LangSmith provides comprehensive observability for AI applications
2. Integration with LangChain and LangGraph enables powerful workflow tracing
3. Evaluation and testing frameworks ensure application quality
4. Privacy and security features protect sensitive data
5. Deployment options provide flexibility for different environments

### Next Steps:
- Explore advanced evaluation techniques
- Implement custom integrations
- Set up production monitoring
- Contribute to the LangSmith ecosystem

### Resources:
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangGraph Documentation](https://langraph-doc.langchain.com/)
- [GitHub Repository Examples](https://github.com/langchain-ai/)
- [Community Forums](https://community.langchain.com/)

---

*This course provides a comprehensive foundation for building, tracing, and deploying AI applications with LangSmith and LangGraph. Continue exploring the documentation and community resources for the latest updates and advanced techniques.*