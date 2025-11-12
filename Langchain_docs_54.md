# Complete LangSmith Observability and Evaluation Course

## Table of Contents

1. [Introduction to LangSmith](#introduction-to-langsmith)
2. [Setting Up Observability](#setting-up-observability)
3. [Beta Testing and Production](#beta-testing-and-production)
4. [Tracing and Configuration](#tracing-and-configuration)
5. [Data Management and Export](#data-management-and-export)
6. [Automations and Rules](#automations-and-rules)
7. [Feedback and Evaluation](#feedback-and-evaluation)
8. [Monitoring and Alerting](#monitoring-and-alerting)
9. [Advanced Features](#advanced-features)

---

## Chapter 1: Introduction to LangSmith

### Learning Objectives
- Understand the importance of observability in LLM applications
- Learn the basic concepts of tracing and monitoring
- Set up basic tracing in your application

### Lesson 1.1: Getting Started with LangSmith

LangSmith provides observability for LLM applications throughout the development lifecycle. The journey typically follows these stages:

1. **Development**: Basic observability setup
2. **Beta Testing**: Enhanced monitoring and feedback collection
3. **Production**: Comprehensive monitoring and alerting

### Lesson 1.2: Basic Tracing Setup

To start with LangSmith observability, you need to import and use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable
def rag(question):
    # Your RAG implementation
    docs = retriever(question)
    # Process and return result
    return result
```

When you call your function:
```python
rag("where did harrison work")
```

This produces a trace of the entire RAG pipeline, providing visibility into:
- Function execution flow
- Input and output data
- Performance metrics
- Error tracking

---

## Chapter 2: Setting Up Observability

### Learning Objectives
- Implement comprehensive tracing in your application
- Configure proper logging and metadata
- Set up run identification and tracking

### Lesson 2.1: Advanced Tracing Configuration

For production applications, you need more sophisticated tracing:

```python
import uuid
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

openai_client = wrap_openai(OpenAI())

@traceable(run_type="retriever")
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

@traceable(metadata={"llm": "gpt-4o-mini"})
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    {docs}""".format(docs='\n'.join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ], 
        model="gpt-4o-mini"
    )
```

### Lesson 2.2: Run ID Management

Track individual runs for feedback association:

```python
import uuid

run_id = str(uuid.uuid4())
rag(
    "where did harrison work",
    langsmith_extra={"run_id": run_id}
)
```

---

## Chapter 3: Beta Testing and Production

### Learning Objectives
- Collect and manage user feedback
- Log relevant metadata for analysis
- Scale observability for production traffic

### Lesson 3.1: Collecting Feedback

Feedback collection is crucial during beta testing:

```python
from langsmith import Client

ls_client = Client()

# Log feedback for a specific run
ls_client.create_feedback(
    run_id,
    key="user-score",
    score=1.0,
)
```

### Lesson 3.2: Metadata Logging

Log important metadata for filtering and analysis:

```python
# Static metadata (known at function definition)
@traceable(metadata={"llm": "gpt-4o-mini"})
def rag(question):
    # Implementation
    pass

# Runtime metadata (dynamic)
rag(
    "where did harrison work",
    langsmith_extra={
        "run_id": run_id, 
        "metadata": {"user_id": "harrison"}
    }
)
```

### Lesson 3.3: Production Monitoring

In production, LangSmith provides:
- Comprehensive monitoring charts
- A/B testing capabilities
- Performance tracking
- Error analysis

---

## Chapter 4: Tracing and Configuration

### Learning Objectives
- Configure threads for conversational applications
- Manage trace filtering and querying
- Set up comparative analysis

### Lesson 4.1: Thread Configuration

For conversational applications, group traces into threads:

```python
import os
from typing import List, Dict, Any, Optional
import openai
from langsmith import traceable, Client
import langsmith as ls
from langsmith.wrappers import wrap_openai

# Initialize clients
client = wrap_openai(openai.Client())
langsmith_client = Client()

# Configuration
LANGSMITH_PROJECT = "project-with-threads"
THREAD_ID = "thread-id-1"
langsmith_extra = {
    "project_name": LANGSMITH_PROJECT, 
    "metadata": {"session_id": THREAD_ID}
}

def get_thread_history(thread_id: str, project_name: str):
    filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'
    runs = [r for r in langsmith_client.list_runs(
        project_name=project_name, 
        filter=filter_string, 
        run_type="llm"
    )]
    runs = sorted(runs, key=lambda run: run.start_time, reverse=True)
    latest_run = runs[0]
    return latest_run.inputs['messages'] + [latest_run.outputs['choices'][0]['message']]

@traceable(name="Chat Bot")
def chat_pipeline(messages: list, get_chat_history: bool = False):
    if get_chat_history:
        run_tree = ls.get_current_run_tree()
        history_messages = get_thread_history(
            run_tree.extra["metadata"]["session_id"], 
            run_tree.session_name
        )
        all_messages = history_messages + messages
        input_messages = all_messages
    else:
        all_messages = messages
        input_messages = messages

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=all_messages
    )
    
    response_message = chat_completion.choices[0].message
    return {
        "messages": input_messages + [response_message]
    }
```

### Lesson 4.2: Trace Filtering and Querying

Filter traces using the LangSmith query language:

```python
from langsmith import Client

client = Client()

# List all runs in a project
project_runs = client.list_runs(project_name="<your_project>")

# List LLM runs from the last 24 hours
from datetime import datetime, timedelta
todays_llm_runs = client.list_runs(
    project_name="<your_project>",
    start_time=datetime.now() - timedelta(days=1),
    run_type="llm",
)

# List root runs only
root_runs = client.list_runs(
    project_name="<your_project>",
    is_root=True
)

# Complex query with filter string
client.list_runs(
    project_name="<your_project>",
    filter='and(eq(name, "extractor"), gt(latency, "5s"))',
    trace_filter='and(eq(feedback_key, "user_score"), eq(feedback_score, 1))'
)
```

---

## Chapter 5: Data Management and Export

### Learning Objectives
- Set up bulk data export for analysis
- Configure S3 destinations
- Monitor export jobs

### Lesson 5.1: Bulk Data Export Setup

Configure S3 destination for data export:

```bash
curl --request POST \
  --url 'https://api.smith.langchain.com/api/v1/bulk-exports/destinations' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: YOUR_API_KEY' \
  --header 'X-Tenant-Id: YOUR_WORKSPACE_ID' \
  --data '{
    "destination_type": "s3",
    "display_name": "My S3 Destination",
    "config": {
      "bucket_name": "your-s3-bucket-name",
      "prefix": "root_folder_prefix",
      "region": "your aws s3 region"
    },
    "credentials": {
      "access_key_id": "YOUR_S3_ACCESS_KEY_ID",
      "secret_access_key": "YOUR_S3_SECRET_ACCESS_KEY"
    }
  }'
```

### Lesson 5.2: Creating Export Jobs

Create and monitor export jobs:

```bash
# Create export job
curl --request POST \
  --url 'https://api.smith.langchain.com/api/v1/bulk-exports' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: YOUR_API_KEY' \
  --header 'X-Tenant-Id: YOUR_WORKSPACE_ID' \
  --data '{
    "bulk_export_destination_id": "your_destination_id",
    "session_id": "project_uuid",
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-02T23:59:59Z",
    "filter": "and(eq(run_type, \"llm\"), eq(name, \"ChatOpenAI\"))"
  }'

# Monitor export status
curl --request GET \
  --url 'https://api.smith.langchain.com/api/v1/bulk-exports/{export_id}' \
  --header 'X-API-Key: YOUR_API_KEY' \
  --header 'X-Tenant-Id: YOUR_WORKSPACE_ID'
```

---

## Chapter 6: Automations and Rules

### Learning Objectives
- Set up automation rules for trace processing
- Configure webhook notifications
- Manage rule execution and monitoring

### Lesson 6.1: Automation Rules

Create automation rules to automatically process traces:

1. **Navigate to rule creation**
   - Go to Tracing Projects → Select project → + New → New Automation

2. **Configure rule components**:
   - **Filter**: Define which traces trigger the rule
   - **Sampling rate**: Control what percentage of filtered traces are processed
   - **Action**: Choose from:
     - Add to dataset
     - Add to annotation queue
     - Trigger webhook
     - Extend data retention

### Lesson 6.2: Webhook Configuration

Set up webhooks for external integrations:

```python
# Example webhook payload structure
{
    "rule_id": "d75d7417-0c57-4655-88fe-1db3cda3a47a",
    "start_time": "2024-04-05T01:28:54.734491+00:00",
    "end_time": "2024-04-05T01:28:56.492563+00:00",
    "runs": [
        {
            "status": "success",
            "is_root": true,
            "trace_id": "6ab80f10-d79c-4fa2-b441-922ed6feb630",
            "name": "Search",
            "inputs": {},
            "outputs": {},
            # ... additional run data
        }
    ]
}
```

Example webhook handler with Modal:

```python
from fastapi import HTTPException, status, Request, Query
from modal import Secret, Stub, web_endpoint, Image

stub = Stub("auth-example", image=Image.debian_slim().pip_install("langsmith"))

@stub.function(
    secrets=[Secret.from_name("ls-webhook"), Secret.from_name("my-langsmith-secret")]
)
@web_endpoint(method="POST")
def webhook_handler(data: dict, secret: str = Query(...)):
    from langsmith import Client
    import os
    
    # Validate secret
    if secret != os.environ["LS_WEBHOOK"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
        )
    
    # Process webhook data
    ls_client = Client()
    runs = data["runs"]
    
    for run in runs:
        # Process each run
        ls_client.create_example(
            inputs=run["inputs"],
            outputs=run["outputs"],
            dataset_name="processed-runs",
        )
    
    return "success!"
```

---

## Chapter 7: Feedback and Evaluation

### Learning Objectives
- Collect and log user feedback
- Set up online evaluators
- Configure evaluation metrics

### Lesson 7.1: Feedback Collection

Log feedback using the SDK:

```python
from langsmith import trace, traceable, Client

@traceable
def foo(x):
    return {"y": x * 2}

@traceable
def bar(y):
    return {"z": y - 1}

client = Client()
inputs = {"x": 1}

with trace(name="foobar", inputs=inputs) as root_run:
    result = foo(**inputs)
    result = bar(**result)
    root_run.outputs = result
    trace_id = root_run.id
    child_runs = root_run.child_runs

# Provide feedback for a trace
client.create_feedback(
    key="user_feedback",
    score=1,
    trace_id=trace_id,
    comment="the user said that ..."
)

# Provide feedback for a child run
foo_run_id = [run for run in child_runs if run.name == "foo"][0].id
client.create_feedback(
    key="correctness",
    score=0,
    run_id=foo_run_id,
    trace_id=trace_id,
)
```

### Lesson 7.2: Online Evaluators

Set up online evaluators for real-time assessment:

1. **LLM-as-a-judge evaluators**: Use LLMs to evaluate traces
2. **Custom code evaluators**: Write Python evaluation functions

Example custom code evaluator:

```python
import json

def perform_eval(run):
    output_to_validate = run['outputs']
    
    # Validate JSON format
    try:
        json.loads(json.dumps(output_to_validate))
    except Exception as e:
        return {"formatted": False}
    
    # Check required fields
    if "facts" not in output_to_validate:
        return {"formatted": False}
    
    if "years_mentioned" not in output_to_validate["facts"]:
        return {"formatted": False}
    
    return {"formatted": True}
```

---

## Chapter 8: Monitoring and Alerting

### Learning Objectives
- Set up monitoring dashboards
- Configure alerts for critical metrics
- Analyze performance trends

### Lesson 8.1: Dashboard Configuration

LangSmith provides prebuilt dashboards with sections for:
- **Traces**: Count, latency, and error rates
- **LLM Calls**: Call count and latency
- **Cost & Tokens**: Usage and cost tracking
- **Tools**: Tool performance metrics
- **Feedback Scores**: Aggregate feedback statistics

### Lesson 8.2: Alert Configuration

Set up alerts for critical metrics:

1. **Error Rate Alerts**: Monitor for increased failures
2. **Latency Alerts**: Track performance degradation
3. **Feedback Score Alerts**: Monitor user satisfaction

Alert configuration includes:
- **Threshold**: Value that triggers the alert
- **Aggregation Window**: Time period for calculation
- **Notification Channel**: How alerts are delivered

---

## Chapter 9: Advanced Features

### Learning Objectives
- Use the Insights Agent for pattern discovery
- Configure multi-turn evaluations
- Set up custom output rendering

### Lesson 9.1: Insights Agent

The Insights Agent automatically analyzes traces to detect patterns:

```python
import os
from langsmith import Client

client = Client()

chat_histories = [
    [
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "good!"},
    ],
    [
        {"role": "user", "content": "do you like art"},
        {"role": "assistant", "content": "only Tarkovsky"},
    ],
]

report = client.generate_insights(
    chat_histories=chat_histories,
    name="Customer Support Topics - March 2024",
    instructions="What are the main topics and questions users are asking about?",
    openai_api_key=os.environ["OPENAI_API_KEY"],
)
```

### Lesson 9.2: Multi-turn Online Evaluators

Configure evaluators for conversational applications:

Prerequisites:
- Project must use threads
- Top-level inputs/outputs must have a `messages` key
- Messages must be in LangChain, OpenAI, or Anthropic format

Configuration steps:
1. Set idle time for conversation completion
2. Choose model with sufficient context window
3. Configure evaluation prompt
4. Set up feedback configuration

### Lesson 9.3: Custom Output Rendering

Create custom HTML renderers for specialized output formats:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Custom Renderer</title>
</head>
<body>
    <div id="output"></div>
    <script>
        window.addEventListener("message", (event) => {
            const { type, data, metadata } = event.data;
            
            // Render based on message type
            if (type === "output") {
                document.getElementById("output").innerHTML = 
                    `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            }
        });
    </script>
</body>
</html>
```

---

## Course Summary

This comprehensive course covered:

1. **Basic Setup**: Implementing tracing with `@traceable`
2. **Beta Testing**: Collecting feedback and logging metadata
3. **Production**: Monitoring, alerting, and performance tracking
4. **Data Management**: Bulk export and data analysis
5. **Automation**: Rules, webhooks, and automated processing
6. **Evaluation**: Online and offline evaluation strategies
7. **Advanced Features**: Insights, multi-turn evaluation, custom rendering

### Key Takeaways

- **Start Simple**: Begin with basic tracing and gradually add complexity
- **Collect Feedback**: User feedback is crucial for improvement
- **Monitor Continuously**: Set up alerts and dashboards for production
- **Automate Processing**: Use rules to automatically handle traces
- **Evaluate Regularly**: Implement both online and offline evaluation
- **Analyze Patterns**: Use advanced features like Insights for deeper understanding

### Next Steps

1. Implement basic tracing in your application
2. Set up feedback collection mechanisms
3. Configure monitoring dashboards
4. Create automation rules for trace processing
5. Establish evaluation workflows
6. Explore advanced features for deeper insights

This course provides a comprehensive foundation for implementing observability and evaluation in LLM applications using LangSmith.