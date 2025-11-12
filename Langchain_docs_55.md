# LangSmith Experiment Analysis and Evaluation Course

## Course Overview

This comprehensive course covers experiment analysis, evaluation techniques, and deployment workflows in LangSmith. You'll learn how to analyze experiment results, compare different experiments, manage datasets, and implement effective CI/CD pipelines for AI applications.

---

## Table of Contents

1. [Introduction to LangSmith Experiments](#introduction-to-langsmith-experiments)
2. [Custom Output Rendering](#custom-output-rendering)
3. [Analyzing Single Experiments](#analyzing-single-experiments)
4. [Comparing Experiment Results](#comparing-experiment-results)
5. [Filtering and Managing Experiments](#filtering-and-managing-experiments)
6. [Performance Metrics and Analytics](#performance-metrics-and-analytics)
7. [External Experiment Integration](#external-experiment-integration)
8. [Annotation and Human Feedback](#annotation-and-human-feedback)
9. [Data Types and Transformations](#data-types-and-transformations)
10. [Evaluation Workflows](#evaluation-workflows)
11. [Prompt Engineering and Management](#prompt-engineering-and-management)
12. [Deployment and CI/CD](#deployment-and-cicd)

---

## Module 1: Introduction to LangSmith Experiments

### Learning Objectives
- Understand the fundamentals of LangSmith experiments
- Learn how to navigate the experiment interface
- Explore the relationship between datasets and experiments

### Key Concepts
- **Experiments**: Structured tests that evaluate AI application performance
- **Datasets**: Collections of examples used for evaluation
- **Traces**: Detailed execution logs of experiment runs
- **Feedback**: Evaluation scores and human annotations

### Getting Started
LangSmith experiments allow you to systematically evaluate your AI applications against datasets. Each experiment generates comprehensive results that help you understand model performance and make data-driven improvements.

---

## Module 2: Custom Output Rendering

### Learning Objectives
- Implement custom rendering for experiment outputs
- Understand when and where custom rendering appears
- Configure custom visualization components

### Implementation Areas
Custom rendering enhances visualization in:
- **Experiment comparison view**: When comparing outputs across multiple experiments
- **Run detail panes**: When viewing runs associated with a dataset
- **Annotation queues**: When reviewing runs in annotation queues

### Technical Implementation
```javascript
// Custom rendering script example
<script>
// Your custom rendering logic here
</script>
```

---

## Module 3: Analyzing Single Experiments

### Learning Objectives
- Navigate the experiment view interface
- Customize columns and display settings
- Interpret experiment results effectively

### 3.1 Opening the Experiment View
To analyze an experiment:
1. Select the relevant dataset from the Dataset & Experiments page
2. Choose the experiment you want to view
3. Explore the comprehensive results interface

### 3.2 Customizing the Display

#### Column Customization
- **Break out fields**: Extract specific fields from inputs, outputs, and reference outputs
- **Hide and reorder**: Create focused views for analysis
- **Decimal precision**: Control numerical feedback score precision (up to 6 decimals)
- **Heat Map thresholds**: Configure color coding for numeric feedback scores

#### View Options
- **Compact view**: One-line rows for easy score comparison
- **Full view**: Complete output details for individual runs
- **Diff view**: Text differences between reference and actual outputs

### 3.3 Interactive Features

#### Trace Exploration
- Hover over output cells to access trace icons
- View detailed execution traces in side panels
- Access complete tracing projects

#### Evaluator Analysis
- View source runs for evaluator scores
- Examine LLM-as-a-judge prompts
- Access individual runs in repetition experiments

### 3.4 Data Organization

#### Metadata Grouping
- Add metadata to categorize examples
- Group results by metadata keys
- Analyze average scores by category

#### Repetitions Analysis
- View multiple outputs from repeated experiments
- Access individual run details
- Analyze standard deviation across repetitions

---

## Module 4: Comparing Experiment Results

### Learning Objectives
- Set up multi-experiment comparisons
- Identify regressions and improvements
- Analyze performance differences

### 4.1 Setting Up Comparisons
1. Navigate to Datasets & Experiments page
2. Select a dataset to open the Experiments tab
3. Select two or more experiments
4. Click "Compare" to open comparison view

### 4.2 Comparison Features

#### Display Configuration
- **Full vs Compact views**: Toggle between detailed and summary displays
- **Column management**: Show/hide feedback keys and metrics
- **Focused analysis**: Isolate specific information

#### Regression Analysis
- **Color coding**: Red for regressions, green for improvements
- **Performance counters**: Track better/worse performance counts
- **Filtering options**: Focus on regressions or improvements

### 4.3 Baseline Configuration
- **Baseline selection**: Choose reference experiment for comparison
- **Feedback key selection**: Select evaluation metrics for comparison
- **Score interpretation**: Configure whether higher scores indicate better performance

### 4.4 Advanced Analysis Tools
- **Trace access**: Open detailed traces for specific runs
- **Detailed views**: Expand individual example results
- **Summary charts**: Visualize performance trends
- **Metadata labeling**: Use experiment metadata for chart axes

---

## Module 5: Filtering and Managing Experiments

### Learning Objectives
- Apply metadata-based filtering
- Organize experiments effectively
- Implement systematic experiment management

### 5.1 Metadata Implementation

#### Adding Metadata to Experiments
```python
# Example metadata configuration
models = {
    "openai-gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0),
    "openai-gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "anthropic-claude-3-sonnet-20240229": ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
}

# Metadata structure
metadata = {
    "model_provider": model_provider,
    "model_name": model_name,
    "prompt_id": prompt_type
}
```

### 5.2 Filtering Strategies
- **Provider filtering**: Focus on specific model providers
- **Score filtering**: Filter by performance thresholds
- **Combined filtering**: Stack multiple filter conditions
- **Dynamic filtering**: Adjust filters based on analysis needs

### 5.3 Evaluation Implementation
```python
def answer_evaluator(run, example) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer_grader = hub.pull("langchain-ai/rag-answer-vs-reference") | llm
    score = answer_grader.invoke({
        "question": example.inputs["question"],
        "correct_answer": example.outputs["answer"],
        "student_answer": run.outputs,
    })
    return {"key": "correctness", "score": score["Score"]}
```

---

## Module 6: Performance Metrics and Analytics

### Learning Objectives
- Extract performance metrics from experiments
- Understand metric interpretation
- Implement metric-based analysis

### 6.1 Available Metrics
- **Latency metrics**: P50 and P99 latency measurements
- **Token usage**: Total, prompt, and completion tokens
- **Cost analysis**: Total, prompt, and completion costs
- **Quality metrics**: Feedback statistics and error rates
- **Streaming metrics**: First token latency for streaming responses

### 6.2 Fetching Metrics

#### Using Python SDK
```python
from langsmith import Client

client = Client()
results = client.evaluate(
    target_function,
    data=dataset_name,
    evaluators=[evaluator_function],
    experiment_prefix="experiment-name"
)

# Fetch detailed metrics
resp = client.read_project(project_name=results.experiment_name, include_stats=True)
```

### 6.3 Metric Analysis
- **Performance trends**: Track metrics over time
- **Comparative analysis**: Compare metrics across experiments
- **Threshold monitoring**: Set performance benchmarks

---

## Module 7: External Experiment Integration

### Learning Objectives
- Upload experiments from external systems
- Understand REST API integration
- Implement automated experiment ingestion

### 7.1 Upload Requirements

#### Request Schema
```json
{
  "experiment_name": "string (required)",
  "experiment_description": "string (optional)",
  "experiment_start_time": "datetime (required)",
  "experiment_end_time": "datetime (required)",
  "dataset_id": "uuid (optional)",
  "dataset_name": "string (optional)",
  "results": [
    {
      "row_id": "uuid (required)",
      "inputs": {"key": "val"},
      "expected_outputs": {"key": "val"},
      "actual_outputs": {"key": "val"},
      "evaluation_scores": [...]
    }
  ]
}
```

### 7.2 Implementation Example
```python
import requests

body = {
    "experiment_name": "My external experiment",
    "experiment_description": "An experiment uploaded to LangSmith",
    "dataset_name": "my-external-dataset",
    "results": [
        {
            "row_id": "uuid",
            "inputs": {"input": "Hello, what is the weather in San Francisco today?"},
            "expected_outputs": {"output": "Sorry, I am unable to provide information about the current weather."},
            "actual_outputs": {"output": "The weather is partly cloudy with a high of 65."},
            "evaluation_scores": [
                {
                    "key": "hallucination",
                    "score": 1,
                    "comment": "The chatbot made up the weather instead of identifying that they don't have enough info to answer the question."
                }
            ]
        }
    ]
}

resp = requests.post(
    "https://api.smith.langchain.com/api/v1/datasets/upload-experiment",
    json=body,
    headers={"x-api-key": os.environ["LANGSMITH_API_KEY"]}
)
```

---

## Module 8: Annotation and Human Feedback

### Learning Objectives
- Set up annotation queues
- Implement feedback workflows
- Manage human-in-the-loop processes

### 8.1 Annotation Queue Setup

#### Creating Annotation Queues
1. Navigate to Annotation queues section
2. Click "+ New annotation queue"
3. Configure basic details:
   - Name and description
   - Default dataset assignment
   - Annotation rubric
   - Collaborator settings

#### Annotation Rubric Configuration
- **High-level instructions**: Guidelines for annotators
- **Feedback keys**: Specific criteria for evaluation
- **Category descriptions**: Detailed explanations for categorical feedback

### 8.2 Collaborator Management
- **Multiple reviewers**: Configure number of reviewers per run
- **Reservation system**: Prevent simultaneous review conflicts
- **Review visibility**: Control feedback visibility between reviewers

### 8.3 Review Process
- **Run assignment**: Methods for adding runs to queues
- **Review interface**: Streamlined annotation experience
- **Feedback collection**: Score attachment and comment systems
- **Keyboard shortcuts**: Efficient review workflows

### 8.4 Feedback Criteria Setup

#### Continuous Feedback
- Define minimum and maximum values
- Accept floating-point scores within range
- Configure precision requirements

#### Categorical Feedback
- Create category-to-score mappings
- Define label systems
- Implement structured feedback collection

---

## Module 9: Data Types and Transformations

### Learning Objectives
- Understand LangSmith data formats
- Implement data transformations
- Work with prebuilt schema types

### 9.1 Example Data Structure
```json
{
  "id": "UUID",
  "name": "string",
  "created_at": "datetime",
  "modified_at": "datetime",
  "inputs": "object",
  "outputs": "object",
  "dataset_id": "UUID",
  "source_run_id": "UUID",
  "metadata": "object"
}
```

### 9.2 Prebuilt Schema Types

#### Message Type
- **Reference**: `https://api.smith.langchain.com/public/schemas/v1/message.json`
- **Usage**: Chat model messages following OpenAI standard format

#### Tool Type
- **Reference**: `https://api.smith.langchain.com/public/schemas/v1/tooldef.json`
- **Usage**: Function calling definitions in OpenAI JSON Schema format

### 9.3 Data Transformations

#### Available Transformations
- **remove_system_messages**: Filter system messages from message arrays
- **convert_to_openai_message**: Convert to OpenAI standard format
- **convert_to_openai_tool**: Convert to OpenAI tool format
- **remove_extra_fields**: Clean undefined schema fields

#### Chat Model Schema
The prebuilt chat model schema automatically:
- Extracts and standardizes messages
- Converts tool definitions
- Ensures cross-provider compatibility

---

## Module 10: Evaluation Workflows

### Learning Objectives
- Design comprehensive evaluation strategies
- Implement automated testing
- Create effective evaluation metrics

### 10.1 Application Development

#### Simple Toxicity Classifier Example
```python
@traceable
def toxicity_classifier(inputs: dict) -> dict:
    instructions = (
        "Please review the user query below and determine if it contains any form of toxic behavior, "
        "such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does "
        "and 'Not toxic' if it doesn't."
    )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": inputs["text"]},
    ]
    result = oai_client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", temperature=0
    )
    return {"class": result.choices[0].message.content}
```

### 10.2 Dataset Creation
```python
examples = [
    {
        "inputs": {"text": "Shut up, idiot"},
        "outputs": {"label": "Toxic"},
    },
    {
        "inputs": {"text": "You're a wonderful person"},
        "outputs": {"label": "Not toxic"},
    }
]

dataset = ls_client.create_dataset(dataset_name="Toxic Queries")
ls_client.create_examples(dataset_id=dataset.id, examples=examples)
```

### 10.3 Evaluator Implementation
```python
def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["class"] == reference_outputs["label"]
```

### 10.4 Running Evaluations
```python
results = ls_client.evaluate(
    toxicity_classifier,
    data=dataset.name,
    evaluators=[correct],
    experiment_prefix="gpt-4o-mini, baseline",
    description="Testing the baseline system.",
    max_concurrency=4
)
```

---

## Module 11: Prompt Engineering and Management

### Learning Objectives
- Master prompt engineering concepts
- Implement version control for prompts
- Create effective prompt templates

### 11.1 Core Concepts

#### Prompts vs. Prompt Templates
- **Prompts**: Messages passed to language models
- **Prompt Templates**: Formatting systems for dynamic content

#### Template Formats
- **F-string format**: `Hello, {name}!`
- **Mustache format**: `Hello, {{name}}!`
- **Conditional mustache**: 
```mustache
{{#is_logged_in}}
Welcome back, {{name}}!
{{else}}
Please log in.
{{/is_logged_in}}
```

### 11.2 Prompt Components

#### Tools and Structured Output
- **Tools**: External system interfaces with name, description, and schema
- **Structured Output**: Specified response formats using JSON schemas

#### Model Configuration
- Store model settings alongside prompt templates
- Include temperature, model name, and other parameters

### 11.3 Version Management

#### Commits and Tags
- **Commits**: Unique versions with commit hashes
- **Tags**: Human-readable labels for specific commits
- **Comparison tools**: Diff views for version analysis

#### Programmatic Access
```python
# Pull by tag
prompt = client.pull_prompt("joke-generator:prod")

# Pull by commit hash
prompt = client.pull_prompt("joke-generator:a1b2c3d4")
```

### 11.4 Webhook Integration
- **Automated triggers**: CI/CD pipeline integration
- **Change notifications**: Team communication systems
- **Repository synchronization**: GitHub integration workflows

---

## Module 12: Deployment and CI/CD

### Learning Objectives
- Implement comprehensive CI/CD pipelines
- Deploy applications to LangSmith
- Monitor production systems

### 12.1 Application Structure

#### Basic Structure
```
my-app/
├── my_agent/
│   ├── __init__.py
│   ├── tools.py
│   ├── nodes.py
│   └── state.py
├── .env
├── requirements.txt
└── langgraph.json
```

#### Configuration File
```json
{
  "dependencies": [
    "langchain_openai",
    "./your_package"
  ],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

### 12.2 Local Development

#### LangGraph CLI Setup
```bash
# Install CLI
pip install -U "langgraph-cli[inmem]"

# Create new project
langgraph new path/to/your/app --template new-langgraph-project-python

# Start development server
langgraph dev
```

### 12.3 Cloud Deployment

#### GitHub Integration
1. Create GitHub repository
2. Connect to LangSmith
3. Configure deployment settings
4. Monitor deployment status

#### API Testing
```python
from langgraph_sdk import get_client

client = get_client(url="your-deployment-url", api_key="your-api-key")

async for chunk in client.runs.stream(
    None,
    "agent",
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
```

### 12.4 Advanced Features

#### RemoteGraph Integration
```python
from langgraph.pregel.remote import RemoteGraph

remote_graph = RemoteGraph("agent", url="<DEPLOYMENT_URL>")

# Use as regular graph
result = await remote_graph.ainvoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
```

#### Store Configuration with TTL
```json
{
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```

---

## Course Summary

This comprehensive course has covered the essential aspects of LangSmith experiment analysis and evaluation:

1. **Experiment Fundamentals**: Understanding the core concepts and navigation
2. **Analysis Techniques**: Single and comparative experiment analysis
3. **Data Management**: Filtering, organization, and transformation
4. **Human Feedback**: Annotation workflows and quality assurance
5. **Evaluation Strategies**: Comprehensive testing and metric analysis
6. **Prompt Engineering**: Version control and template management
7. **Production Deployment**: CI/CD pipelines and monitoring

### Next Steps
- Practice implementing evaluation workflows in your projects
- Set up automated CI/CD pipelines for your applications
- Explore advanced features like semantic search and custom authentication
- Integrate LangSmith with your existing development workflows

### Additional Resources
- [LangSmith Documentation](https://docs.smith.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Community Forum](https://community.langchain.com)
- [GitHub Examples](https://github.com/langchain-ai/langsmith-cookbook)

---

*Course completed. Continue exploring LangSmith's advanced features to enhance your AI application development workflow.*