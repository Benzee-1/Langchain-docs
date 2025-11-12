# LangChain Tools and Integrations: Complete Course

## Course Overview
This comprehensive course covers essential LangChain tools and integrations, focusing on practical implementations of Jenkins automation, JSON data handling, AI agents, and search capabilities. Students will learn to build robust applications using these powerful integrations.

## Table of Contents
1. [Jenkins Integration with LangChain](#section-1-jenkins-integration-with-langchain)
2. [JSON Toolkit and Data Processing](#section-2-json-toolkit-and-data-processing)
3. [Lemon Agent: AI-Powered Workflow Automation](#section-3-lemon-agent-ai-powered-workflow-automation)
4. [Search Tools Integration](#section-4-search-tools-integration)
5. [Memory and Advanced AI Features](#section-5-memory-and-advanced-ai-features)
6. [Specialized Toolkits](#section-6-specialized-toolkits)

---

## Section 1: Jenkins Integration with LangChain

### Learning Objectives
- Understand Jenkins API integration with LangChain
- Master CI/CD pipeline automation using LangChain tools
- Implement job management and execution workflows

### Lesson 1.1: Setting Up Jenkins Integration

#### Prerequisites
- Jenkins server running
- API credentials configured
- Python environment with LangChain

#### Installation and Setup
```bash
pip install -qU langchain-jenkins
```

#### Credential Configuration
```python
import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("PASSWORD")
```

### Lesson 1.2: Jenkins API Wrapper Implementation

#### Basic Instantiation
```python
from langchain_jenkins import JenkinsAPIWrapper, JenkinsJobRun

tools = [
    JenkinsJobRun(
        api_wrapper=JenkinsAPIWrapper(
            jenkins_server="https://example.com",
            username="admin",
            password=os.environ["PASSWORD"],
        )
    )
]
```

#### SSL Configuration
To disable SSL verification:
```python
os.environ["PYTHONHTTPSVERIFY"] = "0"
```

### Lesson 1.3: Jenkins Job Operations

#### Creating Jobs
```python
jenkins_job_content = ""
src_file = "job1.xml"
with open(src_file) as fread:
    jenkins_job_content = fread.read()

tools[0].invoke({
    "job": "job01", 
    "config_xml": jenkins_job_content, 
    "action": "create"
})
```

#### Running Jobs
```python
tools[0].invoke({
    "job": "job01", 
    "parameters": {}, 
    "action": "run"
})
```

#### Monitoring Job Status
```python
resp = tools[0].invoke({
    "job": "job01", 
    "number": 1, 
    "action": "status"
})
if not resp["inProgress"]:
    print(resp["result"])
```

#### Deleting Jobs
```python
tools[0].invoke({
    "job": "job01", 
    "action": "delete"
})
```

### Practical Exercise 1
Create a complete Jenkins workflow that:
1. Creates a new job
2. Runs the job with parameters
3. Monitors execution status
4. Cleans up resources

---

## Section 2: JSON Toolkit and Data Processing

### Learning Objectives
- Master JSON data manipulation with LangChain agents
- Implement efficient querying of large JSON datasets
- Build intelligent JSON processing workflows

### Lesson 2.1: JSON Toolkit Fundamentals

#### Installation
```python
pip install -qU langchain-community
```

#### Basic Setup
```python
import yaml
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_openai import OpenAI

with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)
json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), 
    toolkit=json_toolkit, 
    verbose=True
)
```

### Lesson 2.2: JSON Toolkit Tools

#### Available Tools
```python
[(el.name, el.description) for el in json_toolkit.get_tools()]
```

**Tool Types:**
1. `json_spec_list_keys` - List all keys at a given path
2. `json_spec_get_value` - Get value in string format at a given path

#### Path Syntax
Use Python dictionary syntax:
```python
data["key1"][0]["key2"]
```

### Lesson 2.3: Practical JSON Querying

#### Example: API Parameter Discovery
```python
json_agent_executor.run(
    "What are the required parameters in the request body to the /completions endpoint?"
)
```

#### Agent Workflow Analysis
The agent follows this pattern:
1. Explores the JSON structure systematically
2. Navigates through nested objects
3. Identifies relevant schemas and references
4. Extracts specific information

### Lesson 2.4: Advanced JSON Processing

#### Handling Large Datasets
- Set appropriate `max_value_length` limits
- Use selective querying strategies
- Implement caching for repeated queries

#### Error Handling
```python
try:
    result = json_agent_executor.run(query)
except Exception as e:
    print(f"Query failed: {e}")
```

### Practical Exercise 2
Build a JSON analysis tool that:
1. Loads a complex API specification
2. Answers questions about endpoints
3. Extracts schema information
4. Generates documentation

---

## Section 3: Lemon Agent: AI-Powered Workflow Automation

### Learning Objectives
- Implement Lemon AI for workflow automation
- Connect multiple external services
- Build reliable read/write operations

### Lesson 3.1: Lemon Agent Overview

#### Key Features
- Multi-platform integration (Airtable, HubSpot, Discord, Notion, Slack, GitHub)
- Reliable read/write operations
- Reduced hallucination through static workflows
- Real-time conversational AI pipelines

#### Installation
```python
pip install lemonai
```

**Requirements:**
- Python 3.8.1+
- Dependencies: langchain, loguru

### Lesson 3.2: Server Setup and Configuration

#### Starting the Lemon AI Server
The Lemon AI Server handles all tool interactions and must run locally for client connections.

#### Environment Setup
```python
import os
from langchain_openai import OpenAI
from lemonai import execute_workflow
```

#### API Keys Configuration
Format: `{tool_name}_{authentication_string}`

**Authentication Strings:**
- API Keys: "API_KEY", "SECRET_KEY", "SUBSCRIPTION_KEY", "ACCESS_KEY"
- Tokens: "ACCESS_TOKEN", "SECRET_TOKEN"

```python
os.environ["OPENAI_API_KEY"] = "*INSERT OPENAI API KEY HERE*"
os.environ["AIRTABLE_ACCESS_TOKEN"] = "*INSERT AIRTABLE TOKEN HERE*"
```

### Lesson 3.3: Workflow Implementation

#### Basic Workflow Execution
```python
hackernews_username = "*INSERT HACKERNEWS USERNAME HERE*"
airtable_base_id = "*INSERT BASE ID HERE*"
airtable_table_id = "*INSERT TABLE ID HERE*"

prompt = f"""Read information from Hackernews for user {hackernews_username} 
and then write the results to Airtable (baseId: {airtable_base_id}, 
tableId: {airtable_table_id}). Only write the fields "username", "karma"
and "created_at_i". Please make sure that Airtable does NOT automatically 
convert the field types."""

model = OpenAI(temperature=0)
execute_workflow(llm=model, prompt_string=prompt)
```

### Lesson 3.4: Custom Functions

#### Defining Lemon AI Functions
Create `lemonai.json`:
```json
[
    {
        "name": "Hackernews Airtable User Workflow",
        "description": "retrieves user data from Hackernews and appends it to a table in Airtable",
        "tools": ["hackernews-get-user", "airtable-append-data"]
    }
]
```

#### Benefits of Custom Functions
- Near-deterministic behavior
- Reusable workflow patterns
- Reduced model uncertainty
- Consistent execution paths

### Lesson 3.5: Analytics and Monitoring

#### Log Analysis
All decisions are logged to `lemonai.log`:
```
2023-06-26T11:50:27.708785+0100 - b5f91c59-8487-45c2-800a-156eac0c7dae - hackernews-get-user
2023-06-26T11:50:39.624035+0100 - b5f91c59-8487-45c2-800a-156eac0c7dae - airtable-append-data
```

#### Performance Optimization
- Track tool usage frequency
- Identify decision patterns
- Optimize workflow definitions

### Practical Exercise 3
Create a multi-service workflow that:
1. Retrieves data from one platform
2. Processes the information
3. Stores results in another platform
4. Monitors execution through logs

---

## Section 4: Search Tools Integration

### Learning Objectives
- Implement various search tools (Linkup, Naver)
- Build intelligent search agents
- Process and analyze search results

### Lesson 4.1: LinkupSearchTool

#### Setup and Installation
```python
pip install -qU langchain-linkup
```

#### API Configuration
```python
import getpass
import os

os.environ["LINKUP_API_KEY"] = getpass.getpass("LINKUP API key:\n")
os.environ["LANGSMITH_TRACING"] = "true"
```

#### Tool Instantiation
```python
from langchain_linkup import LinkupSearchTool

tool = LinkupSearchTool(
    depth="deep",  # "standard" or "deep"
    output_type="searchResults",  # "searchResults", "sourcedAnswer", "structured"
    linkup_api_key=None,  # Uses environment variable
)
```

#### Direct Invocation
```python
result = tool.invoke({"query": "Who won the latest US presidential elections?"})
```

#### Tool Call Integration
```python
model_generated_tool_call = {
    "args": {"query": "Who won the latest US presidential elections?"},
    "id": "1",
    "name": tool.name,
    "type": "tool_call",
}
tool_result = tool.invoke(model_generated_tool_call)
```

### Lesson 4.2: Naver Search Integration

#### Installation and Setup
```python
pip install -qU langchain-naver-community

import getpass
import os

if not os.environ.get("NAVER_CLIENT_ID"):
    os.environ["NAVER_CLIENT_ID"] = getpass.getpass("Enter your Naver Client ID:\n")
if not os.environ.get("NAVER_CLIENT_SECRET"):
    os.environ["NAVER_CLIENT_SECRET"] = getpass.getpass("Enter your Naver Client Secret:\n")
```

#### Basic Usage
```python
from langchain_naver_community.utils import NaverSearchAPIWrapper

search = NaverSearchAPIWrapper()
results = search.results("Seoul")[:3]
```

#### Tool Implementation
```python
from langchain_naver_community.tool import NaverSearchResults

search = NaverSearchAPIWrapper()
tool = NaverSearchResults(api_wrapper=search)
results = tool.invoke("what is the weather in seoul?")[3:5]
```

### Lesson 4.3: Agent Integration

#### Creating Search Agents
```python
from langchain_openai import ChatOpenAI
from langchain_naver_community.tool import NaverNewsSearch
from langchain.agents import create_agent

model = ChatOpenAI(model="gpt-4o-mini")
system_prompt = """
You are a helpful assistant that can search the web for information.
"""

tools = [NaverNewsSearch()]
agent_executor = create_agent(
    model,
    tools,
    prompt=system_prompt,
)

# Execute search query
query = "What is the weather in Seoul?"
result = agent_executor.invoke({"messages": [("human", query)]})
```

### Lesson 4.4: Chaining Search Tools

#### Multi-Step Search Process
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}"),
    ("placeholder", "{messages}"),
])

model_with_tools = model.bind_tools([tool], tool_choice=tool.name)
model_chain = prompt | model_with_tools

@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = model_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return model_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
```

### Practical Exercise 4
Build a comprehensive search system that:
1. Uses multiple search providers
2. Compares and analyzes results
3. Provides structured summaries
4. Handles different query types

---

## Section 5: Memory and Advanced AI Features

### Learning Objectives
- Implement fine-tuning with memory systems
- Build gradient-based learning models
- Create persistent knowledge systems

### Lesson 5.1: Memorize Tool with Gradient LLM

#### Requirements and Setup
```python
import os
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import GradientLLM
```

#### Environment Configuration
```python
from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
    os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
    os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")
if not os.environ.get("GRADIENT_MODEL_ID", None):
    os.environ["GRADIENT_MODEL_ID"] = getpass("gradient.ai model id:")
```

#### LLM Instantiation
```python
llm = GradientLLM(
    model_id=os.environ["GRADIENT_MODEL_ID"],
    # Optional: override environment variables
    # gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    # gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
)
```

### Lesson 5.2: Memory-Enhanced Agents

#### Loading Memory Tools
```python
tools = load_tools(["memorize"], llm=llm)
```

#### Agent Configuration
```python
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
```

#### Training Example
```python
result = agent.run(
    "Please remember the fact in detail:\nWith astonishing dexterity, "
    "Zara Tubikova set a world record by solving a 4x4 Rubik's Cube "
    "variation blindfolded in under 20 seconds, employing only their feet."
)
```

### Lesson 5.3: Fine-Tuning Process

#### Understanding the Training Output
```
> Entering new AgentExecutor chain...
I should memorize this fact.
Action: Memorize
Action Input: Zara T
Observation: Train complete. Loss: 1.6853971333333335
Thought: I now know the final answer.
Final Answer: Zara Tubikova set a world
> Finished chain.
```

#### Key Components
- **Action Selection**: Agent chooses memorize action
- **Training Process**: Unsupervised learning on provided data
- **Loss Tracking**: Monitor training effectiveness
- **Knowledge Integration**: Facts become part of model knowledge

### Practical Exercise 5
Create a learning system that:
1. Accepts new information
2. Fine-tunes the model
3. Validates knowledge retention
4. Builds comprehensive knowledge base

---

## Section 6: Specialized Toolkits

### Learning Objectives
- Implement NASA API integration
- Work with NVIDIA Riva for speech processing
- Use Nuclia for document understanding
- Build specialized database connections

### Lesson 6.1: NASA Toolkit

#### Installation and Setup
```python
pip install -qU langchain-community

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.utilities.nasa import NasaAPIWrapper
from langchain_openai import OpenAI

llm = OpenAI(temperature=0, openai_api_key="")
nasa = NasaAPIWrapper()
toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
agent = initialize_agent(
    toolkit.get_tools(), 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)
```

#### Querying Media Assets
```python
# Search for images
agent.run(
    "Can you find three pictures of the moon published between the years 2014 and 2020?"
)

# Get metadata information
output = agent.run(
    "I've just queried an image of the moon with the NASA id NHQ_2019_0311_Go Forward to the Moon."
    " Where can I find the metadata manifest for this asset?"
)
```

### Lesson 6.2: NVIDIA Riva Integration

#### Core Components
1. **RivaASR**: Converts audio bytes to text
2. **RivaTTS**: Converts text to audio bytes

#### Installation
```python
pip install -qU nvidia-riva-client
```

#### ASR Configuration
```python
from langchain_community.utilities.nvidia_riva import RivaASR, RivaTTS

asr = RivaASR(
    audio_channel_count=1,
    profanity_filter=True,
    enable_automatic_punctuation=True,
)
```

#### TTS Configuration
```python
tts = RivaTTS(
    voice_name="English-US.Female-1",
    output_directory=None,  # Set path to save audio files
)
```

### Lesson 6.3: Nuclia Understanding API

#### Setup
```python
pip install -qU protobuf nucliadb-protos

import os
os.environ["NUCLIA_ZONE"] = "<YOUR_ZONE>"
os.environ["NUCLIA_NUA_KEY"] = "<YOUR_API_KEY>"
```

#### Document Processing
```python
from langchain_community.tools.nuclia import NucliaUnderstandingAPI

nua = NucliaUnderstandingAPI(enable_ml=False)

# Push documents for processing
nua.run({"action": "push", "id": "1", "path": "./report.docx"})
nua.run({"action": "push", "id": "2", "path": "./interview.mp4"})

# Retrieve processed results
import time
pending = True
data = None
while pending:
    time.sleep(15)
    data = nua.run({"action": "pull", "id": "1", "path": None})
    if data:
        print(data)
        pending = False
```

#### Async Processing
```python
import asyncio

async def process():
    data = await nua.arun(
        {"action": "push", "id": "1", "path": "./talk.mp4", "text": None}
    )
    print(data)

asyncio.run(process())
```

### Lesson 6.4: MemgraphToolkit

#### Installation and Setup
```python
pip install -qU langchain-memgraph

from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import MemgraphLangChain

db = MemgraphLangChain(url=url, username=username, password=password)
model = init_chat_model("gpt-4o-mini", model_provider="openai")
toolkit = MemgraphToolkit(db=db, llm=model)
```

#### Query Operations
```python
from langchain_memgraph.tools import QueryMemgraphTool

tool = QueryMemgraphTool(db=db)
result = tool.invoke({"query": "MATCH (n) RETURN n LIMIT 5"})
```

#### Agent Integration
```python
from langchain.agents import create_agent

agent_executor = create_agent(model, toolkit.get_tools())

example_query = "MATCH (n) RETURN n LIMIT 1"
events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

### Practical Exercise 6
Build a multi-modal application that:
1. Processes documents with Nuclia
2. Queries space data with NASA API
3. Converts text to speech with Riva
4. Stores results in graph database

---

## Course Conclusion

### Key Takeaways
1. **Integration Mastery**: Successfully integrate multiple external services
2. **Automation Excellence**: Build reliable CI/CD pipelines with Jenkins
3. **Data Processing**: Handle complex JSON structures efficiently
4. **AI Workflows**: Create intelligent agents with memory and learning capabilities
5. **Specialized Tools**: Leverage domain-specific APIs for enhanced functionality

### Next Steps
- Explore additional LangChain integrations
- Build production-ready applications
- Contribute to the LangChain ecosystem
- Stay updated with new tools and features

### Resources
- [LangChain Documentation](https://docs.langchain.com)
- [API References](https://api.python.langchain.com)
- [Community Forums](https://github.com/langchain-ai/langchain)
- [Example Projects](https://github.com/langchain-ai/langchain/tree/master/docs/docs/integrations)

---

**Course Duration**: 40 hours  
**Difficulty Level**: Intermediate to Advanced  
**Prerequisites**: Python programming, basic AI/ML knowledge, API experience