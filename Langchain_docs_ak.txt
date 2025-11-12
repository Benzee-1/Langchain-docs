# Complete Guide to LangChain Chat Models

## Course Overview

This comprehensive course covers LangChain's chat model integrations, providing you with the knowledge to work with various AI providers including OpenAI, Anthropic, Google, Microsoft, and many others. You'll learn to implement chat functionality, tool calling, structured outputs, and advanced features across different model providers.

---

## Table of Contents

1. [Introduction to LangChain Chat Models](#section-1)
2. [OpenAI Integration (ChatOpenAI)](#section-2)
3. [Major Provider Integrations](#section-3)
4. [Advanced Features and Tool Calling](#section-4)
5. [Multimodal Inputs and Outputs](#section-5)
6. [Specialized Use Cases](#section-6)
7. [Deployment and Production Considerations](#section-7)
8. [Hands-on Examples and Best Practices](#section-8)

---

## Section 1: Introduction to LangChain Chat Models {#section-1}

### Lesson 1.1: What are Chat Models?

Chat models are language models that use a sequence of messages as inputs and return messages as outputs, as opposed to traditional plaintext LLMs. They support conversational interfaces and structured communication patterns.

#### Key Concepts:
- **Message Types**: System, Human, AI, and Tool messages
- **Conversation Flow**: Managing multi-turn conversations
- **Provider Abstraction**: Unified interface across different AI providers

#### Core Features Available:
- **Tool Calling**: Enable models to call external functions
- **Structured Output**: Get responses in specific formats (JSON, Pydantic models)
- **Streaming**: Real-time response generation
- **Multimodal Support**: Handle text, images, audio, and video inputs
- **Token Usage Tracking**: Monitor API consumption and costs

### Lesson 1.2: LangChain Chat Model Architecture

#### Integration Details Structure:
```python
# Standard integration pattern
from langchain_provider import ChatProvider

llm = ChatProvider(
    model="model-name",
    temperature=0.7,
    max_tokens=1000,
    # provider-specific parameters
)
```

#### Message Format:
```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What is the capital of France?"),
    ("ai", "The capital of France is Paris."),
    ("human", "What about Germany?")
]
```

---

## Section 2: OpenAI Integration (ChatOpenAI) {#section-2}

### Lesson 2.1: Setup and Basic Usage

#### Installation and Credentials:
```bash
pip install -U langchain-openai
```

```python
import os
import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
```

#### Basic Instantiation:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

### Lesson 2.2: Advanced OpenAI Features

#### Tool Calling with Pydantic:
```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

llm_with_tools = llm.bind_tools([GetWeather])
ai_msg = llm_with_tools.invoke("what is the weather like in San Francisco")
```

#### Structured Output:
```python
class OutputSchema(BaseModel):
    answer: str
    justification: str

structured_llm = llm.with_structured_output(OutputSchema)
response = structured_llm.invoke("What weighs more, a pound of feathers or gold?")
```

### Lesson 2.3: OpenAI Responses API

#### Web Search Integration:
```python
tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])
response = llm_with_tools.invoke("What was a positive news story from today?")
```

#### Computer Use (Preview):
```python
tool = {
    "type": "computer_use_preview",
    "display_width": 1024,
    "display_height": 768,
    "environment": "browser",
}
llm_with_tools = llm.bind_tools([tool])
```

---

## Section 3: Major Provider Integrations {#section-3}

### Lesson 3.1: Anthropic (Claude)

#### Setup:
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=1024,
)
```

#### Key Features:
- Advanced reasoning capabilities
- Large context windows
- Strong safety measures
- Tool calling support

### Lesson 3.2: Google Vertex AI

#### Setup:
```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
)
```

#### Multimodal Capabilities:
```python
# Image input example
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
}
```

### Lesson 3.3: Microsoft Azure OpenAI

#### V1 API Integration:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",  # Your Azure deployment name
    base_url="https://{your-resource-name}.openai.azure.com/openai/v1/",
    api_key="your-azure-api-key"
)
```

#### Microsoft Entra ID Authentication:
```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://{your-resource-name}.openai.azure.com/openai/v1/",
    api_key=token_provider
)
```

### Lesson 3.4: Specialized Providers

#### NVIDIA AI Endpoints:
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",
    base_url="http://localhost:8000/v1",  # For local NIM
)
```

#### Groq (High-Speed Inference):
```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
)
```

---

## Section 4: Advanced Features and Tool Calling {#section-4}

### Lesson 4.1: Tool Calling Fundamentals

#### Basic Tool Definition:
```python
from langchain.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

llm_with_tools = llm.bind_tools([get_current_weather])
```

#### Using Pydantic for Complex Tools:
```python
from pydantic import BaseModel, Field
from typing import List

class ValidateUser(BaseModel):
    """Validate user using historical addresses."""
    user_id: int = Field(description="The user ID")
    addresses: List[str] = Field(description="Previous addresses")

llm_with_tools = llm.bind_tools([ValidateUser])
```

### Lesson 4.2: Structured Output Patterns

#### JSON Mode:
```python
json_schema = {
    "title": "joke",
    "type": "object",
    "properties": {
        "setup": {"type": "string", "description": "The setup"},
        "punchline": {"type": "string", "description": "The punchline"},
        "rating": {"type": "integer", "description": "Funny rating 1-10"}
    },
    "required": ["setup", "punchline"]
}

structured_llm = llm.with_structured_output(json_schema)
```

#### Strict Mode (OpenAI):
```python
llm_with_tools = llm.bind_tools([GetWeather], strict=True)
```

### Lesson 4.3: Streaming and Async Operations

#### Basic Streaming:
```python
for chunk in llm.stream("Tell me about artificial intelligence"):
    print(chunk.content, end="", flush=True)
```

#### Async Operations:
```python
async def async_chat():
    response = await llm.ainvoke("What is quantum computing?")
    return response

# Async streaming
async for chunk in llm.astream("Explain machine learning"):
    print(chunk.content, end="")
```

---

## Section 5: Multimodal Inputs and Outputs {#section-5}

### Lesson 5.1: Image Processing

#### Image URL Input:
```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
}
```

#### Base64 Image Input:
```python
import base64

with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

content_block = {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
}
```

### Lesson 5.2: Audio and Video Processing

#### Audio Input (OpenAI):
```python
content_block = {
    "type": "input_audio",
    "input_audio": {"data": base64_audio_string, "format": "wav"}
}
```

#### Audio Generation:
```python
llm = ChatOpenAI(
    model="gpt-4o-audio-preview",
    model_kwargs={
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"}
    }
)
```

### Lesson 5.3: Document Processing

#### PDF Input:
```python
content_block = {
    "type": "file",
    "file": {
        "filename": "document.pdf",
        "file_data": f"data:application/pdf;base64,{base64_pdf_string}"
    }
}
```

---

## Section 6: Specialized Use Cases {#section-6}

### Lesson 6.1: Local Model Deployment

#### Ollama Integration:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)
```

#### Llama.cpp Integration:
```python
from langchain_community.chat_models import ChatLlamaCpp

llm = ChatLlamaCpp(
    model_path="path/to/model.gguf",
    temperature=0.5,
    n_ctx=10000,
    n_gpu_layers=8,
    verbose=True,
)
```

### Lesson 6.2: Custom Model Servers

#### Baseten Integration:
```python
from langchain_baseten import ChatBaseten

# Using model slug
model = ChatBaseten(
    model="moonshotai/Kimi-K2-Instruct-0905",
    api_key="your-api-key"
)

# Using dedicated URL
model = ChatBaseten(
    model_url="https://model-<id>.api.baseten.co/environments/production/predict",
    api_key="your-api-key"
)
```

### Lesson 6.3: Reasoning Models

#### OpenAI Reasoning Configuration:
```python
reasoning = {
    "effort": "medium",  # 'low', 'medium', or 'high'
    "summary": "auto",   # 'detailed', 'auto', or None
}

llm = ChatOpenAI(model="o1-preview", reasoning=reasoning)
response = llm.invoke("Solve this complex math problem...")

# Access reasoning process
for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(block["reasoning"])
```

---

## Section 7: Deployment and Production Considerations {#section-7}

### Lesson 7.1: Performance Optimization

#### Streaming with Usage Metadata:
```python
llm = ChatOpenAI(model="gpt-4o", stream_usage=True)

for chunk in llm.stream(messages):
    if chunk.usage_metadata:
        print(f"Tokens used: {chunk.usage_metadata}")
```

#### Batch Processing:
```python
batch_messages = [
    [("human", "Translate 'hello' to French")],
    [("human", "Translate 'goodbye' to Spanish")]
]

responses = llm.batch(batch_messages)
```

### Lesson 7.2: Error Handling and Retries

#### Retry Configuration:
```python
llm = ChatOpenAI(
    model="gpt-4o",
    max_retries=3,
    timeout=30,
)
```

#### Custom Error Handling:
```python
try:
    response = llm.invoke(messages)
except Exception as e:
    print(f"Error occurred: {e}")
    # Implement fallback logic
```

### Lesson 7.3: Cost Management

#### Token Usage Monitoring:
```python
response = llm.invoke(messages)
usage = response.usage_metadata
print(f"Input tokens: {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```

#### Model Selection Strategy:
```python
# Use smaller models for simple tasks
simple_llm = ChatOpenAI(model="gpt-4o-mini")

# Use larger models for complex reasoning
complex_llm = ChatOpenAI(model="gpt-4o")
```

---

## Section 8: Hands-on Examples and Best Practices {#section-8}

### Lesson 8.1: Building a Conversational Agent

#### Memory Management:
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

# Use with session
config = {"configurable": {"session_id": "user123"}}
response = conversation.invoke("Hello!", config=config)
```

### Lesson 8.2: Multi-Provider Fallback System

#### Router Implementation:
```python
from langchain_community.chat_models import GPTRouter
from langchain_community.chat_models.gpt_router import GPTRouterModel

# Define multiple models
openai_model = GPTRouterModel(name="gpt-4o", provider_name="openai")
anthropic_model = GPTRouterModel(name="claude-3-sonnet", provider_name="anthropic")

# Create router with fallback
router = GPTRouter(models_priority_list=[openai_model, anthropic_model])
```

### Lesson 8.3: Advanced Prompt Engineering

#### Chain Creation:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} consultant."),
    ("human", "{input}")
])

chain = prompt | llm

response = chain.invoke({
    "domain": "financial planning",
    "input": "How should I invest $10,000?"
})
```

#### Function Calling Chain:
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

tools = [get_weather_tool, search_web_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What's the weather in Paris?"})
```

### Lesson 8.4: Production Deployment Patterns

#### Environment Configuration:
```python
import os
from typing import Optional

class LLMConfig:
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.model_name: str = os.getenv("MODEL_NAME", "gpt-4o")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))

config = LLMConfig()
llm = ChatOpenAI(
    model=config.model_name,
    temperature=config.temperature,
    max_tokens=config.max_tokens
)
```

#### Monitoring and Logging:
```python
import logging

# Set up LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-key"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_invoke(llm, messages):
    try:
        logger.info(f"Invoking LLM with {len(messages)} messages")
        response = llm.invoke(messages)
        logger.info(f"Response received: {len(response.content)} characters")
        return response
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise
```

---

## Course Summary and Next Steps

This comprehensive course has covered:

1. **Foundation**: Understanding chat models and their role in LangChain
2. **Integration**: Working with major AI providers (OpenAI, Anthropic, Google, etc.)
3. **Advanced Features**: Tool calling, structured outputs, multimodal inputs
4. **Specialization**: Local models, reasoning systems, custom deployments
5. **Production**: Performance optimization, error handling, monitoring

### Key Takeaways:

- **Provider Abstraction**: LangChain provides a unified interface across different AI providers
- **Feature Consistency**: Core features like tool calling and streaming work similarly across providers
- **Flexibility**: Choose the right model for your use case (cost, performance, capabilities)
- **Production Ready**: Built-in support for monitoring, retries, and error handling

### Recommended Next Steps:

1. **Practice**: Implement examples from each section with your preferred provider
2. **Explore**: Try different models and compare their capabilities
3. **Build**: Create a production application using the patterns learned
4. **Monitor**: Set up proper logging and monitoring for your deployments
5. **Optimize**: Experiment with different models and configurations for your use case

### Additional Resources:

- [LangChain Documentation](https://docs.langchain.com)
- [LangSmith for Monitoring](https://langsmith.langchain.com)
- [Provider-Specific Documentation](https://docs.langchain.com/integrations)
- [Community Examples](https://github.com/langchain-ai/langchain)

This course provides the foundation for building sophisticated AI applications using LangChain's chat model integrations. Continue exploring and experimenting to master these powerful tools!