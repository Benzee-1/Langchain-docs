# Complete Guide to AI Chat Models and LangChain Integration

## Course Overview
This comprehensive course covers AI message handling, multimodal features, and integration with various AI platforms through LangChain. Students will learn about content blocks, tool calling, caching, and working with different AI providers.

---

## Table of Contents
1. [Introduction to AI Message Systems](#section-1)
2. [Content Blocks and Message Handling](#section-2)
3. [Multimodal AI Features](#section-3)
4. [Tool Calling and Function Integration](#section-4)
5. [Caching and Performance Optimization](#section-5)
6. [AI Platform Integrations](#section-6)
7. [Advanced Features and Best Practices](#section-7)

---

## Section 1: Introduction to AI Message Systems {#section-1}

### Learning Objectives
- Understand AI message structure and formats
- Learn about different content block types
- Master basic AI communication patterns

### 1.1 AI Message Fundamentals

AI messages can either be simple strings or complex content blocks containing multiple data types:

```python
from langchain_anthropic import ChatAnthropic
from typing_extensions import Annotated

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

def get_weather(
    location: Annotated[str, ..., "Location as city and state."]
) -> str:
    """Get the weather at a location."""
    return "It's sunny."

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("Which city is hotter today: LA or NY?")
```

### 1.2 Message Content Structure

A single Anthropic AIMessage contains:
- Text content
- Tool invocations
- Standardized content blocks
- Metadata and usage information

Example response structure:
```python
[
    {'text': "I'll help you compare temperatures...", 'type': 'text'},
    {'id': 'toolu_01CkMaXrgmsNjTso7so94RJq',
     'input': {'location': 'Los Angeles, CA'},
     'name': 'get_weather',
     'type': 'tool_use'}
]
```

---

## Section 2: Content Blocks and Message Handling {#section-2}

### Learning Objectives
- Work with different content block types
- Handle tool calls effectively
- Understand content rendering formats

### 2.1 Content Block Types

#### Text Blocks
Basic text content for conversational responses:
```python
response.content_blocks
# Returns standardized format across providers
```

#### Tool Use Blocks
For function and tool invocations:
```python
ai_msg.tool_calls
# Standardized tool call format:
[{
    'name': 'GetWeather',
    'args': {'location': 'Los Angeles, CA'},
    'id': 'toolu_01Ddzj5PkuZkrjF4tafzu54A'
}]
```

### 2.2 Accessing Content

You can access content in multiple ways:
- `response.content` - Raw content
- `response.content_blocks` - Standardized blocks
- `response.tool_calls` - Tool-specific calls

---

## Section 3: Multimodal AI Features {#section-3}

### Learning Objectives
- Work with images and PDFs
- Implement file uploads
- Handle multimodal content blocks

### 3.1 Image Processing

Claude supports multiple image formats through content blocks:

```python
import anthropic
from langchain_anthropic import ChatAnthropic

client = anthropic.Anthropic()
file = client.beta.files.upload(
    file=("image.png", open("/path/to/image.png", "rb"), "image/png"),
)

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["files-api-2025-04-14"],
)

input_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "file_id": file.id},
    ],
}
```

### 3.2 PDF Document Processing

Handle PDF documents with the Files API:

```python
# Upload PDF
file = client.beta.files.upload(
    file=("document.pdf", open("/path/to/document.pdf", "rb"), "application/pdf"),
)

input_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this document."},
        {"type": "file", "file_id": file.id}
    ],
}
```

### 3.3 Extended Thinking Feature

Some models support step-by-step reasoning:

```python
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    max_tokens=5000,
    thinking={"type": "enabled", "budget_tokens": 2000},
)

response = model.invoke("What is the cube root of 50.653?")
```

---

## Section 4: Tool Calling and Function Integration {#section-4}

### Learning Objectives
- Implement tool calling patterns
- Create custom tools
- Handle tool responses

### 4.1 Basic Tool Implementation

```python
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

@tool
def get_weather(location: str, date: str) -> str:
    """Provide weather for specified location and date."""
    if location == "New York" and date == "2024-12-05":
        return "25 celsius"
    return "32 celsius"

llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])
```

### 4.2 Tool Response Handling

```python
from langchain.messages import ToolMessage

response = llm_with_tools.invoke([HumanMessage(content="What's the weather in NY?")])

if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_weather":
            weather = get_weather.invoke(tool_call["args"])
            tool_message = ToolMessage(content=weather, tool_call_id=tool_call["id"])
```

### 4.3 Token-Efficient Tool Use

Enable optimized tool usage:

```python
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["token-efficient-tools-2025-02-19"],
)
```

---

## Section 5: Caching and Performance Optimization {#section-5}

### Learning Objectives
- Implement prompt caching
- Optimize token usage
- Handle incremental caching

### 5.1 Message Caching

Cache frequently used content to reduce costs:

```python
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a technology expert."},
            {
                "type": "text",
                "text": f"{readme}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    },
    {
        "role": "user",
        "content": "What's LangChain?",
    },
]
```

### 5.2 Extended Caching

For longer cache lifetimes:

```python
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["extended-cache-ttl-2025-04-11"],
)

# Cache for 1 hour instead of 5 minutes
cache_control = {"type": "ephemeral", "ttl": "1h"}
```

### 5.3 Tool Caching

Cache tool definitions:

```python
weather_tool = convert_to_anthropic_tool(get_weather)
weather_tool["cache_control"] = {"type": "ephemeral"}
```

### 5.4 Incremental Caching in Conversations

For conversational applications:

```python
def messages_reducer(left: list, right: list) -> list:
    # Mark last user message for caching
    for i in range(len(right) - 1, -1, -1):
        if right[i].type == "human":
            right[i].content[-1]["cache_control"] = {"type": "ephemeral"}
            break
    return add_messages(left, right)
```

---

## Section 6: AI Platform Integrations {#section-6}

### Learning Objectives
- Connect to multiple AI providers
- Understand platform-specific features
- Implement cross-platform compatibility

### 6.1 Anthropic Claude Integration

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    max_tokens=1024
)
```

### 6.2 OpenAI Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=1024
)
```

### 6.3 Ollama Local Models

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
```

### 6.4 Multi-Modal with Ollama

```python
llm = ChatOllama(model="bakllava", temperature=0)

def prompt_func(data):
    text = data["text"]
    image = data["image"]
    
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }
    
    content_parts = [image_part, {"type": "text", "text": text}]
    return [HumanMessage(content=content_parts)]
```

### 6.5 Cohere Integration

```python
from langchain_cohere import ChatCohere

chat = ChatCohere(
    model="command-r",
    temperature=0.3
)
```

---

## Section 7: Advanced Features and Best Practices {#section-7}

### Learning Objectives
- Implement advanced AI features
- Follow security best practices
- Optimize performance and costs

### 7.1 Citations and RAG

Handle citations in responses:

```python
# Citations automatically generated with document blocks
message = {
    "role": "user",
    "content": [
        {
            "type": "document",
            "source": {"type": "text", "data": "The grass is green. The sky is blue."},
            "title": "My Document",
            "citations": {"enabled": True},
        },
        {"type": "text", "text": "What color is the grass?"},
    ],
}
```

### 7.2 Context Management

Automatically manage context windows:

```python
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["context-management-2025-06-27"],
    context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
)
```

### 7.3 Built-in Tools

#### Web Search
```python
tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
model_with_tools = model.bind_tools([tool])
```

#### Code Execution
```python
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["code-execution-2025-05-22"],
)
tool = {"type": "code_execution_20250522", "name": "code_execution"}
```

### 7.4 Security and Guardrails

Implement input/output validation:

```python
# Input validation for PII
llm = PredictionGuard(
    model="Hermes-2-Pro-Llama-3-8B", 
    predictionguard_input={"pii": "block"}
)

# Output validation for toxicity
llm = PredictionGuard(
    model="Hermes-2-Pro-Llama-3-8B", 
    predictionguard_output={"toxicity": True}
)
```

### 7.5 Performance Best Practices

1. **Use appropriate caching strategies**
2. **Optimize token usage with efficient prompts**
3. **Implement streaming for long responses**
4. **Choose the right model for your use case**
5. **Monitor usage and costs**

### 7.6 Error Handling

```python
try:
    response = llm.invoke(messages)
except ValueError as e:
    if "pii detected" in str(e):
        # Handle PII detection
        pass
    elif "toxicity" in str(e):
        # Handle toxicity detection
        pass
```

---

## Course Summary

This course covered:

1. **AI Message Systems**: Understanding content blocks and message structures
2. **Multimodal Features**: Working with images, PDFs, and files
3. **Tool Integration**: Implementing and managing tool calls
4. **Performance Optimization**: Caching strategies and token efficiency
5. **Platform Integration**: Working with multiple AI providers
6. **Advanced Features**: Citations, context management, and built-in tools
7. **Security**: Implementing guardrails and validation

### Next Steps
- Practice with different AI providers
- Experiment with multimodal features
- Implement caching in your applications
- Explore advanced tool calling patterns
- Build production-ready AI applications

### Resources
- LangChain Documentation
- Provider-specific API docs
- Best practices guides
- Community examples and tutorials

---

*This course provides a comprehensive foundation for working with modern AI chat models and LangChain integration. Continue practicing with real-world projects to master these concepts.*