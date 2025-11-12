# Complete LangChain Integration Course

## Course Overview
This comprehensive course covers LangChain integrations with various AI/ML providers, focusing on LLMs, embedding models, and practical implementation strategies. Students will learn to build context-aware AI applications using different cloud platforms and open-source models.

---

## Table of Contents

1. [Introduction to LangChain Integrations](#section-1-introduction-to-langchain-integrations)
2. [Cloud-Based LLM Integrations](#section-2-cloud-based-llm-integrations)
3. [Open Source and Self-Hosted Solutions](#section-3-open-source-and-self-hosted-solutions)
4. [Embedding Models and Vector Operations](#section-4-embedding-models-and-vector-operations)
5. [Advanced Integration Patterns](#section-5-advanced-integration-patterns)
6. [Production Deployment Strategies](#section-6-production-deployment-strategies)

---

## Section 1: Introduction to LangChain Integrations

### Learning Objectives
- Understand the LangChain ecosystem and integration philosophy
- Learn about different types of model integrations
- Set up basic development environment

### Lesson 1.1: LangChain Integration Overview

LangChain provides a unified interface to work with various AI models and services. The platform supports:

- **Chat Models**: Conversational AI interfaces
- **Text Completion Models**: Traditional completion-based models
- **Embedding Models**: Vector representation models
- **Retrieval Systems**: RAG and search capabilities

### Lesson 1.2: Basic Setup and Authentication

Most integrations require API keys and proper environment setup:

```python
import os
import getpass

# Common pattern for API key setup
if not os.getenv("API_KEY"):
    os.environ["API_KEY"] = getpass.getpass("Enter your API key: ")
```

### Lesson 1.3: Installation Patterns

```bash
# General installation pattern
pip install -qU langchain-[provider]

# Examples:
pip install -qU langchain-openai
pip install -qU langchain-google-genai
pip install -qU langchain-community
```

---

## Section 2: Cloud-Based LLM Integrations

### Learning Objectives
- Master major cloud AI platform integrations
- Understand authentication and configuration patterns
- Implement production-ready cloud solutions

### Lesson 2.1: OpenAI Integration

#### Basic OpenAI Usage
```python
from langchain_openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=256
)

response = llm.invoke("What are the benefits of AI?")
```

#### Azure OpenAI Integration
```python
from langchain_openai import AzureOpenAI

llm = AzureOpenAI(
    deployment_name="gpt-35-turbo-instruct-0914",
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7
)
```

### Lesson 2.2: Google AI Platforms

#### Google AI Studio (Gemini API)
```python
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(
    model="models/text-bison-001",
    google_api_key=api_key
)
```

#### Google Cloud Vertex AI
```python
from langchain_google_vertexai import VertexAI

model = VertexAI(model_name="gemini-2.5-pro")
response = model.invoke("Explain quantum computing")
```

### Lesson 2.3: AWS Integration

#### Amazon Bedrock
```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    credentials_profile_name="bedrock-admin",
    region_name="us-east-1"
)
```

### Lesson 2.4: Microsoft Azure ML
```python
from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint

llm = AzureMLOnlineEndpoint(
    endpoint_url="https://your-endpoint.azure.com/score",
    endpoint_api_type="dedicated",
    endpoint_api_key="your-api-key"
)
```

---

## Section 3: Open Source and Self-Hosted Solutions

### Learning Objectives
- Deploy and manage open-source models
- Understand local hosting strategies
- Implement cost-effective AI solutions

### Lesson 3.1: Hugging Face Integration

```python
from langchain_community.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation"
)
```

### Lesson 3.2: Ollama for Local Models

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull and run models
ollama pull llama2
ollama run llama2
```

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
response = llm.invoke("Explain machine learning")
```

### Lesson 3.3: Llamafile - Single File Deployment

```bash
# Download and setup llamafile
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
```

```python
from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()
response = llm.invoke("Tell me a joke")
```

### Lesson 3.4: C Transformers for GGML Models

```python
from langchain_community.llms import CTransformers

llm = CTransformers(model="marella/gpt-2-ggml")
print(llm.invoke("AI is going to"))
```

---

## Section 4: Embedding Models and Vector Operations

### Learning Objectives
- Master embedding model integrations
- Implement vector search and retrieval
- Build RAG systems

### Lesson 4.1: OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Single text embedding
vector = embeddings.embed_query("Hello world")

# Multiple documents
vectors = embeddings.embed_documents(["Text 1", "Text 2"])
```

### Lesson 4.2: Google AI Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Specify task type for optimization
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)
```

### Lesson 4.3: Hugging Face Embeddings

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Lesson 4.4: Vector Store Integration

```python
from langchain_core.vectorstores import InMemoryVectorStore

# Create vector store
text = "LangChain is the framework for building context-aware reasoning applications"
vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use as retriever
retriever = vectorstore.as_retriever()
retrieved_documents = retriever.invoke("What is LangChain?")
```

---

## Section 5: Advanced Integration Patterns

### Learning Objectives
- Implement complex integration patterns
- Handle multimodal inputs
- Manage safety and content filtering

### Lesson 5.1: Multimodal Capabilities

#### Working with Images (Gemini)
```python
from langchain.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro-vision")

image_message = {
    "type": "image_url",
    "image_url": {"url": "path/to/image.jpg"},
}
text_message = {
    "type": "text",
    "text": "What is shown in this image?",
}

message = HumanMessage(content=[text_message, image_message])
output = llm.invoke([message])
```

#### Video and Audio Processing
```python
# Video processing
media_message = {
    "type": "image_url",
    "image_url": {
        "url": "gs://cloud-samples-data/generative-ai/video/pixel8.mp4",
    },
}

# Audio transcription
audio_message = {
    "type": "image_url",
    "image_url": {
        "url": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
    },
}
```

### Lesson 5.2: Safety and Content Filtering

```python
from langchain_google_genai import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
}

llm = GoogleGenerativeAI(
    model="gemini-pro",
    safety_settings=safety_settings
)
```

### Lesson 5.3: Custom Content Formatters

```python
from langchain_community.llms.azureml_endpoint import ContentFormatterBase
import json

class CustomFormatter(ContentFormatterBase):
    content_type = "application/json"
    accepts = "application/json"
    
    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({
            "inputs": [prompt],
            "parameters": model_kwargs,
        })
        return str.encode(input_str)
    
    def format_response_payload(self, output: bytes) -> str:
        response_json = json.loads(output)
        return response_json[0]["generated_text"]
```

---

## Section 6: Production Deployment Strategies

### Learning Objectives
- Deploy models in production environments
- Implement monitoring and error handling
- Optimize for performance and cost

### Lesson 6.1: Streaming and Async Operations

```python
# Streaming responses
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Async operations
import asyncio

async def async_generate():
    result = await llm.ainvoke("What is AI?")
    return result
```

### Lesson 6.2: Error Handling and Retry Logic

```python
from langchain_core.exceptions import LangChainException

try:
    response = llm.invoke(prompt)
except LangChainException as e:
    print(f"LangChain error: {e}")
    # Implement retry logic
except Exception as e:
    print(f"General error: {e}")
    # Handle other exceptions
```

### Lesson 6.3: Performance Optimization

#### Batch Processing
```python
# Embed multiple documents efficiently
documents = ["Doc 1", "Doc 2", "Doc 3"]
embeddings_batch = embeddings.embed_documents(documents)
```

#### Caching Strategies
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### Lesson 6.4: Cost Management

#### Token Usage Monitoring
```python
# Track token usage
result = model.generate([message])
usage = result.generations[0][0].generation_info.get('usage_metadata')
print(f"Tokens used: {usage['total_token_count']}")
```

#### Model Selection Strategy
```python
# Use appropriate models for tasks
cheap_model = OpenAI(model="gpt-3.5-turbo-instruct")  # For simple tasks
powerful_model = OpenAI(model="gpt-4")  # For complex tasks
```

---

## Practical Exercises

### Exercise 1: Multi-Provider RAG System
Build a RAG system that can switch between different embedding providers based on requirements.

### Exercise 2: Multimodal Chat Application
Create a chat application that can process text, images, and audio inputs.

### Exercise 3: Cost-Optimized Deployment
Design a system that automatically selects the most cost-effective model for each query type.

### Exercise 4: Safety-First AI Assistant
Implement a comprehensive content filtering and safety system for AI responses.

---

## Best Practices Summary

1. **Authentication Management**
   - Use environment variables for API keys
   - Implement proper credential rotation
   - Never hardcode sensitive information

2. **Error Handling**
   - Implement comprehensive exception handling
   - Use retry mechanisms with exponential backoff
   - Log errors for monitoring and debugging

3. **Performance Optimization**
   - Use batch operations when possible
   - Implement caching for frequently used queries
   - Monitor token usage and costs

4. **Security Considerations**
   - Implement content filtering
   - Use safety settings appropriately
   - Validate and sanitize inputs

5. **Production Readiness**
   - Use async operations for better concurrency
   - Implement proper monitoring and alerting
   - Plan for scaling and load management

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google AI Documentation](https://ai.google.dev/)
- [Azure AI Documentation](https://docs.microsoft.com/azure/ai/)
- [Hugging Face Documentation](https://huggingface.co/docs)

---

## Course Conclusion

This course has provided comprehensive coverage of LangChain integrations across major AI platforms and deployment strategies. Students should now be equipped to:

- Integrate various AI models and services
- Build production-ready AI applications
- Implement multimodal AI solutions
- Manage costs and optimize performance
- Deploy secure and scalable AI systems

Continue practicing with different providers and explore advanced features as the LangChain ecosystem continues to evolve.