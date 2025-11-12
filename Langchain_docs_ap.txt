# Oracle AI Vector Search and LangChain Embeddings Course

## Course Overview
This comprehensive course covers Oracle AI Vector Search, LangChain embedding integrations, and various embedding providers. You'll learn how to work with embeddings in Oracle Database, implement vector search capabilities, and integrate with multiple embedding providers through LangChain.

---

## Table of Contents

1. [Introduction to Oracle AI Vector Search](#section-1-introduction-to-oracle-ai-vector-search)
2. [Oracle Database Connection and Setup](#section-2-oracle-database-connection-and-setup)
3. [Embedding Generation and Providers](#section-3-embedding-generation-and-providers)
4. [ONNX Model Integration](#section-4-onnx-model-integration)
5. [Credential Management for Third-Party Providers](#section-5-credential-management-for-third-party-providers)
6. [LangChain Embedding Providers](#section-6-langchain-embedding-providers)
7. [Advanced Embedding Techniques](#section-7-advanced-embedding-techniques)
8. [Production Deployment and Best Practices](#section-8-production-deployment-and-best-practices)

---

## Section 1: Introduction to Oracle AI Vector Search

### Learning Objectives
- Understand the concept of vector search and embeddings
- Learn about Oracle AI Vector Search capabilities
- Explore different modes of operation (Thin vs Thick mode)

### Key Concepts

#### Oracle Database Modes
Oracle's python-oracledb operates in two modes:

**Thin Mode:**
- Default mode that doesn't require Oracle Client libraries
- Comprehensive functionality supporting Python Database API v2.0 Specification
- Lighter footprint and easier deployment

**Thick Mode:**
- Uses Oracle Client libraries for additional functionality
- Enhanced features available when Oracle Client libraries are present
- May be required for specific use cases where thin-mode limitations exist

#### Vector Search Overview
Oracle AI Vector Search provides powerful capabilities for:
- Similarity search using embeddings
- Retrieval-augmented generation (RAG) applications
- Document analysis and semantic search
- Multi-modal embeddings (text and images)

---

## Section 2: Oracle Database Connection and Setup

### Learning Objectives
- Establish connection to Oracle Database
- Configure database credentials
- Understand connection parameters

### Lesson 2.1: Basic Database Connection

```python
import sys
import oracledb

# Update the following variables with your Oracle database credentials and connection details
username = "<username>"
password = "<password>"
dsn = "<hostname>/<service_name>"

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")
    sys.exit(1)
```

### Key Components Explained:
- **Username/Password**: Database authentication credentials
- **DSN**: Data Source Name specifying hostname and service name
- **Error Handling**: Proper exception handling for connection failures

### Best Practices:
1. Store credentials securely using environment variables
2. Implement proper error handling
3. Use connection pooling for production applications
4. Validate connection before proceeding with operations

---

## Section 3: Embedding Generation and Providers

### Learning Objectives
- Understand different embedding providers
- Learn provider selection criteria
- Implement provider-specific configurations

### Lesson 3.1: Provider Options Overview

Oracle AI Vector Search supports multiple embedding providers:

1. **Database Provider**: Uses ONNX models within Oracle Database
2. **Third-Party Providers**:
   - OCI GenAI
   - Hugging Face
   - OpenAI

### Lesson 3.2: Provider Selection Criteria

**Database Provider Advantages:**
- Enhanced security (data stays within database)
- Reduced latency (no external API calls)
- No network dependencies
- Cost efficiency for high-volume operations

**Third-Party Provider Advantages:**
- Access to latest models
- Specialized model capabilities
- Reduced infrastructure management
- Regular model updates

### Lesson 3.3: Configuration Examples

```python
from langchain_community.embeddings.oracleai import OracleEmbeddings

# Database provider configuration
embedder_params_db = {
    "provider": "database", 
    "model": "demo_model"
}

# OCI GenAI provider configuration
embedder_params_oci = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v3.0",
}

# Hugging Face provider configuration
embedder_params_hf = {
    "provider": "huggingface",
    "credential_name": "HF_CRED",
    "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "wait_for_model": "true"
}
```

---

## Section 4: ONNX Model Integration

### Learning Objectives
- Load ONNX models into Oracle Database
- Understand ONNX model requirements
- Configure model parameters

### Lesson 4.1: ONNX Model Loading Process

ONNX (Open Neural Network Exchange) models can be loaded directly into Oracle Database for embedding generation.

```python
from langchain_community.embeddings.oracleai import OracleEmbeddings

# ONNX model configuration
onnx_dir = "DEMO_DIR"
onnx_file = "tinybert.onnx"
model_name = "demo_model"

try:
    OracleEmbeddings.load_onnx_model(conn, onnx_dir, onnx_file, model_name)
    print("ONNX model loaded.")
except Exception as e:
    print("ONNX model loading failed!")
    sys.exit(1)
```

### Lesson 4.2: ONNX Model Benefits

**Security Benefits:**
- Data processing within database boundaries
- No external data transmission
- Compliance with data privacy regulations

**Performance Benefits:**
- Eliminates network latency
- Reduces API call overhead
- Consistent processing times

**Cost Benefits:**
- No per-request charges
- Predictable infrastructure costs
- Scalable within existing database resources

### Prerequisites for ONNX Models:
1. Ensure ONNX file availability in the system
2. Proper directory permissions
3. Sufficient database storage
4. Compatible model format

---

## Section 5: Credential Management for Third-Party Providers

### Learning Objectives
- Create and manage provider credentials
- Implement secure credential storage
- Configure provider-specific authentication

### Lesson 5.1: Credential Creation Process

When using third-party providers, credentials must be established for secure API access.

```python
try:
    cursor = conn.cursor()
    cursor.execute(
        """
        declare
            jo json_object_t;
        begin
            -- HuggingFace Credential Setup
            dbms_vector_chain.drop_credential(credential_name => 'HF_CRED');
            jo := json_object_t();
            jo.put('access_token', '<access_token>');
            dbms_vector_chain.create_credential(
                credential_name => 'HF_CRED',
                params => json(jo.to_string)
            );
            
            -- OCI GenAI Credential Setup
            dbms_vector_chain.drop_credential(credential_name => 'OCI_CRED');
            jo := json_object_t();
            jo.put('user_ocid','<user_ocid>');
            jo.put('tenancy_ocid','<tenancy_ocid>');
            jo.put('compartment_ocid','<compartment_ocid>');
            jo.put('private_key','<private_key>');
            jo.put('fingerprint','<fingerprint>');
            dbms_vector_chain.create_credential(
                credential_name => 'OCI_CRED',
                params => json(jo.to_string)
            );
        end;
        """
    )
    cursor.close()
    print("Credentials created.")
except Exception as ex:
    cursor.close()
    raise
```

### Lesson 5.2: Provider-Specific Requirements

**Hugging Face:**
- Access token required
- Model-specific permissions
- Rate limiting considerations

**OCI GenAI:**
- Multiple OCID parameters
- Private key authentication
- Regional endpoint configuration

**Important Security Notes:**
- Never hardcode credentials in source code
- Use environment variables or secure vaults
- Implement credential rotation policies
- Monitor credential usage

---

## Section 6: LangChain Embedding Providers

### Learning Objectives
- Explore various LangChain embedding providers
- Implement provider-specific configurations
- Compare provider capabilities and use cases

### Lesson 6.1: Core Embedding Generation

```python
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_core.documents import Document

# proxy configuration for third-party providers
proxy = "<proxy>"  # Configure if required

# Initialize embedder with database provider
embedder_params = {"provider": "database", "model": "demo_model"}
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

# Generate embeddings
embed = embedder.embed_query("Hello World!")
print(f"Embedding generated by OracleEmbeddings: {embed}")
```

### Lesson 6.2: Popular LangChain Embedding Providers

#### Ascend Embeddings
```python
from langchain_community.embeddings import AscendEmbeddings

model = AscendEmbeddings(
    model_path="/root/.cache/modelscope/hub/yangjhchs/acge_text_embedding",
    device_id=0,
    query_instruction="Represent this sentence for searching relevant passages: ",
)

# Synchronous embedding
emb = model.embed_query("hello")

# Document embedding
doc_embs = model.embed_documents(
    ["This is a content of the document", "This is another document"]
)

# Asynchronous operations
async_result = await model.aembed_query("hello")
async_docs = await model.aembed_documents(
    ["This is a content of the document", "This is another document"]
)
```

#### AwaDB Embeddings
```python
from langchain_community.embeddings import AwaEmbeddings

# Initialize with default model
embedding = AwaEmbeddings()

# Set specific model
text = "our embedding test"
embedding.set_model("all-mpnet-base-v2")

# Generate embeddings
res_query = embedding.embed_query("The test information")
res_document = embedding.embed_documents(["test1", "another test"])
```

#### Baidu Qianfan Platform
```python
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
import os

# Set environment variables
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"

embed = QianfanEmbeddingsEndpoint()

# Synchronous operations
res = embed.embed_documents(["hi", "world"])

# Asynchronous operations
async def aioEmbed():
    res = await embed.aembed_query("qianfan")
    print(res[:8])

await aioEmbed()

# Custom model deployment
embed_custom = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", 
    endpoint="bge_large_zh"
)
```

### Lesson 6.3: Cloud Provider Embeddings

#### Cloudflare Workers AI
```python
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(".env")
cf_acct_id = os.getenv("CF_ACCOUNT_ID")
cf_ai_token = os.getenv("CF_AI_API_TOKEN")

embeddings = CloudflareWorkersAIEmbeddings(
    account_id=cf_acct_id,
    api_token=cf_ai_token,
    model_name="@cf/baai/bge-small-en-v1.5",
)

# Single embedding
query_result = embeddings.embed_query("test")

# Batch embeddings
batch_result = embeddings.embed_documents(["test1", "test2", "test3"])
```

#### Voyage AI
```python
from langchain_voyageai import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(
    voyage_api_key="[ Your Voyage API key ]", 
    model="voyage-law-2"
)

# Document embedding
documents = [
    "Caching embeddings enables storage or temporary caching of embeddings.",
    "An LLMChain composes basic LLM functionality with PromptTemplate and language model.",
    "A Runnable represents a generic unit of work that can be invoked, batched, streamed."
]

documents_embds = embeddings.embed_documents(documents)

# Query embedding
query = "What's an LLMChain?"
query_embd = embeddings.embed_query(query)
```

### Lesson 6.4: Open Source and Local Embeddings

#### Hugging Face Sentence Transformers
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = "This is a test document."
query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text, "This is not a test document."])
```

#### FastEmbed by Qdrant
```python
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Configuration parameters
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    max_length=512,
    doc_embed_type="default",
    batch_size=256,
)

# Generate embeddings
document_embeddings = embeddings.embed_documents([
    "This is a document", 
    "This is some other document"
])

query_embeddings = embeddings.embed_query("This is a query")
```

---

## Section 7: Advanced Embedding Techniques

### Learning Objectives
- Implement multimodal embeddings
- Optimize embedding performance
- Handle large-scale embedding operations

### Lesson 7.1: Multimodal Embeddings with Jina

```python
from langchain_community.embeddings import JinaEmbeddings
import requests
from PIL import Image

# Text embeddings
text_embeddings = JinaEmbeddings(
    jina_api_key="jina_*", 
    model_name="jina-embeddings-v2-base-en"
)

text = "This is a test document."
query_result = text_embeddings.embed_query(text)

# Multimodal embeddings (text + images)
multimodal_embeddings = JinaEmbeddings(
    jina_api_key="jina_*", 
    model_name="jina-clip-v1"
)

image = "https://avatars.githubusercontent.com/u/126733545?v=4"
description = "Logo of a parrot and a chain on green background"

image_result = multimodal_embeddings.embed_images([image])
description_result = multimodal_embeddings.embed_documents([description])

# Calculate similarity
from numpy import dot, linalg
cosine_similarity = dot(image_result[0], description_result[0]) / (
    linalg.norm(image_result[0]) * linalg.norm(description_result[0])
)
```

### Lesson 7.2: Performance Optimization

#### Intel Optimization (IPEX-LLM)
```python
from langchain_community.embeddings import IpexLLMBgeEmbeddings

# CPU optimization
embedding_model_cpu = IpexLLMBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={},
    encode_kwargs={"normalize_embeddings": True},
)

# GPU optimization
embedding_model_gpu = IpexLLMBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "xpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Runtime configuration for Windows Intel GPU
import os
os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["BIGDL_LLM_XMX_DISABLED"] = "1"  # For Intel Core Ultra integrated GPU
```

#### Quantized Embeddings
```python
from langchain_community.embeddings import QuantizedBgeEmbeddings

# Intel quantized BGE model
model_name = "Intel/bge-small-en-v1.5-sts-int8-static-inc"
encode_kwargs = {"normalize_embeddings": True}

model = QuantizedBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)

text = "This is a test document."
query_result = model.embed_query(text)
doc_result = model.embed_documents([text])
```

### Lesson 7.3: Specialized Embedding Providers

#### Elasticsearch Embeddings
```python
from langchain_elasticsearch import ElasticsearchEmbeddings

# Using Elastic Cloud
embeddings = ElasticsearchEmbeddings.from_credentials(
    "your_model_id",
    es_cloud_id="your_cloud_id",
    es_user="your_user",
    es_password="your_password",
)

# Using existing Elasticsearch connection
from elasticsearch import Elasticsearch
es_connection = Elasticsearch(
    hosts=["https://es_cluster_url:port"], 
    basic_auth=("user", "password")
)

embeddings = ElasticsearchEmbeddings.from_es_connection(
    "model_id",
    es_connection,
)

# Generate embeddings
documents = [
    "This is an example document.",
    "Another example document to generate embeddings for.",
]
document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query("This is a single query.")
```

---

## Section 8: Production Deployment and Best Practices

### Learning Objectives
- Implement production-ready embedding systems
- Configure monitoring and logging
- Optimize for scale and performance

### Lesson 8.1: End-to-End RAG Implementation

```python
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_core.documents import Document

# Production configuration
class ProductionEmbeddingConfig:
    def __init__(self):
        self.connection_pool_size = 10
        self.retry_attempts = 3
        self.timeout_seconds = 30
        self.batch_size = 100

def create_production_embedder(conn, config):
    embedder_params = {
        "provider": "database", 
        "model": "production_model"
    }
    
    return OracleEmbeddings(
        conn=conn, 
        params=embedder_params,
        batch_size=config.batch_size
    )

# Implement with error handling and logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embed_with_retry(embedder, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = embedder.embed_query(text)
            logger.info(f"Successfully embedded text in attempt {attempt + 1}")
            return result
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None
```

### Lesson 8.2: Monitoring and Observability

```python
import time
from contextlib import contextmanager

@contextmanager
def embedding_timer(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{operation_name} completed in {end_time - start_time:.2f} seconds")

# Usage example
with embedding_timer("Document embedding"):
    embeddings = embedder.embed_documents(documents)

# Metrics collection
class EmbeddingMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0
    
    def record_success(self, latency):
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency += latency
    
    def record_failure(self):
        self.total_requests += 1
        self.failed_requests += 1
    
    def get_success_rate(self):
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0
    
    def get_average_latency(self):
        return self.total_latency / self.successful_requests if self.successful_requests > 0 else 0
```

### Lesson 8.3: Deployment Strategies

#### Container Deployment
```dockerfile
# Dockerfile for embedding service
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "embedding_service:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Kubernetes Configuration
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: embedding-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: ORACLE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: oracle-credentials
              key: connection-string
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Lesson 8.4: Security Best Practices

#### Credential Management
```python
import os
from cryptography.fernet import Fernet

class SecureCredentialManager:
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_credential(self, credential):
        return self.cipher.encrypt(credential.encode())
    
    def decrypt_credential(self, encrypted_credential):
        return self.cipher.decrypt(encrypted_credential).decode()
    
    def get_oracle_credentials(self):
        return {
            'username': os.getenv('ORACLE_USERNAME'),
            'password': self.decrypt_credential(os.getenv('ENCRYPTED_ORACLE_PASSWORD')),
            'dsn': os.getenv('ORACLE_DSN')
        }
```

#### Access Control
```python
from functools import wraps
import jwt

def require_authentication(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'Authentication required'}, 401
        
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
        
        return f(*args, **kwargs)
    return decorated_function

@require_authentication
def embed_text_endpoint():
    # Embedding endpoint implementation
    pass
```

### Lesson 8.5: Performance Optimization Strategies

#### Batch Processing
```python
def batch_embed_documents(embedder, documents, batch_size=100):
    """Process documents in batches for optimal performance"""
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        with embedding_timer(f"Batch {i//batch_size + 1}"):
            batch_embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

#### Caching Strategy
```python
import redis
import json
import hashlib

class EmbeddingCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour
    
    def get_cache_key(self, text, model_name):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{model_name}:{text_hash}"
    
    def get_cached_embedding(self, text, model_name):
        cache_key = self.get_cache_key(text, model_name)
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    def cache_embedding(self, text, model_name, embedding):
        cache_key = self.get_cache_key(text, model_name)
        self.redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            json.dumps(embedding)
        )

# Usage with caching
def embed_with_cache(embedder, text, model_name, cache):
    # Check cache first
    cached_embedding = cache.get_cached_embedding(text, model_name)
    if cached_embedding:
        logger.info("Cache hit for embedding")
        return cached_embedding
    
    # Generate new embedding
    embedding = embedder.embed_query(text)
    
    # Cache the result
    cache.cache_embedding(text, model_name, embedding)
    logger.info("Generated and cached new embedding")
    
    return embedding
```

---

## Course Summary and Next Steps

### What You've Learned

This course covered:

1. **Oracle AI Vector Search fundamentals** - Understanding vector search concepts and Oracle's implementation
2. **Database connectivity** - Establishing secure connections to Oracle Database
3. **Embedding providers** - Working with multiple embedding providers including database-native and third-party options
4. **ONNX model integration** - Loading and managing ONNX models within Oracle Database
5. **Credential management** - Secure handling of API keys and authentication
6. **LangChain integration** - Leveraging various LangChain embedding providers
7. **Advanced techniques** - Multimodal embeddings, performance optimization, and quantization
8. **Production deployment** - Scalable, secure, and monitored embedding systems

### Key Takeaways

- **Security First**: Always prioritize data security by using database-native embeddings when possible
- **Performance Optimization**: Consider ONNX models, quantization, and caching for production systems
- **Provider Diversity**: Different providers excel in different use cases - choose appropriately
- **Monitoring**: Implement comprehensive logging and metrics for production systems
- **Scalability**: Design for batch processing and horizontal scaling from the start

### Recommended Next Steps

1. **Hands-on Practice**: Implement a complete RAG system using Oracle AI Vector Search
2. **Performance Testing**: Benchmark different embedding providers for your specific use case
3. **Advanced Features**: Explore multimodal embeddings and specialized models
4. **Integration Projects**: Build end-to-end applications combining embeddings with LLMs
5. **Community Engagement**: Contribute to the Oracle AI Vector Search community and LangChain ecosystem

### Additional Resources

- Oracle AI Vector Search Documentation
- LangChain Community Documentation
- Oracle AI Vector Search End-to-End Demo Guide
- Performance optimization guides for specific embedding providers
- Security best practices for production AI applications

---

## Appendix: Code Examples and Templates

### Complete RAG Implementation Template

```python
# Complete production-ready RAG implementation
import oracledb
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_core.documents import Document
import logging
import os

class ProductionRAGSystem:
    def __init__(self, config):
        self.config = config
        self.connection = self._create_connection()
        self.embedder = self._create_embedder()
        self.logger = self._setup_logging()
    
    def _create_connection(self):
        return oracledb.connect(
            user=os.getenv('ORACLE_USERNAME'),
            password=os.getenv('ORACLE_PASSWORD'),
            dsn=os.getenv('ORACLE_DSN')
        )
    
    def _create_embedder(self):
        embedder_params = {
            "provider": "database",
            "model": self.config.model_name
        }
        return OracleEmbeddings(
            conn=self.connection,
            params=embedder_params
        )
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def index_documents(self, documents):
        """Index documents for search"""
        embeddings = self.embedder.embed_documents(documents)
        # Store embeddings in vector database
        # Implementation depends on your vector storage solution
        pass
    
    def search(self, query, k=5):
        """Search for similar documents"""
        query_embedding = self.embedder.embed_query(query)
        # Perform vector similarity search
        # Return top-k most similar documents
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

# Usage example
config = type('Config', (), {
    'model_name': 'production_embedding_model',
    'batch_size': 100
})()

with ProductionRAGSystem(config) as rag_system:
    documents = ["Document 1", "Document 2", "Document 3"]
    rag_system.index_documents(documents)
    
    results = rag_system.search("search query")
    print(results)
```

This comprehensive course provides a solid foundation for working with Oracle AI Vector Search and LangChain embeddings in both development and production environments.