# LangChain Advanced Retrieval Systems Course

## Course Overview

This comprehensive course covers advanced retrieval systems in LangChain, focusing on vector search, hybrid retrieval methods, and integration with various database systems and services. Students will learn to implement sophisticated RAG (Retrieval-Augmented Generation) systems using cutting-edge technologies.

## Table of Contents

1. [Introduction to Advanced Retrieval Systems](#section-1-introduction)
2. [Pinecone Hybrid Search](#section-2-pinecone-hybrid-search)
3. [Dappier Real-Time Data Retrieval](#section-3-dappier-retrieval)
4. [Graph RAG Systems](#section-4-graph-rag)
5. [Embedchain RAG Framework](#section-5-embedchain)
6. [Specialized Retrievers](#section-6-specialized-retrievers)
7. [Database Integration](#section-7-database-integration)
8. [Performance Optimization](#section-8-performance-optimization)

---

## Section 1: Introduction to Advanced Retrieval Systems {#section-1-introduction}

### Learning Objectives
- Understand the evolution of retrieval systems
- Learn about vector search fundamentals
- Explore hybrid search approaches
- Master RAG architecture patterns

### Key Concepts
- **Vector Search**: Using embeddings to find semantically similar content
- **Hybrid Search**: Combining dense and sparse retrieval methods
- **RAG Architecture**: Retrieval-Augmented Generation patterns
- **Embedding Models**: Dense vector representations of text

### Course Prerequisites
- Basic understanding of LangChain
- Familiarity with Python programming
- Knowledge of machine learning concepts
- Understanding of vector databases

---

## Section 2: Pinecone Hybrid Search {#section-2-pinecone-hybrid-search}

### Lesson 2.1: Setting Up Pinecone

#### Initialize Pinecone Client
```python
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```

#### Connect to Index
```python
index = pc.Index(index_name)
```

### Lesson 2.2: Embeddings and Sparse Encoders

#### Dense Embeddings
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

#### Sparse Encoders
```python
from pinecone_text.sparse import BM25Encoder

# Use default tf-idf values
bm25_encoder = BM25Encoder().default()

# Fit tf-idf values to your own corpus
corpus = ["foo", "bar", "world", "hello"]
bm25_encoder.fit(corpus)

# Store and load values
bm25_encoder.dump("bm25_values.json")
bm25_encoder = BM25Encoder().load("bm25_values.json")
```

### Lesson 2.3: Hybrid Search Retriever

#### Load Retriever
```python
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, 
    sparse_encoder=bm25_encoder, 
    index=index
)
```

#### Add and Search Documents
```python
# Add texts
retriever.add_texts(["foo", "bar", "world", "hello"])

# Perform search
result = retriever.invoke("foo")
print(result[0])  # Document(page_content='foo', metadata={})
```

---

## Section 3: Dappier Real-Time Data Retrieval {#section-3-dappier-retrieval}

### Lesson 3.1: Dappier Overview
Dappier connects LLMs to real-time, rights-cleared, proprietary data from trusted sources, making AI an expert in specialized domains.

#### Key Features
- Real-Time Web Search
- News and Sports Data
- Financial Market Data
- Cryptocurrency Information
- Premium Publisher Content

### Lesson 3.2: Setup and Configuration

#### Installation and API Key
```python
# Install package
pip install -U langchain-dappier

# Set environment variable
export DAPPIER_API_KEY="your-api-key"
```

#### Initialize Retriever
```python
from langchain_dappier import DappierRetriever

retriever = DappierRetriever(data_model_id="dm_01jagy9nqaeer9hxx8z1sk1jx6")
```

### Lesson 3.3: Advanced Usage

#### Query with Parameters
```python
query = "latest tech news"
results = retriever.invoke(query)

# Results include metadata like title, author, source_url, image_url, pubdata
for result in results:
    print(f"Title: {result.metadata['title']}")
    print(f"Content: {result.page_content}")
```

---

## Section 4: Graph RAG Systems {#section-4-graph-rag}

### Lesson 4.1: Introduction to Graph RAG
Graph RAG combines unstructured similarity search with structured traversal of metadata properties, enabling graph-based retrieval over existing vector stores.

#### Benefits
- Link based on existing metadata
- Change links on demand
- Pluggable traversal strategies
- Broad compatibility

### Lesson 4.2: Setup and Installation
```python
pip install -qU langchain-graph-retriever
```

### Lesson 4.3: Graph Traversal Implementation

#### Basic Setup
```python
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

traversal_retriever = GraphRetriever(
    store=vector_store,
    edges=[("habitat", "habitat"), ("origin", "origin")],
    strategy=Eager(k=5, start_k=1, max_depth=2),
)
```

#### Graph vs Standard Retrieval
```python
# Graph traversal
results = traversal_retriever.invoke("what animals could be found near a capybara?")

# Standard retrieval (max_depth=0)
standard_retriever = GraphRetriever(
    store=vector_store,
    edges=[("habitat", "habitat"), ("origin", "origin")],
    strategy=Eager(k=5, start_k=5, max_depth=0),
)
```

---

## Section 5: Embedchain RAG Framework {#section-5-embedchain}

### Lesson 5.1: Embedchain Overview
Embedchain is a RAG framework for creating data pipelines that loads, indexes, retrieves, and syncs data.

#### Installation
```python
pip install -qU embedchain
```

### Lesson 5.2: Creating Retrievers

#### Basic Setup
```python
from langchain_community.retrievers import EmbedchainRetriever

# Create retriever with default options
retriever = EmbedchainRetriever.create()

# Or with custom YAML config
retriever = EmbedchainRetriever.create(yaml_path="config.yaml")
```

#### Adding Data
```python
retriever.add_texts([
    "https://en.wikipedia.org/wiki/Elon_Musk",
    "https://www.forbes.com/profile/elon-musk",
    "https://www.youtube.com/watch?v=RcYjXbSJBN8",
])
```

### Lesson 5.3: Document Retrieval
```python
result = retriever.invoke("How many companies does Elon Musk run and name those?")
```

---

## Section 6: Specialized Retrievers {#section-6-specialized-retrievers}

### Lesson 6.1: Arcee Domain-Adapted Models
```python
from langchain_community.retrievers import ArceeRetriever

retriever = ArceeRetriever(
    model="DALM-PubMed",
    model_kwargs={
        "size": 5,
        "filters": [
            {
                "field_name": "document",
                "filter_type": "fuzzy_search",
                "value": "Einstein",
            }
        ],
    },
)
```

### Lesson 6.2: Activeloop Deep Memory
Deep Memory optimizes vector stores for specific use cases, achieving up to 27% improvement in retrieval accuracy.

#### Key Features
- Neural network layer for query-data matching
- Minimal latency increase
- Cost-effective optimization
- Simple integration

### Lesson 6.3: Bedrock Knowledge Bases
```python
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="PUIJP4EQUA",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)
```

---

## Section 7: Database Integration {#section-7-database-integration}

### Lesson 7.1: Azure AI Search
```python
from langchain_community.retrievers import AzureAISearchRetriever

retriever = AzureAISearchRetriever(
    content_key="content", 
    top_k=1, 
    index_name="langchain-vector-demo"
)
```

### Lesson 7.2: Elasticsearch Integration
```python
from langchain_elasticsearch import ElasticsearchRetriever

# Vector search
def vector_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)
    return {
        "knn": {
            "field": dense_vector_field,
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        }
    }

vector_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=vector_query,
    content_field=text_field,
    url=es_url,
)
```

### Lesson 7.3: Google Cloud SQL Integration
```python
from langchain_google_cloud_sql_mysql import MySQLEngine, MySQLLoader

engine = MySQLEngine.from_instance(
    project_id=PROJECT_ID, 
    region=REGION, 
    instance=INSTANCE, 
    database=DATABASE
)

loader = MySQLLoader(engine=engine, table_name=TABLE_NAME)
```

---

## Section 8: Performance Optimization {#section-8-performance-optimization}

### Lesson 8.1: Hybrid Search Strategies

#### BM25 + Vector Search
```python
def hybrid_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)
    return {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {"match": {text_field: search_query}}
                        }
                    },
                    {
                        "knn": {
                            "field": dense_vector_field,
                            "query_vector": vector,
                            "k": 5,
                            "num_candidates": 10,
                        }
                    },
                ]
            }
        }
    }
```

### Lesson 8.2: Reranking with FlashRank
```python
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)
```

### Lesson 8.3: Memory Optimization
- Product Quantization (NanoPQ)
- Efficient vector storage
- Batch processing strategies
- Caching mechanisms

---

## Course Assignments and Projects

### Assignment 1: Hybrid Search Implementation
Implement a hybrid search system combining BM25 and vector search using Pinecone.

### Assignment 2: Graph RAG System
Build a graph-based retrieval system for a specific domain dataset.

### Assignment 3: Real-time Data Integration
Create a RAG system that incorporates real-time data using Dappier.

### Final Project: Production RAG System
Design and implement a complete production-ready RAG system incorporating multiple retrieval strategies.

---

## Additional Resources

### Documentation Links
- [LangChain Documentation](https://docs.langchain.com)
- [Pinecone Documentation](https://docs.pinecone.io)
- [Dappier Marketplace](https://marketplace.dappier.com)
- [Graph RAG Project Page](https://github.com/datastax/graph-rag)

### Best Practices
1. Choose appropriate embedding models for your domain
2. Implement proper error handling and fallbacks
3. Monitor retrieval performance and accuracy
4. Use caching strategies for frequently accessed data
5. Consider cost optimization in production deployments

### Troubleshooting Common Issues
1. Vector dimension mismatches
2. API rate limiting
3. Memory constraints with large datasets
4. Latency optimization techniques
5. Authentication and security considerations

---

## Conclusion

This course provides comprehensive coverage of advanced retrieval systems in LangChain, preparing students to build sophisticated RAG applications using cutting-edge technologies. The combination of theoretical knowledge and practical implementation ensures students can deploy production-ready systems in real-world scenarios.

## Next Steps
- Explore specialized domain applications
- Investigate custom embedding models
- Study advanced optimization techniques
- Contribute to open-source retrieval projects