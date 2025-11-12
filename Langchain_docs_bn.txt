# LangChain Document Loaders and Integrations Course

## Course Overview

This comprehensive course covers the implementation and usage of various document loaders and integrations within the LangChain ecosystem. Students will learn about different data sources, web scraping techniques, and database integrations essential for building robust AI applications.

---

## Table of Contents

1. [Introduction to LangChain Document Loaders](#section-1-introduction-to-langchain-document-loaders)
2. [Search Engine Integrations](#section-2-search-engine-integrations)
3. [Web Scraping and Browser-Based Loaders](#section-3-web-scraping-and-browser-based-loaders)
4. [Database Integrations](#section-4-database-integrations)
5. [Document Processing and File Formats](#section-5-document-processing-and-file-formats)
6. [API Integrations and Data Sources](#section-6-api-integrations-and-data-sources)
7. [Text Processing and Semantic Layers](#section-7-text-processing-and-semantic-layers)
8. [Advanced Topics and Best Practices](#section-8-advanced-topics-and-best-practices)

---

## Section 1: Introduction to LangChain Document Loaders

### Learning Objectives
- Understand the role of document loaders in the LangChain ecosystem
- Learn about the Document object structure and metadata handling
- Implement basic document loading patterns

### 1.1 What are Document Loaders?

Document loaders are essential components in LangChain that extract and structure data from various sources into a standardized Document format. They serve as the bridge between raw data and AI applications.

### 1.2 Document Structure

Each Document contains:
- `page_content`: The main text content
- `metadata`: Additional information about the source
- `lookup_str`: String for document identification
- `lookup_index`: Numerical index for the document

### 1.3 Basic Implementation Pattern

```python
from langchain_community.document_loaders import SomeLoader

# Initialize loader
loader = SomeLoader(source="path/to/source")

# Load documents
documents = loader.load()

# Inspect structure
print(documents[0].page_content[:100])
print(documents[0].metadata)
```

### Lab Exercise 1.1
Create a simple document loading script using the Copy Paste loader to understand basic Document structure.

---

## Section 2: Search Engine Integrations

### Learning Objectives
- Implement search engine integrations for dynamic data retrieval
- Configure API connections for various search services
- Handle search results and pagination

### 2.1 Brave Search Integration

#### Overview
Brave Search is a privacy-focused search engine that provides comprehensive web indexing capabilities. As of May 2022, it covered over 10 billion pages and serves 92% of search results independently.

#### Key Features
- Own web index with intentionally smaller size to avoid spam
- Ad-free premium model available
- No user data collection by default
- API-based integration for developers

#### Implementation

```python
from langchain_community.document_loaders import BraveSearchLoader

# Setup API key
api_key = "your_brave_api_key"

# Initialize loader with search parameters
loader = BraveSearchLoader(
    query="obama middle name", 
    api_key=api_key, 
    search_kwargs={"count": 3}
)

# Load search results
docs = loader.load()
print(f"Found {len(docs)} documents")

# Examine metadata
for doc in docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Link: {doc.metadata['link']}")
```

#### Search Configuration Options
- `count`: Number of results to retrieve
- `query`: Search terms and operators
- `search_kwargs`: Additional search parameters

### 2.2 Search Result Processing

Understanding how search results are structured and processed is crucial for effective implementation:

```python
# Example search results structure
search_results = [
    {
        'title': "Obama's Middle Name -- My Last Name -- is 'Hussein.' So?",
        'link': 'https://www.cair.com/cair_in_the_news/obamas-middle-name-my-last-name-is-hussein-so/',
        'content': 'Detailed article content...'
    }
]
```

### Lab Exercise 2.1
Implement a Brave Search integration that searches for current events and processes the results for analysis.

---

## Section 3: Web Scraping and Browser-Based Loaders

### Learning Objectives
- Master web scraping techniques using headless browsers
- Implement Browserbase and Browserless integrations
- Handle complex web pages and JavaScript content

### 3.1 Browserbase Integration

#### Overview
Browserbase is a developer platform providing reliable headless browsers for data extraction with advanced features like stealth mode and automatic captcha solving.

#### Key Features
- Serverless infrastructure for browser automation
- Stealth mode with fingerprinting tactics
- Session debugger for troubleshooting
- Live debugging capabilities

#### Setup and Configuration

```python
import os
from langchain_community.document_loaders import BrowserbaseLoader

# Environment variables setup
BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID")

# Initialize loader
loader = BrowserbaseLoader(
    api_key=BROWSERBASE_API_KEY,
    project_id=BROWSERBASE_PROJECT_ID,
    urls=["https://example.com"],
    text_content=False  # Set to True for text-only extraction
)

# Load documents
docs = loader.load()
print(docs[0].page_content[:61])
```

#### Configuration Options
- `urls`: List of URLs to scrape (Required)
- `text_content`: Extract text-only content (Default: False)
- `session_id`: Use existing session (Optional)
- `proxy`: Enable/disable proxy usage (Optional)

### 3.2 Browserless Integration

#### Overview
Browserless provides cloud-based headless Chrome instances for scalable browser automation without infrastructure management.

#### Implementation

```python
from langchain_community.document_loaders import BrowserlessLoader

BROWSERLESS_API_TOKEN = "your_browserless_token"

loader = BrowserlessLoader(
    api_token=BROWSERLESS_API_TOKEN,
    urls=["https://en.wikipedia.org/wiki/Document_classification"],
    text_content=True
)

documents = loader.load()
print(documents[0].page_content[:1000])
```

### 3.3 Docusaurus Integration

#### Overview
Docusaurus is a static-site generator that provides documentation features. The DocusaurusLoader extends SitemapLoader to scrape documentation sites.

#### Implementation

```python
from langchain_community.document_loaders import DocusaurusLoader

# Basic usage
loader = DocusaurusLoader("https://python.langchain.com")
docs = loader.load()

# With URL filtering
loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=["https://python.langchain.com/docs/integrations/document_loaders/sitemap"]
)
documents = loader.load()
```

#### Custom Scraping Rules

```python
from bs4 import BeautifulSoup

def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Remove navigation and header elements
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    
    for element in nav_elements + header_elements:
        element.decompose()
    
    return str(content.get_text())

# Apply custom parsing
loader = DocusaurusLoader(
    "https://python.langchain.com",
    parsing_function=remove_nav_and_header_elements
)
```

### Lab Exercise 3.1
Create a comprehensive web scraping solution that can handle different types of websites using appropriate loaders.

---

## Section 4: Database Integrations

### Learning Objectives
- Connect to various database systems
- Implement efficient data retrieval strategies
- Handle different data formats and structures

### 4.1 Cassandra Integration

#### Overview
Apache Cassandra is a NoSQL, row-oriented, highly scalable database. Starting with version 5.0, it includes vector search capabilities.

#### Key Features
- NoSQL architecture with high availability
- Vector search capabilities (v5.0+)
- Flexible query options
- Customizable data mapping

#### Implementation

```python
from langchain_community.document_loaders import CassandraLoader
from cassandra.cluster import Cluster

# Method 1: Direct session connection
cluster = Cluster()
session = cluster.connect()

loader = CassandraLoader(
    table="movie_reviews",
    session=session,
    keyspace="your_keyspace"
)

docs = loader.load()
print(docs[0])
```

#### Using Cassio for Configuration

```python
import cassio

# Initialize with cassio
cassio.init(contact_points="127.0.0.1", keyspace="your_keyspace")

loader = CassandraLoader(table="movie_reviews")
docs = loader.load()
```

#### Configuration Parameters
- `table`: Target table name
- `session`: Cassandra driver session
- `keyspace`: Database keyspace
- `query`: Custom SQL query
- `page_content_mapper`: Custom content mapping function
- `metadata_mapper`: Custom metadata mapping function

### 4.2 Couchbase Integration

#### Overview
Couchbase is a distributed NoSQL cloud database offering versatility, performance, and scalability for various applications.

#### Setup and Connection

```python
from langchain_community.document_loaders.couchbase import CouchbaseLoader

# Connection parameters
connection_string = "couchbase://localhost"
db_username = "Administrator"
db_password = "Password"

# SQL++ query
query = """
SELECT h.* FROM `travel-sample`.inventory.hotel h
WHERE h.country = 'United States'
LIMIT 1
"""

# Create loader
loader = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query
)

# Load documents
docs = loader.load()
```

#### Custom Field Selection

```python
# Specify content and metadata fields
loader_with_fields = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
    page_content_fields=["address", "name", "city", "description"],
    metadata_fields=["id"]
)

docs = loader_with_fields.load()
```

### Lab Exercise 4.1
Set up database connections and implement efficient data retrieval for a sample application.

---

## Section 5: Document Processing and File Formats

### Learning Objectives
- Handle various document formats (PDF, DOCX, etc.)
- Implement text extraction and processing
- Manage document metadata and structure

### 5.1 Dedoc Integration

#### Overview
Dedoc is an open-source library that extracts texts, tables, and document structure from various file formats including DOCX, XLSX, PPTX, PDF, and images.

#### Supported Formats
- Microsoft Office documents (DOCX, XLSX, PPTX)
- PDF files (with and without text layers)
- Email formats (EML)
- HTML and images
- And many more formats

#### Basic Implementation

```python
from langchain_community.document_loaders import DedocFileLoader

# Basic file loading
loader = DedocFileLoader("./example_data/state_of_the_union.txt")
docs = loader.load()
print(docs[0].page_content[:100])
```

#### Document Splitting Options

```python
# Split by pages
loader = DedocFileLoader(
    "./example_data/layout-parser-paper.pdf",
    split="page",
    pages=":2"  # First two pages
)

# Split by document structure
loader = DedocFileLoader(
    "./document.pdf",
    split="node"  # Split by document nodes
)
```

#### Table Handling

```python
# Enable table processing
loader = DedocFileLoader(
    "./example_data/data.csv",
    with_tables=True  # Default is True
)

docs = loader.load()

# Check for table content
for doc in docs:
    if doc.metadata.get("type") == "table":
        print("HTML Table:", doc.metadata["text_as_html"][:200])
```

#### Attachment Processing

```python
# Process email attachments
loader = DedocFileLoader(
    "./example_data/email_with_attachment.eml",
    with_attachments=True  # Default is False
)

docs = loader.load()

# Check for attachments
for doc in docs:
    if doc.metadata.get("type") == "attachment":
        print("Attachment content:", doc.page_content[:200])
```

### 5.2 Email Processing

#### Using UnstructuredEmailLoader

```python
from langchain_community.document_loaders import UnstructuredEmailLoader

# Basic email loading
loader = UnstructuredEmailLoader("./example_data/fake-email.eml")
data = loader.load()
print(data[0].page_content)

# Retain element structure
loader = UnstructuredEmailLoader(
    "example_data/fake-email.eml", 
    mode="elements"
)
data = loader.load()
```

#### Using OutlookMessageLoader

```python
from langchain_community.document_loaders import OutlookMessageLoader

loader = OutlookMessageLoader("example_data/fake-email.msg")
data = loader.load()
print(f"Subject: {data[0].metadata['subject']}")
print(f"Sender: {data[0].metadata['sender']}")
```

### 5.3 EPub Processing

```python
from langchain_community.document_loaders import UnstructuredEPubLoader

loader = UnstructuredEPubLoader("./example_data/childrens-literature.epub")
data = loader.load()
print(f"Document length: {len(data[0].page_content)}")
```

### Lab Exercise 5.1
Create a document processing pipeline that can handle multiple file formats and extract structured information.

---

## Section 6: API Integrations and Data Sources

### Learning Objectives
- Integrate with external APIs and services
- Handle authentication and rate limiting
- Process structured and unstructured data from various sources

### 6.1 College Confidential Integration

```python
from langchain_community.document_loaders import CollegeConfidentialLoader

loader = CollegeConfidentialLoader(
    "https://www.collegeconfidential.com/colleges/brown-university/"
)
data = loader.load()
print("Loaded college data:", len(data[0].page_content))
```

### 6.2 Confluence Integration

#### Authentication Methods
- Username/API key authentication
- OAuth2 login
- Cookie-based authentication
- Token authentication (on-premises)

#### Implementation with API Token

```python
from langchain_community.document_loaders import ConfluenceLoader

loader = ConfluenceLoader(
    url="https://yoursite.atlassian.com/wiki",
    username="your-username",
    api_key="your-api-token",
    space_key="your-space-key",
    include_attachments=True,
    limit=50
)

documents = loader.load()
```

#### Personal Access Token (On-Premise)

```python
loader = ConfluenceLoader(
    url="https://confluence.yoursite.com/",
    token="your-personal-access-token",
    space_key="your-space-key",
    include_attachments=True,
    limit=50,
    max_pages=50
)

documents = loader.load()
```

### 6.3 Datadog Logs Integration

```python
from langchain_community.document_loaders import DatadogLogsLoader

DD_API_KEY = "your_api_key"
DD_APP_KEY = "your_app_key"

query = "service:agent status:error"

loader = DatadogLogsLoader(
    query=query,
    api_key=DD_API_KEY,
    app_key=DD_APP_KEY,
    from_time=1688732708951,  # Timestamp in milliseconds
    to_time=1688736308951,
    limit=100
)

documents = loader.load()
```

### 6.4 Diffbot Integration

#### Setup and Configuration

```python
import os
from langchain_community.document_loaders import DiffbotLoader

urls = ["https://python.langchain.com/"]

loader = DiffbotLoader(
    urls=urls, 
    api_token=os.environ.get("DIFFBOT_API_TOKEN")
)

documents = loader.load()
```

#### Graph Transformation

```python
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer

diffbot_nlp = DiffbotGraphTransformer(
    diffbot_api_key=os.environ.get("DIFFBOT_API_TOKEN")
)

graph_documents = diffbot_nlp.convert_to_graph_documents(loader.load())
```

### Lab Exercise 6.1
Build an integrated data pipeline that combines multiple API sources for comprehensive data collection.

---

## Section 7: Text Processing and Semantic Layers

### Learning Objectives
- Implement advanced text processing techniques
- Work with semantic layers and knowledge graphs
- Handle complex document structures and metadata

### 7.1 Cube Semantic Layer Integration

#### Overview
Cube provides a semantic layer for building data apps, offering structure and definitions for LLM context enhancement.

#### Implementation

```python
import jwt
from langchain_community.document_loaders import CubeSemanticLoader

# Setup API credentials
api_url = "https://api-example.gcp-us-central1.cubecloudapp.dev/cubejs-api/v1/meta"
cubejs_api_secret = "your-api-secret"
security_context = {}

# Generate JWT token
api_token = jwt.encode(security_context, cubejs_api_secret, algorithm="HS256")

# Create loader
loader = CubeSemanticLoader(api_url, api_token)
documents = loader.load()
```

#### Document Structure
Each document contains:
- `page_content`: Semantic information about data model
- `metadata`: Including table_name, column_name, data_type, etc.

### 7.2 CoNLL-U Format Processing

```python
from langchain_community.document_loaders import CoNLLULoader

loader = CoNLLULoader("example_data/conllu.conllu")
document = loader.load()
print("Processed text:", document[0].page_content)
```

### 7.3 Concurrent Processing

```python
from langchain_community.document_loaders import ConcurrentLoader

# Concurrent file processing
loader = ConcurrentLoader.from_filesystem("example_data/", glob="**/*.txt")
files = loader.load()
print(f"Processed {len(files)} files concurrently")
```

### Lab Exercise 7.1
Implement a semantic processing pipeline that can handle structured data and extract meaningful relationships.

---

## Section 8: Advanced Topics and Best Practices

### Learning Objectives
- Implement advanced document processing patterns
- Understand performance optimization techniques
- Apply best practices for production deployment

### 8.1 Docugami Advanced Integration

#### Semantic Chunking
Docugami provides intelligent chunking based on document semantic structure rather than arbitrary length splitting.

```python
from docugami_langchain.document_loaders import DocugamiLoader

# Basic usage
loader = DocugamiLoader(docset_id="your-docset-id")
chunks = loader.load()

# Advanced configuration
loader.min_text_length = 64
loader.include_xml_tags = True
loader.parent_hierarchy_levels = 3
loader.max_text_length = 1024 * 8  # 8K characters

chunks = loader.load()
```

#### Document QA Implementation

```python
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

# Setup retrieval chain
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)
retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True
)

# Query the system
response = qa_chain("What are the key contract terms?")
```

#### Self-Querying Retriever

```python
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

# Define metadata fields
metadata_field_info = [
    AttributeInfo(
        name=key,
        description=f"The {key} for this chunk",
        type="string"
    )
    for key in chunks[0].metadata
    if key.lower() not in ["id", "xpath", "structure"]
]

# Create self-querying retriever
retriever = SelfQueryRetriever.from_llm(
    llm=OpenAI(temperature=0),
    vectorstore=vectordb,
    document_content_description="Contents of this chunk",
    metadata_field_info=metadata_field_info,
    verbose=True
)
```

### 8.2 Multi-Vector Retrieval Pattern

```python
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

# Setup components
vectorstore = Chroma(collection_name="retrieval", embedding_function=embeddings)
store = InMemoryStore()

# Create retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    search_type=SearchType.mmr,
    search_kwargs={"k": 2}
)

# Add documents
retriever.vectorstore.add_documents(child_chunks)
retriever.docstore.mset(parent_chunks.items())
```

### 8.3 Performance Optimization

#### Batch Processing
```python
# Process documents in batches
def process_documents_batch(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        yield process_batch(batch)
```

#### Caching Strategies
```python
# Implement caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_document_processing(document_path):
    return loader.load_document(document_path)
```

### 8.4 Error Handling and Monitoring

```python
import logging
from typing import List, Optional

def robust_document_loading(
    loader_class, 
    source: str, 
    retry_count: int = 3
) -> Optional[List]:
    """
    Robust document loading with error handling and retries
    """
    for attempt in range(retry_count):
        try:
            loader = loader_class(source)
            return loader.load()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_count - 1:
                logging.error(f"All attempts failed for {source}")
                return None
```

### Lab Exercise 8.1
Build a production-ready document processing system with advanced features, error handling, and performance optimization.

---

## Course Summary and Next Steps

### What You've Learned

1. **Document Loader Fundamentals**: Understanding the core concepts and patterns
2. **Integration Techniques**: Connecting to various data sources and APIs
3. **Advanced Processing**: Implementing semantic analysis and intelligent chunking
4. **Performance Optimization**: Building scalable and robust systems
5. **Best Practices**: Production-ready implementation patterns

### Recommended Next Steps

1. **Build a Portfolio Project**: Create a comprehensive document processing application
2. **Explore Advanced Integrations**: Investigate additional LangChain integrations
3. **Performance Tuning**: Optimize your implementations for production use
4. **Community Contribution**: Contribute to the LangChain ecosystem

### Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Community Forums and Discord](https://discord.gg/langchain)
- [GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangSmith for Monitoring](https://smith.langchain.com/)

### Final Project

Create a comprehensive document processing and analysis system that:
1. Integrates multiple data sources
2. Implements intelligent document chunking
3. Provides semantic search capabilities
4. Includes monitoring and error handling
5. Demonstrates production-ready patterns

---

*This course provides a comprehensive foundation for working with LangChain document loaders and integrations. Continue practicing and exploring to master these powerful tools for building AI applications.*