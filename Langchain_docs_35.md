# Google Cloud Database and Document Management Course

## Course Overview

This comprehensive course covers Google Cloud database services, document management, and various data loading techniques. Students will learn to work with Cloud SQL, Firestore, Spanner, and multiple document loaders for building AI-powered applications.

---

## Module 1: Google Cloud SQL for SQL Server

### 1.1 Introduction to Cloud SQL for SQL Server

Cloud SQL is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. It offers MySQL, PostgreSQL, and SQL Server database engines, allowing you to extend your database application to build AI-powered experiences.

### 1.2 Prerequisites and Setup

**Before You Begin:**
- Create a Google Cloud Project
- Enable the Cloud SQL Admin API
- Create a Cloud SQL for SQL server instance
- Create a Cloud SQL database
- Add an IAM database user to the database (Optional)

**Required Parameters:**
```python
REGION = "us-central1"
INSTANCE = "test-instance"
DB_USER = "sqlserver"
DB_PASSWORD = "password"
DATABASE = "test"
TABLE_NAME = "test-default"
```

### 1.3 Library Installation and Authentication

**Installation:**
```bash
pip install -qU langchain-google-cloud-sql-mssql
```

**Authentication Setup:**
```python
from google.colab import auth
auth.authenticate_user()

# Set Google Cloud Project
PROJECT_ID = "my-project-id"
!gcloud config set project {PROJECT_ID}

# Enable Cloud SQL Admin API
!gcloud services enable sqladmin.googleapis.com
```

### 1.4 MSSQLEngine Connection Pool

The MSSQLEngine configures a SQLAlchemy connection pool to your Cloud SQL database:

```python
from langchain_google_cloud_sql_mssql import MSSQLEngine

engine = MSSQLEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    instance=INSTANCE,
    database=DATABASE,
    user=DB_USER,
    password=DB_PASS,
)
```

### 1.5 Basic Database Operations

**Initialize a Table:**
```python
engine.init_document_table(TABLE_NAME, overwrite_existing=True)
```

**Save Documents:**
```python
from langchain_core.documents import Document
from langchain_google_cloud_sql_mssql import MSSQLDocumentSaver

test_docs = [
    Document(
        page_content="Apple Granny Smith 150 0.99 1",
        metadata={"fruit_id": 1},
    ),
    Document(
        page_content="Banana Cavendish 200 0.59 0",
        metadata={"fruit_id": 2},
    ),
    Document(
        page_content="Orange Navel 80 1.29 1",
        metadata={"fruit_id": 3},
    ),
]

saver = MSSQLDocumentSaver(engine=engine, table_name=TABLE_NAME)
saver.add_documents(test_docs)
```

**Load Documents:**
```python
from langchain_google_cloud_sql_mssql import MSSQLLoader

loader = MSSQLLoader(engine=engine, table_name=TABLE_NAME)
docs = loader.lazy_load()
for doc in docs:
    print("Loaded documents:", doc)
```

### 1.6 Advanced Operations

**Load Documents via Query:**
```python
loader = MSSQLLoader(
    engine=engine,
    query=f"select * from \"{TABLE_NAME}\" where JSON_VALUE(langchain_metadata, '$.fruit_id') = 1;",
)
onedoc = loader.load()
```

**Delete Documents:**
```python
loader = MSSQLLoader(engine=engine, table_name=TABLE_NAME)
docs = loader.load()
saver.delete(docs)
```

### 1.7 Customized Document Management

**Custom Page Content & Metadata:**
```python
# Create custom table structure
engine.init_document_table(
    TABLE_NAME,
    metadata_columns=[
        sqlalchemy.Column(
            "fruit_name",
            sqlalchemy.UnicodeText,
            primary_key=False,
            nullable=True,
        ),
        sqlalchemy.Column(
            "organic",
            sqlalchemy.Boolean,
            primary_key=False,
            nullable=True,
        ),
    ],
    content_column="description",
    metadata_json_column="other_metadata",
    overwrite_existing=True,
)
```

---

## Module 2: Google Cloud SQL for PostgreSQL

### 2.1 Introduction to PostgreSQL on Cloud SQL

Cloud SQL for PostgreSQL is a fully-managed database service that helps you set up, maintain, manage, and administer your PostgreSQL relational databases on Google Cloud Platform.

### 2.2 Setup and Configuration

**Installation:**
```bash
pip install -qU langchain_google_cloud_sql_pg
```

**Basic Configuration:**
```python
REGION = "us-central1"
INSTANCE = "my-primary"
DATABASE = "my-database"
TABLE_NAME = "vector_store"
```

### 2.3 PostgresEngine Setup

```python
from langchain_google_cloud_sql_pg import PostgresEngine

engine = await PostgresEngine.afrom_instance(
    project_id=PROJECT_ID,
    region=REGION,
    instance=INSTANCE,
    database=DATABASE,
)
```

### 2.4 PostgresLoader Operations

**Create Loader:**
```python
from langchain_google_cloud_sql_pg import PostgresLoader

loader = await PostgresLoader.create(engine, table_name=TABLE_NAME)
```

**Load Documents with Custom Columns:**
```python
loader = await PostgresLoader.create(
    engine,
    table_name=TABLE_NAME,
    content_columns=["product_name"],
    metadata_columns=["id"],
)
docs = await loader.aload()
```

**Set Page Content Format:**
```python
loader = await PostgresLoader.create(
    engine,
    table_name="products",
    content_columns=["product_name", "description"],
    format="YAML",
)
docs = await loader.aload()
```

---

## Module 3: Google Cloud Storage

### 3.1 Google Cloud Storage Directory Loader

Google Cloud Storage is a managed service for storing unstructured data.

**Installation:**
```bash
pip install -qU langchain-google-community[gcs]
```

**Basic Usage:**
```python
from langchain_google_community import GCSDirectoryLoader

loader = GCSDirectoryLoader(project_name="aist", bucket="testing-hwc")
documents = loader.load()
```

**Specifying a Prefix:**
```python
loader = GCSDirectoryLoader(
    project_name="aist", 
    bucket="testing-hwc", 
    prefix="fake"
)
```

**Continue on Failure:**
```python
loader = GCSDirectoryLoader(
    project_name="aist", 
    bucket="testing-hwc", 
    continue_on_failure=True
)
```

### 3.2 Google Cloud Storage File Loader

**Load Single Files:**
```python
from langchain_google_community import GCSFileLoader

loader = GCSFileLoader(
    project_name="aist", 
    bucket="testing-hwc", 
    blob="fake.docx"
)
documents = loader.load()
```

**Custom Loader Functions:**
```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    return PyPDFLoader(file_path)

loader = GCSFileLoader(
    project_name="aist", 
    bucket="testing-hwc", 
    blob="fake.pdf", 
    loader_func=load_pdf
)
```

---

## Module 4: Google El Carro for Oracle Workloads

### 4.1 Introduction to El Carro

Google El Carro Oracle Operator offers a way to run Oracle databases in Kubernetes as a portable, open source, community driven, no vendor lock-in container orchestration system.

### 4.2 Setup and Configuration

**Installation:**
```bash
pip install -qU langchain-google-el-carro
```

**Database Connection:**
```python
HOST = "127.0.0.1"
PORT = 3307
DATABASE = "my-database"
TABLE_NAME = "message_store"
USER = "my-user"
PASSWORD = input("Please provide a password: ")
```

### 4.3 ElCarroEngine Operations

**Connection Pool:**
```python
from langchain_google_el_carro import ElCarroEngine

elcarro_engine = ElCarroEngine.from_instance(
    db_host=HOST,
    db_port=PORT,
    db_name=DATABASE,
    db_user=USER,
    db_password=PASSWORD,
)
```

**Initialize Table:**
```python
elcarro_engine.drop_document_table(TABLE_NAME)
elcarro_engine.init_document_table(table_name=TABLE_NAME)
```

### 4.4 Document Operations

**Save Documents:**
```python
from langchain_core.documents import Document
from langchain_google_el_carro import ElCarroDocumentSaver

doc = Document(
    page_content="Banana",
    metadata={"type": "fruit", "weight": 100, "organic": 1},
)

saver = ElCarroDocumentSaver(
    elcarro_engine=elcarro_engine,
    table_name=TABLE_NAME,
)
saver.add_documents([doc])
```

**Load Documents:**
```python
from langchain_google_el_carro import ElCarroLoader

loader = ElCarroLoader(
    elcarro_engine=elcarro_engine, 
    table_name=TABLE_NAME
)
docs = loader.lazy_load()
```

---

## Module 5: Google Firestore

### 5.1 Firestore Native Mode

Firestore is a serverless document-oriented database that scales to meet any demand.

**Installation:**
```bash
pip install -qU langchain-google-firestore
```

**Basic Operations:**
```python
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreSaver

saver = FirestoreSaver()
data = [Document(page_content="Hello, World!")]
saver.upsert_documents(data)
```

**Load Documents:**
```python
from langchain_google_firestore import FirestoreLoader

loader_collection = FirestoreLoader("Collection")
data_collection = loader_collection.load()
```

### 5.2 Firestore in Datastore Mode

**Installation:**
```bash
pip install -upgrade --quiet langchain-google-datastore
```

**Basic Operations:**
```python
from langchain_core.documents import Document
from langchain_google_datastore import DatastoreSaver

saver = DatastoreSaver()
data = [Document(page_content="Hello, World!")]
saver.upsert_documents(data)
```

---

## Module 6: Google Memorystore for Redis

### 6.1 Introduction to Memorystore

Google Memorystore for Redis is a fully-managed service powered by Redis in-memory data store.

**Installation:**
```bash
pip install -upgrade --quiet langchain-google-memorystore-redis
```

### 6.2 Basic Operations

**Save Documents:**
```python
import redis
from langchain_core.documents import Document
from langchain_google_memorystore_redis import MemorystoreDocumentSaver

redis_client = redis.from_url(ENDPOINT)
saver = MemorystoreDocumentSaver(
    client=redis_client,
    key_prefix=KEY_PREFIX,
    content_field="page_content",
)
saver.add_documents(test_docs, ids=doc_ids)
```

**Load Documents:**
```python
from langchain_google_memorystore_redis import MemorystoreDocumentLoader

loader = MemorystoreDocumentLoader(
    client=redis_client,
    key_prefix=KEY_PREFIX,
    content_fields=set(["page_content"]),
)
docs = loader.lazy_load()
```

---

## Module 7: Google Spanner

### 7.1 Introduction to Spanner

Spanner is a highly scalable database that combines unlimited scalability with relational semantics.

**Installation:**
```bash
pip install -upgrade --quiet langchain-google-spanner langchain
```

### 7.2 Basic Operations

**Save Documents:**
```python
from langchain_core.documents import Document
from langchain_google_spanner import SpannerDocumentSaver

saver = SpannerDocumentSaver(
    instance_id=INSTANCE_ID,
    database_id=DATABASE_ID,
    table_name=TABLE_NAME,
)
saver.add_documents(test_docs)
```

**Load Documents:**
```python
from langchain_google_spanner import SpannerLoader

query = f"SELECT * from {TABLE_NAME}"
loader = SpannerLoader(
    instance_id=INSTANCE_ID,
    database_id=DATABASE_ID,
    query=query,
)
docs = loader.lazy_load()
```

---

## Module 8: Document Loaders and Data Sources

### 8.1 Google Speech-to-Text Audio Transcripts

**Installation:**
```bash
pip install -qU langchain-google-community[speech]
```

**Basic Usage:**
```python
from langchain_google_community import SpeechToTextLoader

project_id = "<PROJECT_ID>"
file_path = "gs://cloud-samples-data/speech/audio.flac"
loader = SpeechToTextLoader(project_id=project_id, file_path=file_path)
docs = loader.load()
```

### 8.2 Google Drive Integration

**Installation:**
```bash
pip install -qU langchain-google-community[drive]
```

**Basic Operations:**
```python
from langchain_google_community import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5",
    token_path="/path/to/token/google_token.json",
    recursive=False,
)
docs = loader.load()
```

### 8.3 YouTube Integration

**Audio Transcription:**
```bash
pip install -qU yt_dlp pydub librosa
```

```python
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

urls = ["https://youtu.be/kCc8FmEb1nY"]
save_dir = "~/Downloads/YouTube"

loader = GenericLoader(
    YoutubeAudioLoader(urls, save_dir), 
    OpenAIWhisperParser()
)
docs = loader.load()
```

**Transcript Loading:**
```bash
pip install -qU youtube-transcript-api pytube
```

```python
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg", 
    add_video_info=True
)
docs = loader.load()
```

---

## Module 9: AWS Integration

### 9.1 AWS S3 Integration

**S3 Directory Loader:**
```bash
pip install -qU boto3
```

```python
from langchain_community.document_loaders import S3DirectoryLoader

loader = S3DirectoryLoader("testing-hwc")
docs = loader.load()

# With prefix
loader = S3DirectoryLoader("testing-hwc", prefix="fake")
```

**S3 File Loader:**
```python
from langchain_community.document_loaders import S3FileLoader

loader = S3FileLoader("testing-hwc", "fake.docx")
docs = loader.load()
```

### 9.2 Amazon Textract

**Installation:**
```bash
pip install -qU boto3 langchain-openai tiktoken python-dotenv
pip install -qU "amazon-textract-caller>=0.2.0"
```

**Basic Usage:**
```python
from langchain_community.document_loaders import AmazonTextractPDFLoader

loader = AmazonTextractPDFLoader("example_data/document.jpeg")
documents = loader.load()
```

### 9.3 Athena Integration

**Setup:**
```bash
pip install boto3
```

```python
from langchain_community.document_loaders.athena import AthenaLoader

database_name = "my_database"
s3_output_path = "s3://my_bucket/query_results/"
query = "SELECT * FROM my_table"
profile_name = "my_profile"

loader = AthenaLoader(
    query=query,
    database=database_name,
    s3_output_uri=s3_output_path,
    profile_name=profile_name,
)
documents = loader.load()
```

---

## Module 10: Microsoft Azure Integration

### 10.1 Azure AI Data

**Installation:**
```bash
pip install -qU azureml-fsspec azure-ai-generative
```

```python
from azure.ai.resources.client import AIClient
from azure.identity import DefaultAzureCredential
from langchain_community.document_loaders import AzureAIDataLoader

client = AIClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group_name>",
    project_name="<project_name>",
)

data_asset = client.data.get(name="<data_asset_name>", label="latest")
loader = AzureAIDataLoader(url=data_asset.path)
```

### 10.2 Azure AI Document Intelligence

**Installation:**
```bash
pip install -qU langchain langchain-community azure-ai-documentintelligence
```

```python
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, 
    api_key=key, 
    file_path=file_path, 
    api_model="prebuilt-layout"
)
documents = loader.load()
```

### 10.3 Azure Blob Storage

**Installation:**
```bash
pip install -qU langchain-azure-storage
```

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    "https://<storage-account-name>.blob.core.windows.net",
    "<container-name>",
)
docs = loader.load()
```

---

## Module 11: Microsoft Office Integration

### 11.1 Microsoft OneDrive

**Prerequisites:**
- Register application with Microsoft identity platform
- Obtain CLIENT_ID, CLIENT_SECRET, and DRIVE_ID
- Install o365 package

```bash
pip install o365
```

```python
from langchain_community.document_loaders.onedrive import OneDriveLoader

loader = OneDriveLoader(
    drive_id="YOUR DRIVE ID", 
    folder_path="Documents/clients", 
    auth_with_token=True
)
documents = loader.load()
```

### 11.2 Microsoft Word Documents

**Using Docx2txt:**
```bash
pip install -qU docx2txt
```

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./example_data/fake.docx")
data = loader.load()
```

**Using Unstructured:**
```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

loader = UnstructuredWordDocumentLoader("example_data/fake.docx")
data = loader.load()
```

### 11.3 Microsoft Excel

```bash
pip install -qU langchain-community unstructured openpyxl
```

```python
from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader(
    "./example_data/stanley-cups.xlsx", 
    mode="elements"
)
docs = loader.load()
```

### 11.4 Microsoft SharePoint

**Prerequisites:**
- Register application with Microsoft identity platform
- Configure proper scopes and permissions

```python
from langchain_community.document_loaders.sharepoint import SharePointLoader

loader = SharePointLoader(
    document_library_id="YOUR DOCUMENT LIBRARY ID", 
    folder_path="Documents/marketing", 
    auth_with_token=True
)
documents = loader.load()
```

### 11.5 Microsoft PowerPoint

```bash
pip install unstructured python-magic python-pptx
```

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader("./example_data/fake-power-point.pptx")
data = loader.load()
```

### 11.6 Microsoft OneNote

**Prerequisites:**
- Register application with Microsoft identity platform
- Install msal and bs4 packages

```bash
pip install msal beautifulsoup4
```

```python
from langchain_community.document_loaders.onenote import OneNoteLoader

loader = OneNoteLoader(
    notebook_name="NOTEBOOK NAME", 
    section_name="SECTION NAME", 
    page_title="PAGE TITLE", 
    auth_with_token=True
)
documents = loader.load()
```

---

## Module 12: Additional Data Sources

### 12.1 HuggingFace Dataset Integration

```python
from langchain_community.document_loaders import HuggingFaceDatasetLoader

dataset_name = "imdb"
page_content_column = "text"
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()
```

### 12.2 Image Caption Loading

```bash
pip install -qU transformers langchain_openai langchain_chroma
```

```python
from langchain_community.document_loaders import ImageCaptionLoader

list_image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/image1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/image2.jpg",
]

loader = ImageCaptionLoader(images=list_image_urls)
list_docs = loader.load()
```

### 12.3 URL Loading

**Unstructured URL Loader:**
```bash
pip install -qU unstructured
```

```python
from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.example.com/page1",
    "https://www.example.com/page2",
]
loader = UnstructuredURLLoader(urls)
```

---

## Module 13: Best Practices and Advanced Topics

### 13.1 Authentication Best Practices

- Use environment variables for sensitive credentials
- Implement proper token management
- Follow principle of least privilege
- Use service accounts where appropriate

### 13.2 Error Handling and Retry Logic

```python
try:
    documents = loader.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    # Implement retry logic or fallback mechanism
```

### 13.3 Performance Optimization

- Use lazy loading for large datasets
- Implement pagination for large collections
- Consider async operations where available
- Cache frequently accessed data

### 13.4 Security Considerations

- Encrypt sensitive data at rest and in transit
- Implement proper access controls
- Regular security audits
- Monitor and log access patterns

---

## Module 14: Practical Applications

### 14.1 Building a Document Search System

Combine multiple loaders to create a comprehensive document search system:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents from multiple sources
all_docs = []
all_docs.extend(google_drive_loader.load())
all_docs.extend(sharepoint_loader.load())
all_docs.extend(s3_loader.load())

# Split and embed documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(all_docs)
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)
```

### 14.2 Creating a Multi-Modal RAG System

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = vectorstore.as_retriever(k=5)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

---

## Course Assessment and Projects

### Final Project Options:

1. **Enterprise Document Management System**
   - Integrate multiple cloud storage services
   - Implement advanced search capabilities
   - Create a user-friendly interface

2. **Multi-Modal AI Assistant**
   - Process text, audio, and image content
   - Implement conversational capabilities
   - Deploy to cloud infrastructure

3. **Cross-Platform Data Integration**
   - Connect disparate data sources
   - Implement real-time synchronization
   - Create analytics dashboards

---

## Conclusion

This course has provided comprehensive coverage of Google Cloud database services and document management systems. Students have learned to:

- Work with various Google Cloud database services
- Implement document loading from multiple sources
- Handle authentication and security properly
- Build scalable AI-powered applications
- Apply best practices for production systems

The skills acquired in this course form the foundation for building robust, scalable data management and AI applications in enterprise environments.

---

## Additional Resources

- [Google Cloud Documentation](https://cloud.google.com/docs)
- [LangChain Documentation](https://docs.langchain.com)
- [Microsoft Graph API Documentation](https://docs.microsoft.com/en-us/graph/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Azure Documentation](https://docs.microsoft.com/en-us/azure/)