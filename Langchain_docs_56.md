# LangSmith Deployment & CI/CD Systems Course

## Course Overview
This comprehensive course covers LangSmith deployment strategies, CI/CD pipeline implementation, authentication systems, and observability tools for LLM applications and AI agents.

---

## Module 1: Introduction to LangSmith Deployment

### Learning Objectives
- Understand LangSmith deployment architecture
- Learn about different hosting models
- Explore deployment components and infrastructure

### Lesson 1.1: Deployment Architecture Overview
#### What is LangSmith Deployment?
LangSmith provides a platform for deploying, managing, and monitoring AI agents and LLM applications. The deployment system supports multiple hosting models:

- **Cloud LangSmith**: Fully managed service with direct GitHub integration
- **Self-Hosted/Hybrid**: Container registry-based deployments with control plane management
- **Standalone Servers**: Independent services without UI or control plane

#### Key Components
1. **Agent Server**: API for creating and managing agent-based applications
2. **Control Plane**: Management interface for deployments and configurations
3. **Data Plane**: Runtime environment where applications execute
4. **Studio**: Visual interface for testing and debugging

### Lesson 1.2: Prerequisites and Setup
#### Requirements
- LangGraph graph implementation
- GitHub repository (for Cloud deployments)
- LangSmith API key
- Docker (for local development)
- Dependencies file (requirements.txt or pyproject.toml)

#### Configuration Structure
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.11"
}
```

---

## Module 2: CI/CD Pipeline Implementation

### Learning Objectives
- Design automated deployment pipelines
- Implement testing strategies for AI applications
- Configure continuous integration workflows

### Lesson 2.1: CI/CD Pipeline Architecture
#### Pipeline Components
The CI/CD pipeline implements multiple layers of validation and deployment:

1. **Trigger Sources**:
   - Git push events (code changes)
   - PromptHub updates (prompt template changes)
   - Online evaluation alerts (performance degradation)
   - Manual triggers (testing/emergency deployments)

2. **Testing Layers**:
   - Unit tests (individual node testing)
   - Integration tests (component interaction)
   - End-to-end tests (full graph execution)
   - Offline evaluations (real-world scenario assessment)

### Lesson 2.2: Evaluation Strategies
#### Types of Evaluations
1. **Final Response Evaluation**: Validates agent's final output against expected results
2. **Single Step Evaluation**: Tests individual workflow nodes in isolation
3. **Agent Trajectory Evaluation**: Analyzes complete decision paths and tool usage
4. **Multi-Turn Evaluation**: Tests conversational flows and context maintenance

#### Implementation Example
```python
# Offline evaluation setup
evaluator_config = {
    "final_response": validate_final_output,
    "trajectory": analyze_decision_path,
    "multi_turn": test_conversation_flow
}
```

### Lesson 2.3: GitHub Actions Workflow
#### Workflow Configuration
```yaml
name: LangSmith CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
      - name: Run tests
      - name: Deploy to LangSmith
```

#### Deployment Flow
1. **New Deployment**: PR creation triggers preview deployment
2. **Revision Management**: Merge to main creates production deployment
3. **Testing Integration**: Comprehensive evaluation before promotion

---

## Module 3: Authentication and Access Control

### Learning Objectives
- Implement custom authentication systems
- Design authorization patterns
- Secure agent communications

### Lesson 3.1: Authentication Fundamentals
#### Authentication vs Authorization
- **Authentication**: Verifies user identity (who you are)
- **Authorization**: Determines access permissions (what you can do)

#### Implementation Pattern
```python
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    # Validate credentials
    api_key = headers.get(b"x-api-key")
    if not api_key or not is_valid_key(api_key):
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return {
        "identity": "user-123",
        "is_authenticated": True,
        "permissions": ["read", "write"]
    }
```

### Lesson 3.2: Resource Authorization
#### Authorization Handlers
```python
@auth.on
async def add_owner(ctx: Auth.types.AuthContext, value: dict):
    # Add ownership metadata
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    
    # Return access filter
    return filters
```

#### Access Patterns
1. **Single-Owner Resources**: User-scoped access control
2. **Permission-Based Access**: Role-based permissions
3. **Resource-Specific Handlers**: Granular control per resource type

### Lesson 3.3: OAuth2 Integration
#### Provider Setup
```python
# OAuth2 configuration
@auth.authenticate
async def authenticate(authorization: str | None):
    scheme, token = authorization.split()
    
    # Validate with OAuth provider
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AUTH_PROVIDER_URL}/user",
            headers={"Authorization": authorization}
        )
        user = response.json()
        
    return {
        "identity": user["id"],
        "email": user["email"]
    }
```

---

## Module 4: Deployment Strategies

### Learning Objectives
- Master different deployment methods
- Configure environment variables
- Manage deployment lifecycles

### Lesson 4.1: Cloud Deployment
#### GitHub Integration
1. **Repository Setup**: Configure LangSmith GitHub app access
2. **Deployment Creation**: Use LangSmith UI for direct deployment
3. **Branch Management**: Link deployments to specific branches
4. **Automatic Updates**: Enable push-triggered deployments

#### Configuration Example
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env",
  "deployment_type": "production"
}
```

### Lesson 4.2: Self-Hosted Deployment
#### Docker Image Workflow
```bash
# Build image
langgraph build -t my-agent:latest

# Push to registry
docker push my-registry.com/my-agent:latest

# Deploy via control plane
# Use control plane UI with image URL
```

#### Environment Configuration
```bash
# Required variables
REDIS_URI=redis://hostname:6379/0
DATABASE_URI=postgres://user:pass@host:5432/db
LANGSMITH_API_KEY=your-api-key
LANGGRAPH_CLOUD_LICENSE_KEY=your-license-key
```

### Lesson 4.3: Standalone Servers
#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: my-agent:latest
        env:
        - name: REDIS_URI
          value: "redis://redis:6379"
```

---

## Module 5: Observability and Studio

### Learning Objectives
- Implement comprehensive monitoring
- Use Studio for debugging and testing
- Set up tracing and evaluation systems

### Lesson 5.1: Studio Fundamentals
#### Core Workflows
1. **Run Application**: Execute and observe agent behavior
2. **Manage Assistants**: Configure and select assistant settings
3. **Manage Threads**: View and organize conversation threads

#### Graph vs Chat Mode
- **Graph Mode**: Full-featured execution visualization
- **Chat Mode**: Lightweight conversational interface

### Lesson 5.2: Tracing and Monitoring
#### Automatic Tracing
```python
# LangSmith automatically creates tracing projects
# Configuration in langgraph.json enables observability
{
  "graphs": {"agent": "./graph.py:graph"},
  "env": ".env"  # LANGSMITH_API_KEY enables tracing
}
```

#### Custom Metrics
- Request/response times
- Error rates and patterns
- Resource utilization
- Agent performance metrics

### Lesson 5.3: Debugging Workflows
#### Prompt Iteration
1. **Direct Node Editing**: Modify prompts in graph interface
2. **Playground Testing**: Test individual LLM calls
3. **Configuration Management**: Update assistant settings

#### Thread Analysis
- View execution history
- Fork and replay scenarios
- Edit state for testing
- Compare different runs

---

## Module 6: Advanced Features

### Learning Objectives
- Implement semantic search capabilities
- Configure TTL and data management
- Customize server behavior

### Lesson 6.1: Semantic Search Integration
#### Store Configuration
```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    }
  }
}
```

#### Custom Embedding Functions
```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function for semantic search."""
    # Implementation using preferred embedding model
    return [[0.1, 0.2, ...] for _ in texts]  # dims-dimensional vectors
```

### Lesson 6.2: Data Lifecycle Management
#### TTL Configuration
```json
{
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080  // 7 days in minutes
    }
  }
}
```

#### Checkpoint Management
```json
{
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 10,
      "default_ttl": 43200  // 30 days
    }
  }
}
```

### Lesson 6.3: Server Customization
#### Custom Routes
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/custom-endpoint")
def custom_logic():
    return {"result": "custom response"}
```

#### Middleware Integration
```python
from starlette.middleware.base import BaseHTTPMiddleware

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers['X-Custom-Header'] = 'Hello from middleware!'
        return response

app.add_middleware(CustomHeaderMiddleware)
```

---

## Module 7: Troubleshooting and Best Practices

### Learning Objectives
- Diagnose common deployment issues
- Implement monitoring and alerting
- Follow security best practices

### Lesson 7.1: Common Issues and Solutions
#### Connection Problems
1. **Safari/Brave Issues**: Use Cloudflare tunnel for local development
2. **Graph Edge Problems**: Define explicit routing paths
3. **Authentication Failures**: Verify token validation logic

#### Environment Variables
```bash
# Debugging variables
LOG_LEVEL=DEBUG
LOG_JSON=true
BG_JOB_ISOLATED_LOOPS=true  # For synchronous code
```

### Lesson 7.2: Performance Optimization
#### Resource Management
- **Connection Pooling**: Configure Postgres and Redis pool sizes
- **Background Jobs**: Optimize job isolation and timeouts
- **Memory Management**: Monitor and tune resource allocation

#### Scaling Considerations
```bash
# Scaling configuration
N_JOBS_PER_WORKER=10
LANGGRAPH_POSTGRES_POOL_MAX_SIZE=150
REDIS_MAX_CONNECTIONS=2000
```

### Lesson 7.3: Security Best Practices
#### API Security
1. **Authentication**: Always implement custom auth for production
2. **Authorization**: Use resource-specific access controls
3. **Secrets Management**: Store sensitive data securely

#### Network Security
- Configure CORS policies
- Implement rate limiting
- Use HTTPS in production
- Allowlist IP addresses where needed

---

## Module 8: Integration and Ecosystem

### Learning Objectives
- Integrate with external services
- Connect multiple agents
- Implement MCP (Model Context Protocol) support

### Lesson 8.1: External Integrations
#### API Integrations
```python
def my_node(state, config):
    user_config = config["configurable"].get("langgraph_auth_user")
    token = user_config.get("github_token", "")
    
    # Use token for authenticated API calls
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://api.github.com/user", headers=headers)
    return {"result": response.json()}
```

#### Database Connections
- Use custom middleware for database access
- Implement connection pooling
- Handle transactions properly

### Lesson 8.2: Multi-Agent Systems
#### RemoteGraph Usage
```python
from langgraph_sdk import get_client

# Connect to deployed agent
client = get_client(url="https://your-deployment.com")

# Use in another graph
async def call_remote_agent(state):
    result = await client.runs.create(
        thread_id=state["thread_id"],
        assistant_id="remote-agent",
        input=state["input"]
    )
    return {"remote_result": result}
```

### Lesson 8.3: Advanced Monitoring
#### Custom Dashboards
- Configure project dashboards
- Set up alerting rules
- Monitor key performance indicators

#### Evaluation Pipelines
- Implement continuous evaluation
- Set up A/B testing frameworks
- Monitor model drift and performance

---

## Practical Exercises

### Exercise 1: Basic Deployment Setup
1. Create a simple LangGraph application
2. Configure langgraph.json
3. Deploy to LangSmith Cloud
4. Test via Studio interface

### Exercise 2: CI/CD Pipeline Implementation
1. Set up GitHub Actions workflow
2. Implement testing layers
3. Configure automated deployment
4. Test with sample code changes

### Exercise 3: Authentication System
1. Implement custom authentication
2. Add resource authorization
3. Test with multiple users
4. Integrate OAuth2 provider

### Exercise 4: Advanced Features
1. Configure semantic search
2. Implement TTL policies
3. Add custom routes
4. Set up monitoring dashboards

---

## Assessment and Certification

### Knowledge Check Questions
1. What are the key components of LangSmith deployment architecture?
2. How do you implement resource-specific authorization?
3. What testing strategies are recommended for AI applications?
4. How do you configure semantic search in the BaseStore?

### Practical Assessment
- Deploy a complete LangGraph application with authentication
- Implement CI/CD pipeline with comprehensive testing
- Configure monitoring and observability
- Demonstrate troubleshooting capabilities

---

## Additional Resources

### Documentation Links
- [LangSmith Deployment Guide](https://docs.smith.langchain.com/deployment)
- [LangGraph CLI Reference](https://docs.smith.langchain.com/reference/langgraph-cli)
- [Authentication API Reference](https://docs.smith.langchain.com/auth)

### Community Resources
- LangChain Forum
- GitHub Examples Repository
- Community Discord/Slack

### Next Steps
- Advanced agent architectures
- Multi-modal applications
- Enterprise deployment patterns
- Custom evaluation frameworks

---

*This course provides comprehensive coverage of LangSmith deployment and CI/CD systems, from basic concepts to advanced implementation patterns. Each module builds upon previous knowledge while providing practical, hands-on experience with real-world deployment scenarios.*