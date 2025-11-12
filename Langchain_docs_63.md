# LangChain & LangSmith Platform Comprehensive Course

## Course Overview
This comprehensive course covers LangChain's platform for agent engineering, including LangSmith for observability, evaluation, deployment, and the complete ecosystem for building reliable AI applications.

---

## Section 1: Introduction to LangChain Platform

### Lesson 1.1: Platform Overview
- **What is LangChain?**
  - Platform for agent engineering
  - Framework-agnostic tooling
  - Used by companies like Replit, Clay, Rippling, Cloudflare, Workday
  
- **Core Components:**
  - Open source agent frameworks (Python/TypeScript)
  - LangSmith platform for development and deployment
  - Integrated workflow from development to production

### Lesson 1.2: Security and Compliance
- **Security Standards:**
  - HIPAA compliance
  - SOC 2 Type 2 certification
  - GDPR compliance
  - Trust Center resources

---

## Section 2: LangSmith Platform Setup and Configuration

### Lesson 2.1: Getting Started with LangSmith
- **Account Setup:**
  - Creating an account at smith.langchain.com
  - Login options (Google, GitHub, email)
  - No credit card required for initial setup

- **API Key Management:**
  - Navigating to Settings → API Keys
  - Creating and securing API keys
  - Best practices for API key management

### Lesson 2.2: Platform Setup Options
- **Deployment Options:**
  - Cloud (SaaS) setup
  - Self-hosted deployment
  - Hybrid configurations
  
- **Infrastructure Requirements:**
  - Kubernetes deployment prerequisites
  - Docker requirements
  - Resource allocation planning

### Lesson 2.3: Self-Hosted Configuration
- **TTL and Data Retention:**
  - Enabling automatic TTL cleanup
  - Configuring retention periods (shortlived vs longlived)
  - ClickHouse TTL cleanup jobs
  - Data privacy regulation compliance

- **Configuration Parameters:**
  ```yaml
  config:
    ttl:
      enabled: true
      ttl_period_seconds:
        longlived: "34560000"  # 400 days
        shortlived: "1209600"   # 14 days
  ```

### Lesson 2.4: Advanced Configuration
- **ClickHouse Management:**
  - Cleanup job scheduling (weekends)
  - Minimum expired rows per part configuration
  - Maximum active mutations settings
  - Emergency mutation stopping procedures

- **Backup and Data Management:**
  - File system hard links impact
  - Backup directory management (/var/lib/clickhouse/backup, /var/lib/clickhouse/shadow)
  - External storage integration (S3, etc.)

---

## Section 3: LangGraph CLI and Development Tools

### Lesson 3.1: LangGraph CLI Installation and Setup
- **Installation Requirements:**
  - Docker installation verification
  - CLI installation via pip or JavaScript
  - Verification commands

- **Quick Commands Overview:**
  - `langgraph dev` - Lightweight local development
  - `langgraph build` - Docker image building
  - `langgraph up` - Local Docker deployment
  - `langgraph dockerfile` - Dockerfile generation

### Lesson 3.2: Configuration File Management
- **langgraph.json Configuration:**
  - Schema compliance
  - Required properties (dependencies, graphs)
  - Optional properties (auth, base_image, env, store)

- **Configuration Examples:**
  ```json
  {
    "$schema": "https://langgra.ph/schema.json",
    "dependencies": ["."],
    "graphs": {
      "chat": "chat.graph:graph"
    }
  }
  ```

### Lesson 3.3: Advanced CLI Features
- **Development Mode:**
  - Hot reloading capabilities
  - Debugging port configuration
  - Browser auto-launch options
  - Tunnel support for remote access

- **Build and Deployment:**
  - Multi-platform builds
  - Custom build commands
  - Image tagging strategies
  - Environment variable management

### Lesson 3.4: Store Configuration and Semantic Search
- **BaseStore Setup:**
  - Semantic search integration
  - Embedding model configuration
  - Document field indexing strategies

- **Common Model Dimensions:**
  - OpenAI text-embedding-3-large: 3072
  - OpenAI text-embedding-3-small: 1536
  - Cohere embed-english-v3.0: 1024

- **Time-to-Live (TTL) Configuration:**
  ```json
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080
    }
  }
  ```

---

## Section 4: Observability and Monitoring

### Lesson 4.1: Observability Stack Deployment
- **Prerequisites:**
  - Compute resource requirements
  - Cert-Manager installation
  - OpenTelemetry Operator setup

- **LGTM Stack Components:**
  - Loki for logs
  - Mimir for metrics and alerting
  - Tempo for traces
  - Grafana for monitoring UI

### Lesson 4.2: Prometheus Exporters
- **Metrics Collection:**
  - PostgreSQL metrics (port 9187)
  - Redis metrics (port 9121)
  - Nginx metrics (port 9113)
  - Kube State Metrics (port 8080)

### Lesson 4.3: Full Observability Implementation
- **Resource Allocation:**
  - Loki: 2vCPU/3vCPU + 2Gi/4Gi
  - Mimir: 1vCPU/2vCPU + 2Gi/4Gi
  - Tempo: 1vCPU/2vCPU + 4Gi/6Gi

- **Installation Process:**
  - Helm chart configuration
  - Service deployment verification
  - Post-installation setup

### Lesson 4.4: Grafana Usage and Monitoring
- **Access Configuration:**
  - Password retrieval from Kubernetes secrets
  - Port forwarding setup
  - Dashboard navigation

- **Monitoring Capabilities:**
  - Log analysis
  - Metric visualization
  - Trace investigation
  - Pre-built dashboard utilization

---

## Section 5: Evaluation and Testing

### Lesson 5.1: Evaluation Fundamentals
- **Core Concepts:**
  - Dataset creation and management
  - Evaluator definition
  - Metrics configuration
  - Framework integration

### Lesson 5.2: Runnable Evaluation
- **Setup Requirements:**
  - LangSmith and LangChain installation
  - OpenAI integration
  - Chain configuration

- **Evaluation Process:**
  ```python
  from langsmith import aevaluate, Client
  
  results = await aevaluate(
      chain,
      data=dataset,
      evaluators=[correct],
      experiment_prefix="gpt-4o, baseline",
  )
  ```

### Lesson 5.3: Multi-Turn Interaction Simulation
- **Simulation Components:**
  - Application function requirements
  - Simulated user creation
  - Conversation management

- **Implementation Example:**
  ```python
  from openevals.simulators import run_multiturn_simulation, create_llm_simulated_user
  
  simulator_result = run_multiturn_simulation(
      app=app,
      user=user,
      max_turns=5,
  )
  ```

### Lesson 5.4: Advanced Evaluation Techniques
- **Testing Framework Integration:**
  - pytest integration
  - Vitest/Jest compatibility
  - Trajectory evaluators

- **Custom Evaluation Scenarios:**
  - Persona modification
  - Multiple conversation turns
  - Performance assessment

---

## Section 6: Management and Administrative Tasks

### Lesson 6.1: Organization Management
- **Administrative Operations:**
  - Workspace deletion procedures
  - Organization removal processes
  - Data cleanup requirements

### Lesson 6.2: Database Management
- **PostgreSQL Operations:**
  - Connection requirements
  - Support queries
  - Data integrity maintenance

- **ClickHouse Management:**
  - Statistics generation
  - Query optimization
  - Performance monitoring

### Lesson 6.3: Data Retention and Cleanup
- **Trace Management:**
  - Deletion procedures
  - Storage optimization
  - Compliance maintenance

- **Prerequisites for Operations:**
  - kubectl access
  - Database credentials
  - Port forwarding setup

---

## Section 7: Authentication and Security

### Lesson 7.1: Custom Authentication Setup
- **Authentication Configuration:**
  ```json
  "auth": {
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      }
    }
  }
  ```

### Lesson 7.2: Security Best Practices
- **Access Control:**
  - API key management
  - Header configuration
  - Private conversation setup

- **Middleware Configuration:**
  - Custom middleware order
  - CORS setup
  - Route authentication

---

## Section 8: Deployment and Production

### Lesson 8.1: Production Deployment Strategies
- **Deployment Options:**
  - Cloud deployment with control plane
  - Standalone server deployment
  - Containerized deployments

### Lesson 8.2: CI/CD Pipeline Implementation
- **Automation Setup:**
  - Build automation
  - Testing integration
  - Deployment workflows

### Lesson 8.3: Scaling and Performance
- **Resource Optimization:**
  - Memory management
  - CPU allocation
  - Storage optimization

- **Performance Monitoring:**
  - Metrics collection
  - Alert configuration
  - Capacity planning

---

## Section 9: Studio and Development Environment

### Lesson 9.1: LangSmith Studio Overview
- **Visual Development:**
  - Interface design
  - Application testing
  - End-to-end refinement

### Lesson 9.2: Collaborative Features
- **Team Workflows:**
  - Prompt versioning
  - Collaboration tools
  - Version control integration

---

## Section 10: Advanced Topics and Troubleshooting

### Lesson 10.1: Advanced Configuration
- **HTTP Customization:**
  - Custom routes
  - Middleware configuration
  - API version pinning

### Lesson 10.2: Troubleshooting Common Issues
- **Performance Issues:**
  - Mutation management
  - Resource optimization
  - Error diagnosis

### Lesson 10.3: Integration with External Services
- **Service Connections:**
  - Database integrations
  - Third-party API connections
  - Monitoring system integration

---

## Course Conclusion

### Summary of Key Learning Outcomes
By completing this course, you will have mastered:

1. **Platform Setup and Configuration**
   - Self-hosted and cloud deployment options
   - TTL and data retention management
   - Security and compliance implementation

2. **Development Tools Mastery**
   - LangGraph CLI proficiency
   - Configuration file management
   - Local development workflows

3. **Observability and Monitoring**
   - Comprehensive monitoring stack deployment
   - Metrics collection and analysis
   - Troubleshooting and performance optimization

4. **Evaluation and Testing**
   - Automated evaluation setup
   - Multi-turn conversation simulation
   - Performance assessment techniques

5. **Production Deployment**
   - Scalable deployment strategies
   - Security best practices
   - Maintenance and administration

### Next Steps
- Explore advanced LangChain integrations
- Implement custom evaluators
- Build production-ready AI applications
- Join the LangChain community for continued learning

### Additional Resources
- [LangChain Documentation](https://docs.langchain.com)
- [LangSmith Platform](https://smith.langchain.com)
- [Community Forum](https://community.langchain.com)
- [LangChain Academy](https://academy.langchain.com)
- [Trust Center](https://trust.langchain.com)

---

**Course Duration:** 40-50 hours of comprehensive learning
**Prerequisites:** Basic knowledge of Python/TypeScript, containerization concepts, and AI/ML fundamentals
**Certification:** LangChain Platform Professional (upon completion)