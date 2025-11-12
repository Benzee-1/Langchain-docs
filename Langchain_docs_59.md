# LangSmith & LangGraph Evaluation and Deployment Course

## Course Overview

This comprehensive course covers advanced evaluation techniques, deployment strategies, and testing methodologies for LangChain applications using LangSmith and LangGraph platforms. Students will learn to build, evaluate, and deploy production-ready AI agents and RAG applications.

## Module 1: Evaluation Fundamentals

### Lesson 1.1: Introduction to LangSmith Evaluation

#### Learning Objectives
- Understand the importance of evaluation in AI applications
- Learn about different evaluation approaches and methodologies
- Set up LangSmith for evaluation tasks

#### Key Concepts
- **Evaluation Types**: Final response, trajectory, and single-step evaluations
- **Evaluation Approaches**: LLM-as-judge, similarity-based, and custom evaluators
- **Datasets and Examples**: Creating and managing evaluation datasets

#### Core Components
1. **LLM-as-Judge Evaluators**
   - Using structured outputs for consistent evaluation
   - Grading criteria and instructions design
   - Temperature settings for deterministic results

2. **Custom Evaluators**
   - Building domain-specific evaluation functions
   - Handling multimodal content in evaluations
   - Return types: boolean, numerical, and categorical scores

### Lesson 1.2: Setting Up Evaluation Infrastructure

#### Dataset Creation and Management
```python
from langsmith import Client
client = Client()

# Create dataset
dataset_name = "Your Dataset Name"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
```

#### Basic Evaluation Structure
- Input/Output schema design
- Reference outputs and ground truth
- Metadata and versioning strategies

## Module 2: RAG Application Evaluation

### Lesson 2.1: Comprehensive RAG Evaluation Framework

#### Multi-Dimensional Evaluation Approach
1. **Correctness Evaluation**
   - Factual accuracy assessment
   - Ground truth comparison
   - Structured grading schema

2. **Relevance Evaluation**
   - Question-answer relevance scoring
   - Conciseness and helpfulness metrics
   - Content appropriateness checking

3. **Groundedness Evaluation**
   - Hallucination detection
   - Source document alignment
   - Factual consistency verification

4. **Retrieval Relevance Evaluation**
   - Document retrieval quality
   - Semantic similarity assessment
   - Coverage and completeness metrics

### Lesson 2.2: Implementation Patterns

#### Evaluation Pipeline Setup
```python
# Grade output schema
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grader LLM configuration
grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)
```

#### Target Function Design
- Input preprocessing and formatting
- State management and persistence
- Output standardization

### Lesson 2.3: Advanced RAG Evaluation Techniques

#### Intermediate Step Evaluation
- Retrieval quality assessment
- Generation step validation
- Pipeline component isolation

#### Run Analysis and Debugging
- Trace examination techniques
- Performance bottleneck identification
- Error pattern analysis

## Module 3: Agent Testing with Testing Frameworks

### Lesson 3.1: Integration with Popular Testing Tools

#### Framework Compatibility
- **Pytest Integration**: LangSmith markers and fixtures
- **Vitest/Jest Support**: TypeScript/JavaScript testing
- **Test Organization**: Structured test suites and categories

#### Test Categories
1. **Off-Topic Query Handling**
   - Tool usage validation
   - Response appropriateness
   - Error handling verification

2. **Simple Tool Calling**
   - Parameter accuracy testing
   - Tool selection validation
   - Response format verification

3. **Complex Tool Calling**
   - Multi-step workflow testing
   - Tool chain validation
   - Performance efficiency measurement

### Lesson 3.2: ReAct Agent Testing Strategies

#### Financial Agent Example
```python
@pytest.mark.langsmith
def test_searches_for_correct_ticker() -> None:
    """Test that the model looks up the correct ticker on simple query."""
    query = "What is the price of Apple?"
    t.log_inputs({"query": query})
    expected = "AAPL"
    
    result = agent.nodes["agent"].invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    actual = result["messages"][0].tool_calls[0]["args"]["ticker"]
    
    assert actual == expected
```

#### LLM-as-Judge Implementation
- Groundedness verification
- Source document validation
- Feedback tracing and logging

### Lesson 3.3: Complex Agent Evaluation

#### Multi-Agent System Testing
- Customer support bot example
- Intent classification validation
- Route optimization testing

#### Three-Tier Evaluation Strategy
1. **Final Response Evaluation**: End-to-end performance assessment
2. **Trajectory Evaluation**: Step-by-step validation with partial credit
3. **Single Step Evaluation**: Individual component testing

## Module 4: Deployment and Production Strategies

### Lesson 4.1: LangSmith Deployment Architecture

#### Core Components
- **Agent Server**: Application runtime environment
- **Control Plane**: Deployment management and orchestration
- **Data Plane**: Infrastructure and supporting services

#### Deployment Types
- **Cloud Deployments**: Managed LangSmith infrastructure
- **Hybrid Deployments**: Partial cloud integration
- **Self-Hosted**: Complete on-premises control

### Lesson 4.2: Production Deployment Patterns

#### Configuration Management
```yaml
# langgraph.json example
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env"
}
```

#### Scaling and Performance
- Auto-scaling configurations
- Resource specification and limits
- Load balancing strategies

### Lesson 4.3: Integration Frameworks

#### AutoGen Integration Example
```python
def call_autogen_agent(state: MessagesState):
    messages = convert_to_openai_messages(state["messages"])
    last_message = messages[-1]
    carryover = messages[:-1] if len(messages) > 1 else []
    
    response = user_proxy.initiate_chat(
        autogen_agent,
        message=last_message,
        carryover=carryover
    )
    
    return {"messages": {"role": "assistant", "content": final_content}}
```

#### Multi-Framework Benefits
- Enhanced feature availability
- Production deployment capabilities
- Persistence and memory management

## Module 5: Streaming and Real-time Applications

### Lesson 5.1: Streaming API Implementation

#### Stream Modes and Applications
- **Values Mode**: Full state streaming after each step
- **Updates Mode**: Incremental state changes
- **Messages-Tuple Mode**: LLM token streaming
- **Debug Mode**: Comprehensive execution information

#### Implementation Patterns
```python
async for chunk in client.runs.stream(
    thread_id,
    assistant_id,
    input=inputs,
    stream_mode="updates"
):
    print(chunk.data)
```

### Lesson 5.2: Advanced Streaming Features

#### Subgraph Streaming
- Hierarchical graph execution
- Event filtering and processing
- Performance optimization techniques

#### Custom Data Streaming
- User-defined data transmission
- Event-driven architectures
- Real-time feedback mechanisms

## Module 6: Multimodal Evaluation

### Lesson 6.1: Multimodal Content Handling

#### Attachment Management
- File upload and processing
- MIME type handling
- Binary data optimization

#### Evaluation Strategies
```python
def valid_image_description(outputs: dict, attachments: dict) -> bool:
    """Use an LLM to judge if image description and images are consistent."""
    image_url = attachments["my_img"]["presigned_url"]
    # Implementation details...
    return response.choices[0].message.parsed.description_is_valid
```

### Lesson 6.2: Cross-Modal Evaluation Techniques

#### Audio-Visual Processing
- Speech-to-text evaluation
- OCR accuracy assessment
- Multi-modal consistency checking

#### Performance Considerations
- File size limitations
- Processing efficiency
- Storage optimization

## Module 7: Production Monitoring and Maintenance

### Lesson 7.1: Observability and Tracing

#### LangSmith Integration
- Automatic trace collection
- Performance monitoring
- Error tracking and analysis

#### Custom Monitoring Solutions
- Metric collection strategies
- Alert configuration
- Dashboard development

### Lesson 7.2: Continuous Integration and Deployment

#### CI/CD Pipeline Implementation
- Automated testing integration
- Deployment automation
- Rollback strategies

#### Quality Assurance
- Regression testing
- Performance benchmarking
- Security validation

## Module 8: Advanced Topics and Best Practices

### Lesson 8.1: Enterprise Deployment Patterns

#### Authentication and Security
- API key management
- Access control implementation
- Data privacy considerations

#### Scalability Planning
- Resource allocation strategies
- Performance optimization
- Cost management approaches

### Lesson 8.2: Troubleshooting and Debugging

#### Common Issues and Solutions
- Configuration problems
- Performance bottlenecks
- Integration challenges

#### Diagnostic Techniques
- Log analysis methods
- Trace inspection tools
- Performance profiling

## Course Assessment and Projects

### Capstone Project: Complete RAG Application

Students will build, evaluate, and deploy a comprehensive RAG application featuring:
- Multi-dimensional evaluation framework
- Production-ready deployment configuration
- Comprehensive testing suite
- Performance monitoring implementation

### Assessment Criteria
1. **Technical Implementation** (40%)
   - Code quality and organization
   - Proper evaluation framework implementation
   - Deployment configuration accuracy

2. **Evaluation Design** (30%)
   - Comprehensive evaluation strategy
   - Appropriate metric selection
   - Meaningful test case development

3. **Production Readiness** (20%)
   - Scalability considerations
   - Security implementation
   - Monitoring and observability

4. **Documentation and Presentation** (10%)
   - Clear documentation
   - Effective demonstration
   - Knowledge transfer capability

## Resources and References

### Official Documentation
- LangSmith Evaluation Guide
- LangGraph Deployment Documentation
- Agent Server API Reference

### Community Resources
- LangChain Forum discussions
- GitHub examples and templates
- Community best practices

### Additional Learning Materials
- Video tutorials and walkthroughs
- Interactive notebooks and examples
- Case study analyses

## Prerequisites and Setup

### Technical Requirements
- Python 3.9+ programming experience
- Basic understanding of AI/ML concepts
- Familiarity with API development
- Docker and containerization knowledge

### Software Installation
```bash
pip install -U langgraph langchain[openai] langsmith
pip install -U "langsmith[pytest]"  # For testing integration
```

### Environment Configuration
- OpenAI API key setup
- LangSmith account and API key
- Development environment preparation

This course provides comprehensive coverage of modern AI application evaluation, testing, and deployment using industry-leading tools and methodologies. Students will gain practical experience with real-world scenarios and production-ready implementations.