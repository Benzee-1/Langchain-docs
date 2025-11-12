# LangSmith Comprehensive Course

## Course Overview
This course provides a comprehensive guide to LangSmith, covering observability, evaluation, and development of LLM applications from basic concepts to advanced techniques.

## Table of Contents
1. [Getting Started with LangSmith](#getting-started-with-langsmith)
2. [Observability and Tracing](#observability-and-tracing)
3. [Evaluation Framework](#evaluation-framework)
4. [Advanced Tracing Techniques](#advanced-tracing-techniques)
5. [Dataset Management](#dataset-management)
6. [Prompt Engineering](#prompt-engineering)
7. [Deployment and Production](#deployment-and-production)
8. [Agent Development](#agent-development)

---

## Section 1: Getting Started with LangSmith

### Lesson 1.1: Introduction to LangSmith
- **What is LangSmith?**
  - Platform for developing and evaluating language model applications
  - Compatible with any LLM framework
  - Seamless integration with LangChain

- **Key Features:**
  - Trace logging and debugging
  - Dataset creation and management
  - Evaluation framework
  - Prompt optimization
  - Production monitoring

### Lesson 1.2: Account Setup and Configuration
- **Account Creation:**
  1. Sign up using GitHub, Discord, or email
  2. Verify email address
  3. Create unique API key from Settings Page

- **Environment Variables:**
  ```bash
  export LANGSMITH_TRACING="true"
  export LANGSMITH_API_KEY="<your-api-key>"
  export LANGSMITH_PROJECT="My Project Name"
  export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
  ```

### Lesson 1.3: First Steps with Tracing
- **Basic Tracing Setup:**
  - Using the `@traceable` decorator
  - Automatic trace collection
  - Viewing traces in the UI

---

## Section 2: Observability and Tracing

### Lesson 2.1: Tracing Fundamentals
- **Core Concepts:**
  - Traces: Complete execution flow
  - Runs (Spans): Individual operations
  - Projects: Grouping of traces
  - Metadata and tags

### Lesson 2.2: Manual Instrumentation
- **Using the `@traceable` decorator:**
  ```python
  from langsmith import traceable
  
  @traceable
  def format_prompt(subject):
      return [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"What's a good name for a store that sells {subject}?"}
      ]
  ```

- **Trace Context Manager:**
  ```python
  with ls.trace("Chat Pipeline", "chain", inputs=inputs) as rt:
      output = chat_pipeline("Can you summarize this morning's meetings?")
      rt.end(outputs={"output": output})
  ```

### Lesson 2.3: LLM Call Logging
- **Proper Message Formatting:**
  - LangChain format support
  - OpenAI completions format
  - Anthropic messages format

- **Token and Cost Tracking:**
  ```python
  @traceable(
      run_type="llm",
      metadata={"ls_provider": "my_provider", "ls_model_name": "my_model"}
  )
  def chat_model(messages: list):
      return output
  ```

### Lesson 2.4: Integration Patterns
- **Framework Integrations:**
  - LangChain automatic tracing
  - OpenAI SDK wrapper
  - Custom integrations
  - Next.js applications
  - Vercel AI SDK

---

## Section 3: Evaluation Framework

### Lesson 3.1: Evaluation Concepts
- **Types of Evaluations:**
  - Offline evaluations (datasets)
  - Online evaluations (production)
  - Comparative evaluations
  - Human feedback integration

### Lesson 3.2: Dataset Management
- **Creating Datasets:**
  ```python
  examples = [
      {
          "inputs": {"question": "What is the largest mammal?"},
          "outputs": {"answer": "The blue whale"},
          "metadata": {"source": "Wikipedia"},
      }
  ]
  
  client = Client()
  dataset = client.create_dataset(
      dataset_name="Elementary Animal Questions",
      description="Questions about animal phylogenetics."
  )
  client.create_examples(dataset_id=dataset.id, examples=examples)
  ```

- **Dataset Sources:**
  - Manual creation
  - CSV/JSONL import
  - Production traces
  - Annotation queues

### Lesson 3.3: Evaluator Types
- **Code Evaluators:**
  ```python
  def correct(outputs: dict, reference_outputs: dict) -> bool:
      return outputs["answer"] == reference_outputs["answer"]
  ```

- **LLM-as-a-Judge:**
  ```python
  def valid_reasoning(inputs: dict, outputs: dict) -> bool:
      instructions = """Determine if the reasoning is logically valid..."""
      response = oai_client.beta.chat.completions.parse(
          model="gpt-4o",
          messages=[{"role": "system", "content": instructions}],
          response_format=Response
      )
      return response.choices[0].message.parsed.reasoning_is_valid
  ```

### Lesson 3.4: Running Evaluations
- **Basic Evaluation:**
  ```python
  results = evaluate(
      target_function,
      data="dataset_name",
      evaluators=[correct, valid_reasoning]
  )
  ```

- **Advanced Configuration:**
  - Repetitions for statistical significance
  - Concurrency control
  - Rate limit handling
  - Async evaluation

---

## Section 4: Advanced Tracing Techniques

### Lesson 4.1: Metadata and Tags
- **Adding Context:**
  ```python
  @ls.traceable(
      tags=["my-tag"],
      metadata={"environment": "production", "user_id": "123"}
  )
  def my_function():
      rt = ls.get_current_run_tree()
      rt.metadata["dynamic_key"] = "value"
      rt.tags.extend(["dynamic-tag"])
  ```

### Lesson 4.2: Custom Run Types
- **Retriever Traces:**
  ```python
  @traceable(run_type="retriever")
  def retrieve_docs(query):
      contents = ["Document 1", "Document 2", "Document 3"]
      return [
          {
              "page_content": content,
              "type": "Document",
              "metadata": {"source": "database"}
          }
          for content in contents
      ]
  ```

### Lesson 4.3: Error Handling and Debugging
- **Trace Filtering:**
  - By tags and metadata
  - By error status
  - Time-based filtering
  - Performance metrics

### Lesson 4.4: Production Monitoring
- **Alerts and Rules:**
  - Error rate monitoring
  - Performance degradation
  - Custom metrics
  - Webhook notifications

---

## Section 5: Dataset Management

### Lesson 5.1: Dataset Creation Strategies
- **From Production Data:**
  ```python
  runs = client.list_runs(
      project_name="my_project",
      is_root=True,
      error=False
  )
  
  examples = [{"inputs": run.inputs, "outputs": run.outputs} for run in runs]
  ```

- **Synthetic Data Generation:**
  - AI-generated examples
  - Schema validation
  - Quality control

### Lesson 5.2: Dataset Versioning and Splits
- **Split Management:**
  - Training/validation/test splits
  - Categorical organization
  - Metadata-based filtering

### Lesson 5.3: Quality Assurance
- **Schema Validation:**
  - JSON schema enforcement
  - Data transformations
  - Custom validation rules

---

## Section 6: Prompt Engineering

### Lesson 6.1: Prompt Hub Integration
- **Version Control:**
  - Prompt versioning
  - A/B testing
  - Performance tracking

### Lesson 6.2: Prompt Optimization
- **Iterative Improvement:**
  - Performance metrics analysis
  - Systematic prompt refinement
  - Evaluation-driven optimization

---

## Section 7: Deployment and Production

### Lesson 7.1: Production Setup
- **Environment Configuration:**
  - API key management
  - Project organization
  - Security considerations

### Lesson 7.2: Monitoring and Alerting
- **Dashboard Setup:**
  - Key performance indicators
  - Error tracking
  - Usage analytics

### Lesson 7.3: Scaling Considerations
- **Performance Optimization:**
  - Sampling strategies
  - Batch processing
  - Resource management

---

## Section 8: Agent Development

### Lesson 8.1: Agent Tracing
- **Complex Workflows:**
  - Multi-step processes
  - Tool usage tracking
  - Decision point analysis

### Lesson 8.2: Agent Evaluation
- **Specialized Metrics:**
  - Task completion rates
  - Tool usage effectiveness
  - Reasoning quality assessment

### Lesson 8.3: Agent Debugging
- **Troubleshooting Techniques:**
  - Step-by-step analysis
  - Performance bottleneck identification
  - Error pattern recognition

---

## Practical Exercises

### Exercise 1: Basic Tracing Setup
1. Set up LangSmith account and API keys
2. Create a simple traced function
3. View traces in the UI
4. Add metadata and tags

### Exercise 2: Dataset Creation and Evaluation
1. Create a dataset from examples
2. Build a simple evaluator
3. Run an evaluation
4. Analyze results

### Exercise 3: RAG Application Evaluation
1. Build a RAG application with retrieval tracing
2. Create evaluation metrics for:
   - Answer correctness
   - Retrieval relevance
   - Response groundedness
3. Compare different approaches

### Exercise 4: Production Monitoring
1. Set up production tracing
2. Configure alerts
3. Create performance dashboards
4. Implement error tracking

### Exercise 5: Agent Development and Testing
1. Build a multi-step agent
2. Trace agent decision-making
3. Evaluate agent performance
4. Optimize based on traces

---

## Best Practices Summary

### Development Best Practices
1. **Comprehensive Tracing:** Trace all LLM calls and key application components
2. **Meaningful Metadata:** Add context that helps with debugging and analysis
3. **Structured Evaluation:** Use systematic approaches to measure performance
4. **Version Control:** Track changes to prompts, models, and application logic

### Production Best Practices
1. **Monitoring:** Set up comprehensive monitoring and alerting
2. **Sampling:** Use appropriate sampling to balance observability and cost
3. **Privacy:** Ensure sensitive data is properly handled
4. **Performance:** Optimize tracing overhead for production workloads

### Evaluation Best Practices
1. **Representative Data:** Use datasets that reflect real-world usage
2. **Multiple Metrics:** Evaluate different aspects of performance
3. **Continuous Improvement:** Regularly update evaluations as applications evolve
4. **Human Feedback:** Incorporate human judgment when possible

---

## Resources and References

### Documentation Links
- [LangSmith Official Documentation](https://docs.langchain.com/langsmith)
- [API Reference](https://api.smith.langchain.com/redoc)
- [Python SDK](https://python.langchain.com/docs/langsmith/)
- [TypeScript SDK](https://js.langchain.com/docs/langsmith/)

### Community Resources
- [LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook)
- [Community Forum](https://github.com/langchain-ai/langsmith/discussions)
- [Example Applications](https://github.com/langchain-ai/langsmith-examples)

### Integration Guides
- OpenAI Integration
- Anthropic Integration
- Custom Framework Integration
- Production Deployment Guides

---

## Course Completion Checklist

- [ ] Successfully set up LangSmith account and environment
- [ ] Implemented basic tracing in an application
- [ ] Created and managed datasets
- [ ] Built and ran evaluations
- [ ] Set up production monitoring
- [ ] Developed agent tracing capabilities
- [ ] Completed all practical exercises
- [ ] Applied best practices in a real project

This comprehensive course provides the foundation for effectively using LangSmith to develop, evaluate, and monitor LLM applications in production environments.