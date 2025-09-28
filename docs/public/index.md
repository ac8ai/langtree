# LangTree

!!! warning "Pre-Alpha Software"
    Core functionality not yet implemented. LangTree DSL parsing works, but LangChain execution is not functional.

**LangTree DSL** - A framework for orchestrating hierarchical prompt structures into executable LangChain pipelines through tag-based data forwarding.

LangTree enables **hierarchical prompt orchestration** where prompts are organized as trees of data they describe how to generate. Data flows between tree structures and persists across processing stages.

```python
class TaskCustomerAnalysis(TreeNode):
    """
    ! llm("opus4.1")

    Analyze customer data to improve business operations.
    {PROMPT_SUBTREE}
    """

    class Order(TreeNode):
        """Extract sentiment and key issues from feedback."""
        feedback: str = Field(description="Raw customer feedback")

    orders: list[Order] = Field(description="""
        ! @each[orders]->task.analyzer@{{value.insights=orders.feedback}}*

        Customer orders to process
    """)
```

## Key Features

- **Deep Navigation**: Multi-level path traversal with validation
- **Cross-Tree References**: Data flow between different tree structures
- **Semantic Validation**: Comprehensive test coverage ensures correctness
- **Template Variables**: {PROMPT_SUBTREE}, {COLLECTED_CONTEXT}

## Vision

**Tomorrow**: Enable parallel execution and chaining of structured generation workflows, allowing complex multi-step LLM processes to run efficiently with automatic dependency resolution and deterministic execution order.

**Future**: Evolve into a comprehensive agent orchestration platform where JSON definitions dynamically generate LangTree nodes, enabling agents to coordinate and collaborate through the same powerful validation and execution framework that drives structured generation today.

## Quick Links

- [Getting Started](getting-started.md) - Installation and basic usage
- [LangTree DSL Reference](LangTree DSL-reference.md) - Complete command syntax

```bash
# Install from GitHub
uv add git+https://github.com/ac8ai/langtree
```

## Repository

**GitHub:** [https://github.com/ac8ai/langtree](https://github.com/ac8ai/langtree)