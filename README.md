# LangTree

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red.svg)](https://github.com/ac8ai/langtree)
[![CI](https://github.com/ac8ai/langtree/workflows/CI/badge.svg)](https://github.com/ac8ai/langtree/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)

> ⚠️ **Pre-Alpha Software** - Core functionality not yet implemented. DPCL parsing works, but LangChain execution is not functional.

**Dynamic Prompt Connecting Language (DPCL)** - A framework for orchestrating hierarchical prompt structures into executable LangChain pipelines through tag-based data forwarding.

## What is LangTree?

LangTree enables **hierarchical prompt orchestration** where prompts are organized as trees of data they describe how to generate. Data flows between tree structures and persists across processing stages. Define your data structures as PromptTreeNode classes, then use DPCL (Dynamic Prompt Connecting Language) to control data flow:

### Vision: Advanced Tree Orchestration
```python
class TaskCustomerAnalysis(PromptTreeNode):
    """
    You are analyzing customer data to improve business operations.
    Focus on extracting actionable insights from customer interactions.
    """

class TaskOrderProcessor(TaskCustomerAnalysis):
    """
    ! llm("opus4.1")

    ## Order Processing Phase

    Now process customer orders to extract feedback patterns.
    Look for common themes and sentiment indicators in order feedback.

    {PROMPT_SUBTREE}
    """

    class Order(PromptTreeNode):
        """
        For each order, analyze the customer feedback thoroughly.
        Extract sentiment, key issues, and satisfaction indicators.
        """
        feedback: str = Field(description="Raw customer feedback from order")

    orders: list[Order] = Field(description="Customer orders to process")

class TaskInsightGenerator(TaskCustomerAnalysis):
    """
    ! repeat(3)

    ## Insight Generation Phase

    Transform processed feedback into business recommendations.
    Create specific, measurable improvement suggestions.

    {PROMPT_SUBTREE}

    # Generated so far

    {COLLECTED_CONTEXT}
    """

    class Category(PromptTreeNode):
        """
        Group related insights into logical business categories.
        Each category should address a specific operational area.
        """

        class Insight(PromptTreeNode):
            """
            Generate one specific, actionable business recommendation.
            Include implementation steps and expected outcomes.
            """
            recommendation: str = Field(description="""
                ! repeat(3)

                Detailed business recommendation
            """)

        insights: list[Insight] = Field(description="""
            ! @each[insights]->task.order_processor@{{orders.feedback=insights.recommendation}}* # Generate insights
            
            Insights for this category
        """)

    categories: list[Category] = Field(description="""
        ! llm('gpt5')

        Business insight categories
    """)
    final_report: str = Field(description="""
        ! @all->task.order_processor@{{prompt.final_report=final_report}}*

        Executive summary of all insights
    """)
```

**Advanced Capabilities:**
- **Deep Navigation**: Multi-level path traversal with validation
- **Cross-Tree References**: Data flow between different tree structures
- **Mixed Hierarchies**: Iterable and non-iterable nodes in same structure
- **Semantic Validation**: Comprehensive validation ensures correctness

The framework validates tree structures, parses DPCL commands, and generates deterministic execution pipelines.

## Current Status

**✅ Working:**
- Core DPCL command parsing and syntax validation
- Prompt tree structure management and registration
- Comprehensive semantic validation framework
- Template variable system with conflict resolution
- Field context scoping and inheritance validation

**🚧 In Development:**
- LangChain integration and chain building
- Runtime execution pipeline assembly
- End-to-end prompt execution

## Vision

### Tomorrow
Enable parallel execution and chaining of structured generation workflows, allowing complex multi-step LLM processes to run efficiently with automatic dependency resolution and deterministic execution order.

### Advanced Tree Orchestration
```python
class TaskCustomerAnalysis(PromptTreeNode):
    """
    You are analyzing customer data to improve business operations.
    Focus on extracting actionable insights from customer interactions.
    """

class TaskOrderProcessor(TaskCustomerAnalysis):
    """
    ! llm("opus4.1")

    ## Order Processing Phase

    Now process customer orders to extract feedback patterns.
    Look for common themes and sentiment indicators in order feedback.

    {PROMPT_SUBTREE}
    """

    class Order(PromptTreeNode):
        """
        For each order, analyze the customer feedback thoroughly.
        Extract sentiment, key issues, and satisfaction indicators.
        """
        feedback: str = Field(description="Raw customer feedback from order")

    orders: list[Order] = Field(description="Customer orders to process")

class TaskInsightGenerator(TaskCustomerAnalysis):
    """
    ! repeat(3)

    ## Insight Generation Phase

    Transform processed feedback into business recommendations.
    Create specific, measurable improvement suggestions.

    {PROMPT_SUBTREE}

    # Generated so far

    {COLLECTED_CONTEXT}
    """

    class Category(PromptTreeNode):
        """
        Group related insights into logical business categories.
        Each category should address a specific operational area.
        """

        class Insight(PromptTreeNode):
            """
            Generate one specific, actionable business recommendation.
            Include implementation steps and expected outcomes.
            """
            recommendation: str = Field(description="""
                ! repeat(3)

                Detailed business recommendation
            """)

        insights: list[Insight] = Field(description="""
            ! @each[insights]->task.order_processor@{{orders.feedback=insights.recommendation}}* # Generate insights

            Insights for this category
        """)

    categories: list[Category] = Field(description="""
        ! llm('gpt5')

        Business insight categories
    """)
    final_report: str = Field(description="""
        ! @all->task.order_processor@{{prompt.final_report=final_report}}*

        Executive summary of all insights
    """)
```

### Future
Evolve into a comprehensive agent orchestration platform where JSON definitions dynamically generate LangTree nodes, enabling agents to coordinate and collaborate through the same powerful validation and execution framework that drives structured generation today.

## Installation

```bash
# Install directly from GitHub
uv add git+https://github.com/ac8ai/langtree

# For development
uv add --group dev git+https://github.com/ac8ai/langtree
uv run pytest  # Run test suite
```

## Documentation

**📖 Full Documentation:** [https://ac8.ai/langtree](https://ac8.ai/langtree)

## Contributing

This project is in active development. Contributors welcome! See [documentation](https://ac8.ai/langtree) for complete guides and language specification.

---

**Not ready for production use** - Follow development progress and contribute at [GitHub](https://github.com/ac8ai/langtree)