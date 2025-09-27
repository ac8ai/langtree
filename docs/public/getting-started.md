# Getting Started

!!! warning "Pre-Alpha Status"
    LangTree is in pre-alpha development. The DPCL parser and validation work, but LangChain execution is not yet implemented. This guide shows the intended API once execution is functional.

## Installation

```bash
# Install from GitHub
uv add git+https://github.com/ac8ai/langtree

# Or with pip
pip install git+https://github.com/ac8ai/langtree
```

For development:

```bash
git clone https://github.com/ac8ai/langtree
cd langtree
uv sync --group dev
uv run pytest  # Run test suite
```

## Basic Concepts

LangTree organizes prompts as **trees of data** they describe how to generate. Key concepts:

- **PromptTreeNode**: Base class for all tree structures
- **DPCL Commands**: Control data flow between trees (@each, @all, !llm, !repeat)
- **Template Variables**: Assembly-time resolution ({PROMPT_SUBTREE}, {COLLECTED_CONTEXT})
- **Field Context**: Commands execute within specific field scopes

## Your First Tree

```python
from langtree.prompt import PromptTreeNode
from pydantic import Field

class TaskDocumentAnalyzer(PromptTreeNode):
    """
    Analyze documents to extract key insights and summaries.
    Focus on identifying main themes and actionable information.
    """

    class Document(PromptTreeNode):
        """
        Process a single document thoroughly.
        Extract key points, themes, and relevant details.
        """
        content: str = Field(description="Document content to analyze")

    documents: list[Document] = Field(description="Documents to process")
    summary: str = Field(description="Overall analysis summary")
```

## Adding DPCL Commands

Commands control execution flow and data forwarding:

```python
class TaskAdvancedAnalyzer(PromptTreeNode):
    """
    ! llm("gpt-4")
    ! repeat(2)

    Advanced document analysis with multi-pass processing.
    Generate comprehensive insights from document collections.

    {PROMPT_SUBTREE}
    """

    documents: list[Document] = Field(description="""
        ! @each[documents]->task.processor@{{value.content=documents.content}}

        Documents to analyze in detail
    """)

    final_report: str = Field(description="""
        ! @all->task.summarizer@{{prompt.documents=documents}}*

        Comprehensive analysis report
    """)
```

## Template Variables

Template variables resolve at assembly time:

- `{PROMPT_SUBTREE}`: Inserts child node prompts
- `{COLLECTED_CONTEXT}`: Aggregated context from previous processing

```python
class TaskWithTemplates(PromptTreeNode):
    """
    Process data with context from previous stages.

    # Previous Results
    {COLLECTED_CONTEXT}

    # Current Processing
    {PROMPT_SUBTREE}
    """
```

## Cross-Tree Data Flow

Reference data between different tree structures:

```python
class TaskDataProcessor(PromptTreeNode):
    """Process raw data into structured format."""
    raw_data: list[str] = Field(description="Raw input data")

class TaskInsightGenerator(PromptTreeNode):
    """Generate insights from processed data."""

    insights: list[str] = Field(description="""
        ! @each[insights]->task.data_processor@{{value.insights=insights}}

        Generated insights from processed data
    """)
```

## Running Trees (When Implemented)

```python
# Future API - not yet functional
from langtree import RunStructure

structure = RunStructure()
structure.add(TaskDocumentAnalyzer)

# This will work once LangChain integration is complete
# result = structure.execute(input_data)
```

## Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `!llm(model)` | Specify LLM model | `! llm("gpt-4")` |
| `!repeat(n)` | Repeat processing | `! repeat(3)` |
| `@each[path]` | Iterate over items | `@each[documents]->task.processor` |
| `@all` | Aggregate all data | `@all->task.summarizer` |

## Validation

LangTree validates your tree structures:

```python
structure = RunStructure()
try:
    structure.add(TaskDocumentAnalyzer)
    print("✅ Tree structure is valid")
except ValidationError as e:
    print(f"❌ Validation error: {e}")
```

## Next Steps

- [DPCL Reference](dpcl-reference.md) - Complete command syntax
- [Examples](examples.md) - Real-world patterns
- [API Reference](api.md) - Python API details

## Development Status

Currently working:
- ✅ DPCL parsing and validation
- ✅ Tree structure management
- ✅ Semantic validation

In development:
- 🚧 LangChain integration
- 🚧 Execution pipeline
- 🚧 Runtime data flow