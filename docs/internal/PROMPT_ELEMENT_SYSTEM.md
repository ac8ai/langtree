# Prompt Element System Architecture

**Date:** 2025-10-07
**Status:** Implemented
**Related Files:** `src/langtree/templates/prompt_*.py`, `src/langtree/templates/element_resolution.py`

---

## Overview

The Prompt Element System is the core architecture for prompt generation in LangTree DSL. It represents prompts as structured lists of typed elements (titles, text, templates) that can be parsed from markdown, manipulated programmatically, and serialized back to markdown with proper heading level management.

This architecture replaced the previous string-based approach with a structured, type-safe system that enables:
- ✅ Automatic heading level alignment
- ✅ Template variable resolution with context awareness
- ✅ Hierarchical prompt assembly from leaf to root
- ✅ Proper nesting and hierarchy preservation

---

## Core Concepts

### PromptElement

Base class for all prompt components. Every piece of a prompt is represented as a `PromptElement`.

**Types of PromptElements:**
1. **PromptTitle** - Markdown headings (`#`, `##`, `###`, etc.)
2. **PromptText** - Plain text content
3. **PromptTemplate** - Template variables (`{PROMPT_SUBTREE}`, `{COLLECTED_CONTEXT}`)

**Common Attributes:**
```python
@dataclass
class PromptElement(ABC):
    level: int = 1           # Heading level (1 for #, 2 for ##, etc.)
    optional: bool = False   # Whether this element can be excluded
    line_number: int | None = None  # Source line for error reporting
```

### Element-Based vs String-Based

**Old Approach (String-Based):**
```python
prompt = "# Title\n\nSome text\n\n## Section\n\nMore text"
# ❌ Hard to manipulate heading levels
# ❌ Difficult to identify template variables
# ❌ No structured representation
```

**New Approach (Element-Based):**
```python
elements = [
    PromptTitle(content="Title", level=1),
    PromptText(content="Some text"),
    PromptTitle(content="Section", level=2),
    PromptText(content="More text")
]
# ✅ Easy to adjust levels programmatically
# ✅ Clear structure and types
# ✅ Can serialize to markdown or other formats
```

---

## System Architecture

### Module Structure

```
src/langtree/templates/
├── prompt_structure.py      # Data classes (PromptElement, PromptTitle, etc.)
├── prompt_parser.py         # Markdown → Elements parsing
├── prompt_operations.py     # Element list manipulation (filter, find, etc.)
├── prompt_assembly.py       # Elements → Markdown serialization
└── element_resolution.py    # Template variable resolution & level alignment
```

### Data Flow

```
1. Parse Phase (Startup)
   Docstring (markdown) → parse_docstring_to_elements() → list[PromptElement]
   ↓
   Cached in StructureTreeNode.prompt_elements

2. Resolution Phase (Runtime)
   Parent elements + Child elements → resolve_prompt_subtree_elements()
   ↓
   Adjusted elements with proper levels → render_elements_to_string()
   ↓
   Final prompt (markdown string)
```

---

## Key Components

### 1. Prompt Structure (`prompt_structure.py`)

Defines the data classes for representing prompt elements.

**PromptTitle:**
```python
@dataclass
class PromptTitle(PromptElement):
    content: str  # Heading text (without # markers)
    level: int    # 1 for #, 2 for ##, etc.
```

**PromptText:**
```python
@dataclass
class PromptText(PromptElement):
    content: str  # Plain text content
    level: int    # Inherited from context (for indentation if needed)
```

**PromptTemplate:**
```python
@dataclass
class PromptTemplate(PromptElement):
    variable_name: str             # "PROMPT_SUBTREE" or "COLLECTED_CONTEXT"
    resolved_content: str | None   # Resolved content (set during resolution)
    level: int                     # Level for embedded content
```

### 2. Prompt Parser (`prompt_parser.py`)

Parses markdown strings into structured element lists.

**parse_markdown_to_elements():**
- Input: Raw markdown string
- Output: `list[PromptElement]`
- Detects: Headings, text blocks, template variables
- Handles: Leading whitespace in headings, empty lines, mixed content

**Key Features:**
- Regex-based heading detection: `r'^\s*(#+)\s+(.+)$'`
- Template variable detection: `{PROMPT_SUBTREE}`, `{COLLECTED_CONTEXT}`
- Preserves blank lines and text structure
- Line number tracking for error reporting

### 3. Element Resolution (`element_resolution.py`)

Core module for template variable resolution and heading level alignment.

**parse_docstring_to_elements():**
```python
def parse_docstring_to_elements(
    content: str,
    base_level: int = 1
) -> list[PromptElement]:
    """
    Parse docstring and track heading levels for template variables.

    Key Feature: Template variables inherit level from preceding heading.
    """
```

**Process:**
1. Split docstring into lines
2. Track `current_content_level` as headings are encountered
3. When template variable found, assign `level = current_content_level`
4. When heading found, update `current_content_level = heading_level + 1`

**adjust_element_levels():**
```python
def adjust_element_levels(
    elements: list[PromptElement],
    base_level: int
) -> list[PromptElement]:
    """
    Shift all heading levels to fit under a base level.

    Key Feature: Preserves relative hierarchy within elements.
    """
```

**Process:**
1. Find minimum level among all elements
2. Calculate shift: `level_shift = base_level - min_level`
3. Apply shift to ALL elements uniformly
4. Return adjusted elements

**resolve_prompt_subtree_elements():**
```python
def resolve_prompt_subtree_elements(
    node: StructureTreeNode,
    child_resolutions: dict[str, list[PromptElement]]
) -> list[PromptElement]:
    """
    Resolve template variables in node's elements using child content.

    Key Feature: Embeds child content at correct levels based on template variable position.
    """
```

**Process:**
1. Iterate through node's cached elements
2. When PromptTemplate encountered:
   - Get child elements from child_resolutions
   - Adjust child levels using `adjust_element_levels(child_elements, template.level)`
   - Replace template with adjusted child elements
3. Return fully resolved elements

### 4. Prompt Assembly (`prompt_assembly.py`)

Converts element lists back to markdown strings.

**render_elements_to_string():**
```python
def render_elements_to_string(
    elements: list[PromptElement]
) -> str:
    """
    Convert element list to markdown string.

    Features:
    - PromptTitle → "## Title" (# count = level)
    - PromptText → raw content
    - PromptTemplate → resolved_content (if set)
    - Proper spacing between elements
    """
```

### 5. Prompt Operations (`prompt_operations.py`)

Utility functions for manipulating element lists.

**Functions:**
- `filter_elements_by_type()` - Extract specific element types
- `find_template_variables()` - Locate all template variables
- `get_heading_at_level()` - Find specific heading levels
- `count_elements()` - Count elements by type

---

## Heading Level Alignment Algorithm

### Problem Statement

When embedding child content into parent prompts, heading levels must be adjusted to maintain proper hierarchy:

```python
# Parent docstring with template variable
"""
## Research Process

{PROMPT_SUBTREE}
"""

# Child docstring
"""
## Analysis Method

### Data Sources
"""

# Desired result: Child should be subordinate
"""
## Research Process

### Literature Review     ← child field title (level 3)
#### Analysis Method       ← child heading (level 4, shifted from 2)
##### Data Sources         ← child subheading (level 5, shifted from 3)
"""
```

### Solution: Three-Step Process

**Step 1: Template Variable Level Detection**

During docstring parsing (`parse_docstring_to_elements`):
```python
current_content_level = base_level
for line in docstring_lines:
    if is_heading(line):
        heading_level = count_hashes(line)
        current_content_level = heading_level + 1

    if is_template_variable(line):
        template.level = current_content_level  # ← Key: inherit from context
```

**Step 2: Child Content Adjustment**

When resolving template variables (`resolve_prompt_subtree_elements`):
```python
# Template variable has level 3 (from "## Research Process" at level 2)
template = PromptTemplate(variable_name="PROMPT_SUBTREE", level=3)

# Child elements have levels [1, 2, 3] (from original docstring)
child_elements = [
    PromptTitle("Analysis Method", level=2),
    PromptTitle("Data Sources", level=3)
]

# Adjust to fit under template level
adjusted = adjust_element_levels(child_elements, base_level=3)
# Result: [level 3, level 4] - shifted by 1 to preserve hierarchy
```

**Step 3: Hierarchy Preservation**

In `adjust_element_levels`:
```python
def adjust_element_levels(elements, base_level):
    # Find the minimum level (top of hierarchy)
    min_level = min(elem.level for elem in elements)  # e.g., 2

    # Calculate shift needed
    level_shift = base_level - min_level  # e.g., 3 - 2 = 1

    # Apply uniform shift to ALL elements
    return [
        replace(elem, level=elem.level + level_shift)
        for elem in elements
    ]
    # Original: [2, 3] → Shifted: [3, 4]
    # Relative difference (1) preserved!
```

**Key Insight:** By calculating the shift amount and applying it uniformly, relative heading differences within content are preserved, maintaining structural integrity.

---

## Full Chain Prompt Generation

### get_prompt() Method

The `get_prompt()` method in `StructureTreeNode` orchestrates the full chain prompt generation from leaf to root.

**Algorithm:**
```python
def get_prompt(self) -> str:
    # 1. Build parent chain (leaf → root)
    chain = []
    current = self
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()  # Now root → leaf

    # 2. Resolve bottom-up (leaf to root)
    child_resolutions = {}
    for node in reversed(chain):  # leaf first
        # Resolve this node's elements using child resolutions
        resolved = resolve_node_prompt_elements(
            node,
            child_resolutions,
            previous_values
        )

        # Store for parent to use
        child_resolutions[node_field_name] = resolved

    # 3. Render final elements to string
    final_elements = child_resolutions[self.name]
    return render_elements_to_string(final_elements)
```

**Key Features:**
- ✅ Walks entire parent chain from leaf to root
- ✅ Resolves from bottom up (child content embedded into parent)
- ✅ Each level's elements adjusted to fit parent's hierarchy
- ✅ Template variables resolved with proper context

---

## Testing Strategy

### Test Categories

**Unit Tests (Module-Specific):**
- `test_prompt_structure.py` - Data class validation
- `test_prompt_parser.py` - Markdown parsing correctness
- `test_prompt_assembly.py` - Markdown serialization
- `test_element_resolution.py` - Level adjustment algorithms

**Feature Tests (Cross-Module):**
- `test_heading_level_alignment.py` - Full heading alignment scenarios (9 tests)
  - Template variable level detection
  - Child content embedding
  - Nested TreeNode hierarchy
  - Deep nesting (3+ levels)
  - Edge cases

**Integration Tests:**
- `test_get_prompt_chain.py` - Full chain generation (leaf to root)
- `test_collected_context_bug.py` - Bug regression tests
- `test_treenode_collected_context.py` - TreeNode in COLLECTED_CONTEXT

### Test Coverage

- **214 template tests** total
- **9 heading alignment tests** specifically
- **100% passing** rate
- **Comprehensive coverage** of:
  - Level detection
  - Level adjustment
  - Hierarchy preservation
  - Edge cases (whitespace, deep nesting, varying depths)

---

## Implementation Patterns

### Pattern 1: Immutable Element Manipulation

**Anti-pattern:**
```python
# ❌ Mutating elements directly
for element in elements:
    element.level += 2  # Dangerous!
```

**Correct approach:**
```python
# ✅ Create new elements with updated values
from dataclasses import replace

adjusted = [
    replace(elem, level=elem.level + 2)
    for elem in elements
]
```

### Pattern 2: Context Tracking During Parsing

```python
# Track state as you parse
current_content_level = base_level

for line in lines:
    if is_heading(line):
        # Update context
        current_content_level = heading_level + 1

    if is_template(line):
        # Use tracked context
        template.level = current_content_level
```

### Pattern 3: Bottom-Up Resolution

```python
# Always resolve from leaf to root
child_resolutions = {}

for node in reversed(parent_chain):  # Start from leaf
    # Resolve using child content
    resolved = resolve_node(node, child_resolutions)

    # Store for parent
    child_resolutions[node.name] = resolved
```

---

## Performance Considerations

### Caching Strategy

**Elements are parsed once and cached:**
```python
class StructureTreeNode:
    prompt_elements: list[PromptElement]  # Cached during structure building
```

**Benefits:**
- ✅ No repeated parsing during runtime
- ✅ Elements available for inspection/debugging
- ✅ Fast prompt generation (just adjust + render)

### Lazy Resolution

**Template variables resolved on-demand:**
- Parse time: Create `PromptTemplate` placeholders
- Resolution time: Replace with actual content
- Only resolve what's needed for current prompt

---

## Error Handling

### Parse-Time Errors

**Heading detection issues:**
```python
# Whitespace handling
r'^\s*(#+)\s+(.+)$'  # ✅ Allows leading whitespace
r'^(#+)\s+(.+)$'     # ❌ Fails on indented headings
```

**Template variable validation:**
```python
if variable_name not in ["PROMPT_SUBTREE", "COLLECTED_CONTEXT"]:
    raise ValueError(f"Unknown template variable: {variable_name}")
```

### Resolution-Time Errors

**Missing child content:**
```python
if field_name not in child_resolutions:
    # Log warning or use empty list
    child_elements = []
```

**Level calculation errors:**
```python
if not elements:
    return []  # Early return for empty lists

min_level = min(..., default=1)  # Safe default
```

---

## Future Enhancements

### Planned Features

1. **Conditional Elements:**
   ```python
   PromptElement(optional=True, condition="if field is set")
   ```

2. **Element Metadata:**
   ```python
   PromptElement(metadata={"source": "parent", "depth": 3})
   ```

3. **Custom Element Types:**
   ```python
   class PromptCode(PromptElement):
       language: str
       code: str
   ```

4. **Element Transformers:**
   ```python
   def transform_element(elem: PromptElement) -> PromptElement:
       # Custom transformation logic
   ```

---

## Migration Guide

### From String-Based to Element-Based

**Old code:**
```python
# String manipulation
prompt = node.docstring.replace("{PROMPT_SUBTREE}", child_content)
```

**New code:**
```python
# Element-based resolution
elements = node.prompt_elements
resolved = resolve_prompt_subtree_elements(node, child_resolutions)
prompt = render_elements_to_string(resolved)
```

### Accessing Elements

**During structure building:**
```python
# Elements are automatically parsed and cached
node.prompt_elements  # list[PromptElement]
```

**During prompt generation:**
```python
# Use get_prompt() - handles everything
prompt = node.get_prompt(previous_values)
```

---

## Debugging Tips

### Inspecting Elements

```python
# Print element structure
for i, elem in enumerate(node.prompt_elements):
    print(f"{i}: {type(elem).__name__} level={elem.level}")
    if isinstance(elem, PromptTitle):
        print(f"   Title: {elem.content}")
    elif isinstance(elem, PromptTemplate):
        print(f"   Variable: {elem.variable_name}")
```

### Tracing Level Adjustments

```python
# Before adjustment
print("Before:", [elem.level for elem in elements])

# After adjustment
adjusted = adjust_element_levels(elements, base_level=3)
print("After:", [elem.level for elem in adjusted])
print("Shift:", base_level - min(e.level for e in elements))
```

### Visualizing Prompt Structure

```python
def visualize_elements(elements: list[PromptElement]) -> str:
    lines = []
    for elem in elements:
        indent = "  " * (elem.level - 1)
        if isinstance(elem, PromptTitle):
            lines.append(f"{indent}[{elem.level}] {elem.content}")
        elif isinstance(elem, PromptTemplate):
            lines.append(f"{indent}[{elem.level}] <{elem.variable_name}>")
    return "\n".join(lines)
```

---

## Related Documentation

- **Public Documentation:** [LANGUAGE_SPECIFICATION.md](../public/LANGUAGE_SPECIFICATION.md) - Template Variables and Heading Level Alignment section
- **Test Documentation:** [test_heading_level_alignment.py](/workspaces/langtree/tests/templates/test_heading_level_alignment.py) - Comprehensive test examples
- **Code Review:** [CODE_REVIEW_HEADING_ALIGNMENT.md](/workspaces/langtree/CODE_REVIEW_HEADING_ALIGNMENT.md) - Implementation review

---

## Summary

The Prompt Element System provides a robust, type-safe architecture for prompt generation with automatic heading level alignment. Key achievements:

- ✅ **Structured representation** of prompts as typed elements
- ✅ **Automatic heading level adjustment** maintaining proper hierarchy
- ✅ **Template variable resolution** with context awareness
- ✅ **Full chain generation** from leaf to root
- ✅ **Comprehensive test coverage** with 214 passing tests
- ✅ **Clean separation of concerns** across focused modules
- ✅ **Immutable, functional approach** for safety and clarity

This system forms the foundation for all prompt generation in LangTree DSL, ensuring consistent, readable, and properly structured prompts throughout the framework.
