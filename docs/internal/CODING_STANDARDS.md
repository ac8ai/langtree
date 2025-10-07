# LangTree DSL Coding Standards

## Overview

This document defines the coding standards, conventions, and architectural principles for the LangTree DSL framework. These guidelines ensure consistency, maintainability, and scalability across the codebase.

## Development Practices

### Clarity and Communication Principles

#### Ask Don't Assume Command
**When encountering unclear, ambiguous, or equivocal statements in requirements, specifications, or discussions:**

- **ALWAYS ASK for clarification rather than making assumptions**
- **Document your questions**, assumptions and the clarification received
- **Propose specific interpretations** rather than open-ended questions
- **Example**: Instead of assuming "improve performance" means optimize database queries, ask: "By 'improve performance', do you mean: (a) reduce API response time, (b) optimize database queries, (c) reduce memory usage, or (d) something else?"

This principle applies to:
- Requirements analysis and specification review
- Code review discussions where intent is unclear
- API design decisions with multiple valid interpretations
- LLM interactions where prompts or context could be interpreted multiple ways

**LLM-Specific Guidelines:**
- When generating code, explicitly state assumptions made
- When requirements are ambiguous, provide multiple implementation options
- Always clarify scope and constraints before implementation
- Document decisions made in the absence of clear guidance

## Module Organization Principles

### Single Responsibility Principle
- **One concept per file**: Each module should handle exactly one major concept or responsibility
- **Logical cohesion**: Related functionality should be grouped, but distinct concepts separated
- **Clear boundaries**: Interface definitions, implementations, and utilities have distinct roles

### When to Create New Files vs. Extend Existing
**Create new file when**:
- Introducing a fundamentally different concept (e.g., `registry.py` vs `resolution.py`)
- File would exceed ~500 lines with new functionality
- New functionality requires different import dependencies
- Concept can be understood and tested independently

**Extend existing file when**:
- Adding closely related functionality to existing concept
- New functions/classes are helper utilities for existing code
- Tight coupling makes separation counterproductive

### Package Structure Principles
- **Functional grouping**: Group by what the code does, not how it's implemented
- **Dependency flow**: Lower-level modules should not import from higher-level ones
- **Interface clarity**: Public APIs clearly separated from internal implementation details

## Documentation Standards

### Two-Level Documentation Architecture

#### Level 1: High-Level Design Documentation
**Purpose**: Strategic understanding without implementation details
**Audience**: Architects, new team members, stakeholders
**Content**: 
- Architectural decisions and rationale
- System interactions and data flow
- Design principles and constraints
- **No code snippets or implementation details**

**Examples**: `TAG_BASED_ARCHITECTURE.md`, high-level sections of `COMPREHENSIVE_GUIDE.md`

#### Level 2: Implementation Documentation  
**Purpose**: Practical development guidance with concrete examples
**Audience**: Active developers working on the code
**Content**:
- Code patterns and examples
- API usage and integration details
- Step-by-step implementation guides
- **Heavy references to existing code, minimal copied snippets**

### Docstring Requirements

#### Format Standard
```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Single-line purpose statement.
    
    Detailed explanation of what the function does, its role in the system,
    and any important behavioral notes. Focus on the 'why' and 'what',
    not the 'how' (code should be self-documenting for 'how').
    
    Params:
        param1: Description focusing on purpose and expected format
        param2: Description with any constraints or special behaviors
    
    Returns:
        Description of return value structure and meaning
        
    Raises:
        ExceptionType: When and why this exception occurs
    """
```

#### Required Elements
- **Purpose statement**: One clear sentence describing function's role
- **Context explanation**: How this fits into the larger system
- **All parameters**: Use `Params:` (not `Args:`), describe purpose not just type
- **Return value**: Structure and meaning, not just type
- **Exceptions**: When applicable, describe conditions that trigger them

### Comment Policy

#### When NOT to Comment
- **Obvious operations**: `x = y + 1  # Add 1 to y`
- **Code structure**: `# End of function` or `# Return statement`
- **Implementation details**: Code should be self-documenting through naming

#### When TO Comment
- **Complex algorithms**: Multi-step processes that aren't immediately obvious nor are self-documented by the code
- **Business logic**: Domain-specific rules or requirements
- **TODOs**: Unimplemented features linked to tests
- **Non-obvious design decisions**: Why something is done a particular way

#### TODO Standards
```python
# TODO: Specific action required
# See: tests/module/test_file.py::test_specific_behavior
```
- **Actionable**: Clear what needs to be implemented
- **Test-linked**: Reference to skipped test defining expected behavior
- **Temporary**: Should be removed when feature is implemented

## Code Style Standards

### Import Organization

#### Four-Group Structure
**All imports must be organized into exactly four groups at the top of the file**:

1. **External direct imports**: `import module_name`
2. **External from imports**: `from module_name import item`
3. **Internal direct imports**: `import project.module_name`
4. **Internal from imports**: `from project.module_name import item`

#### Absolute Imports Only
**All imports must use absolute paths**:
- **Never use relative imports**: No `from .module import item` or `from ..parent import item`
- **Always specify full module path**: `from langtree.structure import RunStructure`
- **Consistent with project structure**: Import paths must match actual package hierarchy
- **Exception**: TYPE_CHECKING blocks may use forward references as strings

#### Alphabetical Sorting
**Within each group, sort alphabetically**:
- By module name for direct imports
- By source module name for from imports
- Case-insensitive sorting

#### Example Structure
```python
# Group 1: External direct imports (alphabetical)
import os
import sys
import typing

# Group 2: External from imports (alphabetical by source module)
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Group 3: Internal direct imports (alphabetical)
import langtree.chains
import langtree.models

# Group 4: Internal from imports (alphabetical by source module)
from langtree.parsing.parser import parse_command
from langtree.structure.registry import VariableRegistry
from langtree.structure import RunStructure
from langtree import TreeNode
```
Do not replicate line comments on top of each groups. They are just a part of description.

#### Blank Line Separation
- **One blank line** between each of the four groups
- **Two blank lines** after all imports before the first code

### Return Value Management

#### Single Return Point Principle
**Preferred approach**: Build result throughout function, single return at end
```python
def process_data(input_data):
    result = {
        'status': 'processing',
        'data': input_data,
        'metadata': {}
    }
    
    # ... processing logic that modifies result ...
    
    result['status'] = 'completed'
    return result
```

#### Early Return Exception
**Acceptable for**: Input validation and error conditions at function start
```python
def process_data(input_data):
    # Early validation returns acceptable
    if not input_data:
        return None
    if not isinstance(input_data, dict):
        raise ValueError("Input must be dictionary")
    
    # Main processing with single return
    result = build_result(input_data)
    return result
```

### Nesting Reduction Strategies

#### Guard Clauses
**Use early returns to reduce nesting depth**:
```python
# Preferred: Flat structure with guard clauses
def validate_command(command):
    if not command:
        return create_error("Command required")
    
    if not command.destination:
        return create_error("Destination required")
    
    if not command.mappings:
        return create_error("Mappings required")
    
    return create_success(command)
```

#### Extraction Methods
**Break complex nested logic into smaller functions**:
```python
# Instead of deep nesting, extract logical chunks
def process_complex_data(data):
    if not validate_input(data):
        return None
    
    processed = apply_transformations(data)
    if not processed:
        return None
        
    return finalize_result(processed)
```

### Comprehension Guidelines

#### Line-by-Line for Clarity
**Preferred for complex operations**:
```python
processed_items = [
    {
        'id': item.id,
        'processed_data': transform(item.data),
        'metadata': extract_metadata(item)
    }
    for item in input_items
    if item.is_valid()
]
```

#### Simple One-Liners When Clear
**Acceptable for straightforward transformations**:
```python
valid_ids = [item.id for item in items if item.is_valid()]
```

#### Avoid Over-Complexity
**Don't use comprehensions for**:
- Multiple levels of nesting
- Complex conditional logic
- Side effects or mutations

## Testing Standards

### Test Organization Principles

#### Structure Mirroring
**Test structure should mirror source structure at the module level**:
- One test file per source file (e.g., `module.py` → `test_module.py`)
- Test package hierarchy matches source package hierarchy
- Test class names relate directly to tested classes/modules

**Exception: Feature and Integration Tests**
Additional test files are acceptable for cross-cutting concerns:
- **Feature tests**: Test complete features spanning multiple modules
- **Integration tests**: Test interactions between multiple components
- **Bug regression tests**: Document and prevent specific bugs
- **Behavioral tests**: Document complex behavioral requirements

**Naming conventions for additional tests**:
- Feature tests: `test_<feature_name>.py` (e.g., `test_heading_level_alignment.py`)
- Integration tests: `test_<integration_scenario>.py` (e.g., `test_get_prompt_chain.py`)
- Bug tests: `test_<bug_description>.py` (e.g., `test_collected_context_bug.py`)

**Example structure**:
```
src/langtree/templates/
├── element_resolution.py     # Module
├── prompt_parser.py          # Module
└── variables.py              # Module

tests/templates/
├── test_element_resolution.py         # ✓ Direct mapping
├── test_prompt_parser.py              # ✓ Direct mapping
├── test_variables.py                  # ✓ Direct mapping
├── test_heading_level_alignment.py    # ✓ Feature test (uses multiple modules)
└── test_get_prompt_chain.py           # ✓ Integration test (full chain behavior)
```

#### When to Create New Test Files
**Create new test file when**:
- Testing a new source module (1:1 mapping)
- Existing test file would become unwieldy (>500 lines)
- Testing a feature that spans multiple modules
- Testing integration between components
- Documenting a specific bug with regression tests

#### Test Grouping Within Files
**Group tests by**:
- Tested class or major function
- Related behavioral scenarios
- Setup requirements (similar fixtures)

### Test-Driven Development Approach

#### Implementation Workflow
1. **Define behavior**: Write skipped test describing expected behavior
2. **Link to TODO**: Reference test in code TODO comment
3. **Implement feature**: Write minimal code to make test pass
4. **Remove skip**: Convert test from skipped to passing

#### Test Structure Requirements
```python
class TestSpecificModule:
    def setup_method(self):
        """Create fixtures for this test class."""
        # Arrange phase setup
        
    def test_specific_behavior(self):
        """Test one specific behavioral requirement."""
        # Arrange - Set up test conditions
        # Act - Execute the behavior being tested  
        # Assert - Verify expected outcomes
        
    @pytest.mark.skip("TODO: Feature not implemented")
    def test_planned_feature(self):
        """Test behavior for planned feature."""
        pytest.skip("TODO: Implement feature X - see module.py line Y")
```

### Type System Standards

#### Comprehensive Type Hints
- **All function signatures**: Parameters and return types
- **Class attributes**: Type hints for all fields
- **Complex types**: Use type aliases for clarity
- **Generic types**: Specify container contents where applicable
- **Union with None**: Use `type | None` instead of `Optional[type]`

#### Type Alias Usage
```python
# Define clear aliases for complex types
PromptValue = str | int | float | bool | list | dict | None
ConfigDict = dict[str, Any]
ProcessingResult = tuple[bool, str | None, dict[str, Any]]
```

## Error Handling Standards

### Exception Hierarchy Design

#### Specific Exception Types
**Create domain-specific exceptions**:
- Inherit from appropriate base exception
- Include relevant context in error messages
- Group related errors under common base classes

#### Error Message Standards
- **Specific and actionable**: Include what went wrong and potential fixes
- **Include context**: Relevant identifiers, paths, or values
- **Consistent format**: Similar errors should have similar message patterns

### Validation Patterns

#### Input Validation Strategy
- **Validate early**: Check inputs at function entry points
- **Fail fast**: Don't continue processing with invalid data
- **Clear messages**: Explain what was expected vs. what was received

These standards ensure consistent, maintainable, and scalable code across the LangTree DSL framework while supporting both current development and future evolution.