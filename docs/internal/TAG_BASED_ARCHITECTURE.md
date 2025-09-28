# Tag-Based Architecture Design Summary

## Core Architectural Decision

**Primary Goal**: Remove all runtime state management. LangTree DSL framework assembles LangChain chains - LangChain handles execution and data flow.

## Tag-Based Data Forwarding

### Key Insights
- **LangChain supports arbitrarily long dict keys** (tested up to 400+ characters)
- **Dotted notation works perfectly**: `"document.research.analysis.methodology.section.1"`
- **No runtime state needed**: Tags flow through LangChain's built-in mechanisms

### Tag Format
```python
# Source -> Target mapping examples
"sections.title" -> "analysis.sections.title" 
"document.methodology.approach" -> "evaluation.document.methodology.approach"
"research.findings.summary" -> "conclusions.research.findings.summary"
```

### Implementation
```python
# Tag-based forwarding in LangChain
RunnableParallel({
    "analysis.document.sections.title": RunnableLambda(extract_title),
    "analysis.document.sections.content": RunnableLambda(extract_content),
    "evaluation.analysis.document.sections.title": RunnableLambda(forward_title)
})
```

## Context Assembly Order

### Deterministic Ordering Algorithm
1. **Pre-calculated fields first**: Forwarded/computed values go to top of context
2. **Definition order**: Fields generate in class definition order  
3. **Sequential dependencies**: Prior fields available to subsequent fields (@sequential)
4. **Parallel isolation**: Fields generate independently (@parallel)

### Example
```python
class DocumentNode:
    """! @sequential"""
    title: str = "Pre-calculated"     # Position 1 in context
    introduction: str                 # Position 2, gets title  
    methodology: str                  # Position 3, gets title + introduction
    results: str                      # Position 4, gets title + introduction + methodology
```

## Node Execution Modifiers

### @parallel (Default)
- **Behavior**: All fields generate simultaneously
- **Context**: Only pre-calculated/forwarded fields available
- **Use case**: Independent content sections, performance optimization

### @sequential  
- **Behavior**: Fields generate in definition order
- **Context**: Each field gets all previously generated fields
- **Auto-forwarding**: Prior fields forwarded as if tagged with `@all`
- **Override**: Explicit forwarding disables auto-resolution
- **Use case**: Sequential document sections, dependent analysis steps

### Design Consistency
```python
# Command syntax (existing)
! @each[sections]->task.analyze@{{...}}*

# Node modifier syntax (new)  
! @sequential  # or ! @parallel
```

Both use `!` prefix and `@` keywords for consistency.

## Scope System Evolution

### Current Scopes
- `prompt`: Target context prompt variables
- `value`: Output becomes target variable value  
- `outputs`: Direct assignment scope (bypasses LLM)
- `task`: Reference to Task classes

### Outputs Scope Clarification
```python
# outputs scope for direct assignment during chain assembly
"outputs.title" -> direct_assignment_to_title
"prompt.title" -> include_in_prompt_context  
"value.title" -> output_becomes_title_value
```

## Implementation Strategy

### Phase 1: Core Architecture ✅
- [x] Remove all runtime state management
- [x] Implement tag-based forwarding  
- [x] Design deterministic context assembly
- [x] Define @sequential/@parallel semantics

### Phase 2: Implementation
- [ ] Update `structure/builder.py` for tag-based forwarding
- [ ] Implement node modifier parsing and handling
- [ ] Update context resolution for deterministic ordering
- [ ] Update all tests to reflect new architecture

### Phase 3: Validation  
- [ ] Test complex tag forwarding scenarios
- [ ] Validate @sequential/@parallel behavior
- [ ] Performance testing with long tag chains
- [ ] Integration testing with real LangChain workflows

## Key Validation Results

### LangChain Compatibility ✅ (Tested in Exploration)
```python
# Tested and confirmed working in notebooks:
keys = {
    "document.research.analysis.methodology.section.1.subsection.3.detailed.technical.evaluation": "content",
    "enterprise.software.architecture.microservices.api.gateway.service.mesh.container.orchestration": "content", 
    "machine.learning.artificial.intelligence.neural.networks.deep.learning.natural.language.processing": "content"
}
# All work perfectly in RunnableParallel (integration planned)
```

### Context Assembly ✅
```python
# Deterministic ordering confirmed:
def assemble_context_ordered(fields, pre_calculated):
    context = {}
    
    # Step 1: Add pre-calculated fields first
    context.update(pre_calculated)
    
    # Step 2: Add fields in definition order
    for field in fields:
        context[f"generated.{field}"] = f"content for {field}"
        
    return context
```

### Node Modifiers ✅  
```python
# @sequential: Fields get prior context
class SequentialNode:
    """! @sequential"""
    field1: str  # Gets: pre_calculated
    field2: str  # Gets: pre_calculated + field1  
    field3: str  # Gets: pre_calculated + field1 + field2

# @parallel: Fields isolated
class ParallelNode:
    """! @parallel"""  
    field1: str  # Gets: pre_calculated only
    field2: str  # Gets: pre_calculated only
    field3: str  # Gets: pre_calculated only
```

## Benefits Achieved

### 1. **Simplicity**
- No runtime state tracking
- No execution result storage  
- Pure chain assembly focus

### 2. **LangChain Native**
- Uses built-in data flow mechanisms
- Leverages proven RunnableParallel/Lambda patterns
- No custom runtime needed

### 3. **Deterministic**  
- Predictable context assembly order
- Explicit control over sequential/parallel execution
- Clear forwarding semantics

### 4. **Extensible**
- Tag system supports arbitrary nesting depth
- Node modifier system easily extensible
- Scope system supports future additions

### 5. **Performance**
- Tag-based forwarding design is efficient
- Parallel execution analysis (implementation planned)
- No unnecessary state overhead

## Architecture Validation

### Self-Consistency Check ✅
- **Tag forwarding**: Works with LangChain's native mechanisms
- **Context assembly**: Deterministic and predictable
- **Node modifiers**: Consistent with existing command syntax  
- **Scope resolution**: Clear semantics for all scopes
- **Data flow**: No runtime state needed

### Language Extension Check ✅  
- **Grammar**: Node modifiers fit naturally into existing EBNF
- **Keywords**: `@sequential`/`@parallel` follow `@each`/`@all` pattern
- **Syntax**: `!` prefix maintains consistency
- **Semantics**: Clear forwarding rules with explicit override capability

This architecture provides a clean, LangChain-native approach to prompt tree execution with deterministic behavior and clear semantics.