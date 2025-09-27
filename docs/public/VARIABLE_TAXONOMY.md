# DPCL Variable Taxonomy

## Overview

DPCL contains several distinct types of variables and references that serve different purposes and have different scopes, resolution times, and syntax. This document provides a precise taxonomy to clarify their roles and prevent confusion during implementation.

## Variable Type Categories

### 1. Assembly Variables
**Syntax:** `! variable_name=value`
**Scope:** Chain assembly time (before execution)
**Storage:** Variable registry during chain construction
**Purpose:** Store values for reuse across prompt nodes during chain assembly
**Conflict Policy:** Prohibited - assignment to existing variable name throws ParseError
**Resolution:** Immediate assignment, stored in registry for later reference

**Examples:**
```
! model_key="gpt-4"
! temperature=0.7
! use_cache=true
```

**Rules:**
- Available during chain construction for variable reference resolution
- Not directly interpolated into prompts (unless referenced by Runtime Variables)
- Must have unique names within chain scope
- Support string, number, and boolean values

### 2. Runtime Variables
**Syntax:** `{variable_name}`
**Scope:** Prompt execution time
**Storage:** Context resolution at runtime
**Purpose:** Dynamic content interpolation in prompts during execution
**Conflict Policy:** N/A (resolution-based, not assignment)
**Resolution:** Resolved from available context sources at execution time

**Examples:**
```
"Analyze this data: {data_source}"
"Use model {model_key} with settings {temperature}"
```

**Rules:**
- `{variable_name}` resolves from execution context (current node data)
- **NO access to Assembly Variables** - complete separation between assembly and runtime
- Support dynamic content from execution context sources only

### 3. DPCL Variable Targets
**Syntax:** `@each[variable_name]` or `@all[variable_name]`
**Scope:** Collection iteration during execution
**Storage:** Variable tracking in registry for dependency analysis
**Purpose:** Iterate over collections and track variable usage patterns
**Conflict Policy:** N/A (iteration-based, not assignment)
**Resolution:** Collection iteration with scope-aware context building

**Examples:**
```
@each[item] -> outputs.summary
@all[document] -> task.analysis
```

**Rules:**
- Create iteration scope for collection processing
- Track variable satisfaction sources for dependency analysis
- Support nested iteration contexts

### 4. Scope Context Variables
**Syntax:** `scope.field` (where scope = prompt|value|outputs|task|current_node)
**Scope:** Context-specific data resolution during execution
**Storage:** Scope-specific context resolvers
**Purpose:** Access structured data from specific execution contexts
**Conflict Policy:** N/A (context-based, not assignment)
**Resolution:** Scope-specific resolver classes handle field access

**Examples:**
```
prompt.template
value.user_input
outputs.generated_text
task.instructions
current_node.metadata
```

**Rules:**
- Each scope has dedicated resolver class
- Support hierarchical field access (e.g., `outputs.analysis.summary`)
- Immutable during prompt execution

### 5. Field References
**Syntax:** `field_name` (in resampling contexts) or `[field_name]` (in aggregation commands)
**Scope:** Specific to resampling and aggregation operations
**Storage:** Field metadata and Enum mapping validation
**Purpose:** Reference data fields for aggregation and resampling operations
**Conflict Policy:** Field existence validation at parse time
**Resolution:** Field type validation and Enum-to-number mapping for numerical functions

**Examples:**
```
! @resampled[status]->count
! @resampled[priority_level]->mean  # Enum mapped to numbers
```

**Rules:**
- Only Enum fields supported for numerical aggregation functions
- Non-numerical functions (count, mode, unique) work with any field type
- Automatic `_resampled_value` suffix generation for numerical outputs

## Variable Registry Design

### Storage Structure
```python
class VariableInfo:
    name: str
    value: Union[str, int, float, bool]
    scope: str  # "assembly", "runtime", "field_reference"
    source_type: str  # "assignment", "iteration", "context", "field"
    satisfaction_sources: List[str]
    relationship: str  # "direct", "reference", "iteration", "aggregation"
```

### Registry Interface
```python
class VariableRegistry:
    def store_assembly_variable(self, name: str, value: Any) -> None
    def get_assembly_variable(self, name: str) -> Any
    def check_assembly_conflict(self, name: str) -> bool
    def track_dpcl_target(self, variable_name: str, scope: str) -> None
    def resolve_runtime_variable(self, pattern: str, context: dict) -> Any
    def validate_field_reference(self, field_name: str, function: str) -> None
```

## Runtime Variable Resolution Priority

1. **Current Node Context** - Immediate execution context
2. **Task Context** - Task-level shared data
3. **Outputs Context** - Previous node outputs
4. **Value Context** - Input values and parameters
5. **Prompt Context** - Template and metadata

**Note**: Assembly Variables are NOT included in runtime resolution - they are completely separate.

## Conflict Resolution Rules

### Assembly Variable Conflicts
- **Rule:** Assignment to existing variable name is prohibited
- **Exception:** ParseError at chain construction time
- **Validation:** Check registry before assignment

### Runtime Variable Resolution
- **Rule:** `{variable_name}` resolves from execution context only
- **Scope Search:** Current node → parent → root hierarchy
- **Separation:** NO access to Assembly Variables (complete separation)

### Field Reference Validation
- **Rule:** Numerical aggregation functions require Enum fields
- **Validation:** Parse-time check of field type and function compatibility
- **Error:** ParseError for invalid field/function combinations

## Implementation Guidelines

1. **Parse-time Validation**: Assembly Variable conflicts and field reference validation must occur during parsing
2. **Runtime Resolution**: Runtime Variables and Scope Context Variables resolve during execution
3. **Separation of Concerns**: Each variable type has dedicated handling logic
4. **Error Handling**: Fail-fast for parse-time errors, graceful fallback for runtime resolution
5. **Documentation**: Clear examples and rules for each variable type

## Migration and Compatibility

- Existing `@each`/`@all` commands remain unchanged
- New Assembly Variables add capability without breaking existing functionality
- Runtime Variable resolution maintains backward compatibility
- Field References only apply to new resampling commands

This taxonomy provides the foundation for implementing a robust variable registry system that maintains clear separation between different variable types while supporting the full range of DPCL functionality.