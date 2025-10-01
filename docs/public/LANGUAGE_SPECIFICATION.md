# LangTree DSL Specification

## Overview

LangTree DSL is a domain-specific language for controlling data flow between TreeNode instancesTreeNodeal prompt execution systems. It enables precise specification of how data moves between different levels and components of a prompt tree structure.

## Node Naming Conventions

### Base Class Requirement
- **Required Inheritance**: All node classes must inherit from `TreeNode`
- **Validation**: System validates inheritance at parse-time for proper field detection and validation
- **Example**: `class TaskProcessor(TreeNode):`

### Root Task Nodes
- **Required Pattern**: `TaskCamelCaseName` (must start with `Task`)
- **Tag Conversion**: `TaskCamelCase` → `task.camel_case`
- **Examples**:
  - `TaskProcessor` → `task.processor`
  - `TaskDataProcessor` → `task.data_processor`
  - `TaskOutputAggregator` → `task.output_aggregator`

### Nested Nodes
- **Pattern**: Regular `CamelCase` naming (no `Task` prefix)
- **Context**: Defined as nested classes within task or other nodes
- **Path Resolution**: Accessed through parent path (e.g., `parent.nested_field`)
- **Inheritance**: Must also inherit from `TreeNode`

### Field Type Detection
- **Iterable Fields**: Automatically detected from type annotations
  - Supported: `List[...]`, `list`, `tuple`, other sequence types
  - Not Supported: `Dict`, `Mapping` (dictionary-like structures)
- **None vs Empty Iterables**: For @each commands, use empty iterables instead of None
  - Recommended: `default=[]` for optional list fields
  - Avoid: `default=None` for fields used with @each commands
  - Reason: None values create ambiguity in iteration contexts and field validation
- **Leaf List Limitations**:
  - Limitation: `list[primitive]` fields (e.g., `list[str]`, `list[int]`) are leaf nodes with no traversable subfields
  - Field Access: Cannot access individual elements or subfields within primitive lists
  - Valid: `sections.paragraphs` where `paragraphs: list[Paragraph]` (list of TreeNode)
  - Invalid: `sections.titles[0]` where `titles: list[str]` (list of primitives)
- **Iteration Inference**: System uses field types to determine iteration structure on left side of mappings

## Language Grammar

### Lexical Elements

#### Keywords
- `@each` - Iteration command for n:n relationships
- `@all` - Aggregation command for 1:1 and 1:n relationships
- `@` - Shorthand for `@all`
- `@sequential` - Node modifier for ordered field generation
- `@parallel` - Node modifier for simultaneous field generation
- `together` - Node modifier for simultaneous field generation (shorthand)
- `@resampled` - Resampling aggregation command for Enum fields
- `resample` - Resampling control command
- `llm` - LLM model selection command
- `true` - Boolean literal
- `false` - Boolean literal

#### Operators
- `!` - Command prefix (required)
- `->` - Flow operator (required)
- `=` - Assignment operator (for variables and variable mappings)
- `*` - Multiplicity indicator
- `{{` `}}` - Variable mapping delimiters
- `[` `]` - Inclusion path delimiters (also used for resampling field specification)
- `,` - Variable mapping separator
- `#` - Comment marker (everything after ignored in command lines)

#### Identifiers
- Path components: `[a-zA-Z_][a-zA-Z0-9_]*`
- Dot notation: `identifier(.identifier)*`
- Scope modifiers: `prompt`, `value`, `outputs`, `task`

### Syntax Grammar (EBNF)

```ebnf
command ::= "!" WS* command_body

command_body ::= each_command | all_command | node_modifier | variable_assignment | execution_command | resampling_command | comment_command

(* Strict spacing - no whitespace around operators *)
each_command ::= "@each" inclusion? "->" destination "@{{" mappings "}}" "*"

all_command ::= ("@all" | "@") "->" destination "@{{" mappings "}}" multiplicity?

(* Multiline support within brackets/braces *)
inclusion ::= "[" (path | multiline_path) "]"
mappings ::= mapping ("," mapping)* | multiline_mappings
multiline_mappings ::= WS* mapping (WS* "," WS* mapping)* WS* ","? WS*
multiline_path ::= WS* path_segments WS*

(* Path components - no spaces around dots *)
path ::= identifier ("." identifier)*
path_segments ::= identifier (WS* "." WS* identifier)*

(* Other command types *)
node_modifier ::= "@sequential" | "@parallel" | "together"

variable_assignment ::= identifier "=" value comment?

execution_command ::= identifier "(" (arguments | multiline_arguments) ")" comment?
multiline_arguments ::= WS* value (WS* "," WS* value)* WS* ","? WS*

resampling_command ::= "@resampled" "[" identifier "]" "->" aggregation_function comment?

comment_command ::= "#" [^\n]*

(* Core elements *)
destination ::= path

mapping ::= explicit_mapping | implicit_mapping

explicit_mapping ::= path "=" (path | "*")

implicit_mapping ::= path

multiplicity ::= "*"

identifier ::= [a-zA-Z_][a-zA-Z0-9_]*

value ::= string_literal | number_literal | boolean_literal | identifier

string_literal ::= '"' ([^"\\] | "\\" .)* '"' | "'" ([^'\\] | "\\" .)* "'"

number_literal ::= [0-9]+ ("." [0-9]+)?

boolean_literal ::= "true" | "false"

arguments ::= (value ("," value)*)?

(* Whitespace and comments *)
WS ::= [ \t\n\r]+ | comment_line
comment_line ::= "#" [^\n]* [\n\r]

aggregation_function ::= "mean" | "median" | "mode" | "sum" | "min" | "max"

comment ::= "#" [^\n]*
```

## Command Types

### @each Commands (Many-to-Many)

**Purpose**: Iterate over collections, producing multiple outputs for each input element.

**Syntax**:
```
! @each[inclusion_path]->destination@{{variable_mappings}}*
```

**Components**:
- `inclusion_path`: Path to collection(s) for iteration, relative to command's current working directory (CWD)
- `destination`: Target path for each iteration result (relative to CWD or absolute)
- `variable_mappings`: Data mappings for each iteration
- `*`: Required multiplicity indicator

**Command Working Directory (CWD)**:
- The CWD is the scope where the command is defined
- All relative paths resolve from this CWD
- Absolute paths (starting with `task.`) ignore CWD

**Iteration Path Rules**:
- For `[sections.subsections]`: Both `sections` and `subsections` must be iterable
- For `[documents.sections.paragraphs]`: All three levels must be iterable if specified
- Creates nested iteration over all specified levels

**Complex Iteration Paths**:
Support for mixed iterable/non-iterable segments in iteration paths:
- Pattern: `iterable1.noniterable.iterable2.iterable3`
- Each segment marked as iterable must actually be iterable in the data structure
- Non-iterable segments provide navigation through object relationships
- Example: `sections.metadata.tags.categories` where `sections` and `categories` are iterable, but `metadata` and `tags` are single objects

**Field Context Scoping Rules**:
Commands must respect strict field context scoping for proper data flow:

- **inclusion_path**: Must start with the field where the command is defined
- **destination**: Can reference other subtrees/nodes anywhere in the tree
- **target_path**: Can reference other subtrees/nodes anywhere in the tree
- **source_path**: Must share all iterable parts of the inclusion_path exactly

**Field Context Examples**:
```python
class TaskExample(TreeNode):
    # ✅ VALID: inclusion_path starts with field 'sections'
    sections: list[Section] = Field(description="! @each[sections.paragraphs]->other.analyzer@{{other.results=sections.paragraphs}}*")

    # ❌ INVALID: inclusion_path 'sections.paragraphs' doesn't start with field 'command'
    command: str = Field(description="! @each[sections.paragraphs]->analyzer@{{value.results=sections.paragraphs}}*")

    # ✅ VALID: inclusion_path starts with field 'analysis', all source_paths are subchains of 'analysis'
    analysis: list[str] = Field(description="! @each[analysis]->task.processor@{{task.results=analysis,other.meta=analysis.metadata}}*")
```

**Variable Mapping Constraints**:
```
! @each[sections.subsections]->task.analyzer@{{
    value.results.items=sections.subsections,           # ✅ Exact subchain match
    value.results.content=sections.subsections.text,    # ✅ Valid subchain with field access
    value.results.metadata=sections.subsections.meta    # ✅ Valid subchain with field access
}}*
```

**Critical Subchain Rule**: All source_paths in variable mappings must be subchains of the inclusion_path. For `@each[X.Y.Z]`, every source_path must start with `X.Y.Z` or be `X.Y.Z` itself.

**Left Side (Target) Rules**:
- Must have implicit iteration structure matching the source
- Field types in tree structure determine iteration levels
- No explicit `[]` notation - iteration is inferred from field types
- Example: `value.results.items` where `results` and `items` are implicitly iterable

**Right Side (Source) Rules**:
- Must be subchains of the inclusion_path
- All source_paths in variable mappings must start with the inclusion_path or be subchains of it
- Can access fields within the iteration scope defined by inclusion_path
- Examples for inclusion_path `[sections.subsections]`:
  - `sections.subsections` ✅ (exact match - valid subchain)
  - `sections.subsections.content` ✅ (subchain with field access)
  - `sections.subsections.title` ✅ (subchain with field access)
  - `sections.title` ❌ (NOT a subchain of 'sections.subsections')
  - `subsections` ❌ (NOT a subchain of 'sections.subsections')
  - `other.field` ❌ (completely different path, not a subchain)

**Nesting Level Validation**:
- **At least one mapping must match iteration level exactly**: For `@each[A.B.C]` creating N iteration levels, at least one LHS field must have exactly N nesting levels
- **No mapping can exceed iteration level**: No LHS field can have more nesting levels than the iteration depth
- **Iteration level counting**: Only iterable field segments count toward iteration levels (non-iterable segments are ignored)

**Nesting Examples**:
```
# @each[sections.paragraphs] creates 2 iteration levels
! @each[sections.paragraphs]->task@{{value.results=sections.paragraphs}}*     # ✅ results must have 2 levels
! @each[sections.paragraphs]->task@{{value.title=sections.title}}*            # ❌ title has 0 levels, no mapping matches 2
! @each[sections.paragraphs]->task@{{value.deep=sections.paragraphs}}*        # ❌ deep has 3+ levels, exceeds iteration depth
```

**Examples with Field Context Scoping**:
```python
class DocumentProcessor(TreeNode):
    # ✅ Command in 'documents' field can use inclusion_path starting with 'documents'
    documents: list[Document] = Field(description="! @each[documents]->task.summarize@{{value.summary=documents.content}}*")

    # ✅ Command in 'sections' field can use inclusion_path starting with 'sections'
    sections: list[Section] = Field(description="""
        ! @each[sections.paragraphs]->analyzer@{{
            value.analysis.segments=sections.paragraphs,
            value.analysis.metadata.title=sections.title
        }}*
    """)
```

**Semantics**:
- Iterate over all levels specified in inclusion path
- Left side implicitly tracks iteration through field types
- Right side must maintain consistent root throughout mappings
- Generate one output per deepest iteration level

### @all Commands (One-to-One and One-to-Many)

**Purpose**: Process entire subtrees as single operations.

**Syntax**:
```
! @all->destination@{{variable_mappings}}     # 1:1 relationship
! @all->destination@{{variable_mappings}}*    # 1:n relationship
! @->destination@{{variable_mappings}}        # Shorthand syntax
```

**Components**:
- `destination`: Target path for operation result (relative to CWD or absolute)
- `variable_mappings`: Data mappings for operation
- `*`: Optional multiplicity indicator for 1:n relationships

**Command Working Directory (CWD)**:
- Same as @each: the scope where the command is defined
- Relative paths like `summary` resolve to fields in current node
- Absolute paths like `task.summarize_analysis` specify exact target

**Source Path Scoping Rules**:
@all commands can only reference data from:
1. **The exact field containing the command** - ensures data locality
2. **Wildcard `*`** - represents entire current node

**Forbidden Source Patterns**:
- ❌ Sibling field references (breaks locality)
- ❌ Longer paths from containing field (breaks "entire subtree" semantics)
- ❌ External field references (breaks predictability)

**Implicit Mapping for Fields**:
When the target variable name matches the containing field name, the mapping can be implicit:
```
! @all->summary@{{outputs.main_analysis}}*  # Equivalent to outputs.main_analysis=main_analysis
```
This shorthand requires:
- The command is in the `main_analysis` field
- Target variable name matches the containing field name
- Only valid for single field forwarding (not with multiple mappings)

**Examples**:
```python
class TaskExample(TreeNode):
    # ✅ Valid - RHS matches containing field
    summary: str = Field(description="! @->task.analyzer@{{prompt.source_data=summary}}")

    # ✅ Valid - RHS matches containing field
    input_data: str = Field(description="! @all->task.process@{{value.data=input_data}}*")

    # ✅ Valid - wildcard represents entire node
    analysis: str = Field(description="! @->task.analyze@{{prompt.context=*}}")

    # ✅ Valid - implicit mapping (field name = target variable name)
    main_analysis: str = Field(description="! @all->summary@{{outputs.main_analysis}}*")
```

**Semantics**:
- Process entire current subtree
- Without `*`: Single output (1:1)
- With `*`: Multiple outputs (1:n)
- Cannot have inclusion brackets (no iteration)
- CWD determines scope for relative paths

### Variable Assignment Commands

**Purpose**: Define assembly-time variables for chain configuration and flow control.

**Syntax**:
```
! variable_name=value  # Basic assignment
! variable_name=value  # comment  
```

**Components**:
- `variable_name`: Valid identifier ([a-zA-Z_][a-zA-Z0-9_]*)
- `value`: String literal (quoted) or number literal (unquoted)
- `# comment`: Optional comment (ignored during parsing)

**Examples**:
```
! count=5                    # Integer variable
! threshold=2.5              # Float variable  
! name="analysis_task"       # String variable
! title='Multi-word title'   # String with single quotes
! path="file with \"quotes\"" # Escaped quotes
! debug=true                 # Boolean variable
! verbose=false              # Boolean variable
! override=true              # Boolean for command arguments
```

**Semantics**:
- Variables are assembly-time only (available during chain construction)
- Variable scope: Available from definition node through all descendant nodes
- Name conflict prohibition: Cannot conflict with field names in same subtree
- Type coercion: Unquoted values must be valid numbers (int/float) or booleans (true/false)

### Execution Commands

**Purpose**: Execute operations and control chain behavior.

**Syntax**:
```
! command_name(arguments)  # Basic execution
! command_name(arg1, arg2) # Multiple arguments
! command_name()           # No arguments
```

**Components**:
- `command_name`: Valid identifier for registered command
- `arguments`: Comma-separated list of variables or literals
- Variable resolution: Arguments resolve to variable values or literal values

**Built-in Commands**:
- `resample(n_times)`: Execute subtree n times for aggregation
- `llm(model_key, override=false)`: Select LLM model for subtree execution

**Command Registry**:
- **Current implementation**: Built-in commands only for simplicity and reliability
- **Future extensibility**: Plugin system may be added later to support custom commands
- **Validation**: All commands validated at parse-time for existence and argument compatibility

**Examples**:
```
! resample(5)        # Literal argument
! resample(count)    # Variable argument  
! resample(iterations) # Variable from current scope
! llm("gpt-4")       # Select model for subtree
! llm("claude-3", override=true)  # Override model for entire subtree
! llm(model_name, override=false) # Variable arguments
```

**Semantics**:
- Commands affect execution behavior of current subtree
- Argument resolution follows variable scope rules
- Parse-time validation of command existence and argument types

### LLM Model Selection Commands

**Purpose**: Configure LLM model for prompt tree execution.

**Syntax**:
```
! llm(model_key)                    # Basic model selection
! llm(model_key, override=boolean)  # Model selection with override control
```

**Components**:
- `model_key`: String literal or variable referencing model name from LLMProvider.list_models()
- `override`: Optional boolean controlling inheritance behavior (default: false)

**Semantics**:
- **Model Resolution**: `model_key` must resolve to valid model name from LLMProvider registry
- **Scope Inheritance**: 
  - `override=false` (default): Model applies to current node/field only
  - `override=true`: Model applies to entire subtree, overriding child node model selections
- **Default Model**: If no `llm()` command specified, uses first model from LLMProvider.list_models()

**Examples**:
```
! llm("gpt-4")                    # Use GPT-4 for this node only
! llm("claude-3", override=true)  # Use Claude-3 for entire subtree
! llm(model_var)                  # Use variable-defined model
! llm("fast-model", override=false) # Explicit non-override (same as default)
```

**Model Inheritance Rules**:
1. **Root Level**: Uses default model (first from list_models()) if no llm() command
2. **Child Inheritance**: Inherits parent model unless:
   - Child has its own llm() command, OR
   - Parent used override=true (takes precedence over child llm() commands)
3. **Override Hierarchy**: override=true propagates down entire subtree
4. **Field Level**: llm() commands in field descriptions apply to that field's generation only

**Validation**:
- Model key must exist in LLMProvider registry at parse-time
- Boolean values validated for override parameter
- Argument count validated (1-2 arguments required)

### Resampling Aggregation Commands

**Purpose**: Specify how Enum fields should be aggregated during resampling operations.

**Syntax**:
```
! @resampled[field_name]->aggregation_function
```

**Components**:
- `field_name`: Name of Enum field in current node
- `aggregation_function`: Aggregation method for multiple execution results

**Aggregation Functions**:
- **Numerical functions** (require numerical enum values):
  - `mean`: Mean of enum numerical values
  - `median`: Middle value when sorted  
  - `min`/`max`: Minimum/maximum numerical values
- **Universal functions** (work with any enum):
  - `mode`: Most frequently occurring enum value

**Examples**:
```
! @resampled[rating]->mean        # Numerical aggregation
! @resampled[status]->mode        # Most frequent value
! @resampled[priority]->median    # Middle value
```

**Semantics**:
- Only Enum fields can be specified (validated at parse-time)
- Numerical functions require enum with numerical values (e.g., `Rating.HIGH = 3`)
- Active only during resampling operations (when `resample(n)` is in effect)
- Numerical aggregation creates additional `field_name_resampled_value: float` output field
- Non-numerical aggregation only populates original enum field

**Output Behavior**:
- **Normal execution**: Resampling commands ignored, LLM fills enum field directly
- **Resampling with numerical aggregation**:
  ```python
  {
      "rating": "MEDIUM",           # Converted from numerical result  
      "rating_resampled_value": 2.3  # Raw aggregated number
  }
  ```
- **Resampling with non-numerical aggregation**:
  ```python
  {
      "rating": "HIGH"  # Most frequent value, no numerical field
  }
  ```

### Node Modifier Commands

**Purpose**: Control field generation behavior and execution order within nodes.

**Syntax**:
```
! @sequential  # Process fields in declaration order
! @parallel    # Process all fields simultaneously  
! together     # Shorthand for @parallel
```

**Components**:
- `@sequential`: Process fields one after another, making previous field results available to subsequent fields
- `@parallel`: Process all fields simultaneously with shared context
- `together`: Equivalent to `@parallel` but more concise

**Examples**:
```
class AnalysisNode(TreeNode):
    """! @sequential"""
    overview: str = Field(description="High-level overview")
    details: str = Field(description="Detailed analysis - has access to overview")
    
class ReportNode(TreeNode):  
    """! @parallel"""
    summary: str = Field(description="Executive summary")
    technical: str = Field(description="Technical details")
    
class QuickNode(TreeNode):
    """! together"""  # Same as @parallel
    intro: str
    body: str
```

**Semantics**:
- **@sequential**: Fields processed in order, each field gets context from all previously processed fields
- **@parallel/@together**: All fields processed simultaneously with same initial context
- **Default behavior**: If no modifier specified, defaults to @sequential
- **Context sharing**: @sequential builds incremental context; @parallel shares static context

### Comment Support

**Purpose**: Add human-readable annotations to command lines and provide standalone documentation.

**Syntax**:
```
! # This is a standalone comment
!# Another standalone comment
! command # This is an inline comment
! variable=value # Another inline comment
```

**Comment Types**:

#### Standalone Comments
- **Syntax**: `! # comment` or `!# comment`
- **Purpose**: Pure documentation lines that don't execute commands
- **Examples**: `! # Initialize analysis phase`, `!# TODO: optimize performance`

#### Inline Comments
- **Syntax**: `command # comment`
- **Purpose**: Annotate existing commands with explanations
- **Support**: All command types (variable assignment, execution, @each, @all, etc.)
- **Examples**:
  - `! model="gpt-4" # Use high-quality model`
  - `! @each[items]->task@{{value.data=items}}* # Process each item`

#### Multiline Comments
- **Context**: Within variable mappings `@{{...}}` blocks
- **Processing**: Quote-aware parsing (comments inside strings are preserved)
- **Example**:
```
! @each[items]->task@{{
    # This is a comment within the mapping
    value.data=items, # Another comment
    value.title="Item with # symbol" # Hash in string preserved
}}*
```

**Rules**:
- Comments start with `#` character
- Everything after `#` is ignored during parsing (except inside quoted strings)
- Whitespace before `#` is trimmed
- Comments are for human readability only
- Quote-aware: `#` symbols inside quoted strings are not treated as comments

## Variable System

LangTree DSL implements a sophisticated variable system with five distinct types, each serving different purposes with different scopes, resolution times, and syntax:

### Variable Type Taxonomy

#### 1. Assembly Variables (`! var=value`)
**Scope:** Chain assembly time (before execution)  
**Storage:** Variable registry during chain construction  
**Purpose:** Store values for reuse across prompt nodes during chain assembly  
**Conflict Policy:** Prohibited - assignment to existing variable name throws ParseError  

#### 2. Runtime Variables (`{var}`)
**Scope:** Prompt execution time
**Storage:** Context resolution at runtime
**Purpose:** Dynamic content interpolation in prompts during execution
**Conflict Policy:** N/A (resolution-based, not assignment)  

#### 3. LangTree DSL Variable Targets (`@each[var]` / `@all[var]`)
**Scope:** Collection iteration during execution  
**Storage:** Variable tracking in registry for dependency analysis  
**Purpose:** Iterate over collections and track variable usage patterns  
**Conflict Policy:** N/A (iteration-based, not assignment)  

#### 4. Scope Context Variables (`scope.field`)
**Scope:** Context-specific data resolution during execution  
**Storage:** Scope-specific context resolvers  
**Purpose:** Access structured data from specific execution contexts  
**Conflict Policy:** N/A (context-based, not assignment)  

#### 5. Field References (`[field]`)
**Scope:** Specific to resampling and aggregation operations  
**Storage:** Field metadata and Enum mapping validation  
**Purpose:** Reference data fields for aggregation and resampling operations  
**Conflict Policy:** Field existence validation at parse time  

### Assembly Variables

**Purpose**: Configuration variables available during chain construction.

**Declaration**: Using variable assignment commands (`! var=value`)

**Scope Rules**:
- Available from definition node through all descendant nodes in subtree
- Resolution by variable name: `<variable_name>` resolves to nearest definition upward in tree
- Resolution by full path: Full tag path for explicit reference when needed

**Conflict Prohibition**:
- Variable names cannot conflict with field names in same subtree
- Parse-time validation throws exception on conflicts
- Ensures unambiguous resolution and clear error messages

**Examples**:
```
! count=5              # Define assembly variable
! resample(count)      # Use in command argument
! threshold=2.5        # Available to all child nodes
```

**Separation from Runtime**:
- Assembly variables: Known at parse-time, used for flow control
- Runtime variables ({variable}): Resolved at execution time in prompts
- No bridging between assembly and runtime contexts

### Runtime Variables

**Purpose**: Dynamic content interpolation in prompts during execution.

**User Syntax**:
- `{variable_name}` - Single token only, no dots allowed
- **Naming Rule**: Variables must contain at least one lowercase letter
- Valid examples: `{model_name}`, `{temperature}`, `{analysisType}`, `{DataSource}`
- **Invalid - no lowercase**: `{MODEL_NAME}`, `{DATA_SOURCE}`, `{CONFIG_1}`, `{OUTPUT_2}`
- **Invalid - dots**: `{task.model}`, `{prompt.data}` - users cannot use dots

**Internal System Expansion**:
- User writes: `{model_name}` in prompts/docstrings
- System expands based on context:
  - **Root context**: `{model_name}` → `{prompt__external__model_name}`
  - **Node context**: `{model_name}` in `analyzer` → `{prompt__analyzer__model_name}`
  - **Deep nesting**: `{model_name}` in `processors.sentiment` → `{prompt__processors__sentiment__model_name}`
- Purpose: Enables LangChain variable forwarding while keeping user syntax simple
- **Constraint**: User variable names cannot contain `__` (reserved for system use)

**Resolution**:
- Runtime variables resolve to actual field values on the current node
- **Scope override rule**: `prompt` scope overrides `task` - they never combine
- Examples:
  - Root: `{title}` → `{prompt__external__title}`
  - Node: `{result}` in `analyzer` → `{prompt__analyzer__result}`
  - Deep: `{score}` in `processors.sentiment` → `{prompt__processors__sentiment__score}`

**Separation from Assembly Variables**:
- Assembly variables (`! var=value`): Parse-time only, for chain configuration
- Runtime variables (`{var}`): Execution-time only, for prompt content
- No bridging between these contexts

### Conflict Resolution Rules

#### Assembly Variable Conflicts
- **Rule:** Assignment to existing variable name is prohibited
- **Exception:** ParseError at chain construction time
- **Validation:** Check registry before assignment

#### Runtime Variable Resolution
- **Rule:** `{variable_name}` resolves from execution context only
- **Scope Search:** Current node → parent → root hierarchy  
- **Error:** Clear error messages for undefined variable references

#### Field Reference Validation
- **Rule:** Numerical aggregation functions require Enum fields
- **Validation:** Parse-time check of field type and function compatibility
- **Error:** ParseError for invalid field/function combinations

### Variable Separation Principles

**Assembly vs Runtime Variables:**
- Assembly variables: Known at parse-time, used for flow control
- Runtime variables: Resolved at execution time in prompts
- No bridging between assembly and runtime contexts

**Registry Separation:**
- Assembly Variables: Simple key-value storage with conflict detection
- LangTree DSL Variable Targets: Complex dependency tracking with satisfaction sources
- Each type has dedicated storage and resolution logic

### Implementation Guidelines

The variable system implementation follows these architectural principles:

1. **Parse-time Validation**: Assembly Variable conflicts and Field Reference validation occur during parsing
2. **Runtime Resolution**: Runtime Variables and Scope Context Variables resolve during execution
3. **Separation of Concerns**: Each variable type has dedicated handling logic
4. **Error Handling**: Fail-fast for parse-time errors, graceful fallback for runtime resolution
5. **No Backward Compatibility**: Clean implementation with no legacy support

For detailed registry architecture and implementation patterns, see [VARIABLE_TAXONOMY.md](VARIABLE_TAXONOMY.md).

### Scope Context Variables

Scope modifiers control how data flows in @each/@all commands. They appear as prefixes in variable mappings: `scope.field_name`.

#### `prompt` Scope
- **Purpose**: Forward data to template variables in target prompts
- **Syntax**: `prompt.variable_name=source`
- **Effect**: Data becomes available as `{variable_name}` in target's prompts/docstrings
- **Example**: `prompt.source_data=summary` → `{source_data}` available in target
- **Scope override**: `prompt` overrides `task` - they are mutually exclusive
- **Internal**: Expands to `{prompt__<target_path>__variable_name}` for LangChain
- **Multiple Sources**: When multiple sources target same `prompt.variable_name`, they are automatically numbered as `{variable_name_1}`, `{variable_name_2}`, etc. in the target prompt
- **Collection behavior**: Unlike `value` scope (which should have single source), `prompt` scope is designed for data accumulation from multiple sources

#### `value` Scope
- **Purpose**: Direct value assignment - sets field value without LLM generation
- **Syntax**: `value.field_name=source`
- **Effect**: Target field gets source value directly (LLM never generates this field)
- **Example**: `value.title=sections.title` → title field set to sections.title value
- **Scope override**: `value` overrides other scopes - mutually exclusive
- **Use case**: Pre-populate fields with known values
- **Multiple Sources**: Multiple sources targeting same `value.field_name` indicate a conflict (field can only have one value) and should be flagged as a validation warning

#### `outputs` Scope
- **Purpose**: Override calculated values during prompt assembly
- **Syntax**: `outputs.field_name=source`
- **Effect**: Temporarily replaces field value when building prompts (not permanent)
- **Example**: `outputs.main_analysis=main_analysis` → override list with single element
- **Scope override**: `outputs` overrides other scopes - mutually exclusive
- **Collection behavior**: When multiple sources send to same `outputs.field`, they should be collected together rather than replacing each other
- **Use case**: Iteration scenarios where you need different context per iteration

#### `task` Scope
- **Purpose**: Reference fields in the tree structure
- **Syntax**: `task.node_name.field_name`
- **Effect**: Access data stored in tree nodes
- **Example**: `task.analyzer.results` → access results field of analyzer node
- **Scope override**: `task` is the default scope, overridden by others
- **Note**: This is for data storage/access, not template variables

#### `prompt.external` Scope (Special Case)
- **Purpose**: Forward data to external prompts (system prompts, additional prompts)
- **Syntax**: `@all->prompt.external@{{variable_name=source}}`
- **Effect**: Makes data available to non-tree prompts
- **Example**: `@all->prompt.external@{{context_data=*}}`
- **Resolution**: External prompts use `{variable_name}` → `{prompt__external__variable_name}`

### Multiple Source Handling

Different scopes handle multiple sources targeting the same variable differently:

- **`prompt` scope**: ✅ **Expected** - Creates numbered variables (`{var_1}`, `{var_2}`, etc.) for data accumulation
- **`value` scope**: ⚠️ **Conflict** - Field can only have one value; multiple sources flagged as validation warning
- **`outputs` scope**: ✅ **Expected** - Collects all sources for context assembly
- **`task` scope**: N/A - References existing data, doesn't receive forwarded data

### Scope Resolution

Scope modifiers are extracted from all path components:

#### Path Component Types
1. **Inclusion Path** (`pathX`): `@each[scope.path]`
2. **Destination Path** (`pathA`): `->scope.path`
3. **Target Path** (`pathB`): `{{scope.path=...}}`
4. **Source Path** (`pathC`): `{{...=scope.path}}`

#### Resolution Rules
- If path starts with known scope modifier: extract scope, use remainder as path
- If path starts with unknown identifier: treat entire path as non-scoped
- Single character prefixes allowed as unknown scopes
- Nested paths supported: `scope.deeply.nested.path`

### Field Type Resolution

LangTree DSL processes different field types during content assembly to generate appropriate headings and structure:

**Collection vs Single Types**:
```python
# Collection type - include node name heading
var_name: list[NodeName] = Field(description="...")
→ # Var Name
  <field description>
  ## Node Name  
  <docstring>

# Single type - skip redundant node name heading  
var_name: NodeName = Field(description="...")
→ # Var Name
  <field description>
  <docstring>
```

**Heading Level Adjustments**:
- System detects heading level of section containing template variables
- Child content heading levels adjusted accordingly for proper nesting
- Tree traversal must be resolved level-by-level to determine correct depth

**Title Generation**:
- Field names converted to titles (e.g., `main_analysis` → `# Main Analysis`)
- Resolution always starts with title (class name or field name) followed by description
- Collection types include both field title and node type title for clarity

## Validation Rules

### Parse-Time Validation

LangTree DSL enforces strict validation during parsing to catch errors early and ensure predictable behavior:

#### Variable Assignment Validation
- **Name format**: Variable names must match `[a-zA-Z_][a-zA-Z0-9_]*` (Python identifier rules)
- **Value types**: 
  - Quoted strings: Support single (`'`) or double (`"`) quotes with escape sequences (`\"`, `\'`)
  - Numbers: Unquoted values must be valid integers or floats
  - Booleans: Unquoted `true` or `false` (case-sensitive)
  - **Error condition**: Unquoted non-numeric, non-boolean values throw parse-time exception
- **Conflict detection**: Variable names cannot conflict with field names in same subtree

#### Execution Command Validation  
- **Command existence**: Referenced commands must be registered in command registry
- **Argument types**: Command arguments validated against expected parameter types
- **Argument resolution**: Variable arguments must resolve to valid variables in scope

#### Resampling Command Validation
- **Field existence**: `@resampled[field_name]` must reference existing field in current node
- **Field type restriction**: Referenced field must be Enum type (not str, int, list, dict, etc.)
- **Function compatibility**:
  - Numerical functions (`mean`, `median`, `sum`, `min`, `max`) require enum with numerical values
  - Universal functions (`mode`) work with any enum type
- **Enum numerical mapping**: For numerical functions, validate that enum has accessible numerical values

#### Error Handling
- **Fail-fast strategy**: Parse-time errors immediately halt processing with descriptive messages
- **Variable conflicts**: Immediate exception when variable names conflict with field names in same subtree
- **Descriptive messages**: Parse-time errors include specific details about what failed and why
- **Location information**: Error messages reference specific line/command where validation failed
- **Suggested fixes**: When possible, error messages suggest correct syntax or alternatives

### Runtime Validation

#### Variable Resolution
- **Scope lookup**: Variables resolved using hierarchical scope rules (current node → parent → root)
- **Conflict prohibition**: Runtime throws exception if variable conflicts detected during resolution
- **Missing variables**: Clear error messages for undefined variable references

#### Command Execution
- **Argument validation**: Runtime validation of argument values and types
- **State consistency**: Validation that commands execute in valid context (e.g., resampling commands only active during resampling)

## Template Variables

LangTree DSL provides two special template variables for automatic prompt assembly and context injection. These variables use the same `{var}` syntax as Runtime Variables but have reserved names and are automatically resolved by the system. They cannot be used as Assembly Variables or user-defined Runtime Variables.

**Naming Reservation**: All variable names without lowercase letters are reserved for template variables. This means:
- `{PROMPT_SUBTREE}` ✅ - Valid template variable
- `{COLLECTED_CONTEXT}` ✅ - Valid template variable
- `{OUTPUT}` ❌ - Reserved namespace (no lowercase), but not a valid template variable (will error)
- `{DATA_SOURCE}` ❌ - Reserved namespace (no lowercase), but not a valid template variable (will error)
- `{OUTPUT_1}` ❌ - Reserved namespace (no lowercase with number), but not a valid template variable (will error)
- `{outputData}` ✅ - Valid runtime variable (has lowercase letters)

### {PROMPT_SUBTREE}

**Purpose**: Placeholder for child field content in parent docstrings.

**Automatic Addition**: If `{PROMPT_SUBTREE}` is not present in a node's docstring, it is automatically appended at the end during parse time.

**Resolution**: Replaced with assembled content from child fields, including:
- Field names converted to titles (e.g., `main_analysis` → `# Main Analysis`)  
- Field descriptions from `Field(description="...")` 
- Proper heading level adjustments based on context depth

**Spacing Rules**: Must have empty lines before and after: `"text\n\n{PROMPT_SUBTREE}\n\nmore text"`

### {COLLECTED_CONTEXT}

**Purpose**: Placeholder for forwarded context data in prompts.

**Manual Addition**: Only added during "prompt with context" resolution if not explicitly present in docstring.

**Resolution**: Replaced with context data from previous generations and forwarded outputs.

**Automatic Fallback**: If `{COLLECTED_CONTEXT}` is not in docstring, system appends `"\n\n# Context\n\n{COLLECTED_CONTEXT}"` when context is needed.

**Spacing Rules**: Same as `{PROMPT_SUBTREE}` - requires empty lines before and after.

### Heading Level Detection

**Level Calculation**: System detects heading level of section containing template variables and adjusts child content accordingly.

**Tree Traversal**: Must be resolved level-by-level to determine correct heading depth for each node.

**Title Generation**: Resolution always starts with title (class name or field name) followed by description.

### Assembly Types

**Context Only**: Clean context with proper headings, no prompt content mixed in.

**Prompt Only**: Docstring with `{PROMPT_SUBTREE}` resolved to child field content.

**Prompt with Context**: 
- If `{COLLECTED_CONTEXT}` exists in docstring: resolved in place
- If `{COLLECTED_CONTEXT}` missing: automatically appended as `"# Context"` section

## Prompt Context Assembly

### Outputs Section in Prompts
When a chain is executed, the target node's prompt includes an **"Outputs"** section containing all data mapped via `outputs` scope assignments. This section aggregates:

1. **Tree-traversal data**: Fields collected automatically from the path (root → current node)
   - Sequential nodes: One field per node on the path OR all fields from parent to current
   - Parallel nodes: All fields from nodes at that level + forwarded parent data
   
2. **Manual forwarding**: Explicitly forwarded data via `outputs.field=source` commands

3. **Prior subchain results**: Results from previously executed subchains in the dependency graph

### Field Context Assembly Rules
When generating a field, its context includes all available field values in this priority order:

1. **External forwarded data** (from other nodes via `outputs` scope assignments)
2. **Pre-calculated fields** (fields with default values in current node)  
3. **Previously processed fields** (for `@sequential` nodes only)
4. **Future fields with known values** (fields defined later but already computed)

**Critical Behavior**: Any field with a known value appears in the context **before** the currently processing field, regardless of its definition order in the class.

### Context Assembly Order
```
# Example prompt structure (planned implementation):
# Context
{general_contextual_information}

# Outputs  
{outputs_section_with_forwarded_data_and_subchain_results}

# Task  
{current_node_docstring_and_instructions}

# Output
{output_format_specifications}

# Input
{direct_input_data}
```

The **Outputs section** (planned feature) will provide the LLM with all relevant intermediate results and forwarded data to inform its generation of the current field.

**Current Implementation**: Uses single "Context" section for all contextual data. The dedicated "Outputs" section is planned for future implementation to separate general context from execution results.

### Data Flow Examples

#### Sequential Processing (Automatic Forwarding)
```python
class DocumentAnalysis(TreeNode):
    """! @sequential"""
    overview: str = Field(description="First field - gets only external context")
    methodology: str = Field(description="Gets overview in context automatically")
    results: str = Field(description="Gets overview + methodology in context automatically")
    # No explicit forwarding needed - @sequential does it automatically
```

#### Manual Forwarding (Cross-Node)
```python
class SummaryTask(TreeNode):
    analysis: str = Field(description="! @each[sections]->summary@{{outputs.section_data=sections}}")
    # summary prompt will include "Outputs" section with all section_data
```

#### Context Assembly with Forward References
```python
class ReportGeneration(TreeNode):
    """! @sequential"""
    # If conclusion is already generated by another chain, it appears BEFORE title
    title: str = Field(description="Gets conclusion in context if available")  
    body: str = Field(description="Gets title + conclusion (if available) in context")
    conclusion: str = Field(description="! @->external_conclusion@{{value.conclusion=analysis}}")
```

**Context Priority**: Fields with known values (from external forwarding or previous execution) appear **before** the currently processing field in the context, regardless of their definition order.

## Variable Mappings

### Explicit Assignment
```
target_path = source_path
```

**Examples**:
```
prompt.title = sections.title
value.content = document.body
outputs.result = analysis.summary
```

### Implicit Assignment
```
target_path
```

**Behavior**: Source path inferred from target variable name.

**Examples**:
```
prompt.target_data    # Equivalent to: prompt.target_data = target_data
value.title          # Equivalent to: value.title = title
outputs.main_analysis  # Equivalent to: outputs.main_analysis = main_analysis
```

**Restrictions**:
- Only works when target variable name matches a field in current scope
- Cannot be mixed with explicit assignments in the same command
- All mappings must be either implicit OR explicit, never both

### Wildcard Assignment
```
target_path = *
```

**Purpose**: Forward entire current subtree to target.

**Restrictions**:
- Only valid in `@all` commands
- Cannot be used with `@each` commands
- Cannot be combined with multiple mappings
- Considered an explicit assignment (cannot mix with implicit)

**Examples**:
```
! @->task.analyze@{{prompt.context=*}}
! @all->destination@{{value.data=*}}
```

### Multiple Mappings
```
mapping1, mapping2, mapping3
```

**Rules**:
- All mappings must be same type (all implicit OR all explicit)
- Cannot mix: `{{outputs.field1, value.field2=source}}` ❌
- Valid: `{{outputs.field1, outputs.field2}}` ✅ (all implicit)
- Valid: `{{value.field1=source1, value.field2=source2}}` ✅ (all explicit)

**Examples**:
```
# All explicit
{{prompt.title=sections.title, value.content=sections.content, outputs.meta=sections.metadata}}

# All implicit (if fields exist in current scope)
{{prompt.title, prompt.summary, prompt.metadata}}
```

## Path Syntax

### Basic Paths
- Simple identifier: `title`
- Dotted notation: `sections.title`
- Deep nesting: `document.structure.sections.subsections.content`

### Scoped Paths
- With scope: `prompt.variable`
- Without scope: `variable`
- Mixed: `prompt.deeply.nested.variable`

### Special Cases
- Single characters allowed: `p.a`, `x.y.z`
- Unknown scopes treated as regular paths: `custom.field`
- Empty components not allowed: `prompt.` (invalid)

## Relationship Types

### One-to-One (1:1)
- **Commands**: `@all` without `*`
- **Behavior**: Single input → Single output
- **Example**: `! @->task.process@{{value.result=input}}`

### One-to-Many (1:n)
- **Commands**: `@all` with `*`
- **Behavior**: Single input → Multiple outputs
- **Example**: `! @all->task.generate@{{prompt.seed=topic}}*`

### Many-to-Many (n:n)
- **Commands**: `@each` with `*` (required)
- **Behavior**: Multiple inputs → Multiple outputs (one per input)
- **Example**: `! @each[items]->task.process@{{value.item=items}}*`

## Validation Rules

### Required Elements
1. Commands must start with `!`
2. Commands must contain `->`
3. Variable mappings cannot be empty
4. `@each` commands require `*` multiplicity
5. Proper bracket/brace matching

### Forbidden Patterns
1. `@all` commands cannot have inclusion brackets: `! @all[items]->...` ❌
2. `@each` commands with empty inclusion: `! @each[]->...` ❌
3. Wildcard with `@each`: `! @each[items]->task@{{value=*}}*` ❌
4. Multiple mappings with wildcard: `! @->task@{{a=*,b=c}}` ❌
5. Empty variable names: `! @->task@{{=value}}` ❌
6. Empty source paths: `! @->task@{{prompt.var=}}` ❌

### Warning Patterns (Valid but Unusual)
1. Unknown scope modifiers: `unknown.field` ⚠️
2. Single character identifiers: `p.a=b` ⚠️
3. Very deep nesting: `a.b.c.d.e.f.g` ⚠️

## Whitespace Handling

### Strict Whitespace Rules

LangTree DSL enforces strict whitespace rules to ensure consistent, unambiguous command syntax:

#### Prohibited Whitespace (Zero Tolerance)
- **Around dots (`.`)**: `document.sections` not `document . sections`
- **Around `@`**: `@each` not `@ each` or ` @each`
- **Before `[`**: `@each[items]` not `@each [items]`
- **After `]`**: `[items]->` not `[items] ->`
- **Around `->` (arrow)**: `]->task` not `] ->` or `-> task`
- **Before `{{`**: `task@{{` not `task @{{`
- **After `}}`**: `}}*` not `}} *` (when followed by another token)

#### Allowed Whitespace
- **After `!`**: `!    @each[items]...` ✅ (spaces after command prefix allowed)
- **Inside `{{...}}`**: `{{value.item=items,prompt.context=data}}` (flexible spacing for multiline - see below)
- **Inside `[...]`**: `[document.sections.subsections]` (flexible spacing for multiline - see below)
- **Inside `(...)`**: `(arg1,arg2)` (flexible spacing for execution commands)

### Multiline Command Support

LangTree DSL supports multiline commands within bracket/brace/parenthesis contexts following Python-inspired syntax:

#### Multiline Contexts
Multiline continuation is **only** allowed within:
- **Inclusion brackets**: `[...]`
- **Variable mapping braces**: `{{...}}`
- **Execution argument parentheses**: `(...)`

#### Multiline Rules
```python
# Valid multiline variable mappings
! @each[items]->task@{{
    value.title=items.title,
    value.content=items.content,
    prompt.metadata=items.meta
}}*

# Valid multiline inclusion paths
! @each[
    document.sections.subsections
]->task@{{value.data=subsections}}*

# Valid multiline execution arguments  
! llm(
    "gpt-4-complex",
    override=true
)
```

#### Continuation Rules
- **Implicit continuation**: No explicit markers needed within bracket/brace/parenthesis pairs
- **Context-aware**: Multiline only within `[]`, `{{}}`, `()` constructs
- **Empty lines allowed**: Within multiline contexts for readability
- **Trailing commas allowed**: `{{value.item=items,}}` ✅
- **Comments allowed**: Within multiline contexts

#### Normalization
Multiline commands are normalized to single-line equivalents before parsing:
- Leading/trailing whitespace stripped from continuation lines
- Empty lines and comments within contexts removed
- Resulting single-line command follows strict whitespace rules

### Examples

#### Valid Commands
```python
"! @each[items]->task@{{value.item=items}}*"                    # Perfect strict spacing
"! @each[document.sections]->task.analyzer@{{value.title=sections.title}}*"  # Complex paths
"! @all->task.process@{{prompt.context=*}}*"                   # @all with wildcard
"!    @each[items]->task@{{value.item=items}}*"               # Spaces after ! allowed
```

#### Invalid Commands  
```python
"! @each [items]->task@{{value.item=items}}*"                  # ❌ Space before [
"! @each[items] ->task@{{value.item=items}}*"                  # ❌ Space after ]
"! @each[items]-> task@{{value.item=items}}*"                  # ❌ Space after ->
"! @ each[items]->task@{{value.item=items}}*"                  # ❌ Space after @
"! @each[items]->task @{{value.item=items}}*"                  # ❌ Space before @{{
"! @each[items]->task@{{value.item=items}} *"                  # ❌ Space after }}
"! @each[document . sections]->task@{{value.item=items}}*"     # ❌ Space around dots
```

### Error Messages
```python
CommandParseError: Space around '.' not allowed - use 'document.sections' not 'document . sections'
CommandParseError: Space around '@' not allowed - use '@each' not '@ each' or ' @each'  
CommandParseError: Space before '[' not allowed - use '@each[items]' not '@each [items]'
CommandParseError: Space after ']' not allowed - use '[items]->' not '[items] ->'
CommandParseError: Space around '->' not allowed - use ']->' not '] ->' or '-> '
CommandParseError: Space before '{{' not allowed - use 'task@{{' not 'task @{{'
CommandParseError: Space after '}}' not allowed - use '}}*' not '}} *'
```

## Semantic Constraints

### Context Dependencies
- `@each` without inclusion requires annotated variable in context
- Scope resolution depends on target context structure
- Path validation depends on available variables

### Type Compatibility

LangTree DSL implements comprehensive type compatibility checking for variable mappings to ensure safe data transformations during prompt assembly.

#### Compatibility Levels

**IDENTICAL**: Same types, no conversion needed
- `str` → `str`, `int` → `int`, `ConfigObject` → `ConfigObject`

**SAFE**: Safe conversions that preserve data integrity
- `int` → `float` (numeric widening)
- `TreeNode` → `dict` (serialization via model_dump())
- Any type → `str` (universal string conversion for COLLECTED_CONTEXT)

**LOSSY**: Potentially lossy conversions requiring explicit opt-in
- `float` → `int` (precision loss possible)

**UNSAFE**: Risky conversions requiring development mode
- Experimental or debugging conversions

**FORBIDDEN**: Never allowed conversions
- Incompatible structural types

#### Validation Strategies

**STRICT** (default): Only IDENTICAL and SAFE conversions allowed
```python
structure = RunStructure()  # Default strategy
```

**PERMISSIVE**: Allows LOSSY conversions with warnings
```python
structure = RunStructure({"validation_strategy": "permissive"})
```

**DEVELOPMENT**: Allows all conversions including UNSAFE
```python
structure = RunStructure({"validation_strategy": "development"})
```

#### String Conversion Rules

**Universal String Compatibility**: All types can convert to string for COLLECTED_CONTEXT assembly
- Primitives: Direct string conversion (`str(value)`)
- Collections: JSON serialization
- TreeNodes: JSON serialization via `model_dump()`

**String Parsing** (configurable): String-to-type conversions
- **JSON**: `str` → `dict`, `list` for valid JSON strings
- **Numeric**: `str` → `int`, `float` for valid numeric strings
- **Boolean**: `str` → `bool` for "true"/"false"/"1"/"0" (case insensitive)

#### Union Type Handling

**Deferred Conversion Strategy**: For union types, conversion attempts are made at runtime
- **Priority-Ordered Resolution**: Union members sorted by conversion priority, not random order
- **First successful conversion is used** after priority sorting
- **Enables flexible type handling** without early commitment

**Union Member Priority Order**:
1. **Exact type matches**: `isinstance(value, member_type)` - no conversion needed
2. **TreeNode types**: Structured data priority over generic dict serialization
3. **Specific numeric types**: `int` for integer values; `bool` only for actual boolean values
4. **General types**: `float`, `bool` (fallback), `str` for broader compatibility
5. **Structured types**: `dict`, `list`, `tuple`, `set`
6. **Remaining types**: All other union members

**Examples**:
- `int | float` with value `42` → tries `int` first (exact match) → `42` not `42.0`
- `bool | float` with value `1` → tries `float` first → `1.0` not `True`
- `bool | float` with value `True` → tries `bool` first (exact match) → `True`

#### Configuration Options

**Factory Pattern**: Type mapping configuration supports multiple input formats
```python
# Dict override
RunStructure({"allow_string_parsing": False})

# TypeMappingConfig instance
config = TypeMappingConfig(validation_strategy=ValidationStrategy.STRICT)
RunStructure(config)

# Default settings
RunStructure()  # Uses default TypeMappingConfig
```

**Available Settings**:
- `validation_strategy`: STRICT | PERMISSIVE | DEVELOPMENT
- `allow_string_parsing`: Enable/disable string-to-type conversions
- `allow_treenode_dict_conversion`: Enable/disable TreeNode ↔ dict conversion
- `strict_bool_parsing`: Restrict bool parsing to "true"/"false"/"1"/"0"

#### Error Messages

Type incompatibility errors provide detailed context:
```
Type incompatibility in variable mapping from task.source.field_name (ConfigComplex)
to task.target.config (dict[str, str]): forbidden conversion not allowed by strict strategy
```

- Source and target field types
- Compatibility level assessment
- Current validation strategy
- Clear remediation guidance

#### Performance Configuration

Type validation can be disabled for performance-critical scenarios:
```python
# Disable type validation entirely
RunStructure({"enable_type_validation": False})

# Custom validation strategy
RunStructure({"validation_strategy": "development", "enable_type_validation": True})
```

**Note**: Disabling validation removes assembly-time type checking. Type conversions still occur during result collection phase.

## Error Handling

### Syntax Errors
- **Missing required elements**: Clear error messages for `!`, `->`, `*`
- **Malformed structure**: Specific errors for brackets, braces, operators
- **Invalid characters**: Report position and expected tokens

### Semantic Errors
- **Scope validation**: Unknown required scopes reported
- **Path validation**: Invalid path structure reported
- **Type mismatches**: Source/target incompatibility reported

### Example Error Messages
```
CommandParseError: Command must start with '!'
CommandParseError: @each commands require '*' multiplicity indicator
CommandParseError: @each commands cannot use wildcard (*) in variable mappings
CommandParseError: Empty variable name in mapping
CommandParseError: Wildcard (*) cannot be used with multiple variable mappings
```

## Tag-Based Data Forwarding

### Overview
LangTree DSL uses tag-based forwarding for data flow between nodes during chain assembly. Variables are mapped to arbitrarily long dotted keys that LangChain passes between chain components.

### Tag Generation
- **Source format**: `node.field` or `scope.path.to.field`
- **Target format**: `destination.node.field` or full tag path
- **Length limits**: No practical limit (tested up to 400+ characters)
- **Characters**: Alphanumeric, dots, underscores supported

### Tag Examples
```python
# Simple tags
"document.title" -> "analysis.document.title"
"sections.content" -> "summary.sections.content"

# Complex nested tags  
"research.methodology.approach" -> "evaluation.research.methodology.approach"
"document.sections.analysis.technical.evaluation" -> "results.document.sections.analysis.technical.evaluation"
```

### Context Assembly Order
Context assembly follows **deterministic ordering** for consistent prompt generation:

1. **Pre-calculated fields**: Forwarded/computed values go first
2. **Definition order**: Fields generate in class definition order  
3. **Sequential forwarding**: For @sequential nodes, prior fields available to subsequent fields
4. **Parallel isolation**: For @parallel nodes, fields generate independently

## Node Execution Modifiers

### @sequential Modifier
Controls field generation order within a node.

**Syntax**: Add to node docstring:
```python
class AnalysisNode:
    """
    ! @sequential
    Generate analysis sections in order
    """
    overview: str
    methodology: str  
    results: str
```

**Behavior**:
- Fields generate in definition order
- Each field gets context from all previously generated fields
- Previously generated fields forwarded as if tagged with `@all`
- Explicit forwarding overrides auto-resolution

**Use cases**:
- Sequential document sections
- Dependent analysis steps
- Iterative refinement processes

### @parallel Modifier  
**Default execution mode** - fields generate simultaneously.

**Syntax**: Add to node docstring:
```python
class SummaryNode:
    """
    ! @parallel  
    Generate summary sections simultaneously
    """
    key_points: str
    conclusions: str
    recommendations: str
```

**Behavior**:
- All fields generate simultaneously  
- No automatic forwarding between fields
- Only explicit forwarding applies
- Pre-calculated/forwarded fields available to all

**Use cases**:
- Independent content sections
- Parallel analysis branches
- Performance optimization

### Context Flow Rules

#### Context Assembly Priority Order
When building context for a field, values are included in this priority order:

1. **External forwarded fields**: Data from other nodes/chains (highest priority)
2. **Pre-calculated fields**: Fields with default values in current node  
3. **Previously generated fields**: Fields generated earlier in current node (for @sequential)
4. **Future fields with known values**: Fields defined later but already computed by other chains

**Key Insight**: Any field with a known value appears in context **before** the currently processing field, regardless of definition order.

#### Sequential Node Context Assembly
```python
class DocumentNode:
    """! @sequential"""
    title: str = "Research Analysis"  # Pre-calculated - available to all fields
    
    # If 'conclusion' was forwarded from another chain, it appears BEFORE introduction
    introduction: str  # Context: [title] + [conclusion if available]
    methodology: str   # Context: [title] + [conclusion if available] + [introduction]  
    results: str       # Context: [title] + [conclusion if available] + [introduction] + [methodology]
    
    # Even though defined last, if generated by external chain, appears early in context
    conclusion: str = Field(description="! @->external@{{value.conclusion=data}}")
```

#### Parallel Node Context Assembly
```python
class SummaryNode:
    """! @parallel"""
    # All fields get same context: [external forwarded] + [pre-calculated] + [explicit forwarded]
    overview: str        # Context: [external_data] + [title] + [analysis_results]
    details: str         # Context: [external_data] + [title] + [analysis_results] (same as overview)
    recommendations: str # Context: [external_data] + [title] + [analysis_results] (same as overview)
    
    title: str = "Summary Report"  # Pre-calculated - available to all parallel fields
```

#### External Forwarding Override
```python
# External forwarding takes precedence over automatic sequential forwarding
! @each[sections]->summary.node@{{prompt.section_titles=sections.title}}*

class SummaryNode:
    """! @sequential"""  
    overview: str        # Context: [section_titles] (external) - no auto-forwarding from other fields
    details: str         # Context: [section_titles] (external) + [overview] (sequential)
```

#### Cross-Node Dependencies
```python
# Manual forwarding between subtrees
! @->conclusions@{{prompt.analysis_results=analysis.results}}

class ConclusionsNode:
    """! @parallel"""    # No auto-resolution
    summary: str         # Gets only analysis_results  
    recommendations: str # Gets only analysis_results
```

## Language Extensions

### Future Scope Modifiers
The scope system is extensible for additional modifiers:
```
@each[custom.items]->task@{{myScope.result=items}}*
```

### Custom Command Types
Grammar allows for future command extensions:
```
! @future_command[params]->destination@{{mappings}}
```

### Advanced Path Features
Support for future path enhancements:
```
scope.path[index].subpath
scope.path{filter}.subpath
```

## Implementation Notes

### Parser Requirements
- Regex-based parsing for performance
- Incremental validation for early error detection
- Immutable result structures for thread safety
- Comprehensive error reporting with position information

### Runtime Requirements
- Context-aware scope resolution
- Type checking for variable assignments
- Iterator protocol support for `@each` commands
- Lazy evaluation for performance optimization

## Examples Collection

### Basic Usage
```
# Simple 1:1 operation
! @->task.summarize@{{prompt.text=content}}

# Collection processing
! @each[sections]->task.analyze@{{value.title=sections.title}}*

# Multiple outputs
! @all->task.generate@{{prompt.seed=topic}}*
```

### Advanced Patterns
```
# Multiple scopes in one command
! @each[prompt.items]->value.processor@{{outputs.result=task.data}}*

# Deep nesting
! @each[doc.chapters.sections]->task.deep_analysis@{{value.path=doc.chapters.sections.title}}*

# Wildcard forwarding
! @->task.comprehensive@{{prompt.context=*}}
```

### Error Examples
```
# Missing multiplicity
! @each[items]->task@{{value=items}}     # ❌ Missing *

# Invalid wildcard usage  
! @each[items]->task@{{value=*}}*        # ❌ Wildcard with @each

# Empty mapping
! @->task@{{}}                           # ❌ Empty mappings
```

This specification provides the complete formal definition of the Action Chaining Language, enabling consistent implementation and usage across different systems.
