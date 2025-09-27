# DESIGN EXPLORATION - WORKING SKETCHPAD

> **⚠️ IMPORTANT: This document contains EXPLORATORY IDEAS ONLY**  
> **This is NOT authoritative documentation. Do not treat these ideas as specifications.**  
> **For authoritative framework documentation, see COMPREHENSIVE_GUIDE.md**

## Purpose
This document serves as a working sketchpad for brainstorming and exploring potential DPCL framework enhancements. Ideas here are:
- **Experimental** - may not be implemented
- **Unvalidated** - may conflict with actual implementation
- **Evolving** - subject to change or abandonment
- **Research-oriented** - for exploring possibilities

For actual framework usage, implementation details, and authoritative specifications, refer to `COMPREHENSIVE_GUIDE.md`.

---

## Exploratory Goal Statement

Transform a declarative hierarchical prompt/command specification (DPCL annotated `PromptTreeNode` classes) into a deterministic, dependency-ordered DAG of LLM chains that: (a) automatically assembles contextual prompts from cleaned structural sources, (b) propagates and merges intermediate structured outputs via declared variable mappings, (c) expands multiplicity directives (`@each`, `*`) into parallel/iterative executions, (d) tracks provenance & dependency state for every declared variable, and (e) produces a final aggregated structured artifact plus a transparent execution + dataflow report.

**Note**: These are exploratory success criteria, not committed features.

Out-of-Scope (Current Phase): Token budgeting, streaming partial execution UI, advanced error recovery policies.

---

## Implementation Status Update (September 2025)

### ✅ **Completed Core Implementation**

#### 1. Command Parser System (86% Coverage)
- **Whitespace Validation**: Strict spacing rules enforced per LANGUAGE_SPECIFICATION.md
- **Command Parsing**: Full DPCL command syntax support (`@each`, `@all`, variable mappings)
- **Error Handling**: Comprehensive parse-time validation with detailed error messages
- **Unicode Support**: Proper handling of Unicode characters in quoted strings
- **Test Coverage**: 367 passing parser tests with edge case validation

#### 2. Runtime Variable System (ARCHITECTURALLY INTEGRATED)
- **Correct Syntax**: Runtime variables use `{var}` syntax (not `{{var}}`)
- **Template Variables**: `{PROMPT_SUBTREE}` and `{COLLECTED_CONTEXT}` reserved and functional
- **Scope Resolution**: Multi-level context resolution (node → parent → root)
- **Assembly Separation**: Assembly variables properly isolated from runtime context
- **Architectural Integration**: Now uses existing resolve_runtime_variables() pipeline instead of regex scanning
- **Proper Content Processing**: Uses node.clean_docstring and clean_field_descriptions (already processed)
- **Documentation Updated**: LANGUAGE_SPECIFICATION.md and VARIABLE_TAXONOMY.md corrected

#### 3. Template Variable Integration (92% Coverage)
- **DPCL Integration**: Template variables work with extracted commands
- **String Representation**: Command objects properly serializable for tests
- **Field Resolution**: Field names converted to proper titles (rich_input → "Rich Input")
- **Error Boundaries**: Template processing failures don't break command extraction

#### 4. LangChain Integration Layer (77% Coverage)
- **Chain Building**: Most integration tests passing (56 passed, 8 failed)
- **Prompt Assembly**: Multi-section prompt generation (system, context, task, output, input)
- **Execution Planning**: Topological dependency ordering functional
- **Field Descriptions**: Output section generation with proper field documentation

### 📊 **Test Results Progress**
- **Current Status**: 483 passing, 33 failing, 83 skipped (September 2025 update)
- **Key Achievement**: Fixed template variable vs DPCL syntax conflicts with 2-step processing
- **Recent Progress**: Command extraction fixes reduced failures from 37 to 33 tests
- **Architecture Clarification**: Separated assembly-time vs runtime variable systems
- **Critical Gap**: Context resolution and integration remain primary blockers

### 🚧 **Remaining Work**
- **Edge Case Refinement**: Some integration test edge cases need adjustment
- **Performance Testing**: Large tree execution validation (currently skipped)
- **Advanced Features**: 91 skipped tests represent planned future enhancements

### 📋 **Remaining Exploratory Items**
The following sections contain exploratory ideas that remain unimplemented and may be superseded by actual implementation decisions.

---
## 1. Current Chain Output Semantics (`llm/chains.py`)

### 1.1 `prepare_chain`
Constructs `ChatPromptTemplate` with two messages:
- System: `prompt_system`
- Human: Composite block (# Context / # Task / # Output / model fields description / # Input / optional Samples / optional signature placeholder)

Returns a Runnable pipeline:
- Template -> LLM (optionally bound with an `output_parser=structured_output` – currently expects a parser object, but code passes a `BaseModel` which may require adapting to LangChain’s `PydanticOutputParser`).
- Optional final `StrOutputParser` if `parse_as_string=True` (mutually exclusive with structured output).

### 1.2 Execution Inputs
The human template contains placeholders that must be satisfied by an input dict (e.g., `document`, `signature`, `sample_i`, etc.). Chain invocation returns:
- If structured_output bound correctly: parsed structure (intended typed object).
- Else: raw model output (string or message) or string if final parser appended.

### 1.3 `sample_chain`
Builds N sibling executions of an input `chain` with UUID signatures, injects those as `{sample_i}` variables into a higher‑level `sampling_chain` (which is itself another `prepare_chain`). Final output (optionally `StrOutputParser`) is a synthesized re‑prompt summarizing samples. No orchestration with prompt tree metadata yet.

### 1.4 Multi‑Tree / Prompt Tree Integration – Current Gap
There is no direct consumption of `RunStructure` / command graph here:
- No automated assembly of composite prompt from hierarchical clean docstrings.
- No variable dependency tracking feeding structured outputs into downstream chain inputs.
- No expansion logic for multiplicity (`@each`, `*`).

Result: Present chains are flat, manually orchestrated, not yet reflective of the DPCL execution graph.

---
## 2. ✅ Template Variable System - IMPLEMENTED

> **✅ IMPLEMENTATION COMPLETE**: The template variable system has been fully implemented and tested. This section documents the achieved functionality.

### 2.1 Implemented Features
- **Template Variables**: `{PROMPT_SUBTREE}` and `{COLLECTED_CONTEXT}` fully functional
- **Spacing Validation**: Strict empty line requirements with detailed error reporting
- **Variable Detection**: Robust parsing and position tracking
- **Integration**: Works with DPCL command syntax and StructureTreeNode hierarchy
- **Error Handling**: Comprehensive malformed syntax and spacing violation detection

### 2.2 Template Variable Resolution
**Implemented Algorithm**:
1. **Detection**: `detect_template_variables()` finds all template variables and their positions
2. **Validation**: `validate_template_variable_spacing()` enforces empty line requirements
3. **Resolution**: 
   - `{PROMPT_SUBTREE}` → Resolves to field descriptions as markdown headers
   - `{COLLECTED_CONTEXT}` → Placeholder for context assembly (future implementation)
4. **Processing**: `process_template_variables()` orchestrates detection, validation, and resolution

### 2.3 Variable Separation (Assembly vs Template)
**Enforced Rules**:
- **Template Variables**: `{VARIABLE_NAME}` - Resolved at assembly time
- **Assembly Variables**: `! variable_name=value` - Parse-time configuration, different resolution scope
- **Conflict Detection**: Validates no naming conflicts between variable types

### 2.4 Test Coverage Achievement
**Comprehensive Testing** (94% coverage):
- **Unit Tests**: All core functions (`detect_template_variables`, `process_template_variables`, etc.)
- **Edge Cases**: Empty content, malformed syntax, spacing violations, multiple variables
- **Integration**: DPCL command integration, StructureTreeNode hierarchy, error handling
- **Architecture**: Variable separation, fail-fast validation, specification compliance

### 2.5 Testing Strategy Implemented
- ✅ **Unit**: `detect_template_variables()` tested with all edge cases and position accuracy
- ✅ **Unit**: Spacing validation with comprehensive violation detection
- ✅ **Integration**: Template variable resolution with real StructureTreeNode instances  
- ✅ **Property**: Deterministic position calculation and error reporting
- ✅ **Contract**: Variable separation enforcement and conflict detection

---
## 3. Subtree + Template Merge Sketch

Goal: Convert a node path (e.g., `task.analyze_comparison.main_analysis`) into a finalized prompt segment by layering contextual texts.

### 3.1 Inputs
- `StructureTreeNode` target.
- Ancestor chain of nodes (MRO + field descriptions).
- Cleaned docstrings (`clean_docstring`), cleaned field descriptions.
- Variable substitution environment (resolved runtime variables from scope resolvers).

### 3.2 Assembly Algorithm
1. Ascend to root collecting (top→down): for each ancestor class (excluding `PromptTreeNode`): its cleaned docstring if non‑empty.
2. For each edge (parent -> child) incorporate the child field description that leads toward target.
3. Append target node’s own cleaned docstring (if used as a prompt boundary) and any directly addressed field description (if a field prompt).
4. Resolve inline placeholders `{var}` via `string.Formatter` backed by a scoped resolver:
   - Attempt order: prompt scope > outputs > value > task/global > current_node.
   - On failure: emit sentinel or raise (configurable).
5. Concatenate with double newline separators; trim redundant blank lines.
6. Optionally apply command filters to inject / transform segments (e.g., `@each` contexts). For multiplicity nodes, the assembly becomes a template repeated per iteration value.

### 3.3 Example (Analyze Comparison `main_analysis`)
Segments:
- Ancestor docstring: `TaskProcessor` (intro + style)
- Field description for `main_analysis` (contains `@each` command + explanation)
- Subsection docstrings for `SectionProcessor` etc. become nested iteration templates.

### 3.4 Testing Strategy (Sketch)
- Unit: `assemble_prompt(node_tag)` returns deterministic ordered list of segments before join.
- Unit: Placeholder resolution – feed mock runtime store with values; assert substitution.
- Param: Multiplicity expansion – simulate list length 0/1/N for `@each` and verify rendered count.
- Property: Idempotency – running assembly twice without runtime changes yields same string.
- Contract: For every extracted command, verify referenced node tag appears in at least one assembled prompt (ensures no lost context).

---
## 4. Large Chain Integration Plan

### 4.1 Task Graph Extraction
Root tasks / conceptual nodes:
- Summaries: `TaskDocumentSummarizer`, `TaskContentSummarizer` (produce prompt variables: `source_data`, `target_data` summaries).
- Comparative Analyses: `TaskDocumentProcessor`, `TaskComparisonProcessor`, `TaskInsightExtractor` (depend on base documents or summaries; commands reference a future `task.output_aggregator`).
- Aggregation: `TaskOutputAggregator` consumes outputs of prior tasks (placeholders: `{task_processor}`, `{task_comparison}`, `{task_insights}`).

### 4.2 Command‑Derived Edges (From Inline Commands)
Examples:
- `! @->task.summarize_analysis@{{prompt.task_analyze_comparison=*}}` (ComparisonAnalysis contributes a prompt field to summarization task).
- `! @all->summary@{{outputs.main_analysis}}*` (Fan‑in mapping from all `main_analysis` sections to `summary`).
- `! @each[main_analysis]->summary@{{outputs.main_analysis=main_analysis}}*` (Iterative expansion per section into summary rows).

### 4.3 Proposed Execution Ordering (Heuristic)
1. Summarize raw documents (if those tasks exist; if not, treat raw inputs as external).
2. Parallel: Compare Data / Analyze Differences / Generate Insights (share base inputs).
3. Aggregate: Summarize Analysis.
4. Downstream consumer chains (not yet defined) – e.g., presentation, scoring.

### 4.4 Planning Pipeline
```python
rs = RunStructure()
rs.add(TaskDocumentSummarizer)
rs.add(TaskContentSummarizer)
rs.add(TaskDocumentProcessor)
rs.add(TaskComparisonProcessor)
rs.add(TaskInsightExtractor)
rs.add(TaskOutputAggregator)

# Phase 2 (future): rs.resolve_deferred_contexts()
plan = rs.get_execution_plan()  # augmented with dependency resolution in future
```

Augment `plan` with DAG edges derived from:
- Destination targets of commands (edge: source_node -> destination.node_path root).
- Variable mappings referencing outputs of another node (edge: producer -> consumer).

### 4.5 Chain Construction Layer
For each `chain_step` in a topologically sorted list:
1. Assemble prompt text via subtree merge algorithm.
2. Determine structured output model (the node’s Pydantic class) if the node is a leaf “execution” node.
3. Build LangChain Runnable: `prompt_template | model | (optional parser)`.
4. Wrap with a post‑processor that:
   - Stores raw + parsed output into runtime `outputs` store.
   - Updates variable dependency tracking for any variables satisfied by this node.

Multiplicity Handling:
- If command type `each` with multiplicity target list: map step expands into `RunnableMap` or `RunnableSequence` producing list outputs aggregated or fanned into destination variables.

### 4.6 Sample Pseudocode
```python
def build_runtime_chains(rs: RunStructure, model_provider):
    dag = derive_dag(rs)
    chains = {}
    for node_tag in topo_sort(dag):
        node = rs.get_node(node_tag)
        prompt_segments = assemble_prompt(rs, node_tag)
        template = ChatPromptTemplate.from_template('\n\n'.join(prompt_segments))
        llm = model_provider.for_node(node_tag)
        parser = pydantic_parser_for(node.field_type) if node.field_type else StrOutputParser()
        chain = template | llm | parser | RunnableLambda(partial(store_outputs, rs=rs, node_tag=node_tag))
        chains[node_tag] = expand_if_multiplicity(chain, node)
    return compose_final(chains, dag)
```

### 4.7 Post‑Execution Validation
After each node execution:
- Attempt dependency resolution on variables referencing this node's outputs.
- If all required variables for a downstream node are satisfied, that node becomes runnable (lazy scheduling possible).
- At end: produce `dependency_report` (satisfied vs unresolved with reasons) + final aggregated structured responses.

### 4.8 Open Design Decisions
| Topic | Options | Preferred (Tentative) |
|-------|---------|------------------------|
| Fan‑in merging | First, reduce(list), policy map | Policy map (configurable) |
| Wildcard handling | External injection, whole object snapshot | Whole object snapshot + provenance tag |
| Prompt caching | Always rebuild, cache by hash | Hash segments for deterministic caching |
| Error propagation | Fail-fast, accumulate | Accumulate with severity tiers |

---
## 5. Execution Plan v2 (Draft Fields)
```json
{
  "nodes": [
    {"tag": "task.document_summarizer", "kind": "llm", "outputs": ["summary"], "deps": []},
    {"tag": "task.content_summarizer", "kind": "llm", "outputs": ["summary"], "deps": []},
    {"tag": "task.comparer", "kind": "llm", "deps": ["task.document_summarizer", "task.content_summarizer"], "fanout": "main_analysis[*]"},
    {"tag": "task.comparison_processor", "kind": "llm", "deps": ["task.document_summarizer", "task.content_summarizer"]},
    {"tag": "task.insight_extractor", "kind": "llm", "deps": ["task.document_summarizer", "task.content_summarizer"]},
    {"tag": "task.output_aggregator", "kind": "llm", "deps": ["task.comparer", "task.comparison_processor", "task.insight_extractor"], "aggregate": true}
  ],
  "variables": {
    "outputs.main_analysis": {"status": "pending", "sources": ["main_analysis"], "relationship": "n:n"}
  }
}
```

---
## 6. Implementation Progress & Next Steps

### ✅ **Completed (September 2025)**
1. ✅ **Template Variable System**: Comprehensive implementation with 94% test coverage
   - `{PROMPT_SUBTREE}` and `{COLLECTED_CONTEXT}` resolution
   - Spacing validation and error handling
   - Integration with StructureTreeNode hierarchy
   - Variable separation enforcement (Assembly vs Template)

2. ✅ **Test Infrastructure**: Robust testing framework
   - 483 passing tests across the codebase (80% pass rate)
   - 83 skipped tests for planned features
   - Comprehensive error handling and edge case coverage
   - **Recent Progress**: Template variable vs DPCL parsing conflicts resolved

3. ✅ **Variable System Architecture**: Five-type variable system defined and partially implemented
   - Assembly Variables, Runtime Variables, DPCL Variable Targets, Scope Context Variables, Field References
   - Clear separation between parse-time and runtime resolution

### 🚧 **In Progress**
4. ✅ **Command Parser Integration**: Template variable vs DPCL syntax conflicts resolved
5. **Context Resolution**: Address remaining 33 failing tests focusing on path resolution and integration
6. ✅ **Documentation Updates**: COMPREHENSIVE_GUIDE.md and DESIGN_EXPLORATION.md updated with current status

### 📋 **Future Exploratory Work** 
7. Implement chain assembly utilities to convert validated structures to LangChain chains
8. Enhance context resolution functions for complex reference validation during assembly
9. Add `assemble_prompt` utility leveraging cleaned docstrings & descriptions
10. Add `pydantic_parser_for` factory (wrap LangChain `PydanticOutputParser`)
11. Build DAG derivation from commands + registry for chain assembly ordering
12. Implement variable dependency engine for assembly-time validation
13. Integrate chains builder using plan ordering & multiplicity expansion

**Note**: Items 7-13 remain exploratory and may be superseded by implementation decisions.

---
## 7. Current Implementation Risks & Considerations
- **Test Quality vs Coverage**: High coverage achieved but integration depth needs improvement
- **Specification Alignment**: Template variables implemented, but broader DPCL integration still developing
- **Architecture Validation**: Variable separation enforced, but dependency DAG construction remains exploratory
- **Documentation Sync**: Need to ensure all docs reflect current implementation status vs exploratory ideas

---
(End of sketchpad – iterate here before modifying formal design docs.)
