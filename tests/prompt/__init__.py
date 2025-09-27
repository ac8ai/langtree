"""Prompt package test suite.

This package contains tests aligned with the modular layout of
`langtree.prompt`.

Implemented test modules:
 - test_structure.py: Tree construction (PromptTreeNode, RunStructure) + pending target tracking.
 - test_registry.py: Variable & pending target registries (registration, satisfaction, resolution semantics).
 - test_resolution.py: Core context & scope resolution (current/value/outputs/task/global/prompt) + error handling.
 - test_context_resolution.py: Extended Phase 2 scenarios (cross‑tree, deep nesting, forward refs, circular patterns).
 - test_integration.py: End‑to‑end execution summary, validation coverage, wildcard/list navigation, execution plan generation.

Placeholder:
 - test_utils.py: Reserved for future utility-specific tests (currently empty).

Historical notes:
 - Legacy monolithic `test_prompt_structure_enhanced.py` decomposed into focused modules above.
 - Previously referenced `test_scopes.py` / `test_validation.py` are intentionally merged into existing files; references removed.
 - A forthcoming `test_todos.py` (skipped tests) will track unimplemented TODO behaviors.
"""