"""
DPCL Prompt Package

This package provides modular components for building and managing
Dynamic Prompt Connecting Language (DPCL) prompt tree structures.

Public API exports maintain compatibility with the original prompt_structure.py module.
"""

# Core structure classes
# Registry classes for variable and target management
from langtree.prompt.registry import (
    PendingTarget,
    PendingTargetRegistry,
    VariableInfo,
    VariableRegistry,
)

# Resolution functions
from langtree.prompt.resolution import resolve_deferred_contexts

# Scope classes for context resolution
from langtree.prompt.scopes import (
    ContextResolver,
    ContextResolverFactory,
    CurrentNodeResolver,
    CurrentNodeScope,
    GlobalTreeResolver,
    OutputsContextResolver,
    OutputsScope,
    PromptContextResolver,
    PromptScope,
    Scope,
    TargetNodeResolver,
    TaskScope,
    ValueContextResolver,
    ValueScope,
    get_scope,
)
from langtree.prompt.structure import (
    PromptTreeNode,
    PromptValue,
    ResolutionResult,
    RunStructure,
    StructureTreeNode,
    StructureTreeRoot,
)

# Utility functions
from langtree.prompt.utils import extract_commands, get_root_tag

# Validation functions
from langtree.prompt.validation import validate_comprehensive, validate_tree

# Make the main classes available at package level
__all__ = [
    # Core classes
    "PromptTreeNode",
    "StructureTreeNode",
    "StructureTreeRoot",
    "RunStructure",
    "PromptValue",
    "ResolutionResult",
    # Scope classes
    "Scope",
    "PromptScope",
    "ValueScope",
    "OutputsScope",
    "TaskScope",
    "CurrentNodeScope",
    "get_scope",
    "ContextResolver",
    "ContextResolverFactory",
    "CurrentNodeResolver",
    "ValueContextResolver",
    "OutputsContextResolver",
    "GlobalTreeResolver",
    "TargetNodeResolver",
    "PromptContextResolver",
    # Registry classes
    "VariableInfo",
    "VariableRegistry",
    "PendingTarget",
    "PendingTargetRegistry",
    # Validation functions
    "validate_tree",
    "validate_comprehensive",
    # Resolution functions
    "resolve_deferred_contexts",
    # Utility functions
    "extract_commands",
    "get_root_tag",
]
