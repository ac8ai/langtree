"""
DPCL Prompt Package

This package provides modular components for building and managing 
Dynamic Prompt Connecting Language (DPCL) prompt tree structures.

Public API exports maintain compatibility with the original prompt_structure.py module.
"""

# Core structure classes
from langtree.prompt.structure import (
    PromptTreeNode,
    StructureTreeNode, 
    StructureTreeRoot,
    RunStructure,
    PromptValue,
    ResolutionResult
)

# Scope classes for context resolution
from langtree.prompt.scopes import (
    Scope,
    PromptScope,
    ValueScope,
    OutputsScope,
    TaskScope,
    CurrentNodeScope,
    get_scope,
    ContextResolver,
    ContextResolverFactory,
    CurrentNodeResolver,
    ValueContextResolver,
    OutputsContextResolver,
    GlobalTreeResolver,
    TargetNodeResolver,
    PromptContextResolver
)

# Registry classes for variable and target management
from langtree.prompt.registry import (
    VariableInfo,
    VariableRegistry,
    PendingTarget,
    PendingTargetRegistry
)

# Validation functions
from langtree.prompt.validation import (
    validate_tree,
    validate_comprehensive
)

# Resolution functions
from langtree.prompt.resolution import (
    resolve_deferred_contexts
)

# Utility functions
from langtree.prompt.utils import (
    extract_commands,
    get_root_tag
)

# Make the main classes available at package level
__all__ = [
    # Core classes
    'PromptTreeNode',
    'StructureTreeNode', 
    'StructureTreeRoot',
    'RunStructure',
    'PromptValue',
    'ResolutionResult',
    
    # Scope classes
    'Scope',
    'PromptScope',
    'ValueScope',
    'OutputsScope',
    'TaskScope',
    'CurrentNodeScope',
    'get_scope',
    'ContextResolver',
    'ContextResolverFactory',
    'CurrentNodeResolver',
    'ValueContextResolver',
    'OutputsContextResolver',
    'GlobalTreeResolver',
    'TargetNodeResolver',
    'PromptContextResolver',
    
    # Registry classes
    'VariableInfo',
    'VariableRegistry',
    'PendingTarget',
    'PendingTargetRegistry',
    
    # Validation functions
    'validate_tree',
    'validate_comprehensive',
    
    # Resolution functions
    'resolve_deferred_contexts',
    
    # Utility functions
    'extract_commands',
    'get_root_tag'
]