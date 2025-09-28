"""
LangTree DSL execution components.

This package provides runtime execution, context resolution, and LangChain
integration for the LangTree DSL framework.
"""

from langtree.execution.integration import LangTreeChainBuilder
from langtree.execution.resolution import resolve_deferred_contexts
from langtree.execution.scopes import (
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

__all__ = [
    "LangTreeChainBuilder",
    "resolve_deferred_contexts",
    "ContextResolver",
    "ContextResolverFactory",
    "CurrentNodeResolver",
    "CurrentNodeScope",
    "GlobalTreeResolver",
    "OutputsContextResolver",
    "OutputsScope",
    "PromptContextResolver",
    "PromptScope",
    "Scope",
    "TargetNodeResolver",
    "TaskScope",
    "ValueContextResolver",
    "ValueScope",
    "get_scope",
]
