"""
Scope classes for LangTree DSL context resolution and variable management.

This module contains all scope classes that define how variables and contexts
are resolved within the LangTree DSL prompt tree execution system.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langtree.prompt.structure import StructureTreeNode


class Scope(ABC):
    """Abstract base class for scope objects."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        pass

    @abstractmethod
    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        pass

    @abstractmethod
    def resolve(self, path: str, context: dict):
        """Resolve a path within this scope's context."""
        pass


class PromptScope(Scope):
    """Scope for accessing prompt fields within the current context."""

    def __init__(self, node_tag: str | None = None):
        """
        Initialize PromptScope for prompt field access.

        Creates a scope that enables access to prompt fields within the
        current execution context, optionally targeting a specific node.

        Params:
            node_tag: Optional tag to identify the specific prompt node context
        """
        self.node_tag = node_tag

    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        return "prompt"

    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        if self.node_tag:
            return f"prompt[{self.node_tag}].{path}"
        else:
            return f"prompt.{path}"

    def resolve(self, path: str, context: dict):
        """Resolve path in current node's prompt context."""
        run_structure = context["run_structure"]
        node_tag = context["node_tag"]
        return run_structure._resolve_in_current_prompt_context(path, node_tag)


class ValueScope(Scope):
    """Scope for accessing value fields from tree nodes."""

    def __init__(self, root_tag: str | None = None):
        """
        Initialize ValueScope for accessing value fields from tree nodes.

        Creates a scope that enables access to value fields and data attributes
        from nodes within the prompt tree structure.

        Params:
            root_tag: Optional root tag to identify the base context for value resolution
        """
        self.root_tag = root_tag

    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        return "value"

    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        if self.root_tag:
            return f"value[{self.root_tag}].{path}"
        else:
            return f"value.{path}"

    def resolve(self, path: str, context: dict):
        """Resolve path in current node's value context."""
        run_structure = context["run_structure"]
        node_tag = context["node_tag"]
        return run_structure._resolve_in_value_context(path, node_tag)


class OutputsScope(Scope):
    """Scope for accessing outputs from executed nodes."""

    def __init__(self, root_tag: str | None = None):
        """Initialize OutputsScope.

        Params:
            root_tag: Optional root tag to identify the base context
        """
        self.root_tag = root_tag

    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        return "outputs"

    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        if self.root_tag:
            return f"outputs[{self.root_tag}].{path}"
        else:
            return f"outputs.{path}"

    def resolve(self, path: str, context: dict):
        """Resolve path in current node's outputs context."""
        run_structure = context["run_structure"]
        node_tag = context["node_tag"]
        return run_structure._resolve_in_outputs_context(path, node_tag)


class TaskScope(Scope):
    """Scope for accessing task-specific context and data."""

    def __init__(self, task_name: str | None = None):
        """Initialize TaskScope.

        Params:
            task_name: Optional task name to identify the specific task
        """
        self.task_name = task_name

    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        return "task"

    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        if self.task_name:
            return f"task[{self.task_name}].{path}"
        else:
            return f"task.{path}"

    def resolve(self, path: str, context: dict):
        """Resolve path in global/task tree context."""
        run_structure = context["run_structure"]
        # For task scope, we need to reconstruct the full task path
        full_task_path = f"task.{path}" if path else "task"
        return run_structure._resolve_in_global_tree_context(full_task_path)


class CurrentNodeScope(Scope):
    """Scope for accessing the current node's context and data."""

    def get_name(self) -> str:
        """Get the scope name for display purposes."""
        return "current_node"

    def get_full_path(self, path: str) -> str:
        """Get the full path string for this scope with the given path."""
        return f"current_node.{path}"

    def resolve(self, path: str, context: dict):
        """Resolve path in current node context."""
        run_structure = context["run_structure"]
        node_tag = context["node_tag"]
        return run_structure._resolve_in_current_node_context(path, node_tag)


# Global scope singletons
_SCOPES = {
    "prompt": PromptScope(),
    "value": ValueScope(),
    "outputs": OutputsScope(),
    "task": TaskScope(),
    "current_node": CurrentNodeScope(),
}


def get_scope(scope_name: str, **kwargs) -> Scope:
    """Factory function to create scope objects.

    Params:
        scope_name: The name of the scope to create
        **kwargs: Additional arguments for scope initialization (ignored for singletons)

    Returns:
        An instance of the appropriate scope class

    Raises:
        ValueError: If scope_name is not recognized
    """
    scope = _SCOPES.get(scope_name)
    if scope is None:
        raise ValueError(
            f"Unknown scope: {scope_name}. Available: {list(_SCOPES.keys())}"
        )
    return scope


class ContextResolver(ABC):
    """Abstract base class for context resolvers."""

    @abstractmethod
    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve a context path to its full representation."""
        pass


class CurrentNodeResolver(ContextResolver):
    """Resolver for current_node context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve current_node context path."""
        return f"current_node.{path}"


class ValueContextResolver(ContextResolver):
    """Resolver for value context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve value context path."""
        return f"value.{path}"


class OutputsContextResolver(ContextResolver):
    """Resolver for outputs context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve outputs context path."""
        return f"outputs.{path}"


class GlobalTreeResolver(ContextResolver):
    """Resolver for global tree context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve global tree context path."""
        return f"global.{path}"


class TargetNodeResolver(ContextResolver):
    """Resolver for target_node context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve target_node context path."""
        if target_node:
            return f"target_node[{target_node.name}].{path}"
        else:
            return f"target_node[unknown].{path}"


class PromptContextResolver(ContextResolver):
    """Resolver for prompt context references."""

    def resolve(
        self,
        path: str,
        scope: Optional["StructureTreeNode"] = None,
        target_node: Optional["StructureTreeNode"] = None,
    ) -> str:
        """Resolve prompt context path."""
        if scope and hasattr(scope, "name"):
            node_tag = scope.name
        else:
            node_tag = None

        if node_tag:
            # Current prompt context
            return f"current_prompt[{node_tag}].{path}"
        elif target_node:
            # Target prompt context
            return f"target_prompt[{target_node.name}].{path}"
        else:
            return f"prompt_context[unknown].{path}"


class ContextResolverFactory:
    """Factory for creating appropriate context resolvers."""

    _resolvers = {
        "current_node": CurrentNodeResolver(),
        "value": ValueContextResolver(),
        "outputs": OutputsContextResolver(),
        "global_tree": GlobalTreeResolver(),
        "target_node": TargetNodeResolver(),
        "prompt": PromptContextResolver(),
    }

    @classmethod
    def get_resolver(cls, context_type: str) -> ContextResolver:
        """Get the appropriate resolver for the context type."""
        if context_type not in cls._resolvers:
            raise ValueError(
                f"Unknown context type: {context_type}. Available: {list(cls._resolvers.keys())}"
            )
        return cls._resolvers[context_type]
