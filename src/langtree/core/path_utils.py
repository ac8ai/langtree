"""
Common path resolution utilities for the LangTree DSL framework.

This module provides unified path resolution functionality that eliminates
duplication across parsing, execution, and structure modules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langtree.execution.scopes import Scope


class ScopeModifier(Enum):
    """Valid scope modifiers for variable mapping and path resolution."""

    PROMPT = "prompt"
    VALUE = "value"
    OUTPUTS = "outputs"
    TASK = "task"


@dataclass
class PathComponents:
    """Result of splitting a path into its components."""

    first_part: str
    remainder: str
    has_remainder: bool

    @classmethod
    def split_path(cls, path: str) -> "PathComponents":
        """
        Split a path at the first dot separator.

        Params:
            path: Path string to split (e.g., "prompt.sections.title")

        Returns:
            PathComponents with first_part, remainder, and has_remainder flag

        Examples:
            "prompt.sections.title" -> PathComponents("prompt", "sections.title", True)
            "simple_path" -> PathComponents("simple_path", "", False)
        """
        if not path or "." not in path:
            return cls(first_part=path, remainder="", has_remainder=False)

        first_part, remainder = path.split(".", 1)
        return cls(first_part=first_part, remainder=remainder, has_remainder=True)


@dataclass
class ResolvedPath:
    """
    Unified path representation with scope resolution.

    This replaces the duplicate ResolvedPath classes in parsing modules
    and provides consistent path handling across the framework.
    """

    scope_modifier: ScopeModifier | None
    path_remainder: str
    original_path: str
    scope_instance: Optional["Scope"] = None

    def __str__(self) -> str:
        """Return the original path string."""
        return self.original_path

    @property
    def has_scope(self) -> bool:
        """Check if this path has a scope modifier."""
        return self.scope_modifier is not None

    @property
    def scoped_path(self) -> str:
        """Return the path with scope modifier if present."""
        if self.scope_modifier:
            return f"{self.scope_modifier.value}.{self.path_remainder}"
        return self.path_remainder

    def resolve(self, context: dict) -> Any:
        """
        Resolve this path using its scope instance.

        Params:
            context: Resolution context dictionary

        Returns:
            Resolved value if scope is available, None otherwise
        """
        if self.scope_instance:
            return self.scope_instance.resolve(self.path_remainder, context)
        return None

    # Compatibility properties for old parser.py interface
    @property
    def scope(self) -> Optional["Scope"]:
        """Compatibility property for scope instance access."""
        return self.scope_instance

    @scope.setter
    def scope(self, value: Optional["Scope"]) -> None:
        """Compatibility setter for scope instance."""
        self.scope_instance = value

    @property
    def path(self) -> str:
        """Compatibility property for path remainder access."""
        return self.path_remainder

    @path.setter
    def path(self, value: str) -> None:
        """Compatibility setter for path remainder."""
        self.path_remainder = value

    @property
    def original(self) -> str:
        """Compatibility property for original path access."""
        return self.original_path

    @original.setter
    def original(self, value: str) -> None:
        """Compatibility setter for original path."""
        self.original_path = value


class PathResolver:
    """
    Unified path resolution utilities.

    This class consolidates all path resolution logic that was previously
    duplicated across parsing/path_resolver.py, parsing/parser.py, and
    other modules.
    """

    @staticmethod
    def resolve_path(path: str) -> ResolvedPath:
        """
        Resolve scope modifier from a path string.

        Params:
            path: Path string that may start with a scope modifier

        Returns:
            ResolvedPath with scope and remainder separated

        Examples:
            "prompt.sections.title" -> ResolvedPath(PROMPT, "sections.title", "prompt.sections.title")
            "simple_path" -> ResolvedPath(None, "simple_path", "simple_path")
        """
        if not path:
            return ResolvedPath(
                scope_modifier=None, path_remainder="", original_path=path
            )

        components = PathComponents.split_path(path)

        if not components.has_remainder:
            return ResolvedPath(
                scope_modifier=None, path_remainder=path, original_path=path
            )

        try:
            scope = ScopeModifier(components.first_part)
            return ResolvedPath(
                scope_modifier=scope,
                path_remainder=components.remainder,
                original_path=path,
            )
        except ValueError:
            # First part is not a valid scope modifier
            return ResolvedPath(
                scope_modifier=None, path_remainder=path, original_path=path
            )

    @staticmethod
    def resolve_path_with_cwd(path: str, cwd: str) -> ResolvedPath:
        """
        Resolve a path with Command Working Directory (CWD) support.

        Params:
            path: Path string that may be relative or absolute
            cwd: Command Working Directory - the scope where command is defined

        Returns:
            ResolvedPath with scope, CWD resolution, and remainder separated

        Rules:
            - Absolute paths (starting with scope modifier or task.): ignore CWD
            - Relative paths: resolve from CWD
            - CWD is the node tag where the command is defined
        """
        if not path:
            return ResolvedPath(
                scope_modifier=None, path_remainder="", original_path=path
            )

        components = PathComponents.split_path(path)

        if components.has_remainder:
            # Check if it's a scope modifier
            try:
                scope = ScopeModifier(components.first_part)
                # Has scope modifier - it's absolute, ignore CWD
                return ResolvedPath(
                    scope_modifier=scope,
                    path_remainder=components.remainder,
                    original_path=path,
                )
            except ValueError:
                # Not a scope modifier, check if it starts with 'task.'
                if components.first_part == "task":
                    # Absolute path starting with task. - ignore CWD
                    return ResolvedPath(
                        scope_modifier=None, path_remainder=path, original_path=path
                    )

        # No scope modifier and doesn't start with 'task.' - it's relative
        # Apply CWD resolution
        if cwd and path:
            resolved_path = f"{cwd}.{path}"
        else:
            resolved_path = path or cwd or ""

        return ResolvedPath(
            scope_modifier=None, path_remainder=resolved_path, original_path=path
        )

    @staticmethod
    def resolve_path_with_scope_instance(path: str) -> ResolvedPath:
        """
        Resolve a path and create scope instance for immediate use.

        This method combines path resolution with scope instance creation,
        replacing the _resolve_path methods in parsing modules.

        Params:
            path: The path to resolve (e.g., "prompt.title", "value.content")

        Returns:
            ResolvedPath object with scope instance and processed path
        """
        resolved = PathResolver.resolve_path(path)

        # Create scope instance if we have a scope modifier
        if resolved.scope_modifier:
            # Import here to avoid circular imports
            from langtree.execution.scopes import get_scope

            scope_instance = get_scope(resolved.scope_modifier.value)
            resolved.scope_instance = scope_instance
        else:
            # No scope modifier - use current_node scope (implicit scope)
            from langtree.execution.scopes import get_scope

            scope_instance = get_scope("current_node")
            resolved.scope_instance = scope_instance
            # For implicit scope, use full path
            resolved.path_remainder = resolved.original_path

        return resolved

    @staticmethod
    def is_valid_scope(scope_name: str) -> bool:
        """
        Check if a string is a valid scope modifier.

        Params:
            scope_name: String to check

        Returns:
            True if scope_name is a valid scope modifier
        """
        try:
            ScopeModifier(scope_name)
            return True
        except ValueError:
            return False

    @staticmethod
    def split_path_components(path: str) -> list[str]:
        """
        Split a path into all its components.

        Params:
            path: Path to split (e.g., "task.analysis.sections.title")

        Returns:
            List of path components

        Examples:
            "task.analysis.sections.title" -> ["task", "analysis", "sections", "title"]
            "simple" -> ["simple"]
        """
        if not path:
            return []
        return path.split(".")

    @staticmethod
    def resolve_variable_mapping_with_cwd(
        target: str, source: str, cwd: str
    ) -> "VariableMapping":
        """
        Resolve both target and source paths in a variable mapping with CWD support.

        Params:
            target: Target path string
            source: Source path string
            cwd: Command Working Directory

        Returns:
            VariableMapping with CWD-resolved paths
        """
        resolved_target = PathResolver.resolve_path_with_cwd(target, cwd)
        resolved_source = PathResolver.resolve_path_with_cwd(source, cwd)

        return VariableMapping(
            target_path=target,
            source_path=source,
            resolved_target=resolved_target,
            resolved_source=resolved_source,
            original_target=target,
            original_source=source,
        )


@dataclass
class VariableMapping:
    """
    Enhanced variable mapping with unified path resolution.

    This replaces duplicate VariableMapping and EnhancedVariableMapping
    classes across modules.
    """

    target_path: str
    source_path: str
    resolved_target: ResolvedPath
    resolved_source: ResolvedPath
    original_target: str
    original_source: str

    @classmethod
    def create(cls, target: str, source: str) -> "VariableMapping":
        """
        Create a VariableMapping with automatic path resolution.

        Params:
            target: Target path string
            source: Source path string

        Returns:
            VariableMapping with paths resolved
        """
        resolved_target = PathResolver.resolve_path_with_scope_instance(target)
        resolved_source = PathResolver.resolve_path_with_scope_instance(source)

        return cls(
            target_path=target,
            source_path=source,
            resolved_target=resolved_target,
            resolved_source=resolved_source,
            original_target=target,
            original_source=source,
        )

    @classmethod
    def create_with_cwd(cls, target: str, source: str, cwd: str) -> "VariableMapping":
        """
        Create a VariableMapping with CWD-aware path resolution.

        Params:
            target: Target path string
            source: Source path string
            cwd: Command Working Directory

        Returns:
            VariableMapping with CWD-resolved paths
        """
        resolved_target = PathResolver.resolve_path_with_cwd(target, cwd)
        resolved_source = PathResolver.resolve_path_with_cwd(source, cwd)

        return cls(
            target_path=target,
            source_path=source,
            resolved_target=resolved_target,
            resolved_source=resolved_source,
            original_target=target,
            original_source=source,
        )


def validate_path_format(path: str, path_type: str = "path") -> None:
    """
    Validate basic path format requirements.

    Params:
        path: Path string to validate
        path_type: Type description for error messages

    Raises:
        ValueError: If path format is invalid
    """
    if not path or not isinstance(path, str):
        raise ValueError(f"{path_type} must be a non-empty string")

    if path.strip() != path:
        raise ValueError(f"{path_type} must not have leading or trailing whitespace")
