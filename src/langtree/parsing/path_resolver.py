"""
Enhanced path resolution for Action Chaining Language.

This module provides scope analysis utilities and maintains compatibility
with existing analysis functions. Core path resolution functionality has
been moved to langtree.core.path_utils.
"""

from dataclasses import dataclass

from langtree.core.path_utils import (
    PathResolver,
    ResolvedPath,
    VariableMapping,
)


@dataclass
class EnhancedVariableMapping:
    """
    Enhanced variable mapping with full path resolution.

    This is a compatibility class that wraps the core VariableMapping
    for use in analysis functions.
    """

    target_path: ResolvedPath
    source_path: ResolvedPath
    original_target: str
    original_source: str

    @classmethod
    def from_core_mapping(cls, mapping: VariableMapping) -> "EnhancedVariableMapping":
        """Create from core VariableMapping instance."""
        return cls(
            target_path=mapping.resolved_target,
            source_path=mapping.resolved_source,
            original_target=mapping.original_target,
            original_source=mapping.original_source,
        )


@dataclass
class EnhancedParsedCommand:
    """Enhanced parsed command with full path resolution."""

    command_type: str  # "each" or "all"
    destination_path: ResolvedPath  # pathA with scope resolution
    inclusion_path: ResolvedPath | None = None  # pathX with scope resolution
    variable_mappings: list[EnhancedVariableMapping] | None = None
    has_multiplicity: bool = False
    is_wildcard_assignment: bool = False

    def __post_init__(self):
        if self.variable_mappings is None:
            self.variable_mappings = []


def resolve_variable_mapping(target: str, source: str) -> EnhancedVariableMapping:
    """
    Resolve both target and source paths in a variable mapping.

    This is a compatibility function that uses core utilities.
    """
    resolved_target = PathResolver.resolve_path(target)
    resolved_source = PathResolver.resolve_path(source)

    return EnhancedVariableMapping(
        target_path=resolved_target,
        source_path=resolved_source,
        original_target=target,
        original_source=source,
    )


def resolve_variable_mapping_with_cwd(
    target: str, source: str, cwd: str
) -> EnhancedVariableMapping:
    """
    Resolve both target and source paths in a variable mapping with CWD support.

    This is a compatibility function that uses core utilities.
    """
    resolved_target = PathResolver.resolve_path_with_cwd(target, cwd)
    resolved_source = PathResolver.resolve_path_with_cwd(source, cwd)

    return EnhancedVariableMapping(
        target_path=resolved_target,
        source_path=resolved_source,
        original_target=target,
        original_source=source,
    )
