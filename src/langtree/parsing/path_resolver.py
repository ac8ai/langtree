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


def analyze_scope_coverage():
    """Analyze how well the current parser handles scopes vs. what it should do."""
    from langtree.parsing.parser import parse_command

    test_commands = [
        "! @each[prompt.sections]->value.task@{{outputs.title=task.sections.title}}*",
        "! @->prompt.destination@{{value.field=outputs.source}}",
        "! @each[value.items]->task.processor@{{prompt.result=value.data}}*",
    ]

    print("=== SCOPE COVERAGE ANALYSIS ===")

    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{i}. Command: {cmd}")

        # Current parser behavior
        try:
            current_result = parse_command(cmd)
            print("   Current parser:")
            print(
                f"   - Inclusion: {current_result.inclusion_path} (scope: NOT EXTRACTED)"
            )
            print(
                f"   - Destination: {current_result.destination_path} (scope: NOT EXTRACTED)"
            )
            for j, mapping in enumerate(current_result.variable_mappings):
                # The current parser resolves scope indirectly via resolved_target/resolved_source
                target_scope = (
                    mapping.resolved_target.scope.get_name()
                    if mapping.resolved_target and mapping.resolved_target.scope
                    else "None"
                )
                source_scope = (
                    mapping.resolved_source.scope.get_name()
                    if mapping.resolved_source and mapping.resolved_source.scope
                    else "None"
                )
                print(
                    f"   - Mapping {j + 1}: {mapping.target_path} = {mapping.source_path}"
                )
                print(
                    f"     Target scope: {target_scope}, Source scope: {source_scope}"
                )
        except Exception as e:
            print(f"   Current parser: ERROR - {e}")

        # Enhanced resolver behavior
        print("   Enhanced resolver:")

        # Parse components manually for demonstration
        if "each[" in cmd:
            inclusion_part = cmd.split("each[")[1].split("]")[0]
            resolved_inclusion = PathResolver.resolve_path(inclusion_part)
            print(
                f"   - Inclusion: {resolved_inclusion.path_remainder} (scope: {resolved_inclusion.scope_modifier.value if resolved_inclusion.scope_modifier else 'None'})"
            )

        dest_part = cmd.split("->")[1].split("@")[0]
        resolved_dest = PathResolver.resolve_path(dest_part)
        print(
            f"   - Destination: {resolved_dest.path_remainder} (scope: {resolved_dest.scope_modifier.value if resolved_dest.scope_modifier else 'None'})"
        )

        # Parse mappings
        mapping_part = cmd.split("{{")[1].split("}}")[0]
        if "=" in mapping_part:
            target, source = mapping_part.split("=", 1)
            resolved_mapping = resolve_variable_mapping(target.strip(), source.strip())
            target_scope = (
                resolved_mapping.target_path.scope_modifier.value
                if resolved_mapping.target_path.scope_modifier
                else "None"
            )
            source_scope = (
                resolved_mapping.source_path.scope_modifier.value
                if resolved_mapping.source_path.scope_modifier
                else "None"
            )
            print(
                f"   - Mapping: {resolved_mapping.target_path.path_remainder} = {resolved_mapping.source_path.path_remainder}"
            )
            print(f"     Target scope: {target_scope}, Source scope: {source_scope}")


if __name__ == "__main__":
    analyze_scope_coverage()
