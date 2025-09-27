"""
Enhanced path resolution for dynamic prompt connecting language.

This module provides proper scope resolution for all path components as specified
in the documentation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScopeModifier(Enum):
    """Valid scope modifiers for variable mapping."""
    PROMPT = "prompt"
    VALUE = "value"
    OUTPUTS = "outputs"
    TASK = "task"


@dataclass
class ResolvedPath:
    """Represents a path with its scope modifier extracted."""
    scope_modifier: Optional[ScopeModifier]
    path_remainder: str  # The path after scope is extracted
    original_path: str   # The original full path
    
    def __str__(self) -> str:
        if self.scope_modifier:
            return f"{self.scope_modifier.value}.{self.path_remainder}"
        return self.path_remainder
    
    @property
    def has_scope(self) -> bool:
        """Check if this path has a scope modifier."""
        return self.scope_modifier is not None


@dataclass
class EnhancedVariableMapping:
    """Enhanced variable mapping with full path resolution."""
    target_path: ResolvedPath      # pathB with scope resolution
    source_path: ResolvedPath      # pathC with scope resolution
    original_target: str           # Original target string
    original_source: str           # Original source string


@dataclass
class EnhancedParsedCommand:
    """Enhanced parsed command with full path resolution."""
    command_type: str              # "each" or "all"
    destination_path: ResolvedPath # pathA with scope resolution
    inclusion_path: Optional[ResolvedPath] = None  # pathX with scope resolution
    variable_mappings: Optional[list[EnhancedVariableMapping]] = None
    has_multiplicity: bool = False
    is_wildcard_assignment: bool = False
    
    def __post_init__(self):
        if self.variable_mappings is None:
            self.variable_mappings = []


class PathResolver:
    """Resolves scope modifiers from all path components."""
    
    @staticmethod
    def resolve_path(path: str) -> ResolvedPath:
        """
        Resolve scope modifier from a path string.
        
        Params:
            path: Path string that may start with a scope modifier
            
        Returns:
            ResolvedPath with scope and remainder separated
        """
        if not path or '.' not in path:
            return ResolvedPath(
                scope_modifier=None,
                path_remainder=path,
                original_path=path
            )
        
        first_part = path.split('.', 1)[0]
        remainder = path.split('.', 1)[1]
        
        try:
            scope = ScopeModifier(first_part)
            return ResolvedPath(
                scope_modifier=scope,
                path_remainder=remainder,
                original_path=path
            )
        except ValueError:
            # First part is not a valid scope modifier
            return ResolvedPath(
                scope_modifier=None,
                path_remainder=path,
                original_path=path
            )
    
    @staticmethod
    def resolve_path_with_cwd(path: str, cwd: str) -> ResolvedPath:
        """
        Resolve a path with Command Working Directory (CWD) support.

        Args:
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
                scope_modifier=None,
                path_remainder="",
                original_path=path
            )

        # First check if it has a scope modifier or starts with 'task.'
        if '.' in path:
            first_part = path.split('.', 1)[0]

            # Check if it's a scope modifier
            try:
                scope = ScopeModifier(first_part)
                # Has scope modifier - it's absolute, ignore CWD
                remainder = path.split('.', 1)[1]
                return ResolvedPath(
                    scope_modifier=scope,
                    path_remainder=remainder,
                    original_path=path
                )
            except ValueError:
                # Not a scope modifier, check if it starts with 'task.'
                if first_part == 'task':
                    # Absolute path starting with task. - ignore CWD
                    return ResolvedPath(
                        scope_modifier=None,
                        path_remainder=path,
                        original_path=path
                    )

        # No scope modifier and doesn't start with 'task.' - it's relative
        # Apply CWD resolution
        if cwd and path:
            resolved_path = f"{cwd}.{path}"
        else:
            resolved_path = path or cwd or ""

        return ResolvedPath(
            scope_modifier=None,
            path_remainder=resolved_path,
            original_path=path
        )

    @staticmethod
    def resolve_variable_mapping(target: str, source: str) -> EnhancedVariableMapping:
        """Resolve both target and source paths in a variable mapping."""
        resolved_target = PathResolver.resolve_path(target)
        resolved_source = PathResolver.resolve_path(source)

        return EnhancedVariableMapping(
            target_path=resolved_target,
            source_path=resolved_source,
            original_target=target,
            original_source=source
        )

    @staticmethod
    def resolve_variable_mapping_with_cwd(target: str, source: str, cwd: str) -> EnhancedVariableMapping:
        """Resolve both target and source paths in a variable mapping with CWD support."""
        resolved_target = PathResolver.resolve_path_with_cwd(target, cwd)
        resolved_source = PathResolver.resolve_path_with_cwd(source, cwd)

        return EnhancedVariableMapping(
            target_path=resolved_target,
            source_path=resolved_source,
            original_target=target,
            original_source=source
        )


def analyze_scope_coverage():
    """Analyze how well the current parser handles scopes vs. what it should do."""
    from langtree.commands.parser import parse_command
    
    test_commands = [
        '! @each[prompt.sections]->value.task@{{outputs.title=task.sections.title}}*',
        '! @->prompt.destination@{{value.field=outputs.source}}',
        '! @each[value.items]->task.processor@{{prompt.result=value.data}}*'
    ]
    
    print("=== SCOPE COVERAGE ANALYSIS ===")
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{i}. Command: {cmd}")
        
        # Current parser behavior
        try:
            current_result = parse_command(cmd)
            print("   Current parser:")
            print(f"   - Inclusion: {current_result.inclusion_path} (scope: NOT EXTRACTED)")
            print(f"   - Destination: {current_result.destination_path} (scope: NOT EXTRACTED)")
            for j, mapping in enumerate(current_result.variable_mappings):
                # The current parser resolves scope indirectly via resolved_target/resolved_source
                target_scope = mapping.resolved_target.scope.get_name() if mapping.resolved_target and mapping.resolved_target.scope else 'None'
                source_scope = mapping.resolved_source.scope.get_name() if mapping.resolved_source and mapping.resolved_source.scope else 'None'
                print(f"   - Mapping {j+1}: {mapping.target_path} = {mapping.source_path}")
                print(f"     Target scope: {target_scope}, Source scope: {source_scope}")
        except Exception as e:
            print(f"   Current parser: ERROR - {e}")
        
        # Enhanced resolver behavior
        print("   Enhanced resolver:")
        
        # Parse components manually for demonstration
        if 'each[' in cmd:
            inclusion_part = cmd.split('each[')[1].split(']')[0]
            resolved_inclusion = PathResolver.resolve_path(inclusion_part)
            print(f"   - Inclusion: {resolved_inclusion.path_remainder} (scope: {resolved_inclusion.scope_modifier.value if resolved_inclusion.scope_modifier else 'None'})")
        
        dest_part = cmd.split('->')[1].split('@')[0]
        resolved_dest = PathResolver.resolve_path(dest_part)
        print(f"   - Destination: {resolved_dest.path_remainder} (scope: {resolved_dest.scope_modifier.value if resolved_dest.scope_modifier else 'None'})")
        
        # Parse mappings
        mapping_part = cmd.split('{{')[1].split('}}')[0]
        if '=' in mapping_part:
            target, source = mapping_part.split('=', 1)
            resolved_mapping = PathResolver.resolve_variable_mapping(target.strip(), source.strip())
            target_scope = resolved_mapping.target_path.scope_modifier.value if resolved_mapping.target_path.scope_modifier else 'None'
            source_scope = resolved_mapping.source_path.scope_modifier.value if resolved_mapping.source_path.scope_modifier else 'None'
            print(f"   - Mapping: {resolved_mapping.target_path.path_remainder} = {resolved_mapping.source_path.path_remainder}")
            print(f"     Target scope: {target_scope}, Source scope: {source_scope}")


if __name__ == "__main__":
    analyze_scope_coverage()
