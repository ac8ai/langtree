"""
Registry classes for managing variables and pending targets in LangTree.

This module contains registry classes that track variable satisfaction
relationships and handle forward references in the prompt tree structure.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langtree.execution.scopes import Scope
from langtree.parsing.parser import CommandType

if TYPE_CHECKING:
    from langtree.parsing.parser import ParsedCommand
    from langtree.structure.builder import (
        RunStructure,
        StructureTreeNode,
    )

# Valid scope prefixes for target paths
# These must match the scopes defined in execution.scopes
VALID_SCOPE_PREFIXES = frozenset({"value", "outputs", "task", "prompt"})


class AssemblyVariableConflictError(Exception):
    """Raised when attempting to assign to an existing Assembly Variable."""

    def __init__(self, variable_name: str, existing_value: Any, new_value: Any):
        self.variable_name = variable_name
        self.existing_value = existing_value
        self.new_value = new_value
        super().__init__(
            f"Assembly Variable '{variable_name}' already exists with value {existing_value!r}, cannot reassign to {new_value!r}"
        )


@dataclass
class AssemblyVariable:
    """Information about an Assembly Variable defined during chain construction."""

    name: str
    value: str | int | float | bool
    source_node_tag: str
    defined_at_line: int = 0  # For debugging/error reporting

    def __str__(self) -> str:
        return f"{self.name}={self.value!r}"


class AssemblyVariableRegistry:
    """Registry for Assembly Variables defined with ! var=value syntax.

    Assembly Variables are parse-time configuration values that:
    - Are available from definition node through all descendant nodes
    - Cannot be reassigned (conflict prohibited)
    - Support string, number, and boolean values
    - Are used for command arguments and flow control
    - Are separate from Runtime Variables ({{var}} patterns)
    """

    def __init__(self):
        self.variables: dict[str, AssemblyVariable] = {}

    def store_variable(
        self,
        name: str,
        value: str | int | float | bool,
        source_node_tag: str,
        defined_at_line: int = 0,
    ) -> None:
        """
        Store an Assembly Variable, checking for conflicts.

        Assembly Variables are parse-time configuration values that must be unique
        across the prompt tree. This method enforces the no-reassignment rule that
        ensures consistent behavior across all descendant nodes.

        Params:
            name: Variable name that must be valid identifier syntax
            value: Variable value supporting string, number, or boolean types
            source_node_tag: Tag of the node where this variable was defined
            defined_at_line: Line number for debugging purposes, defaults to 0

        Raises:
            AssemblyVariableConflictError: If variable already exists with different value
        """
        if name in self.variables:
            existing = self.variables[name]
            raise AssemblyVariableConflictError(name, existing.value, value)

        self.variables[name] = AssemblyVariable(
            name=name,
            value=value,
            source_node_tag=source_node_tag,
            defined_at_line=defined_at_line,
        )

    def get_variable(self, name: str) -> AssemblyVariable | None:
        """
        Get an Assembly Variable by name.

        Provides access to complete variable information including metadata
        such as source node and definition line for debugging purposes.

        Params:
            name: Variable name to lookup in the registry

        Returns:
            AssemblyVariable instance if found, None if variable does not exist
        """
        return self.variables.get(name)

    def get_variable_value(self, name: str) -> str | int | float | bool | None:
        """
        Get just the value of an Assembly Variable.

        Convenience method for accessing variable values when metadata
        is not needed. Commonly used in command argument resolution
        and template substitution contexts.

        Params:
            name: Variable name to lookup in the registry

        Returns:
            Variable value (string, number, or boolean) if found, None if variable does not exist
        """
        var = self.get_variable(name)
        return var.value if var else None

    def has_variable(self, name: str) -> bool:
        """Check if an Assembly Variable exists.

        Params:
            name: Variable name to check

        Returns:
            True if variable exists, False otherwise
        """
        return name in self.variables

    def check_conflict(self, name: str) -> bool:
        """Check if a variable name would conflict with existing variables.

        Params:
            name: Variable name to check

        Returns:
            True if conflict exists, False if name is available
        """
        return name in self.variables

    def list_variables(self) -> list[AssemblyVariable]:
        """Get a list of all Assembly Variables.

        Returns:
            List of all stored AssemblyVariable objects
        """
        return list(self.variables.values())

    def get_variables_for_node(self, node_tag: str) -> list[AssemblyVariable]:
        """Get all variables defined by a specific node.

        Params:
            node_tag: Tag of the source node

        Returns:
            List of AssemblyVariable objects defined by that node
        """
        return [
            var for var in self.variables.values() if var.source_node_tag == node_tag
        ]

    def resolve_variable_reference(
        self, pattern: str
    ) -> str | int | float | bool | None:
        """Resolve a variable reference pattern like <variable_name>.

        Params:
            pattern: Variable reference pattern (with or without < > brackets)

        Returns:
            Variable value if found, None otherwise
        """
        clean_name = pattern.strip("<>")
        return self.get_variable_value(clean_name)


@dataclass
class SourceInfo:
    """Information about a single source that satisfies a variable."""

    source_node_tag: str
    source_field_path: str
    command_type: (
        CommandType | None
    )  # CommandType.EACH, CommandType.ALL, or None for direct assignment
    has_multiplicity: bool = False

    def get_relationship_type(self) -> str:
        """Determine the relationship type for this specific source."""
        if self.command_type == CommandType.EACH and self.has_multiplicity:
            return "n:n"
        elif self.command_type == CommandType.ALL and self.has_multiplicity:
            return "1:n"
        elif self.command_type == CommandType.ALL and not self.has_multiplicity:
            return "1:1"
        elif self.command_type == CommandType.EACH and not self.has_multiplicity:
            return "1:1"
        else:
            return "1:1"  # Direct assignment (None command_type)


@dataclass
class VariableInfo:
    """Information about a variable and its satisfaction sources."""

    variable_path: str
    scope: Scope
    sources: list[SourceInfo] = field(default_factory=list)

    def get_relationship_types(self) -> list[str]:
        """Get all relationship types from all sources."""
        return [source.get_relationship_type() for source in self.sources]

    def get_source_node_tags(self) -> list[str]:
        """Get all source node tags."""
        return [source.source_node_tag for source in self.sources]

    def get_satisfaction_sources(self) -> list[str]:
        """Get all source field paths."""
        return [source.source_field_path for source in self.sources]

    def is_satisfied(self) -> bool:
        """Check if this variable has any satisfaction sources."""
        return len(self.sources) > 0

    def has_multiple_sources(self) -> bool:
        """Check if this variable has multiple satisfaction sources."""
        return len(self.sources) > 1

    def get_scope_name(self) -> str:
        """Get the scope name for display purposes only."""
        return self.scope.get_name()


class VariableRegistry:
    """Registry tracking declared LangTree Variable Targets and their satisfaction sources.

    This registry is specifically for LangTree Variable Targets from @each/@all commands,
    NOT for Assembly Variables (! var=value). Those are handled by AssemblyVariableRegistry.

    A LangTree Variable Target is identified by (scope_name + '.' + variable_path). Each entry records:
    - Declared scope (`Scope` object) & original source node tag
    - Command type & multiplicity indicators for relationship classification
    - List of satisfaction source paths (raw strings) – wildcard "*" is always considered satisfied

    This registry does NOT currently validate semantic existence of sources beyond a shallow
    on‑demand check in `is_source_satisfied` (invoked by higher level planning/validation routines).
    """

    def __init__(self):
        self.variables: dict[str, VariableInfo] = {}

    def register_variable(
        self,
        variable_path: str,
        scope: Scope,
        source_node_tag: str,
        command_type: str = "",
        has_multiplicity: bool = False,
    ):
        """
        Register a variable target derived from a LangTree command mapping.

        Creates a new entry in the registry for tracking a variable declared
        by @each/@all commands. This establishes the variable as a target
        that can receive satisfaction sources from other nodes.

        Params:
            variable_path: Scope-relative variable identifier without scope prefix
            scope: Scope object defining the target context (prompt/value/outputs/task/current_node)
            source_node_tag: Tag of the originating node where this command was declared
            command_type: Parsed command type label, typically "each" or "all"
            has_multiplicity: Whether the command included asterisk multiplicity flag
        """
        scope_name = scope.get_name()
        full_var_name = f"{scope_name}.{variable_path}"

        if full_var_name not in self.variables:
            self.variables[full_var_name] = VariableInfo(
                variable_path=variable_path,
                scope=scope,
            )

    def add_satisfaction_source(
        self,
        variable_path: str,
        scope: Scope,
        source_node_tag: str,
        source_path: str,
        command_type: CommandType | None = None,
        has_multiplicity: bool = False,
    ):
        """
        Associate a satisfaction source with an existing variable.

        Links a rich source object to a previously registered variable target,
        preserving complete command context for each source.

        Params:
            variable_path: Target variable path in scope-relative format
            scope: Scope object that must match the registered variable
            source_node_tag: Tag of the node providing this source
            source_path: Raw source path string, may include wildcard "*"
            command_type: Command type that created this source ("each", "all", etc.)
            has_multiplicity: Whether the source command has multiplicity flag
        """
        scope_name = scope.get_name()
        full_var_name = f"{scope_name}.{variable_path}"

        if full_var_name in self.variables:
            # Check if this exact source already exists
            existing_source = None
            for source in self.variables[full_var_name].sources:
                if (
                    source.source_node_tag == source_node_tag
                    and source.source_field_path == source_path
                ):
                    existing_source = source
                    break

            # Only add if this exact source doesn't already exist
            if existing_source is None:
                new_source = SourceInfo(
                    source_node_tag=source_node_tag,
                    source_field_path=source_path,
                    command_type=command_type,
                    has_multiplicity=has_multiplicity,
                )
                self.variables[full_var_name].sources.append(new_source)

    def is_source_satisfied(
        self,
        source_path: str,
        run_structure: "RunStructure",
        source_node_tag: str | None = None,
    ) -> bool:
        """
        Heuristically determine if a satisfaction source path is resolvable.

        Performs lightweight validation to check if a source path can be resolved
        within the current tree structure. This is used during planning to assess
        variable satisfaction without full semantic resolution.

        Logic:
            - Wildcard "*" always considered satisfied
            - If source_node_tag provided, attempts attribute existence on that node instance
            - For dotted paths, splits final segment as field candidate and checks node existence

        Params:
            source_path: Raw path string provided in the variable mapping
            run_structure: Active RunStructure instance for tree queries
            source_node_tag: Optional node tag for local resolution context

        Returns:
            True if heuristically satisfied or resolvable, False otherwise
        """
        if source_path == "*":
            return True

        try:
            if source_node_tag:
                source_node = run_structure.get_node(source_node_tag)
                if source_node and source_node.field_type:
                    node_instance = source_node.field_type()
                    if hasattr(node_instance, source_path):
                        return True

            if "." in source_path:
                node_path, field_path = source_path.rsplit(".", 1)
                node = run_structure.get_node(node_path)
                if node and node.field_type:
                    node_instance = node.field_type()
                    return hasattr(node_instance, field_path)

            return False
        except Exception:
            return False

    def get_truly_unsatisfied_variables(
        self, run_structure: "RunStructure"
    ) -> list[VariableInfo]:
        """Return variables lacking any valid (resolvable) satisfaction source.

        A variable with sources all failing `is_source_satisfied` is considered truly unsatisfied.
        """
        truly_unsatisfied = []
        for var_info in self.variables.values():
            if not var_info.sources:
                truly_unsatisfied.append(var_info)
            else:
                has_valid_source = False
                for source in var_info.sources:
                    if self.is_source_satisfied(
                        source.source_field_path, run_structure, source.source_node_tag
                    ):
                        has_valid_source = True
                        break

                if not has_valid_source:
                    truly_unsatisfied.append(var_info)

        return truly_unsatisfied

    def get_unsatisfied_variables(self) -> list[VariableInfo]:
        """Return variables with an empty satisfaction source list (syntactic unsatisfied set)."""
        return [
            var_info
            for var_info in self.variables.values()
            if not var_info.is_satisfied()
        ]

    def get_multiply_satisfied_variables(self) -> list[VariableInfo]:
        """Return variables with >1 satisfaction source (potential ambiguity / merge case)."""
        return [
            var_info
            for var_info in self.variables.values()
            if var_info.has_multiple_sources()
        ]


@dataclass
class PendingTarget:
    """Information about a pending target resolution."""

    target_path: str
    command: "ParsedCommand"
    source_node_tag: str


@dataclass
class ResolvedTarget:
    """Information about a resolved target.

    Tracks fields that have incoming data forwarded to them via commands.
    This enables efficient lookup when generating {COLLECTED_CONTEXT} to check
    which fields have incoming data.

    Note: If a field is in this registry, it means data is being forwarded TO it.
    Fields that are directly generated at their own node won't be in this registry.
    """

    target_path: str  # Full path to the target field (e.g., "task.analysis.summary")
    source_node_tag: str  # Where the data comes from
    command: "ParsedCommand"  # The command that forwards data to this field


class ResolvedTargetRegistry:
    """Registry for tracking resolved (populated) fields with node-level indexing.

    Maintains a record of fields that have received data through commands,
    indexed by node tag for efficient COLLECTED_CONTEXT generation.

    Handles all scope prefixes (value.*, outputs.*, task.*) by extracting
    the node portion from target paths.

    Usage:
        # When processing a command that forwards data to a target
        registry.add_resolved(target_path, source_tag, command)

        # Get all fields for a node (for COLLECTED_CONTEXT)
        node_fields = registry.get_resolved_for_node("task.analysis")
        # Returns: {"summary": [ResolvedTarget, ...], "title": [...]}
    """

    def __init__(self):
        # Node-level index: node_tag -> {field_name -> [ResolvedTarget, ...]}
        # Example: {"task.analysis": {"summary": [ResolvedTarget, ...], "title": [...]}}
        self.targets_by_node: dict[str, dict[str, list[ResolvedTarget]]] = {}

    def add_resolved(
        self,
        target_path: str,
        source_node_tag: str,
        command: "ParsedCommand",
    ):
        """
        Record that a field has incoming data forwarded to it.

        Params:
            target_path: Full path to the target field (e.g., "value.analysis.summary", "outputs.task.processor.title")
            source_node_tag: Tag of the node providing the data
            command: The command that forwards data to this target
        """
        # Parse target_path to extract node and field
        node_tag, field_name = self._parse_target_path(target_path)

        # Initialize nested dicts if needed
        if node_tag not in self.targets_by_node:
            self.targets_by_node[node_tag] = {}
        if field_name not in self.targets_by_node[node_tag]:
            self.targets_by_node[node_tag][field_name] = []

        # Add resolved target
        self.targets_by_node[node_tag][field_name].append(
            ResolvedTarget(target_path, source_node_tag, command)
        )

    def _parse_target_path(self, target_path: str) -> tuple[str, str]:
        """
        Parse target path to extract node tag and field name.

        Handles scope prefixes (value.*, outputs.*, task.*, prompt.*) by stripping
        them and extracting the underlying node tag and field name.

        Examples:
            "value.analysis.summary" -> ("analysis", "summary")
            "outputs.task.processor.title" -> ("task.processor", "title")
            "task.analysis.summary" -> ("task.analysis", "summary")
            "analysis.summary" -> ("analysis", "summary")

        Params:
            target_path: Full path including optional scope prefix

        Returns:
            Tuple of (node_tag, field_name)
        """
        parts = target_path.split(".")

        # Check if first part is a scope modifier
        if parts[0] in VALID_SCOPE_PREFIXES:
            remaining = parts[1:]
        else:
            remaining = parts

        # Last part is field name, rest is node tag
        if len(remaining) < 2:
            # Malformed path - should have at least node.field
            raise ValueError(
                f"Invalid target path: {target_path}. Expected format: [scope.]node[.subnode].field"
            )

        field_name = remaining[-1]
        node_tag = ".".join(remaining[:-1])

        return node_tag, field_name

    def get_resolved_for_node(self, node_tag: str) -> dict[str, list[ResolvedTarget]]:
        """
        Get all resolved fields for a specific node.

        Useful for generating COLLECTED_CONTEXT - returns all fields that have
        incoming data for the specified node.

        Params:
            node_tag: Tag of the node (e.g., "task.analysis", "processor")

        Returns:
            Dictionary mapping field names to lists of ResolvedTarget entries.
            Empty dict if no fields have incoming data.

        Example:
            >>> registry.get_resolved_for_node("task.analysis")
            {"summary": [ResolvedTarget(...), ...], "title": [ResolvedTarget(...)]}
        """
        return self.targets_by_node.get(node_tag, {})

    def get_resolved_for_field(self, field_path: str) -> list[ResolvedTarget]:
        """
        Get all resolved targets for a specific field path.

        Params:
            field_path: Full path to the field (e.g., "task.analysis.summary", "value.analysis.summary")

        Returns:
            List of ResolvedTarget entries showing what data is being sent to this field.
            Empty list if no data is being sent to this field.
        """
        node_tag, field_name = self._parse_target_path(field_path)
        return self.targets_by_node.get(node_tag, {}).get(field_name, [])

    def has_incoming_data(self, field_path: str) -> bool:
        """
        Check if a field has any incoming data.

        Params:
            field_path: Full path to the field

        Returns:
            True if at least one source sends data to this field, False otherwise
        """
        resolved = self.get_resolved_for_field(field_path)
        return len(resolved) > 0

    def get_all_resolved_paths(self) -> list[str]:
        """
        Get all field paths that have been resolved.

        Returns:
            List of all field paths that have incoming data (with original scope prefixes preserved)
        """
        paths = []
        for node_tag, fields in self.targets_by_node.items():
            for field_name, targets in fields.items():
                # Use the original target_path from ResolvedTarget
                for target in targets:
                    if target.target_path not in paths:
                        paths.append(target.target_path)
        return paths


class PendingTargetRegistry:
    """Registry for forward destination targets (waiting list pattern).

    Tracks commands referencing destination task paths not yet materialized in the structure.
    When a node appears, `resolve_pending` retrieves and removes all matching entries
    (exact path and descendants). Caller processes resolved commands via callback.
    """

    def __init__(self):
        self.pending_targets: dict[str, list[PendingTarget]] = {}

    def add_pending(
        self, target_path: str, command: "ParsedCommand", source_node_tag: str
    ):
        """
        Record a command awaiting destination node availability.

        Registers a command that references a forward destination node not yet
        present in the tree structure. This enables deferred processing until
        the target node is added to the tree.

        Params:
            target_path: Fully qualified destination path, e.g., "task.analyze.subtask"
            command: Parsed LangTree command object containing the referencing logic
            source_node_tag: Tag of the originating node where this command was declared
        """
        if target_path not in self.pending_targets:
            self.pending_targets[target_path] = []

        self.pending_targets[target_path].append(
            PendingTarget(target_path, command, source_node_tag)
        )

    def resolve_pending(self, resolved_path: str) -> list[PendingTarget]:
        """
        Return and remove all pending targets satisfied by a newly added path.

        Retrieves commands that were waiting for a specific node path and removes
        them from the pending registry. This enables batch processing of forward
        references when their target nodes become available.

        Matching rules:
            - Exact match: pending path equals resolved path
            - Descendant matches: pending path starts with resolved_path + '.'

        Params:
            resolved_path: Path of the newly added node that may satisfy pending commands

        Returns:
            List of PendingTarget objects that were waiting for this path
        """
        resolved = []

        # Check for exact matches and partial matches
        for target_path in list(self.pending_targets.keys()):
            if target_path == resolved_path or target_path.startswith(
                resolved_path + "."
            ):
                resolved.extend(self.pending_targets.pop(target_path))

        return resolved

    def resolve_target(
        self, target_path: str, target_node: "StructureTreeNode"
    ) -> list[PendingTarget]:
        """
        Resolve a specific target path and process its pending commands.

        This method finds all pending commands waiting for the given target path,
        removes them from the pending registry, and returns them for processing.

        Params:
            target_path: The target path that has been resolved
            target_node: The actual node that satisfies the target path

        Returns:
            List of PendingTarget objects that were waiting for this target
        """
        resolved = self.resolve_pending(target_path)
        # Caller processes resolved commands (see builder._complete_pending_command_processing)
        return resolved
