"""
Registry classes for managing variables and pending targets in LangTree.

This module contains registry classes that track variable satisfaction
relationships and handle forward references in the prompt tree structure.
"""

# Group 2: External from imports (alphabetical by source module)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

# Group 4: Internal from imports (alphabetical by source module)
from langtree.prompt.exceptions import NodeInstantiationError, FieldTypeError, PathValidationError, NodeTagValidationError
from langtree.prompt.scopes import Scope

if TYPE_CHECKING:
    from langtree.commands.parser import ParsedCommand
    from langtree.prompt.structure import RunStructure, StructureTreeNode, PromptTreeNode


class AssemblyVariableConflictError(Exception):
    """Raised when attempting to assign to an existing Assembly Variable."""
    def __init__(self, variable_name: str, existing_value: Any, new_value: Any):
        self.variable_name = variable_name
        self.existing_value = existing_value
        self.new_value = new_value
        super().__init__(f"Assembly Variable '{variable_name}' already exists with value {existing_value!r}, cannot reassign to {new_value!r}")


@dataclass
class AssemblyVariable:
    """Information about an Assembly Variable defined during chain construction."""
    name: str
    value: Union[str, int, float, bool]
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
    
    def store_variable(self, name: str, value: Union[str, int, float, bool], 
                      source_node_tag: str, defined_at_line: int = 0) -> None:
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
            defined_at_line=defined_at_line
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
    
    def get_variable_value(self, name: str) -> Union[str, int, float, bool, None]:
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
        return [var for var in self.variables.values() if var.source_node_tag == node_tag]
    
    def resolve_variable_reference(self, pattern: str) -> Union[str, int, float, bool, None]:
        """Resolve a variable reference pattern like <variable_name>.
        
        Params:
            pattern: Variable reference pattern (with or without < > brackets)
            
        Returns:
            Variable value if found, None otherwise
        """
        clean_name = pattern.strip('<>')
        return self.get_variable_value(clean_name)


@dataclass
class VariableInfo:
    """Information about a variable and its satisfaction sources."""
    variable_path: str
    scope: Scope
    source_node_tag: str
    satisfaction_sources: list[str]
    command_type: str = ""  # "each", "all", or ""
    has_multiplicity: bool = False
    
    def get_relationship_type(self) -> str:
        """Determine the relationship type based on command and multiplicity."""
        if self.command_type == "each" and self.has_multiplicity:
            return "n:n"
        elif self.command_type == "all" and self.has_multiplicity:
            return "1:n"
        elif self.command_type == "all" and not self.has_multiplicity:
            return "1:1"
        else:
            return "unknown"
    
    def is_satisfied(self) -> bool:
        """Check if this variable has any satisfaction sources."""
        return len(self.satisfaction_sources) > 0
    
    def has_multiple_sources(self) -> bool:
        """Check if this variable has multiple satisfaction sources."""
        return len(self.satisfaction_sources) > 1
    
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
    
    def register_variable(self, variable_path: str, scope: Scope, source_node_tag: str, 
                         command_type: str = "", has_multiplicity: bool = False):
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
                source_node_tag=source_node_tag,
                satisfaction_sources=[],
                command_type=command_type,
                has_multiplicity=has_multiplicity
            )
    
    def add_satisfaction_source(self, variable_path: str, scope: Scope, source_path: str):
        """
        Associate a satisfaction source path with an existing variable.

        Links a source path to a previously registered variable target,
        building the mapping that will be used during resolution to determine
        where variable values come from.

        Params:
            variable_path: Target variable path in scope-relative format
            scope: Scope object that must match the registered variable
            source_path: Raw source path string, may include wildcard "*"
        """
        scope_name = scope.get_name()
        full_var_name = f"{scope_name}.{variable_path}"
        
        if full_var_name in self.variables:
            if source_path not in self.variables[full_var_name].satisfaction_sources:
                self.variables[full_var_name].satisfaction_sources.append(source_path)
    
    def is_source_satisfied(self, source_path: str, run_structure: 'RunStructure', source_node_tag: str | None = None) -> bool:
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
    
    def get_truly_unsatisfied_variables(self, run_structure: 'RunStructure') -> list[VariableInfo]:
        """Return variables lacking any valid (resolvable) satisfaction source.

        A variable with sources all failing `is_source_satisfied` is considered truly unsatisfied.
        """
        truly_unsatisfied = []
        for var_info in self.variables.values():
            if not var_info.satisfaction_sources:
                truly_unsatisfied.append(var_info)
            else:
                has_valid_source = False
                for source in var_info.satisfaction_sources:
                    if self.is_source_satisfied(source, run_structure, var_info.source_node_tag):
                        has_valid_source = True
                        break
                
                if not has_valid_source:
                    truly_unsatisfied.append(var_info)
        
        return truly_unsatisfied
    
    def get_unsatisfied_variables(self) -> list[VariableInfo]:
        """Return variables with an empty satisfaction source list (syntactic unsatisfied set)."""
        return [var_info for var_info in self.variables.values() if not var_info.is_satisfied()]
    
    def get_multiply_satisfied_variables(self) -> list[VariableInfo]:
        """Return variables with >1 satisfaction source (potential ambiguity / merge case)."""
        return [var_info for var_info in self.variables.values() if var_info.has_multiple_sources()]


@dataclass
class PendingTarget:
    """Information about a pending target resolution."""
    target_path: str
    command: 'ParsedCommand'
    source_node_tag: str


class PendingTargetRegistry:
    """Registry for forward destination targets (waiting list pattern).

    Tracks commands referencing destination task paths not yet materialized in the structure.
    When a node appears, `resolve_pending` retrieves and removes all matching entries
    (exact path and descendants), allowing a future callback to finalize resolution (TODO).
    """
    
    def __init__(self):
        self.pending_targets: dict[str, list[PendingTarget]] = {}
    
    def add_pending(self, target_path: str, command: 'ParsedCommand', source_node_tag: str):
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
            if target_path == resolved_path or target_path.startswith(resolved_path + '.'):
                resolved.extend(self.pending_targets.pop(target_path))
        
        return resolved
    
    def resolve_target(self, target_path: str, target_node: 'StructureTreeNode') -> list[PendingTarget]:
        """
        Resolve a specific target path and process its pending commands.
        
        This method finds all pending commands waiting for the given target path,
        removes them from the pending registry, and returns them for processing.
        
        Args:
            target_path: The target path that has been resolved
            target_node: The actual node that satisfies the target path
            
        Returns:
            List of PendingTarget objects that were waiting for this target
        """
        resolved = self.resolve_pending(target_path)
        
        # TODO: Process the resolved commands with the target node
        # This would involve updating the commands with the resolved target
        # and potentially triggering further processing
        
        return resolved


# Common validation utilities
def _validate_path_and_node_tag(path: str, node_tag: str) -> None:
    """
    Validate path and node_tag parameters for resolution methods.
    
    Common validation logic shared across registry methods to ensure
    consistent error handling and parameter validation.
    
    Params:
        path: The path string to validate for basic format requirements
        node_tag: The node tag string to validate for basic format requirements
        
    Raises:
        PathValidationError: If path format is invalid
        NodeTagValidationError: If node_tag format is invalid
    """
    if not path or not isinstance(path, str):
        raise PathValidationError(path, "must be a non-empty string")
    
    if not node_tag or not isinstance(node_tag, str):
        raise NodeTagValidationError(node_tag, "must be a non-empty string")


def _get_node_instance(node: 'StructureTreeNode', node_tag: str) -> 'PromptTreeNode':
    """
    Get an instance of a node's field type for data access.
    
    Creates an instance of the PromptTreeNode class associated with
    a structure tree node, enabling access to node data and attributes
    for resolution and validation purposes.
    
    Params:
        node: The structure tree node containing the field_type reference
        node_tag: The node tag used for error reporting and debugging
        
    Returns:
        An instance of the node's field_type (PromptTreeNode subclass)
        
    Raises:
        FieldTypeError: When node has no field_type defined
        NodeInstantiationError: When node cannot be instantiated due to initialization errors
    """
    if not node.field_type:
        raise FieldTypeError(node_tag)
    
    try:
        return node.field_type()
    except Exception as e:
        raise NodeInstantiationError(node_tag, str(e))