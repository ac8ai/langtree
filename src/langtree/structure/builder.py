"""
Core structure classes for LangTree DSL prompt tree building and management.

This module contains the main classes for building and managing the
prompt tree structure, including node classes and the main RunStructure
class that coordinates tree building and command processing.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, get_args, get_origin

from pydantic import BaseModel

from langtree.core.tree_node import TreeNode
from langtree.exceptions import DuplicateTargetError, FieldTypeError
from langtree.parsing.parser import (
    ExecutionCommand,
    NodeModifierCommand,
    ParsedCommand,
    ResamplingCommand,
    VariableAssignmentCommand,
    parse_command,
)
from langtree.structure.registry import (
    AssemblyVariableRegistry,
    PendingTarget,
    PendingTargetRegistry,
    VariableRegistry,
)
from langtree.templates.utils import extract_commands, get_root_tag
from langtree.templates.variables import process_template_variables

# Type aliases
PromptValue = str | int | float | bool | list | dict | None
ParsedCommandUnion = (
    VariableAssignmentCommand
    | ExecutionCommand
    | ResamplingCommand
    | NodeModifierCommand
    | ParsedCommand
)


ResolutionResult = PromptValue | TreeNode


@dataclass
class StructureTreeNode:
    """Node in the structure tree representing a parsed prompt component."""

    name: str
    field_type: type[TreeNode] | None = None
    parent: Optional["StructureTreeNode"] = None
    children: dict[str, "StructureTreeNode"] = field(default_factory=dict)
    clean_docstring: str | None = field(default=None)
    clean_field_descriptions: dict[str, str] = field(default_factory=dict)
    extracted_commands: list[ParsedCommandUnion] = field(default_factory=list)


@dataclass
class StructureTreeRoot(StructureTreeNode):
    """Root node for structure tree."""

    def __init__(self, name: str):
        super().__init__(name=name, field_type=None, parent=None)


class RunStructure:
    """Coordinator for building and managing LangTree DSL prompt tree structures.

    This framework assembles LangChain chains from LangTree DSL-annotated prompt trees.
    It builds the structural representation during assembly phase - LangChain
    handles all execution and output management.

    Responsibilities:
    - Build structural tree of `TreeNode` subclasses for chain assembly
    - Extract and parse LangTree DSL commands from docstrings and field descriptions
    - Register variables and pending (forward‑referenced) target destinations
    - Provide execution plan scaffolding for chain composition
    - Generate validation reports for command compatibility

    This is a chain assembly framework - not a runtime execution system.
    The assembled chains will be LangChain Runnables that handle their own execution.
    """

    def __init__(self):
        self._root_nodes = {}
        self._variable_registry = VariableRegistry()
        self._pending_target_registry = PendingTargetRegistry()
        self._assembly_variable_registry = AssemblyVariableRegistry()

    def add(self, tree: type[TreeNode]) -> None:
        """Add a root prompt tree node class to the structure.

        This performs a depth‑first traversal over the supplied `TreeNode` subclass,
        creating `StructureTreeNode` entries, extracting & parsing any LangTree DSL commands
        from docstrings / field descriptions, and registering variables & pending targets.

        Params:
            tree: A subclass of `TreeNode` representing a root task/tree entry.

        Raises:
            ValueError: If class naming convention does not match expected kind (via `get_root_tag`).
            ValueError: If a field lacks a type annotation or uses an unsupported type pattern.
        """
        tag = get_root_tag(tree)
        designation = tag.split(".")[0]
        if designation not in self._root_nodes:
            root = StructureTreeRoot(name=designation)
            self._root_nodes[designation] = root
        self._process_subtask(tree, self._root_nodes[designation], tag)

    def _process_subtask(
        self, subtree: type[TreeNode], parent: StructureTreeNode, tag: str
    ) -> None:
        """Recursively materialize a subtree rooted at `subtree` into the structure.

        Side effects:
        - Adds a `StructureTreeNode` entry under `parent`.
        - Extracts & parses commands from class docstring and field descriptions.
        - Registers variables & pending targets through registries.
        - Traverses nested `TreeNode` field annotations, including generics like `list[MyNode]`.

        Params:
            subtree: The `TreeNode` subclass being processed.
            parent: The parent `StructureTreeNode` into which this node is inserted.
            tag: Fully qualified hierarchical tag (e.g., `task.analysis.sections`).

        Raises:
            ValueError: If a field annotation is missing or unsupported.
        """
        node = StructureTreeNode(name=tag, field_type=subtree, parent=parent)
        field_name = tag.split(".")[-1]

        # Check for duplicate target definitions
        if field_name in parent.children:
            existing_node = parent.children[field_name]
            if existing_node.field_type == subtree:
                raise DuplicateTargetError(
                    tag, existing_node.field_type.__name__, subtree.__name__
                )

        parent.children[field_name] = node

        resolved_targets = self._pending_target_registry.resolve_pending(tag)
        for resolved_target in resolved_targets:
            self._complete_pending_command_processing(resolved_target)

        if subtree.__doc__:
            commands, clean_content = extract_commands(subtree.__doc__)

            processed_content = process_template_variables(clean_content, node)
            node.clean_docstring = processed_content

            for command_str in commands:
                parsed_command = parse_command(command_str)
                node.extracted_commands.append(parsed_command)

                # Validate field context scoping for docstring commands (field_name = None)
                self._validate_field_context_scoping(parsed_command, None, tag)

                self._process_command(parsed_command, tag, None)

        for field_name_inner, field_def in subtree.model_fields.items():
            if field_def.description:
                commands, clean_content = extract_commands(field_def.description)
                if clean_content:
                    processed_content = process_template_variables(clean_content, node)
                    node.clean_field_descriptions[field_name_inner] = processed_content

                for command_str in commands:
                    parsed_command = parse_command(command_str)
                    node.extracted_commands.append(parsed_command)

                    # Validate field context scoping for field-level commands
                    self._validate_field_context_scoping(
                        parsed_command, field_name_inner, tag
                    )

                    # Validate that iteration commands are only defined on iterable fields
                    if (
                        hasattr(parsed_command, "command_type")
                        and parsed_command.command_type
                    ):
                        self._validate_command_field_compatibility(
                            parsed_command, field_name_inner, field_def, tag
                        )

                    self._process_command(parsed_command, tag, field_name_inner)

        for field_name_inner, field_def in subtree.model_fields.items():
            field_tag = f"{tag}.{field_name_inner}"
            annotation = field_def.annotation
            origin = get_origin(annotation)
            args = get_args(annotation)
            if annotation is None:
                raise FieldTypeError(field_tag, "is missing type annotation")

            # Validate inheritance: reject BaseModel classes that don't inherit from TreeNode
            self._validate_field_inheritance(field_tag, annotation, origin, args)

            if issubclass(annotation, TreeNode):
                # For TreeNode fields, validate command compatibility before processing subtask
                if field_def.description:
                    commands, _ = extract_commands(field_def.description)
                    for command_str in commands:
                        parsed_command = parse_command(command_str)
                        if (
                            hasattr(parsed_command, "command_type")
                            and parsed_command.command_type
                        ):
                            self._validate_command_field_compatibility(
                                parsed_command, field_name_inner, field_def, tag
                            )

                self._process_subtask(annotation, node, field_tag)
            elif origin is not None and args is not None:
                for type_candidate in args:
                    if issubclass(type_candidate, TreeNode):
                        self._process_subtask(type_candidate, node, field_tag)
            elif origin is None and args is None:
                raise FieldTypeError(
                    field_tag, f"has unsupported field type: {field_def.annotation}"
                )

    def _process_command(
        self, command: ParsedCommandUnion, source_node_tag: str, field_name: str = None
    ) -> None:
        """Register structural + variable metadata for a parsed command.

        Current behavior (Phase 1):
        - For LangTree DSL commands (ParsedCommand): Resolves destination paths and registers variables
        - For other commands: Store for later processing during execution
        - Defers semantic/context resolution to later phases (see `resolution.py`).

        Params:
            command: Parsed LangTree DSL command (already syntax‑validated by parser layer).
            source_node_tag: Tag of the node from which the command originated.
            field_name: Name of the field containing the command (None for docstring commands).

        Raises:
            None directly (parser has already validated). Future phases may raise on semantic issues.
        """

        if isinstance(command, ParsedCommand):
            self._process_acl_command(command, source_node_tag, field_name)
        elif isinstance(command, VariableAssignmentCommand):
            self._process_variable_assignment(command, source_node_tag)
        elif isinstance(
            command, ExecutionCommand | ResamplingCommand | NodeModifierCommand
        ):
            pass

    def _process_acl_command(
        self, command: ParsedCommand, source_node_tag: str, field_name: str = None
    ) -> None:
        """Process LangTree DSL commands (@each, @all) with destinations and variable mappings."""

        # Step 1: Validate inclusion field exists if this is an @each command
        if command.inclusion_path:
            self._validate_inclusion_field(command.inclusion_path, source_node_tag)
            # Step 1.2: Validate base field is iterable for @each commands
            if command.command_type.value == "@each":
                self._validate_inclusion_base_field_iterable(
                    command.inclusion_path, source_node_tag
                )

        # Step 1.5: Validate task target completeness
        self._validate_task_target_completeness(command, source_node_tag)

        # Step 2: Process destination path
        destination_path = command.destination_path
        if command.resolved_destination and command.resolved_destination.scope:
            if command.resolved_destination.scope.get_name() == "task":
                full_destination = f"task.{command.resolved_destination.path}"
            else:
                full_destination = destination_path
        else:
            full_destination = destination_path

        target_node = self.get_node(full_destination)
        if target_node is None:
            self._pending_target_registry.add_pending(
                full_destination, command, source_node_tag
            )
        else:
            # Target exists - validate cross-tree iteration count matching for @each commands
            if command.inclusion_path and command.command_type.value == "each":
                self._validate_cross_tree_iteration_matching(
                    command, source_node_tag, full_destination
                )

        # Step 3: Validate variable mappings and register them
        if command.variable_mappings:
            # Phase 3a: Validate @all command RHS scoping rules FIRST (before field existence checks)
            self._validate_all_command_rhs_scoping(command, field_name, source_node_tag)

            # Phase 3b: Validate variable source fields exist (VariableSourceValidationError)
            for variable_mapping in command.variable_mappings:
                self._validate_variable_source_field(
                    variable_mapping.source_path, source_node_tag, command
                )

            # Phase 3c: Validate loop nesting constraints between inclusion path and variable mappings
            self._validate_variable_mapping_nesting(command, source_node_tag)

            # Phase 3d: Validate subchain matching for source paths AFTER field existence is confirmed
            self._validate_subchain_matching(command, source_node_tag)

        for variable_mapping in command.variable_mappings:
            if variable_mapping.resolved_target:
                target_scope = variable_mapping.resolved_target.scope
                variable_path = variable_mapping.resolved_target.path
                source_path = variable_mapping.source_path

                # Validate variable target structure exists
                self._validate_variable_target_structure(
                    variable_path, target_scope, source_node_tag, full_destination
                )

                # Register the variable target
                self._variable_registry.register_variable(
                    variable_path=variable_path,
                    scope=target_scope,
                    source_node_tag=source_node_tag,
                    command_type=command.command_type,  # Convert enum to string
                    has_multiplicity=command.has_multiplicity,
                )

                # Add satisfaction source
                self._variable_registry.add_satisfaction_source(
                    variable_path=variable_path,
                    scope=target_scope,
                    source_node_tag=source_node_tag,
                    source_path=source_path,
                    command_type=command.command_type,
                    has_multiplicity=command.has_multiplicity,
                )

        # NOTE: Context resolution is deferred until after tree building is complete
        # This prevents failures when trying to resolve forward references or non-existent fields
        # during tree construction. Context resolution will happen in validate_tree() or
        # when preparing execution chains.

    def _process_variable_assignment(
        self, command: VariableAssignmentCommand, source_node_tag: str
    ) -> None:
        """Process Assembly Variable assignment commands (! var=value).

        Stores Assembly Variables in the registry with conflict detection per LANGUAGE_SPECIFICATION.md.
        Assembly Variables are available from definition node through all descendant nodes.

        Params:
            command: Variable assignment command containing name and value
            source_node_tag: Tag of the node where this variable was defined

        Raises:
            AssemblyVariableConflictError: If variable already exists with different value
            LangTreeDSLError: If variable name conflicts with reserved template variables or field names
        """
        from langtree.exceptions import LangTreeDSLError
        from langtree.structure.registry import AssemblyVariableConflictError
        from langtree.templates.variables import (
            validate_template_variable_conflicts,
        )

        # Check for reserved template variable conflicts first
        template_conflicts = validate_template_variable_conflicts(
            "", {command.variable_name}
        )
        if template_conflicts:
            raise ValueError(f"Reserved variable name: {template_conflicts[0]}")

        # Check for field name conflicts per LANGUAGE_SPECIFICATION.md
        source_node = self.get_node(source_node_tag)
        if source_node and source_node.field_type:
            field_names = set(source_node.field_type.model_fields.keys())
            if command.variable_name in field_names:
                raise LangTreeDSLError(
                    f"Assembly Variable '{command.variable_name}' conflicts with field name "
                    f"in {source_node_tag}. Variable names cannot conflict with field names in same subtree."
                )

        try:
            self._assembly_variable_registry.store_variable(
                name=command.variable_name,
                value=command.value,
                source_node_tag=source_node_tag,
                defined_at_line=0,  # TODO: Add line number tracking if needed
            )
        except AssemblyVariableConflictError:
            # Re-raise with context for better error reporting
            raise

    def _complete_pending_command_processing(
        self, pending_target: PendingTarget
    ) -> None:
        """Complete processing of a command that was waiting for its target to be resolved.

        This method performs the comprehensive resolution and validation workflow for commands
        that were deferred due to forward references. It integrates with context resolution,
        variable mapping validation, and variable registry updates.

        Params:
            pending_target: The pending target with command and source information
        """
        # Now that the target exists, we can complete the deferred processing
        command = pending_target.command
        source_node_tag = pending_target.source_node_tag

        # Add the command to the source node's extracted_commands if not already there
        source_node = self.get_node(source_node_tag)
        if source_node and command not in source_node.extracted_commands:
            source_node.extracted_commands.append(command)

        try:
            # Step 1: Invoke inclusion context resolution if applicable
            if command.inclusion_path and command.resolved_inclusion:
                self._resolve_inclusion_context(command, source_node_tag)

            # Step 2: Invoke destination context resolution now that target node exists
            target_node = self.get_node(pending_target.target_path)
            if target_node:
                self._resolve_destination_context(
                    command, target_node, pending_target.target_path
                )

            # Step 2.5: Validate cross-tree iteration count matching for @each commands
            if (
                target_node
                and command.inclusion_path
                and command.command_type.value == "each"
            ):
                self._validate_cross_tree_iteration_matching(
                    command, source_node_tag, pending_target.target_path
                )

            # Step 3: Resolve each variable mapping (source + target) semantically
            for variable_mapping in command.variable_mappings:
                self._resolve_variable_mapping_context(
                    variable_mapping,
                    command,
                    source_node_tag,
                    pending_target.target_path,
                )

            # Step 4: Update variable registry entries from syntactic to semantic satisfaction
            self._update_variable_registry_satisfaction(command, source_node_tag)

        except Exception as e:
            # Let validation errors bubble up - these are intended to fail the operation
            from langtree.exceptions import FieldValidationError

            if isinstance(e, FieldValidationError):
                raise

            # For other exceptions, continue swallowing for now
            # TODO: Implement command.resolution_errors tracking
            pass

    def _resolve_inclusion_context(
        self, command: ParsedCommand, source_node_tag: str
    ) -> None:
        """
        Resolve inclusion context for @each commands with inclusion paths.

        Validates that the inclusion path points to an iterable field and performs
        semantic validation to ensure the command can execute successfully.

        Params:
            command: The parsed command with inclusion path information
            source_node_tag: Tag of the source node for context resolution

        Raises:
            ValueError: When inclusion path is invalid or not iterable
        """
        # Basic validation - check if inclusion path exists and is accessible
        # TODO: Implement actual iterable validation logic
        pass

    def _resolve_destination_context(
        self, command: ParsedCommand, target_node: StructureTreeNode, target_path: str
    ) -> None:
        """
        Resolve destination context now that target node exists.

        Validates that destination path exists, is accessible, and type-compatible
        with the variable mappings that will target it.

        Params:
            command: The parsed command with destination information
            target_node: The target structure tree node that was just added
            target_path: Full path to the target node

        Raises:
            ValueError: When destination is incompatible with variable mappings
        """
        # Validate that target node has a field_type (required for variable mapping)
        if not target_node.field_type:
            raise ValueError(
                f"Target node '{target_path}' has no field_type defined - cannot map variables"
            )

        # For each variable mapping, check if the target field exists
        for variable_mapping in command.variable_mappings:
            if (
                variable_mapping.resolved_target
                and variable_mapping.resolved_target.scope
            ):
                scope_name = variable_mapping.resolved_target.scope.get_name()
                field_path = variable_mapping.resolved_target.path

                # For 'value' scope, check if the field exists in the target node's field_type
                if scope_name == "value":
                    try:
                        # Check if field exists in the target node's model
                        if hasattr(target_node.field_type, "model_fields"):
                            if field_path not in target_node.field_type.model_fields:
                                raise ValueError(
                                    f"Target field '{field_path}' not found in {target_path}"
                                )
                    except Exception as e:
                        raise ValueError(
                            f"Cannot validate target field '{field_path}' in {target_path}: {e}"
                        )

                # TODO: Add validation for other scopes (outputs, prompt, task)

    def _resolve_variable_mapping_context(
        self,
        variable_mapping,
        command: ParsedCommand,
        source_node_tag: str,
        target_path: str,
    ) -> None:
        """
        Resolve a single variable mapping semantically.

        Validates that source paths exist and are reachable from source node context.
        Validates that target variables exist in destination and are type-compatible.

        Params:
            variable_mapping: The variable mapping to resolve
            command: The parent command containing this mapping
            source_node_tag: Tag of the source node for context resolution
            target_path: Full path to the target node

        Raises:
            ValueError: When variable mapping is invalid or incompatible
        """
        # Validate source path exists and is accessible
        source_node = self.get_node(source_node_tag)
        if not source_node or not source_node.field_type:
            raise ValueError(
                f"Source node '{source_node_tag}' not found or has no field_type"
            )

        # Check source field accessibility
        if variable_mapping.resolved_source and variable_mapping.resolved_source.scope:
            scope_name = variable_mapping.resolved_source.scope.get_name()
            field_path = variable_mapping.resolved_source.path

            # For 'prompt' scope, check if the field exists in the source node
            if scope_name == "prompt":
                if hasattr(source_node.field_type, "model_fields"):
                    if field_path not in source_node.field_type.model_fields:
                        raise ValueError(
                            f"Source field '{field_path}' not found in {source_node_tag}"
                        )

            # TODO: Add validation for other source scopes (value, outputs, task)

        # Track outputs collection for multiple sources to same field
        if variable_mapping.resolved_target and variable_mapping.resolved_target.scope:
            target_scope_name = variable_mapping.resolved_target.scope.get_name()
            if target_scope_name == "outputs":
                # Track this for collection
                target_node = self.get_node(target_path)
                if target_node:
                    from langtree.execution.resolution import _track_outputs_collection

                    _track_outputs_collection(
                        self,
                        variable_mapping.resolved_target.path,
                        variable_mapping.resolved_source.path
                        if variable_mapping.resolved_source
                        else "*",
                        source_node_tag,
                        target_node,
                    )

        # Target validation is handled by _resolve_destination_context
        # TODO: Add type compatibility checking between source and target

    def _update_variable_registry_satisfaction(
        self, command: ParsedCommand, source_node_tag: str
    ) -> None:
        """
        Update variable registry entries from syntactic to semantic satisfaction.

        Marks variables as semantically satisfied with resolved source paths.
        Updates relationship types (1:1, 1:n, n:n) based on command analysis.

        Params:
            command: The resolved command with variable mappings
            source_node_tag: Tag of the source node for registry updates
        """
        # TODO: Implement variable registry satisfaction updates
        pass

    def get_assembly_variable_registry(self) -> AssemblyVariableRegistry:
        """Get the Assembly Variable Registry for cross-module variable access.

        Returns:
            The AssemblyVariableRegistry instance containing all Assembly Variables
        """
        return self._assembly_variable_registry

    def resolve_runtime_variable(
        self, variable_path: str, current_node: StructureTreeNode
    ) -> str:
        """
        Resolve runtime variables {{variable}} for content processing.

        This method provides the core runtime variable resolution capability required by
        the LANGUAGE_SPECIFICATION.md. It resolves variables from execution context only.

        Params:
            variable_path: The variable path to resolve (e.g., "field", "task.field")
            current_node: Current node context for variable resolution

        Returns:
            Resolved variable value as string

        Raises:
            RuntimeVariableError: When variable resolution fails
        """
        try:
            # Import resolution functions from resolution module
            from langtree.execution.resolution import _resolve_regular_variable

            # Regular runtime variable ({{variable}}) - execution context only
            return _resolve_regular_variable(self, variable_path, current_node)

        except Exception as e:
            from langtree.exceptions import RuntimeVariableError

            raise RuntimeVariableError(
                f"Failed to resolve runtime variable '{{{{{variable_path}}}}}': {str(e)}"
            )

    def get_node(self, tag: str) -> StructureTreeNode | None:
        """Retrieve a node by fully qualified tag.

        Params:
            tag: Dot‑separated hierarchical tag (e.g., `task.analysis.sections`).

        Returns:
            The `StructureTreeNode` if present; otherwise `None`.
        """
        path = tag.split(".")
        node = self._root_nodes.get(path[0])
        for name in path[1:]:
            if node is None:
                break
            node = node.children.get(name)
        return node

    def get_prompt_sequence(self, name: str) -> list[str]:
        """Construct ordered prompt content segments leading to a descendant field.

        This traverses parent classes (MRO up to `TreeNode`) and collects:
        - Parent class docstrings (raw currently; TODO processing hooks pending)
        - Target field description found in the terminal node.

        Params:
            name: Fully qualified path to a field (e.g., `task.analysis.sections`).

        Returns:
            Ordered list of textual segments (docstrings / field descriptions) forming a composite prompt context.

        TODO:
            Integrate cleaned docstring/field description content rather than raw unprocessed strings.
            See: tests/langtree/prompt/test_todos.py::test_cleaned_prompt_content_integration
        """
        from itertools import pairwise

        path = name.split(".")
        node = self._root_nodes[path[0]]
        prompts = []
        for tag, tag_next in pairwise(path[1:]):
            node = node.children.get(tag)
            type_mro = node.field_type.__mro__
            for parent in type_mro[: type_mro.index(TreeNode)]:
                parent_descr = parent.__doc__ or ""  # TODO: process prompt
                # See: tests/langtree/prompt/test_todos.py::test_prompt_template_processing
                prompts.append(parent_descr)
            field_descr = node.field_type.model_fields[
                tag_next
            ].description  # TODO: process prompt
            # See: tests/langtree/prompt/test_todos.py::test_field_description_processing
            prompts.append(field_descr)
        return prompts

    def get_execution_summary(self) -> dict:
        """Summarize current variable + target graph state.

        Returns:
            dict with counts and relationship distribution:
            - total_variables
            - satisfied_variables (>=1 satisfaction source)
            - unsatisfied_variables (no sources or invalid sources)
            - pending_targets (unresolved destination references)
            - relationship_types: counts by 1:1 / 1:n / n:n / unknown
        """
        return {
            "total_variables": len(self._variable_registry.variables),
            "satisfied_variables": len(
                [
                    v
                    for v in self._variable_registry.variables.values()
                    if v.is_satisfied()
                ]
            ),
            "unsatisfied_variables": len(
                self._variable_registry.get_unsatisfied_variables()
            ),
            "pending_targets": len(self._pending_target_registry.pending_targets),
            "relationship_types": {
                "1:1": len(
                    [
                        source
                        for v in self._variable_registry.variables.values()
                        for source in v.sources
                        if source.get_relationship_type() == "1:1"
                    ]
                ),
                "1:n": len(
                    [
                        source
                        for v in self._variable_registry.variables.values()
                        for source in v.sources
                        if source.get_relationship_type() == "1:n"
                    ]
                ),
                "n:n": len(
                    [
                        source
                        for v in self._variable_registry.variables.values()
                        for source in v.sources
                        if source.get_relationship_type() == "n:n"
                    ]
                ),
            },
        }

    def get_execution_plan(self) -> dict:
        """Produce a preliminary (heuristic) execution plan description.

        Current heuristic (placeholder):
        - Each node with >=1 commands becomes a chain step (no dependency ordering yet).
        - Unsatisfied variables become `external_inputs` entries.
        - Pending destinations become `unresolved_issues`.
        - Satisfied variable sources recorded as `variable_flows`.

        Returns:
            dict with keys: `chain_steps`, `external_inputs`, `variable_flows`, `unresolved_issues`.

        NOTE:
            This does not yet perform full topological ordering or multiplicity expansion; future work will refine.
        """
        plan = {
            "chain_steps": [],
            "external_inputs": [],
            "variable_flows": [],
            "unresolved_issues": [],
        }

        # Identify external inputs (unsatisfied variables)
        unsatisfied_vars = self._variable_registry.get_truly_unsatisfied_variables(self)
        for var_info in unsatisfied_vars:
            scope_name = var_info.get_scope_name()
            full_name = (
                f"{scope_name}.{var_info.variable_path}"
                if scope_name
                else var_info.variable_path
            )
            # Get the first source node tag if available
            first_source_tag = (
                var_info.sources[0].source_node_tag if var_info.sources else "unknown"
            )
            first_relationship_type = (
                var_info.sources[0].get_relationship_type()
                if var_info.sources
                else "unknown"
            )
            plan["external_inputs"].append(
                {
                    "variable": full_name,
                    "source_node": first_source_tag,
                    "scope": scope_name,
                    "path": var_info.variable_path,
                    "required_type": first_relationship_type,
                }
            )

        # Identify unresolved targets as blocking issues
        for (
            target_path,
            pending_list,
        ) in self._pending_target_registry.pending_targets.items():
            plan["unresolved_issues"].append(
                {
                    "type": "unresolved_target",
                    "target": target_path,
                    "referenced_by": [
                        pending.source_node_tag for pending in pending_list
                    ],
                    "command_count": len(pending_list),
                }
            )

        # Analyze variable flows and satisfaction relationships
        for var_name, var_info in self._variable_registry.variables.items():
            if var_info.is_satisfied():
                for source in var_info.sources:
                    plan["variable_flows"].append(
                        {
                            "from": source.source_field_path,
                            "to": var_name,
                            "target_node": source.source_node_tag,
                            "relationship_type": source.get_relationship_type(),
                            "scope": var_info.get_scope_name(),
                        }
                    )

        # Organize nodes into potential execution steps
        # This is a simplified version - could be enhanced with dependency analysis
        processed_nodes = set()

        # First, collect all nodes that are referenced as targets
        referenced_nodes = set()
        for node_name, node in self._root_nodes.items():
            self._collect_referenced_nodes(node, referenced_nodes)

        # Add all nodes to execution plan - both nodes with commands and referenced target nodes
        for node_name, node in self._root_nodes.items():
            self._add_node_to_execution_plan(
                node, plan, processed_nodes, referenced_nodes
            )

        return plan

    def _collect_referenced_nodes(
        self, node: StructureTreeNode, referenced_nodes: set
    ) -> None:
        """Helper method to collect all nodes that are referenced as command targets."""
        if hasattr(node, "extracted_commands") and node.extracted_commands:
            for command in node.extracted_commands:
                if hasattr(command, "destination_path") and command.destination_path:
                    # Try to find the actual target node
                    target_node = self.get_node(command.destination_path)
                    if target_node:
                        referenced_nodes.add(target_node.name)
                    else:
                        # Try alternative paths for the destination
                        from langtree.templates.utils import underscore

                        try:
                            underscore_name = underscore(command.destination_path)
                            if "_" in underscore_name:
                                parts = underscore_name.split("_", 1)
                                if len(parts) == 2:
                                    alt_path = f"{parts[0]}.{parts[1]}"
                                    target_node = self.get_node(alt_path)
                                    if target_node:
                                        referenced_nodes.add(target_node.name)
                        except Exception:
                            pass

        # Recursively check children
        for child in node.children.values():
            self._collect_referenced_nodes(child, referenced_nodes)

    def _add_node_to_execution_plan(
        self,
        node: StructureTreeNode,
        plan: dict,
        processed_nodes: set,
        referenced_nodes: set,
    ) -> None:
        """Helper method to add a node and its children to the execution plan."""
        if node.name in processed_nodes:
            return

        processed_nodes.add(node.name)

        # Add this node as an execution step if it has commands OR if it's referenced by other commands
        has_commands = hasattr(node, "extracted_commands") and node.extracted_commands
        is_referenced = node.name in referenced_nodes

        if has_commands or is_referenced:
            step = {
                "node_tag": node.name,
                "commands": len(getattr(node, "extracted_commands", [])),
                "clean_prompt": getattr(node, "clean_docstring", None),
                "field_descriptions": getattr(node, "clean_field_descriptions", {}),
                "dependencies": [],  # Could be enhanced with actual dependency analysis
                "is_terminal": not has_commands
                and is_referenced,  # Mark terminal nodes
            }
            plan["chain_steps"].append(step)

        # Recursively process children
        for child in node.children.values():
            self._add_node_to_execution_plan(
                child, plan, processed_nodes, referenced_nodes
            )

    # Validation methods (delegated to validation module)
    def validate_tree(self) -> dict[str, list[str]]:
        """
        Run basic structural validation on the prompt tree.

        Performs fundamental validation checks including unresolved targets,
        unsatisfied variables, and multiply satisfied variables. This is a
        lightweight validation pass that focuses on structural integrity
        rather than semantic correctness.

        Returns:
            Dictionary with validation results containing keys:
            - 'unresolved_targets': List of target paths that couldn't be resolved
            - 'unsatisfied_variables': List of variables that lack sources
            - 'multiply_satisfied_variables': List of variables with multiple sources
        """
        from langtree.structure.validation import validate_tree

        return validate_tree(self)

    def validate_comprehensive(self) -> dict:
        """
        Perform comprehensive semantic validation of the prompt tree.

        Executes a full suite of validation checks beyond basic structural
        validation, including semantic analysis, dependency validation, and
        error categorization. This provides the complete validation picture
        required for execution planning and error reporting.

        Returns:
            Dictionary with comprehensive validation results containing:
            - 'circular_dependencies': List of detected circular references
            - 'unresolved_targets': Target paths that couldn't be resolved
            - 'unsatisfied_variables': Variables without valid sources
            - 'invalid_scopes': Scope references that are malformed or unreachable
            - 'malformed_commands': Commands with syntax or structural errors
            - 'impossible_mappings': Variable mappings that cannot be satisfied
            - 'self_references': Commands that reference themselves
            - 'error_summary': Synthesized summary of all validation issues
        """
        from langtree.structure.validation import validate_comprehensive

        return validate_comprehensive(self)

    # Resolution methods (delegated to resolution module)
    def resolve_deferred_contexts(self) -> dict:
        """
        Attempt Phase 2 context resolution for all deferred commands.

        Executes context resolution for commands that were deferred during tree
        construction due to forward references or missing targets. This is a
        best-effort operation that attempts to resolve as many contexts as
        possible without failing on unresolvable references.

        Returns:
            Dictionary containing resolution results and status information

        Raises:
            ResolutionError: When critical resolution failures prevent execution
        """
        from langtree.execution.resolution import resolve_deferred_contexts

        return resolve_deferred_contexts(self)

    def _resolve_in_current_node_context(self, path: str, node_tag: str):
        """
        Resolve a path within the current node's context.

        Attempts to resolve a given path by looking within the specified node's
        available fields and attributes. This is used for resolving references
        that should be satisfied by the current node's content or metadata.

        Params:
            path: The field path to resolve within the node context
            node_tag: Identifier of the node to search within

        Returns:
            Resolved value if path exists, None otherwise
        """
        from langtree.execution.resolution import _resolve_in_current_node_context

        return _resolve_in_current_node_context(self, path, node_tag)

    def _resolve_scope_segment_context(self, path: str, node_tag: str):
        """
        Resolve a scoped path segment with explicit scope prefix.

        Handles resolution of paths that include scope prefixes (e.g., 'prompt.',
        'outputs.', 'value.'). Routes the resolution to the appropriate scope
        resolver based on the prefix and validates scope accessibility.

        Params:
            path: The scoped path to resolve (includes scope prefix)
            node_tag: Identifier of the node providing the resolution context

        Returns:
            Resolved value if scope and path are valid, None otherwise

        Raises:
            ScopeError: When scope prefix is invalid or inaccessible
        """
        from langtree.execution.resolution import _resolve_scope_segment_context

        return _resolve_scope_segment_context(self, path, node_tag)

    def _resolve_in_value_context(self, path: str, node_tag: str):
        """Resolve a path within the value context."""
        from langtree.execution.resolution import _resolve_in_value_context

        return _resolve_in_value_context(self, path, node_tag)

    def _resolve_in_outputs_context(self, path: str, node_tag: str):
        """Resolve a path within the outputs context."""
        from langtree.execution.resolution import _resolve_in_outputs_context

        return _resolve_in_outputs_context(self, path, node_tag)

    def _resolve_in_global_tree_context(self, path: str):
        """Resolve a path within the global tree context."""
        from langtree.execution.resolution import _resolve_in_global_tree_context

        return _resolve_in_global_tree_context(self, path)

    def _resolve_in_target_node_context(self, path: str, target_node):
        """Resolve a path within a target node context."""
        from langtree.execution.resolution import _resolve_in_target_node_context

        return _resolve_in_target_node_context(self, path, target_node)

    def _resolve_in_current_prompt_context(self, path: str, node_tag: str):
        """Resolve a path within the current prompt context."""
        from langtree.execution.resolution import _resolve_in_current_prompt_context

        return _resolve_in_current_prompt_context(self, path, node_tag)

    def resolve_full_path(self, full_path: str) -> Any:
        """
        Resolve a full path like 'task.node.field' directly to its value.

        This is a convenience method that combines node and field resolution
        into a single call. It automatically splits the path and routes to
        the appropriate resolution method.

        Params:
            full_path: Complete path like 'task.node.field' or 'task.node'

        Returns:
            The resolved value at the path

        Raises:
            ValueError: When path format is invalid
            KeyError: When path doesn't exist in the tree

        Examples:
            result = run_structure.resolve_full_path("task.analysis.title")
            node = run_structure.resolve_full_path("task.analysis")
        """
        if not full_path or not isinstance(full_path, str):
            raise ValueError("Path must be a non-empty string")

        # Check if it's a global tree path (has at least one dot)
        if "." in full_path:
            # Try as global tree path first
            try:
                return self._resolve_in_global_tree_context(full_path)
            except (KeyError, ValueError) as e:
                # If global resolution fails, try splitting into node.field
                parts = full_path.rsplit(".", 1)
                if len(parts) == 2:
                    node_path, field_name = parts
                    try:
                        return self._resolve_in_current_node_context(
                            field_name, node_path
                        )
                    except (KeyError, ValueError):
                        pass
                # Re-raise original error
                raise e
        else:
            # Single component - treat as field in current context if we have one
            raise ValueError(
                f"Path '{full_path}' must contain at least one dot for node.field format"
            )

    def _validate_variable_source_field(
        self, source_path: str, source_node_tag: str, command: Any = None
    ) -> None:
        """
        Validate that variable source field exists in referenced structure.

        Validates that source paths in variable mappings (e.g., "sections.title", "nonexistent_field")
        actually exist in the referenced structure's Pydantic model.

        Params:
            source_path: Path to the source field (e.g., "sections.title", "nonexistent_field")
            source_node_tag: Tag of the node where this source is referenced

        Raises:
            VariableSourceValidationError: If source field does not exist
        """

        # Skip validation for wildcard sources
        if source_path == "*":
            return

        # Handle @each vs @all commands differently per LANGUAGE_SPECIFICATION.md
        if command and hasattr(command, "command_type"):
            if (
                command.command_type.value == "each"
                and hasattr(command, "inclusion_path")
                and command.inclusion_path
            ):
                # @each commands: validate iteration variable field existence
                inclusion_parts = command.inclusion_path.split(".")
                iteration_collection = inclusion_parts[
                    -1
                ]  # e.g., "sections" from "document.sections"

                path_components = source_path.split(".")
                if path_components[0] == iteration_collection:
                    # This is an iteration variable access (e.g., "sections.title" for @each[sections])
                    # Validate that the dotted field exists on the iteration item type
                    remaining_components = path_components[
                        1:
                    ]  # Skip iteration variable name itself
                    if remaining_components:
                        iteration_item_type = self._get_iteration_item_type(
                            source_node_tag, iteration_collection
                        )
                        self._validate_field_path_exists(
                            remaining_components,
                            iteration_item_type,
                            source_path,
                            source_node_tag,
                        )
                    return
            elif command.command_type.value == "all":
                # @all commands: already handled by _validate_all_command_rhs_scoping()
                # No dotted variables allowed in RHS at all - strict locality principle
                return

        source_node = self.get_node(source_node_tag)
        if not source_node or not source_node.field_type:
            return  # Skip validation if source node not found

        path_components = source_path.split(".")

        # Handle scoped variables (e.g., prompt.data, value.field, outputs.result)
        known_scopes = ["prompt", "value", "outputs", "task"]
        if len(path_components) >= 2 and path_components[0] in known_scopes:
            scope_prefix = path_components[0]
            remaining_path = ".".join(path_components[1:])

            if scope_prefix == "prompt":
                # For prompt scope, validate that the remaining path exists in the current node
                remaining_components = path_components[1:]
                if hasattr(source_node.field_type, "model_fields"):
                    self._validate_field_path_exists(
                        remaining_components,
                        source_node.field_type,
                        remaining_path,
                        source_node_tag,
                    )
                return
            elif scope_prefix in ["value", "outputs", "task"]:
                # For other scopes, we defer validation to runtime resolution
                # as these depend on execution context
                return

        # Validate the complete path using same logic as inclusion field validation
        if hasattr(source_node.field_type, "model_fields"):
            self._validate_field_path_exists(
                path_components, source_node.field_type, source_path, source_node_tag
            )

    def _validate_variable_target_structure(
        self,
        variable_path: str,
        target_scope,
        source_node_tag: str,
        target_node_tag: str = None,
    ) -> None:
        """
        Validate that variable target structure can be satisfied.

        Validates that target variable paths (e.g., "main_analysis.title") can be
        created by checking if the target structure exists in the appropriate scope.

        Params:
            variable_path: Path to the target variable (e.g., "main_analysis.title")
            target_scope: Scope object for the target variable
            source_node_tag: Tag of the node where this target is declared
            target_node_tag: Tag of the target node where this variable will be resolved

        Raises:
            VariableTargetValidationError: If target structure cannot be satisfied
        """
        from langtree.exceptions import VariableTargetValidationError

        # For value scope, check if the path can be created in the current node structure
        if target_scope.get_name() == "value":
            path_components = variable_path.split(".")
            if len(path_components) > 1:
                # Check if the parent structure exists (e.g., "main_analysis" in "main_analysis.title")
                # UNCOMMENTED FIX: Use target node context, not source node context
                if target_node_tag:
                    target_node = self.get_node(target_node_tag)
                    validation_node = target_node
                else:
                    # Fallback to source node if target not provided (backward compatibility)
                    validation_node = self.get_node(source_node_tag)

                # ORIGINAL CODE (BUGGY): validates against source node instead of target node
                # source_node = self.get_node(source_node_tag)
                # validation_node = source_node
                # validation_tag = source_node_tag

                if validation_node and validation_node.field_type:
                    first_component = path_components[0]
                    if (
                        hasattr(validation_node.field_type, "model_fields")
                        and first_component
                        not in validation_node.field_type.model_fields
                    ):
                        raise VariableTargetValidationError(
                            target_path=variable_path,
                            source_node=source_node_tag,
                            reason=f"target structure '{first_component}' does not exist",
                        )

    def _validate_inclusion_field(
        self, inclusion_path: str, source_node_tag: str
    ) -> None:
        """
        Validate that inclusion field exists in the source structure.

        Validates that @each inclusion paths (e.g., "books.chapters") reference
        fields that actually exist in the source node structure.

        Params:
            inclusion_path: Path to the inclusion field (e.g., "books.chapters")
            source_node_tag: Tag of the node where this inclusion is declared

        Raises:
            FieldValidationError: If inclusion field does not exist
        """
        from typing import get_args, get_origin

        from langtree.exceptions import FieldValidationError

        source_node = self.get_node(source_node_tag)
        if not source_node or not source_node.field_type:
            return  # Skip validation if source node not found or has no type

        path_components = inclusion_path.split(".")
        current_type = source_node.field_type

        # Validate each component of the path step by step
        for i, component in enumerate(path_components):
            # Check if this component exists as a field in current type
            if not hasattr(current_type, "model_fields"):
                # Can't validate further if not a TreeNode
                break

            if component not in current_type.model_fields:
                partial_path = ".".join(path_components[: i + 1])
                raise FieldValidationError(
                    field_path=partial_path,
                    container=current_type.__name__,
                    message=f"inclusion field '{component}' does not exist",
                )

            # Move to the next type in the path
            field_def = current_type.model_fields[component]
            field_type = field_def.annotation

            # Only continue validation if there are more components
            if i < len(path_components) - 1:
                # Handle list types by extracting the element type
                if get_origin(field_type) is list:
                    args = get_args(field_type)
                    if args and hasattr(args[0], "model_fields"):
                        current_type = args[0]
                    else:
                        break  # Can't validate further if not a TreeNode
                elif hasattr(field_type, "model_fields"):
                    current_type = field_type
                else:
                    # The field exists but we can't validate nested paths within it
                    break

    def _validate_inclusion_base_field_iterable(
        self, inclusion_path: str, source_node_tag: str
    ) -> None:
        """
        Validate that the base field in inclusion path is iterable for @each commands.

        For @each commands like "@each[config.items]", validates that the base field 'config'
        is iterable (list type), not a single object. This prevents confusions where users
        try to iterate over single objects.

        Params:
            inclusion_path: Path to the inclusion field (e.g., "config.items")
            source_node_tag: Tag of the node where this inclusion is declared

        Raises:
            FieldValidationError: If base field is not iterable
        """
        from typing import get_origin

        from langtree.exceptions import FieldValidationError

        source_node = self.get_node(source_node_tag)
        if not source_node or not source_node.field_type:
            return  # Skip validation if source node not found or has no type

        path_components = inclusion_path.split(".")
        base_field = path_components[0]

        # Check if base field exists
        if not hasattr(source_node.field_type, "model_fields"):
            return  # Can't validate if not a TreeNode

        if base_field not in source_node.field_type.model_fields:
            # This should be caught by _validate_inclusion_field, but just in case
            return

        # Check if base field is iterable
        field_def = source_node.field_type.model_fields[base_field]
        field_type = field_def.annotation

        # For @each commands, the base field must be iterable (list type)
        if get_origin(field_type) is not list:
            raise FieldValidationError(
                field_path=base_field,
                container=source_node_tag,
                message=f"base field '{base_field}' must be iterable (list type) for @each command. "
                f"Found {field_type}. Use @all for single objects or make field a list.",
                command_context=f"@each[{inclusion_path}] requires base field to be list type",
            )

    def _validate_command_field_compatibility(
        self, command: "ParsedCommand", field_name: str, field_def, source_node_tag: str
    ) -> None:
        """
        Validate that command types are compatible with the field they're defined on.

        For iteration commands (@each), the field must be iterable (list type) unless
        it's a cross-tree reference where results are sent to another structure.

        Params:
            command: Parsed command to validate
            field_name: Name of field where command is defined
            field_def: Field definition from Pydantic model
            source_node_tag: Tag of the node containing the field

        Raises:
            FieldValidationError: If command type incompatible with field type
        """
        from typing import get_origin

        from langtree.exceptions import FieldValidationError

        # Only validate iteration commands
        if not hasattr(command, "command_type") or command.command_type.value != "each":
            return

        field_type = field_def.annotation
        origin = get_origin(field_type)

        # Check if this is a cross-tree reference (destination goes to different registered structure)
        is_cross_tree = False
        if hasattr(command, "destination_path") and command.destination_path:
            dest_path = command.destination_path
            # Cross-tree means the destination is a different registered structure node
            if source_node_tag != dest_path:
                destination_node = self.get_node(dest_path)
                is_cross_tree = destination_node is not None

        # For @each commands on same-tree references, the field must be iterable (list type)
        # For cross-tree references, the field doesn't need to be iterable since results go elsewhere
        if origin is not list and not is_cross_tree:
            command_context = (
                f"@each[{command.inclusion_path}]->{command.destination_path}@{{...}}*"
            )

            raise FieldValidationError(
                field_path=field_name,
                container=source_node_tag,
                message=f"iteration command @each cannot be defined on non-iterable field of type {field_type}. "
                f"@each creates multiple iterations which require an iterable field (list type). "
                f"Use @all for single objects or change field to list type.",
                command_context=command_context,
            )

    def _validate_field_path_exists(
        self,
        path_components: list[str],
        field_type: type,
        source_path: str,
        source_node_tag: str,
    ) -> None:
        """
        Helper method to validate that a field path exists in the given type structure.

        Reused by both inclusion field validation and variable source validation.

        Params:
            path_components: List of path components (e.g., ["sections", "title"])
            field_type: The Pydantic model type to validate against
            source_path: Original source path for error messages
            source_node_tag: Node tag for error context

        Raises:
            VariableSourceValidationError: If source field does not exist
        """
        from langtree.exceptions import VariableSourceValidationError

        current_type = field_type

        for i, component in enumerate(path_components):
            if not hasattr(current_type, "model_fields"):
                # Can't validate further - not a Pydantic model
                # If there are remaining components to check, this is an error
                remaining_path = ".".join(path_components[i:])
                raise VariableSourceValidationError(
                    source_path=source_path,
                    structure_type=f"{current_type.__name__ if hasattr(current_type, '__name__') else str(current_type)}",
                    command_context=f"Cannot access '{remaining_path}' on {current_type.__name__ if hasattr(current_type, '__name__') else str(current_type)} type",
                )

            if component not in current_type.model_fields:
                raise VariableSourceValidationError(
                    source_path=source_path,
                    structure_type=current_type.__name__,
                    command_context=source_node_tag,
                )

            # Get the field type for next iteration
            field_def = current_type.model_fields[component]
            field_annotation = field_def.annotation

            # Handle list/sequence types: list[ItemType] -> ItemType
            if hasattr(field_annotation, "__origin__"):
                if field_annotation.__origin__ is list and hasattr(
                    field_annotation, "__args__"
                ):
                    if field_annotation.__args__:
                        current_type = field_annotation.__args__[0]
                        continue

            # Direct type assignment
            current_type = field_annotation

    def _validate_variable_mapping_nesting(
        self, command: "ParsedCommand", source_node_tag: str
    ) -> None:
        """
        Validate loop nesting constraints between inclusion path and variable mappings.

        Per LANGUAGE_SPECIFICATION.md:
        - LHS must have iteration structure matching or fewer than source
        - At least one mapping must match iteration level
        - RHS must start from iteration root path
        - None can exceed iteration level
        """
        from langtree.exceptions import FieldValidationError

        # Calculate iteration levels from inclusion path
        iteration_levels = 0
        iteration_root = None

        if command.inclusion_path:
            # Count actual iterable levels in inclusion path, not just path components
            path_components = command.inclusion_path.split(".")
            iteration_levels = self._count_iterable_levels_in_path(
                path_components, source_node_tag
            )
            iteration_root = path_components[0]

        # Track nesting levels for all LHS mappings
        lhs_nesting_levels = []

        for variable_mapping in command.variable_mappings:
            # Calculate LHS nesting level from variable path like "value.results.items"
            if (
                variable_mapping.resolved_target
                and variable_mapping.resolved_target.path
            ):
                lhs_path = variable_mapping.resolved_target.path
                lhs_components = lhs_path.split(".")

                # For value scope, we can't validate against source node types since
                # target fields are in destination node. Skip detailed LHS validation
                # but still track for "at least one must match" requirement
                # For ALL scopes, only validate nesting for existing LHS fields
                # Non-existing LHS fields are destination fields that will be validated during assembly

                # Check if the field actually exists in the source node
                source_node = self.get_node(source_node_tag)
                field_exists = False
                if (
                    source_node
                    and source_node.field_type
                    and hasattr(source_node.field_type, "model_fields")
                ):
                    first_component = lhs_components[0] if lhs_components else ""
                    field_exists = (
                        first_component in source_node.field_type.model_fields
                    )

                if field_exists:
                    # Field exists in source - validate its nesting level (applies to ALL scopes)
                    lhs_nesting = self._count_iterable_levels(
                        lhs_components, source_node_tag
                    )
                    lhs_nesting_levels.append(lhs_nesting)

                    # Validate: no LHS can exceed iteration level
                    if lhs_nesting > iteration_levels:
                        raise FieldValidationError(
                            field_path=lhs_path,
                            container=source_node_tag,
                            message=f"has {lhs_nesting} nesting levels which exceeds iteration level {iteration_levels} from inclusion path '{command.inclusion_path}'",
                        )
                # else: Field doesn't exist - it's a destination field, skip LHS-RHS nesting validation
                # This will be validated during assembly when destination structure is known (for ALL scopes)

            # Validate: RHS must start from iteration root OR be a field in the same context
            if iteration_root and command.inclusion_path:
                rhs_path = variable_mapping.source_path

                # Skip scope-modified paths (handled by different validation)
                if "." in rhs_path and rhs_path.split(".")[0] in (
                    "prompt",
                    "value",
                    "outputs",
                    "task",
                ):
                    continue

                # Allow if path starts with iteration root (subchain)
                if rhs_path.startswith(iteration_root):
                    continue

                # Allow if path starts with iteration variable
                # For @each[document.sections], allow "sections.title" (sections is the iteration variable)
                inclusion_parts = command.inclusion_path.split(".")
                iteration_collection = inclusion_parts[
                    -1
                ]  # e.g., "sections" from "document.sections"
                if rhs_path.split(".")[0] == iteration_collection:
                    continue

                # TODO: This exception needs specification clarification
                # Allow same-context field access for value scope (value forwarding pattern)
                # This enables forwarding context data along with iteration data
                # NOTE: This may be too permissive and should be refined based on use cases
                current_mapping_scope = None
                for mapping in command.variable_mappings:
                    if mapping.source_path == rhs_path and mapping.resolved_target:
                        current_mapping_scope = mapping.resolved_target.scope
                        break

                if (
                    current_mapping_scope
                    and current_mapping_scope.get_name() == "value"
                ):
                    # Check if it's a valid field in the same context
                    source_node = self.get_node(source_node_tag)
                    if (
                        source_node
                        and source_node.field_type
                        and hasattr(source_node.field_type, "model_fields")
                        and rhs_path in source_node.field_type.model_fields
                    ):
                        continue

                # If none of the above, fail
                raise FieldValidationError(
                    field_path=rhs_path,
                    container=source_node_tag,
                    message=f"must start from iteration root '{iteration_root}' when using @each[{command.inclusion_path}] "
                    f"or be a field in the same context",
                )

        # Check for cross-tree references before applying local nesting validation
        has_cross_tree_reference = False
        for variable_mapping in command.variable_mappings:
            if variable_mapping.resolved_target:
                target_scope = variable_mapping.resolved_target.scope
                variable_path = variable_mapping.resolved_target.path

                # Cross-tree references in value scope with nested paths skip local nesting validation
                if (
                    target_scope
                    and target_scope.get_name() == "value"
                    and "." in variable_path
                    and not variable_path.startswith(command.inclusion_path or "")
                ):
                    has_cross_tree_reference = True
                    break

        # Skip local nesting validation for cross-tree references - they are handled by cross-tree iteration matching
        if has_cross_tree_reference:
            return

        # Validate: at least one mapping must match iteration level exactly (if we have iteration and mappings)
        if iteration_levels > 0 and lhs_nesting_levels:
            if not any(level == iteration_levels for level in lhs_nesting_levels):
                raise FieldValidationError(
                    field_path=command.inclusion_path,
                    container=source_node_tag,
                    message=f"requires at least one variable mapping to match iteration level {iteration_levels}, but found nesting levels: {lhs_nesting_levels}",
                )

    def _validate_field_inheritance(
        self, field_tag: str, annotation: type, origin: type, args: tuple
    ) -> None:
        """
        Validate that field types use proper inheritance (TreeNode, not raw BaseModel).

        Per framework requirements, all model classes must inherit from TreeNode.
        Raw BaseModel inheritance is not allowed in the LangTree DSL framework.
        Also validates against list[list[...]] antipatterns.

        Params:
            field_tag: Full tag path of the field for error reporting
            annotation: The field's type annotation
            origin: Generic origin type (e.g., list, Union)
            args: Generic type arguments

        Raises:
            FieldTypeError: If a BaseModel class that doesn't inherit from TreeNode is found
            FieldValidationError: If list[list[...]] antipattern is detected
        """

        # Validate against bare collection types (list, dict, set without type parameters)
        self._validate_against_bare_collection_types(
            field_tag, annotation, origin, args
        )

        # TODO: Add list[list[...]] antipattern validation

        # Check the annotation itself
        self._check_single_type_inheritance(field_tag, annotation)

        # Check generic type arguments (e.g., list[SomeModel])
        if origin is not None and args is not None:
            for type_candidate in args:
                self._check_single_type_inheritance(field_tag, type_candidate)

    def _check_single_type_inheritance(
        self, field_tag: str, type_to_check: type
    ) -> None:
        """
        Check a single type for proper inheritance.

        Params:
            field_tag: Field tag for error reporting
            type_to_check: Type to validate

        Raises:
            FieldTypeError: If the type is BaseModel but not TreeNode
        """

        # Skip built-in types and primitives
        if not hasattr(type_to_check, "__module__") or type_to_check.__module__ in (
            "builtins",
            "typing",
        ):
            return

        # If it's a class that inherits from BaseModel
        try:
            if issubclass(type_to_check, BaseModel) and not issubclass(
                type_to_check, TreeNode
            ):
                raise FieldTypeError(
                    field_tag,
                    f"uses BaseModel inheritance ({type_to_check.__name__}) which is not allowed. "
                    f"All model classes must inherit from TreeNode, not raw BaseModel.",
                )
        except TypeError:
            # issubclass can raise TypeError for some types like generics
            pass

    def _get_iteration_item_type(
        self, source_node_tag: str, iteration_collection: str
    ) -> type:
        """
        Extract the item type from a list field for iteration validation.

        For iteration field like 'sections: list[Section]', extract 'Section' type
        to validate dotted field access like 'sections.title' against Section fields.

        Params:
            source_node_tag: Tag of the node containing the iteration field
            iteration_collection: Name of the iteration field (e.g., "sections")

        Returns:
            The item type for iteration (e.g., Section from list[Section])

        Raises:
            VariableSourceValidationError: If iteration field not found or not a list type
        """
        from langtree.exceptions import VariableSourceValidationError

        source_node = self.get_node(source_node_tag)
        if not source_node or not hasattr(source_node.field_type, "model_fields"):
            raise VariableSourceValidationError(
                source_path=iteration_collection,
                structure_type="unknown",
                command_context=f"Cannot validate iteration on node {source_node_tag}",
            )

        field_def = source_node.field_type.model_fields.get(iteration_collection)
        if not field_def:
            raise VariableSourceValidationError(
                source_path=iteration_collection,
                structure_type=source_node.field_type.__name__,
                command_context=f"Iteration field '{iteration_collection}' not found",
            )

        # Extract item type from list[ItemType]
        field_annotation = field_def.annotation
        if (
            hasattr(field_annotation, "__origin__")
            and field_annotation.__origin__ is list
        ):
            if hasattr(field_annotation, "__args__") and field_annotation.__args__:
                return field_annotation.__args__[0]

        raise VariableSourceValidationError(
            source_path=iteration_collection,
            structure_type=source_node.field_type.__name__,
            command_context=f"Field '{iteration_collection}' is not a list type for iteration",
        )

    def _validate_against_bare_collection_types(
        self, field_tag: str, annotation: type, origin: type, args: tuple
    ) -> None:
        """
        Validate that collection types have proper type parameters.

        Bare collection types like list, dict, set without type parameters are underspecified
        and should be rejected to maintain architectural integrity.

        Params:
            field_tag: Full tag path of the field for error reporting
            annotation: The field's type annotation
            origin: Generic origin type (e.g., list, Union)
            args: Generic type arguments

        Raises:
            FieldValidationError: If bare collection types are detected
        """
        from langtree.exceptions import FieldValidationError

        bare_collection_types = (list, dict, set)

        # Check if this is a bare collection type (no type parameters)
        if annotation in bare_collection_types:
            collection_name = annotation.__name__
            raise FieldValidationError(
                field_path=field_tag,
                container="field type validation",
                message=f"Bare collection type '{collection_name}' is underspecified. "
                f"Use proper type parameters like '{collection_name}[str]' or '{collection_name}[str, int]' instead.",
            )

        # Check if origin is a bare collection type (this handles typing.List, typing.Dict, etc.)
        if origin in bare_collection_types and (args is None or len(args) == 0):
            collection_name = origin.__name__
            raise FieldValidationError(
                field_path=field_tag,
                container="field type validation",
                message=f"Bare collection type '{collection_name}' is underspecified. "
                f"Use proper type parameters like '{collection_name}[str]' or '{collection_name}[str, int]' instead.",
            )

    def _count_iterable_levels(
        self, path_components: list[str], source_node_tag: str
    ) -> int:
        """
        Count the actual iterable levels in a field path by examining types.

        For loop nesting validation, we need to count how many nested list levels
        a field path actually has. For example:
        - results: list[str] = 1 level
        - results: list[list[str]] = 2 levels
        - title: str = 0 levels

        This uses Pydantic type introspection to check actual field types.
        """
        # Skip 'value' if present - it's not part of the field structure
        if path_components and path_components[0] == "value":
            field_components = path_components[1:]
        else:
            field_components = path_components

        if not field_components:
            return 0

        source_node = self.get_node(source_node_tag)
        if not source_node:
            return 0

        current_type = source_node.field_type

        # Walk through the path components to find the target field type
        for component in field_components:
            if not hasattr(current_type, "model_fields"):
                # Cannot continue traversal - this should not happen for source fields
                raise ValueError(
                    f"Cannot count iterable levels: '{current_type}' has no model_fields while processing '{'.'.join(field_components)}'"
                )

            if component not in current_type.model_fields:
                # Field doesn't exist - this should not happen for source fields after field existence check
                from langtree.exceptions import FieldValidationError

                raise FieldValidationError(
                    field_path=".".join(field_components),
                    container=source_node_tag,
                    message=f"field '{component}' does not exist in {current_type.__name__ if hasattr(current_type, '__name__') else str(current_type)} during nesting level counting",
                )

            field_def = current_type.model_fields[component]
            field_annotation = field_def.annotation

            # Move to the field type for analysis
            current_type = field_annotation

        # Now count how many iterable levels this final type has
        # This includes both direct list nesting and TreeNode field traversal
        return self._count_nested_iterations(current_type)

    def _count_nested_iterations(self, field_type: type) -> int:
        """
        Count total iteration levels by examining both list nesting and TreeNode hierarchies.

        For example:
        - list[str] = 1 level
        - list[SomeNode] where SomeNode.items: list[str] = 2 levels (can traverse as field.items)
        - list[list[str]] = invalid pattern (should fail - no traversable field names)
        """
        from typing import get_args, get_origin

        # Check if this is a list type
        origin = get_origin(field_type)
        if origin is list:
            args = get_args(field_type)
            if not args:
                return 1

            inner_type = args[0]
            inner_origin = get_origin(inner_type)

            # If inner type is also a list, continue counting (will naturally fail during field resolution)
            if inner_origin is list:
                # Count this as one level, inner list will be counted recursively
                return 1 + self._count_nested_iterations(inner_type)

            # If inner type is a TreeNode, count its maximum iterable depth
            if hasattr(inner_type, "model_fields"):
                max_inner_depth = 0
                for field_name, field_def in inner_type.model_fields.items():
                    try:
                        inner_depth = self._count_nested_iterations(
                            field_def.annotation
                        )
                        max_inner_depth = max(max_inner_depth, inner_depth)
                    except Exception:
                        continue
                return 1 + max_inner_depth
            else:
                # Inner type is not iterable (str, int, etc.)
                return 1
        else:
            # Not a list - check if it's a TreeNode with iterable fields
            if hasattr(field_type, "model_fields"):
                max_depth = 0
                for field_name, field_def in field_type.model_fields.items():
                    try:
                        depth = self._count_nested_iterations(field_def.annotation)
                        max_depth = max(max_depth, depth)
                    except Exception:
                        continue
                return max_depth
            else:
                return 0

    def _count_iterable_levels_in_path(
        self, path_components: list[str], source_node_tag: str
    ) -> int:
        """
        Count actual iterable levels in an inclusion path by examining field types.

        For example, in companies.metadata.departments.items:
        - companies: list[Company] -> iterable (1)
        - metadata: CompanyMetadata -> not iterable (0)
        - departments: list[Department] -> iterable (1)
        - items: list[Item] -> iterable (1)
        Total iterable levels: 3

        This differs from path component count (4) because not all path segments are iterable.
        """
        if not path_components:
            return 0

        source_node = self.get_node(source_node_tag)
        if not source_node or not source_node.field_type:
            return 0

        current_type = source_node.field_type
        iterable_count = 0

        # Walk through path components and count iterable levels
        for component in path_components:
            if not hasattr(current_type, "model_fields"):
                break

            if component not in current_type.model_fields:
                break

            field_def = current_type.model_fields[component]
            field_annotation = field_def.annotation

            # Check if this field is iterable (list type)
            if (
                hasattr(field_annotation, "__origin__")
                and field_annotation.__origin__ is list
            ):
                iterable_count += 1
                # Get the inner type for next iteration
                if hasattr(field_annotation, "__args__") and field_annotation.__args__:
                    current_type = field_annotation.__args__[0]
                else:
                    break
            else:
                # Non-iterable field - move to its type but don't count
                current_type = field_annotation

            # Stop if we can't introspect further
            if not hasattr(current_type, "model_fields"):
                break

        return iterable_count

    def _validate_task_target_completeness(
        self, command: "ParsedCommand", source_node_tag: str
    ) -> None:
        """
        Validate that task targets are complete (not just 'task').

        Per LANGUAGE_SPECIFICATION.md:
        - Destinations like '->task' are incomplete and should be caught
        - Complete destinations like '->task.analyzer' should pass
        """
        from langtree.exceptions import FieldValidationError

        destination_path = command.destination_path

        # Check if destination is just 'task' without a specific task name
        if destination_path == "task":
            # Build command context for error chaining
            command_type = "@each" if command.inclusion_path else "@all"
            inclusion_part = (
                f"[{command.inclusion_path}]" if command.inclusion_path else ""
            )
            command_context = (
                f"{command_type}{inclusion_part}->{command.destination_path}@{{...}}"
            )

            raise FieldValidationError(
                field_path=destination_path,
                container=source_node_tag,
                message="is an incomplete task target. Use specific task names like 'task.analyzer' instead of just 'task'",
                command_context=command_context,
            )

    def _validate_field_context_scoping(
        self, command, field_name: str, source_node_tag: str
    ) -> None:
        """
        Validate field context scoping for @each commands.

        LangTree DSL rule: inclusion_path (tag1) must ALWAYS start with field_name - no exceptions.
        This applies regardless of where the destination_path (tag2) points.
        """
        from langtree.exceptions import FieldValidationError

        # Only validate ParsedCommand types with inclusion_path (@each commands)
        if not hasattr(command, "inclusion_path") or not command.inclusion_path:
            return

        inclusion_path = command.inclusion_path

        # @each commands are FORBIDDEN in docstrings (field_name = None)
        if field_name is None:
            raise FieldValidationError(
                field_path=inclusion_path,
                container=source_node_tag,
                message="@each commands are not allowed in class docstrings. "
                "@each commands must be defined in field descriptions where they can reference the field context.",
                command_context=f"@each[{command.inclusion_path}] in docstring of {source_node_tag}",
            )

        # LangTree DSL scoping rule: inclusion_path must ALWAYS start with field_name
        if not inclusion_path.startswith(field_name):
            # Build command context for error chaining
            command_context = f"@each[{command.inclusion_path}]->{command.destination_path or 'current'}@{{...}}* in field '{field_name}'"

            raise FieldValidationError(
                field_path=inclusion_path,
                container=source_node_tag,
                message=f"inclusion_path '{inclusion_path}' must start with field '{field_name}' where command is defined. "
                f"Commands can only reference the field they are defined in (LangTree DSL scoping rule)",
                command_context=command_context,
            )

    def _validate_subchain_matching(
        self, command: "ParsedCommand", source_node_tag: str
    ) -> None:
        """
        Validate RHS paths using last-matching-iterable algorithm.

        Uses enhanced iterable depth counting algorithm to replace complex anchor-based validation.
        Validates that RHS paths (source_paths) have proper structural relationships with inclusion_path.
        """
        from langtree.parsing.parser import CommandParser

        # Only validate @each commands with inclusion_path and variable mappings
        if not command.inclusion_path or not command.variable_mappings:
            return

        # Get the source node type for validation
        source_tree_node = self.get_node(source_node_tag)
        if not source_tree_node or not source_tree_node.field_type:
            return  # Node validation will catch this separately

        # Create instance with default values for type introspection
        # The algorithm only examines field types, not values
        try:
            source_node_type = source_tree_node.field_type
            # Try to create instance with model_construct (bypasses validation)
            source_node = source_node_type.model_construct()
        except Exception:
            # Fall back to trying default constructor
            try:
                source_node = source_node_type()
            except Exception:
                return  # Can't create instance for validation

        # Collect RHS paths from variable mappings, filtering scope-modified paths
        rhs_paths = []
        for variable_mapping in command.variable_mappings:
            source_path = variable_mapping.source_path

            # Skip wildcard and scope-modified paths - these have different validation rules
            if source_path == "*":
                continue
            if "." in source_path and source_path.split(".")[0] in (
                "prompt",
                "value",
                "outputs",
                "task",
            ):
                continue  # Skip scoped paths from structural validation

            # COMMENTED OUT: Skip iteration variable paths from structural validation
            # For @each[document.sections], skip "sections.title" as it refers to iteration variable
            # inclusion_parts = command.inclusion_path.split('.')
            # iteration_collection = inclusion_parts[-1]  # e.g., "sections" from "document.sections"
            # if source_path.split('.')[0] == iteration_collection:
            #     continue  # Skip iteration variable paths from structural validation

            # Skip same-node field references in value scope
            if variable_mapping.resolved_target:
                target_scope = variable_mapping.resolved_target.scope.get_name()
                if target_scope == "value" and self._is_same_node_field(
                    source_path, source_node_tag
                ):
                    continue
                if target_scope in ("prompt", "outputs"):
                    continue  # These scopes have broader validation rules

            # Note: Do NOT skip sibling field references - these need subchain validation
            # Sibling fields that diverge from inclusion path should be caught and rejected

            # Add to structural validation
            rhs_paths.append(source_path.split("."))

        # Apply last-matching-iterable validation if we have paths to validate
        if rhs_paths:
            parser = CommandParser()
            inclusion_path_components = command.inclusion_path.split(".")

            try:
                result = parser.validate_last_matching_iterable(
                    source_node, inclusion_path_components, rhs_paths
                )
                if not result["valid"]:
                    from langtree.exceptions import FieldValidationError

                    command_context = f"@each[{command.inclusion_path}]->{command.destination_path}@{{...}}* in field '{command.inclusion_path.split('.')[0]}'"

                    # Extract detailed error message from parser
                    error_detail = result.get(
                        "error", result.get("reason", "validation failed")
                    )
                    raise FieldValidationError(
                        field_path=command.inclusion_path,
                        container=source_node_tag,
                        message=error_detail,
                        command_context=command_context,
                    )
            except Exception as e:
                # Convert parser errors to field validation errors, preserving detailed messages
                from langtree.exceptions import FieldValidationError

                command_context = f"@each[{command.inclusion_path}]->{command.destination_path}@{{...}}* in field '{command.inclusion_path.split('.')[0]}'"

                # Preserve the original error message which contains the keywords expected by tests
                raise FieldValidationError(
                    field_path=command.inclusion_path,
                    container=source_node_tag,
                    message=str(e),
                    command_context=command_context,
                )

    def _is_same_node_field(self, source_path: str, source_node_tag: str) -> bool:
        """
        Check if source_path refers to a field in the same node.

        Params:
            source_path: The path to check (e.g., 'processed', 'items.title')
            source_node_tag: The tag of the source node

        Returns:
            True if source_path is a field in the same node as source_node_tag
        """
        try:
            source_node = self._root_nodes.get(source_node_tag)
            if not source_node:
                return False

            # Get the field type for this node
            field_type = source_node.field_type
            if not field_type:
                return False

            # Check if the source_path is a valid field in this node
            path_components = source_path.split(".")
            self._validate_field_path_exists(
                path_components, field_type, source_path, source_node_tag
            )
            return True

        except Exception:
            # If validation fails, it's not a valid same-node field
            return False

    def _is_sibling_field_reference(
        self, source_path: str, inclusion_path: str
    ) -> bool:
        """
        Check if source_path and inclusion_path are sibling fields in the same structure.

        Example: items.name and items.values are sibling fields of the same Item structure.

        Params:
            source_path: The source path (e.g., 'items.name')
            inclusion_path: The inclusion path (e.g., 'items.values')

        Returns:
            True if they are sibling fields in the same parent structure
        """
        source_parts = source_path.split(".")
        inclusion_parts = inclusion_path.split(".")

        # Must have at least 2 parts each to have a parent and field
        if len(source_parts) < 2 or len(inclusion_parts) < 2:
            return False

        # Check if they share the same parent path
        source_parent = ".".join(source_parts[:-1])
        inclusion_parent = ".".join(inclusion_parts[:-1])

        # Sibling fields have the same parent path but different final fields
        return (
            source_parent == inclusion_parent
            and source_parts[-1] != inclusion_parts[-1]
        )

    def _validate_cross_tree_iteration_matching(
        self, command: "ParsedCommand", source_node_tag: str, target_path: str
    ) -> None:
        """
        Validate that cross-tree iteration counts match between source and target structures.

        Cross-tree references require that the source iteration pattern matches the target
        structure's iteration capacity. Non-iterable spacing differences are ignored -
        only the count of iterable levels matters for execution compatibility.

        Params:
            command: ParsedCommand with cross-tree reference to validate
            source_node_tag: Tag of the source node containing the command
            target_path: Full path to the target task being referenced

        Raises:
            FieldValidationError: When source and target have mismatched iteration counts
        """
        from langtree.exceptions import FieldValidationError

        # Validate that inclusion_path exists
        if not command.inclusion_path:
            return

        # Extract inclusion path components for source iteration count
        inclusion_components = command.inclusion_path.split(".")
        source_iteration_count = self._count_iterable_levels_in_path(
            inclusion_components, source_node_tag
        )

        # Count iterations in target structure by finding maximum iteration depth
        target_node = self.get_node(target_path)
        if not target_node or not target_node.field_type:
            return

        # Find the maximum iteration depth among all fields in target
        target_iteration_count = 0
        for field_name, _ in target_node.field_type.model_fields.items():
            try:
                # Try to find the deepest iterable path starting from this field
                max_depth = self._find_max_iteration_depth(
                    field_name, target_node.field_type, target_path
                )
                target_iteration_count = max(target_iteration_count, max_depth)
            except Exception:
                continue

        # For @each commands with multiplicity, the target receives individual items
        # so iteration count mismatch is expected and valid (expansion pattern)
        is_each_with_multiplicity = (
            hasattr(command, "command_type")
            and command.command_type.value == "each"
            and hasattr(command, "has_multiplicity")
            and command.has_multiplicity
        )

        # Validate iteration counts match (except for @each expansion patterns)
        if (
            source_iteration_count != target_iteration_count
            and not is_each_with_multiplicity
        ):
            command_context = f"@each[{command.inclusion_path}]->{command.destination_path}@{{...}}* in field '{command.inclusion_path.split('.')[0]}'"

            raise FieldValidationError(
                field_path=command.inclusion_path,
                container=source_node_tag,
                message=f"Cross-tree iteration count mismatch: source has {source_iteration_count} iteration levels "
                f"but target structure '{target_path}' has {target_iteration_count} iteration levels. "
                f"Cross-tree references require matching iteration counts for execution compatibility.",
                command_context=command_context,
            )

    def _find_max_iteration_depth(
        self, field_name: str, parent_type: type, context_tag: str
    ) -> int:
        """
        Find the maximum iteration depth starting from a field by traversing nested TreeNode types.

        Params:
            field_name: Name of the field to start traversal from
            parent_type: The TreeNode type containing the field
            context_tag: Context tag for path resolution

        Returns:
            Maximum number of nested iteration levels found
        """
        field_def = parent_type.model_fields.get(field_name)
        if not field_def:
            return 0

        # Check if this field itself is iterable
        field_type = field_def.annotation
        from typing import get_args, get_origin

        origin = get_origin(field_type)

        if origin is list:
            # This field is iterable - count 1 and recurse into its element type
            args = get_args(field_type)
            if args and hasattr(args[0], "model_fields"):
                # Element type is a TreeNode - find max depth in its fields
                element_type = args[0]
                max_nested_depth = 0
                for nested_field_name, _ in element_type.model_fields.items():
                    nested_depth = self._find_max_iteration_depth(
                        nested_field_name, element_type, context_tag
                    )
                    max_nested_depth = max(max_nested_depth, nested_depth)
                return 1 + max_nested_depth
            else:
                # Element type is not a TreeNode - just 1 iteration level
                return 1
        else:
            # Field is not iterable itself - check if it's a TreeNode we can recurse into
            if hasattr(field_type, "model_fields"):
                max_nested_depth = 0
                for nested_field_name, _ in field_type.model_fields.items():
                    nested_depth = self._find_max_iteration_depth(
                        nested_field_name, field_type, context_tag
                    )
                    max_nested_depth = max(max_nested_depth, nested_depth)
                return max_nested_depth
            else:
                # Non-iterable primitive field
                return 0

    def _validate_all_command_rhs_scoping(
        self,
        command: "ParsedCommand",
        field_name: str = None,
        source_node_tag: str = None,
    ) -> None:
        """
        Validate @all command RHS scoping rules per LANGUAGE_SPECIFICATION.md.

        ARCHITECTURAL PRINCIPLE: @all commands can only reference:
        - Field-level commands: The exact field containing the command OR wildcard (*)
        - Docstring-level commands: Only wildcard (*) OR implicit mapping (equivalent to wildcard)

        FORBIDDEN PATTERNS:
        - Sibling field references (breaks locality)
        - Longer paths from containing field (breaks semantics)
        - External field references (breaks predictability)
        - Docstring commands referencing specific fields (breaks encapsulation)

        Params:
            command: The parsed @all command to validate
            field_name: Name of field containing command, or None for docstring commands
            source_node_tag: Tag of node where command is defined

        Raises:
            FieldValidationError: If @all RHS scoping rules are violated
        """
        from langtree.exceptions import FieldValidationError

        # Only validate @all commands with variable mappings
        if command.command_type.value != "all" or not command.variable_mappings:
            return

        # Get the source node to validate field references
        source_tree_node = self.get_node(source_node_tag)
        if not source_tree_node or not source_tree_node.field_type:
            return  # Node validation will catch this separately

        for variable_mapping in command.variable_mappings:
            source_path = variable_mapping.source_path

            # Allow wildcard - represents entire current node
            if source_path == "*":
                continue

            if field_name is not None:
                # Field-level command: check if source_path matches containing field exactly
                if source_path == field_name:
                    continue

                # Field-level command with invalid RHS
                raise FieldValidationError(
                    field_path=field_name,
                    container=source_node_tag,
                    message=f"@all command references '{source_path}' but @all commands can only reference the exact field containing the command ('{field_name}') or wildcard (*). RHS must be containing field only.",
                )
            else:
                # Docstring-level command: only wildcard allowed
                raise FieldValidationError(
                    field_path="<docstring>",
                    container=source_node_tag,
                    message=f"@all command in docstring references '{source_path}' but docstring @all commands can only use wildcard (*) for data locality.",
                )

    def list_runtime_variables(self) -> list[str]:
        """
        List all runtime variables that need values at invoke time.

        Returns simple list of fully expanded variable names like:
        ['prompt__node__variable_name', 'prompt__other__var']

        Uses existing runtime variable resolution system instead of regex scanning.
        """
        import re

        from langtree.execution.resolution import resolve_runtime_variables

        runtime_vars = []

        def scan_node(node: StructureTreeNode, path: str):
            """Recursively scan nodes for runtime variables using proper resolution."""
            # Use processed clean content, not raw docstrings
            content_sources = []

            # Add clean docstring if available
            if hasattr(node, "clean_docstring") and node.clean_docstring:
                content_sources.append(node.clean_docstring)

            # Add clean field descriptions if available
            if (
                hasattr(node, "clean_field_descriptions")
                and node.clean_field_descriptions
            ):
                content_sources.extend(node.clean_field_descriptions.values())

            # Process each content source using existing resolution system
            for content in content_sources:
                if content:
                    try:
                        # Use existing resolve_runtime_variables to get expanded content
                        # Use deferred validation for variable listing (don't error on undefined vars)
                        expanded_content = resolve_runtime_variables(
                            content, self, node, validate=False
                        )

                        # Extract expanded variable patterns from resolved content
                        expanded_var_pattern = re.compile(r"\{(prompt__[^}]+)\}")
                        found_vars = expanded_var_pattern.findall(expanded_content)
                        runtime_vars.extend(found_vars)

                    except Exception:
                        # If resolution fails, continue to next content
                        # (This maintains existing behavior where resolution errors don't break enumeration)
                        continue

            # Recurse into children
            for child_name, child_node in node.children.items():
                scan_node(child_node, f"{path}.{child_name}")

        # Start from all root nodes
        for root_name, root_node in self._root_nodes.items():
            scan_node(root_node, root_name)

        return list(set(runtime_vars))  # Remove duplicates
