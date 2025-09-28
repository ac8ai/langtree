"""
Resolution module for LangTree DSL context and path resolution.

This module contains methods for resolving contexts, paths, and references
within the LangTree DSL prompt tree structure, including deferred context resolution,
scope-based path navigation, and cross-tree references.
"""

import re
from typing import TYPE_CHECKING, Any, cast

from langtree.execution.scopes import get_scope
from langtree.structure.utils import get_node_instance, validate_path_and_node_tag

if TYPE_CHECKING:
    from langtree.parsing.parser import ParsedCommand
    from langtree.structure.builder import (
        PromptValue,
        ResolutionResult,
        RunStructure,
        StructureTreeNode,
    )


def resolve_deferred_contexts(run_structure: "RunStructure") -> dict:
    """
    Perform context resolution that was deferred during tree building.

    This method attempts to resolve all command contexts now that the tree
    is complete. It's useful for validation and execution planning.

    Params:
        run_structure: The RunStructure instance to resolve contexts for

    Returns:
        Dictionary with resolution results and any errors encountered.
    """
    resolution_results = {
        "successful_resolutions": [],
        "failed_resolutions": [],
        "resolution_errors": [],
    }

    # Iterate root nodes and attempt resolution for each command encountered
    for node_name, node in run_structure._root_nodes.items():
        _resolve_node_contexts_recursive(run_structure, node, resolution_results)

    return resolution_results


def _resolve_node_contexts_recursive(
    run_structure: "RunStructure", node: "StructureTreeNode", results: dict
) -> None:
    """
    Helper method to recursively resolve contexts for all commands in a node tree.

    Traverses the node tree depth-first, attempting context resolution for each
    command found. Accumulates resolution results and errors for later analysis.

    Params:
        run_structure: The RunStructure instance containing the tree
        node: Current structure tree node being processed
        results: Dictionary to accumulate resolution results and errors
    """
    if hasattr(node, "extracted_commands"):
        for command in node.extracted_commands:
            try:
                # Attempt to resolve the command context now
                target_node = (
                    run_structure.get_node(command.destination_path)
                    if command.destination_path
                    else None
                )
                _resolve_command_context(run_structure, command, node.name, target_node)
                results["successful_resolutions"].append(
                    {
                        "node": node.name,
                        "command": command.destination_path,
                        "type": "deferred_resolution",
                    }
                )
            except Exception as e:
                results["failed_resolutions"].append(
                    {
                        "node": node.name,
                        "command": command.destination_path,
                        "error": str(e),
                    }
                )
                results["resolution_errors"].append(str(e))

    # Recurse into children
    for child_node in node.children.values():
        _resolve_node_contexts_recursive(run_structure, child_node, results)


def _resolve_command_context(
    run_structure: "RunStructure",
    command: "ParsedCommand",
    source_node_tag: str,
    target_node: "StructureTreeNode | None",
) -> None:
    """
    Perform Phase 2 context resolution for a LangTree DSL command.

    Resolves inclusion paths, destination paths, and variable mappings within their
    proper context scopes. This is called after the tree structure is complete
    to finalize all path references and variable bindings.

    Params:
        run_structure: The RunStructure instance containing the complete tree
        command: The parsed LangTree DSL command requiring context resolution
        source_node_tag: The tag of the node where this command originated
        target_node: The target node for the command, may be None for pending targets

    Returns:
        None: Currently updates internal state, future versions should return resolution results
    """

    # Inclusion path context (@each) resolution (noop placeholder)
    if command.inclusion_path and command.resolved_inclusion:
        _resolve_inclusion_context(
            run_structure, command.resolved_inclusion, source_node_tag
        )

    # Resolve destination path context
    _resolve_destination_context(
        run_structure,
        command.resolved_destination if command.resolved_destination else None,
        command.destination_path,
        target_node,
    )

    # Resolve variable mapping contexts
    for mapping in command.variable_mappings:
        if mapping.resolved_target and mapping.resolved_source:
            _resolve_variable_mapping_context(
                run_structure,
                mapping.resolved_target,
                mapping.resolved_source,
                source_node_tag,
                target_node,
            )


def _resolve_inclusion_context(
    run_structure: "RunStructure", resolved_inclusion: Any, source_node_tag: str
):
    """
    Resolve inclusion path in context of source node.

    Validates that the inclusion path exists and is iterable for @each commands.
    This is essential for command execution that needs to iterate over collections
    within the source node's data context.

    Params:
        run_structure: The RunStructure instance containing the tree
        resolved_inclusion: Resolved inclusion path info containing path and scope
        source_node_tag: Tag of the node where the inclusion command originates

    Returns:
        dict: Resolution result with metadata for execution planning

    Raises:
        ValueError: When inclusion path doesn't exist or isn't iterable
    """
    if not resolved_inclusion or not hasattr(resolved_inclusion, "path"):
        raise ValueError("Invalid resolved inclusion - missing path information")

    inclusion_path = resolved_inclusion.path
    inclusion_scope = resolved_inclusion.scope

    try:
        # Use appropriate scope resolution to get the inclusion target
        if inclusion_scope:
            scope = get_scope(inclusion_scope.get_name())
            context = {"node_tag": source_node_tag, "run_structure": run_structure}
            inclusion_value = scope.resolve(inclusion_path, context)
        else:
            # Default to current node context
            inclusion_value = _resolve_in_current_node_context(
                run_structure, inclusion_path, source_node_tag
            )

        # Validate that the inclusion target is iterable
        if not hasattr(inclusion_value, "__iter__") or isinstance(inclusion_value, str):
            raise ValueError(
                f"Inclusion path '{inclusion_path}' resolves to non-iterable value: {type(inclusion_value)}"
            )

        # Return metadata for execution planning
        iterable_length = None
        item_type = "unknown"

        # Safe length check for sized collections
        if isinstance(inclusion_value, list | dict | tuple | set):
            try:
                iterable_length = len(inclusion_value)
            except (TypeError, AttributeError):
                iterable_length = None

        # Safe item type detection for indexable collections
        if isinstance(inclusion_value, list | tuple) and inclusion_value:
            try:
                item_type = type(inclusion_value[0]).__name__
            except (IndexError, TypeError, AttributeError):
                item_type = "unknown"
        elif isinstance(inclusion_value, dict) and inclusion_value:
            # For dicts, get the type of the first value
            try:
                first_value = next(iter(inclusion_value.values()))
                item_type = type(first_value).__name__
            except (StopIteration, TypeError, AttributeError):
                item_type = "unknown"

        return {
            "inclusion_path": inclusion_path,
            "source_node": source_node_tag,
            "scope": inclusion_scope.get_name() if inclusion_scope else "current_node",
            "iterable_length": iterable_length,
            "item_type": item_type,
        }

    except Exception as e:
        raise ValueError(
            f"Failed to resolve inclusion path '{inclusion_path}' in node '{source_node_tag}': {e}"
        )


def _resolve_destination_context(
    run_structure: "RunStructure",
    resolved_destination: Any,
    destination_path: str,
    target_node: "StructureTreeNode | None",
):
    """Resolve destination path in context of target structure.

    Validates that the destination exists or can be created in the target structure.
    Essential for ensuring command mappings point to valid destinations.

    Params:
        run_structure: The RunStructure instance
        resolved_destination: The resolved destination path info (ResolvedPath object)
        destination_path: The raw destination path string
        target_node: The target node, may be None for pending targets

    Returns:
        dict: Resolution result with validation status and metadata

    Raises:
        ValueError: When destination path is invalid or incompatible
    """
    if not destination_path:
        raise ValueError("Destination path cannot be empty")

    # Handle pending targets (forward references)
    if target_node is None:
        # Target doesn't exist yet - mark as pending resolution
        return {
            "destination_path": destination_path,
            "status": "pending",
            "reason": "Target node not yet available in tree structure",
            "requires_resolution": True,
        }

    # For existing targets, validate structure compatibility
    try:
        if resolved_destination and resolved_destination.scope:
            # Scoped destination - validate the scope context
            scope_name = resolved_destination.scope.get_name()

            if scope_name == "task":
                # Task scope destinations refer to other tree nodes
                full_path = f"task.{resolved_destination.path}"
                referenced_node = run_structure.get_node(full_path)
                if not referenced_node:
                    return {
                        "destination_path": destination_path,
                        "status": "invalid",
                        "reason": f'Referenced task node "{full_path}" not found',
                        "requires_resolution": False,
                    }
            elif scope_name in ["value", "outputs", "prompt"]:
                # These scopes refer to chain assembly contexts - validate basic structure
                if not target_node.field_type:
                    raise ValueError(
                        f"Target node has no field type for {scope_name} scope validation"
                    )

                # Basic validation - the scope should be accessible in target context
                # More detailed validation will happen during chain assembly
                pass

        else:
            # Unscoped destination - validate as direct node reference
            if not target_node.field_type:
                raise ValueError("Target node has no field type defined")

        return {
            "destination_path": destination_path,
            "status": "valid",
            "target_node": target_node.name,
            "target_type": target_node.field_type.__name__
            if target_node.field_type
            else "unknown",
            "scope": resolved_destination.scope.get_name()
            if resolved_destination and resolved_destination.scope
            else "direct",
            "requires_resolution": False,
        }

    except Exception as e:
        return {
            "destination_path": destination_path,
            "status": "error",
            "reason": f"Validation failed: {e}",
            "requires_resolution": False,
        }


def _resolve_variable_mapping_context(
    run_structure: "RunStructure",
    resolved_target: Any,
    resolved_source: Any,
    source_node_tag: str,
    target_node: "StructureTreeNode | None",
):
    """Resolve variable mapping between source and target contexts.

    Validates that source paths can be resolved and target paths are compatible.
    Essential for ensuring variable mappings will work during execution.

    Params:
        run_structure: The RunStructure instance
        resolved_target: The resolved target path info (ResolvedPath object)
        resolved_source: The resolved source path info (ResolvedPath object)
        source_node_tag: The tag of the source node
        target_node: The target node, may be None for pending targets

    Returns:
        dict: Resolution result with mapping validation status and metadata

    Raises:
        ValueError: When mapping is fundamentally invalid
    """
    if not resolved_target or not resolved_source:
        raise ValueError(
            "Both target and source must be resolved for mapping validation"
        )

    mapping_result = {
        "target_path": resolved_target.path,
        "source_path": resolved_source.path,
        "source_node": source_node_tag,
        "target_scope": resolved_target.scope.get_name()
        if resolved_target.scope
        else "current_node",
        "source_scope": resolved_source.scope.get_name()
        if resolved_source.scope
        else "current_node",
        "validations": {},
    }

    # Validate source path resolution
    try:
        if resolved_source.path == "*":
            # Wildcard source - always valid, represents whole object
            mapping_result["validations"]["source"] = {
                "status": "valid",
                "type": "wildcard",
                "reason": "Wildcard source always resolves to complete object",
            }
        else:
            # Attempt to resolve source path in its context
            if resolved_source.scope:
                scope = get_scope(resolved_source.scope.get_name())
                context = {"node_tag": source_node_tag, "run_structure": run_structure}
                try:
                    source_value = scope.resolve(resolved_source.path, context)
                    mapping_result["validations"]["source"] = {
                        "status": "valid",
                        "type": type(source_value).__name__,
                        "reason": "Source path successfully resolved",
                    }
                except Exception as e:
                    mapping_result["validations"]["source"] = {
                        "status": "warning",
                        "type": "unknown",
                        "reason": f"Source path resolution deferred: {e}",
                    }
            else:
                # Default to current node context
                try:
                    source_value = _resolve_in_current_node_context(
                        run_structure, resolved_source.path, source_node_tag
                    )
                    mapping_result["validations"]["source"] = {
                        "status": "valid",
                        "type": type(source_value).__name__,
                        "reason": "Source path successfully resolved in current context",
                    }
                except Exception as e:
                    mapping_result["validations"]["source"] = {
                        "status": "warning",
                        "type": "unknown",
                        "reason": f"Source path resolution deferred: {e}",
                    }
    except Exception as e:
        mapping_result["validations"]["source"] = {
            "status": "error",
            "type": "unknown",
            "reason": f"Source validation failed: {e}",
        }

    # Validate target path compatibility
    try:
        if target_node is None:
            # Target node pending - defer validation
            mapping_result["validations"]["target"] = {
                "status": "pending",
                "reason": "Target node not yet available for validation",
            }
        else:
            if resolved_target.scope:
                scope_name = resolved_target.scope.get_name()

                if scope_name == "outputs":
                    # Outputs scope - validate structure compatibility for chain assembly
                    # Track collection for outputs.field assignments
                    _track_outputs_collection(
                        run_structure,
                        resolved_target.path,
                        resolved_source.path,
                        source_node_tag,
                        target_node,
                    )
                    mapping_result["validations"]["target"] = {
                        "status": "valid",
                        "scope": "outputs",
                        "reason": "Outputs scope accepts any values during LangChain execution (collection supported)",
                    }
                elif scope_name == "prompt":
                    # Prompt scope - validate prompt variable structure
                    mapping_result["validations"]["target"] = {
                        "status": "valid",
                        "scope": "prompt",
                        "reason": "Prompt scope accepts variable assignments",
                    }
                elif scope_name == "value":
                    # Value scope - validate against target node structure
                    try:
                        _resolve_in_target_node_context(
                            run_structure, resolved_target.path, target_node
                        )
                        mapping_result["validations"]["target"] = {
                            "status": "valid",
                            "scope": "value",
                            "reason": "Target path exists in node value structure",
                        }
                    except Exception as e:
                        mapping_result["validations"]["target"] = {
                            "status": "warning",
                            "scope": "value",
                            "reason": f"Target path validation deferred: {e}",
                        }
                else:
                    # Other scopes
                    mapping_result["validations"]["target"] = {
                        "status": "valid",
                        "scope": scope_name,
                        "reason": f"Target scope {scope_name} validation passed",
                    }
            else:
                # Unscoped target - validate against target node structure
                try:
                    _resolve_in_target_node_context(
                        run_structure, resolved_target.path, target_node
                    )
                    mapping_result["validations"]["target"] = {
                        "status": "valid",
                        "scope": "direct",
                        "reason": "Target path exists in node structure",
                    }
                except Exception as e:
                    mapping_result["validations"]["target"] = {
                        "status": "warning",
                        "scope": "direct",
                        "reason": f"Target validation deferred: {e}",
                    }
    except Exception as e:
        mapping_result["validations"]["target"] = {
            "status": "error",
            "reason": f"Target validation failed: {e}",
        }

    # Determine overall mapping status
    source_status = mapping_result["validations"]["source"]["status"]
    target_status = mapping_result["validations"]["target"]["status"]

    if source_status == "error" or target_status == "error":
        mapping_result["overall_status"] = "error"
    elif source_status == "pending" or target_status == "pending":
        mapping_result["overall_status"] = "pending"
    elif source_status == "warning" or target_status == "warning":
        mapping_result["overall_status"] = "warning"
    else:
        mapping_result["overall_status"] = "valid"

    return mapping_result


def _resolve_in_current_node_context(
    run_structure: "RunStructure", path: str, node_tag: str
) -> "PromptValue":
    """Resolve a dot path against the current node's model instance.

    Resolution semantics:
      - Wildcard '*' returns the entire node instance
      - Dotted segments traverse attributes / dict keys
      - Lists: final segment may project attribute across list items
      - Missing optional / defaulted fields may yield None when acceptable

    Params:
        run_structure: Active `RunStructure` containing the node.
        path: Dot-delimited path or '*' for whole instance.
        node_tag: Tag identifying the source node in the structure.

    Returns:
        The resolved value (any JSON-serializable / Pydantic field type) or entire node instance for '*'.

    Raises:
        ValueError: Node not found or malformed path.
        KeyError: Non-existent intermediate / terminal path component.
        AttributeError: Attribute access failure during traversal.
    """
    validate_path_and_node_tag(path, node_tag)

    # Wildcard '*' returns entire node instance
    if path == "*":
        # Return the entire node instance for wildcard (cast to PromptValue)
        node = run_structure.get_node(node_tag)
        if not node:
            raise ValueError(f"Node with tag '{node_tag}' not found in tree")
        return cast("PromptValue", get_node_instance(node, node_tag))

    # Acquire node
    node = run_structure.get_node(node_tag)
    if not node:
        raise ValueError(f"Node with tag '{node_tag}' not found in tree")

    # Navigate path components
    node_instance = get_node_instance(node, node_tag)

    try:
        return _navigate_object_path(
            node_instance, path, "current node context", node_tag
        )
    except KeyError as e:
        # Missing field fallback if optional
        if _is_missing_field_acceptable(path, node_instance):
            return None  # Return None for acceptable missing fields
        else:
            raise e  # Re-raise the original error


def _is_missing_field_acceptable(path: str, node_instance: object) -> bool:
    """Check if a missing field is acceptable (e.g., optional field)."""
    try:
        # Check if the first component of the path exists as a field but is None/empty
        first_component = path.split(".")[0]
        if hasattr(node_instance, first_component):
            value = getattr(node_instance, first_component)
            # If it's None or an empty collection, missing sub-paths are acceptable
            if value is None or (isinstance(value, list | dict) and len(value) == 0):
                return True

        # Check if it's defined in model fields but not set
        model_fields = getattr(node_instance.__class__, "model_fields", {})
        if first_component in model_fields:
            field_info = model_fields[first_component]
            # If field has a default or is optional, missing is acceptable
            if hasattr(field_info, "default") or not getattr(
                field_info, "is_required", True
            ):
                return True

        return False
    except Exception:
        return False


def _resolve_scope_segment_context(
    run_structure: "RunStructure", path: str, node_tag: str
) -> "ResolutionResult":
    """Route scope-prefixed paths to appropriate context resolvers using scope objects.

    Params:
        run_structure: The RunStructure instance
        path: The path with or without scope prefix (e.g., 'prompt.field', 'value.data', 'field')
        node_tag: The tag/identifier of the current node in the tree

    Returns:
        Result from the appropriate scope resolver - type depends on which resolver is used.
    """
    validate_path_and_node_tag(path, node_tag)

    # Build base context for scope handlers
    context = {"node_tag": node_tag, "run_structure": run_structure}

    # Scope-prefixed path dispatch
    if "." in path:
        potential_scope, rest_of_path = path.split(".", 1)

        # Check if first segment is a known scope and create scope immediately
        if potential_scope == "prompt":
            scope = get_scope("prompt")
        elif potential_scope == "value":
            scope = get_scope("value")
        elif potential_scope == "outputs":
            scope = get_scope("outputs")
        elif potential_scope == "task":
            scope = get_scope("task")
        else:
            # Not a scope - default to current node context
            scope = get_scope("current_node")
            rest_of_path = path  # Use the full path

        return scope.resolve(rest_of_path, context)

    # Unscoped -> current node
    current_node_scope = get_scope("current_node")
    return current_node_scope.resolve(path, context)


def _resolve_in_value_context(
    run_structure: "RunStructure", path: str, node_tag: str
) -> "PromptValue":
    """Resolve a path in the node's value context during chain assembly.

    Currently identical to current-node context since we're building chains,
    not executing them. This is a placeholder for potential future divergence
    where assembly-time values differ from structural defaults.

    Params:
        run_structure: Active structure.
        path: Value path relative to node.
        node_tag: Node tag.

    Returns:
        Resolved value.

    Raises:
        ValueError: Node not found / invalid path.
        KeyError: Missing path component.
    """
    validate_path_and_node_tag(path, node_tag)

    # Acquire node
    node = run_structure.get_node(node_tag)
    if not node:
        raise ValueError(f"Node with tag '{node_tag}' not found in tree")

    # During chain assembly, value context is same as current node context
    node_instance = get_node_instance(node, node_tag)
    return _navigate_object_path(node_instance, path, "value context", node_tag)


def _track_outputs_collection(
    run_structure: "RunStructure",
    target_path: str,
    source_path: str,
    source_node_tag: str,
    target_node: "StructureTreeNode | None",
) -> None:
    """Track outputs collection for multiple sources sending to same outputs field.

    When multiple sources send values to the same outputs.field path, this function
    tracks them for collection during chain assembly and execution.

    Params:
        run_structure: The RunStructure instance
        target_path: The path within outputs scope (e.g., 'results' for outputs.results)
        source_path: The source path being mapped
        source_node_tag: The tag of the source node
        target_node: The target node receiving the outputs
    """
    if target_node is None:
        return  # Can't track for pending targets

    # Initialize collection tracking if needed
    if not hasattr(run_structure, "_outputs_collection"):
        run_structure._outputs_collection = {}

    # Create collection key
    target_node_path = _get_node_path(run_structure, target_node)
    collection_key = f"{target_node_path}.outputs.{target_path}"

    # Track this source for the collection
    if collection_key not in run_structure._outputs_collection:
        run_structure._outputs_collection[collection_key] = []

    # Add this source to the collection
    source_info = {
        "source_node": source_node_tag,
        "source_path": source_path,
        "collected_at": "assembly_time",
    }

    if source_info not in run_structure._outputs_collection[collection_key]:
        run_structure._outputs_collection[collection_key].append(source_info)


def _resolve_in_outputs_context(
    run_structure: "RunStructure", path: str, node_tag: str
) -> "PromptValue":
    """Resolve a path in the node's outputs context during chain assembly.

    During chain assembly, outputs context represents where LLM execution results
    will be stored when chains run. This function manages collection of outputs
    from multiple sources sending to the same outputs.field path.

    Collection behavior: When multiple sources send to same outputs.field,
    they should be collected together rather than replacing each other.

    Params:
        run_structure: Active structure.
        path: Outputs path.
        node_tag: Node tag.

    Returns:
        List of collected values if multiple sources target same path,
        single value if only one source, or None if no outputs yet.

    Raises:
        ValueError: Node not found.
    """
    validate_path_and_node_tag(path, node_tag)

    # Check if we need to track collected outputs
    if not hasattr(run_structure, "_outputs_collection"):
        run_structure._outputs_collection = {}

    # Create key for this outputs path
    collection_key = f"{node_tag}.outputs.{path}"

    # During chain assembly, we track what will be collected
    # Actual collection happens during LangChain execution
    if collection_key in run_structure._outputs_collection:
        # Return the collection for this path
        return run_structure._outputs_collection[collection_key]

    # No collection yet for this path
    return None


def _resolve_in_global_tree_context(
    run_structure: "RunStructure", path: str
) -> "ResolutionResult":
    """Resolve path in global tree context.

    Params:
        run_structure: The RunStructure instance
        path: The global tree path to resolve (e.g., 'task.analysis.sections')

    Returns:
        Either a complete TreeNode instance or a field value from any node in the tree.
        Type depends on whether path points to a node or a field within a node.

        # TODO: Could use overloads to distinguish node vs field access at compile time
        # @overload def _resolve_in_global_tree_context(path: NodePath) -> TreeNode
        # @overload def _resolve_in_global_tree_context(path: FieldPath) -> PromptValue        Raises:
        ValueError: When path is empty or invalid format
        KeyError: When path doesn't exist in global tree
        AttributeError: When attempting to access non-existent attributes

    Implementation details:
    1. Resolve node references across entire tree structure
    2. Support forward references with pending registry integration
    3. Return actual node objects or data from cross-tree paths
    4. Handle cross-tree path resolution (task.analysis.sections)
    References: EXECUTION_DESIGN.md destination path resolution requirements.
    """
    # Validate path parameter
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Accept forms: task.node, task.node.field (all paths must start with task.)

    # Derive node vs field slices
    path_parts = path.split(".")

    # All paths must start with 'task.'
    if path_parts[0] != "task" or len(path_parts) < 2:
        raise ValueError(
            f"Path '{path}' must start with 'task.' and have at least one node component"
        )

    # Remove 'task.' prefix and reconstruct the actual node path
    node_path_parts = path_parts[1:]

    # Node.field pattern
    if len(node_path_parts) > 1:
        # Try treating all but the last part as the node path
        potential_node_path_parts = node_path_parts[:-1]
        field_name = node_path_parts[-1]

        # Reconstruct the full node path (add back task prefix if it was there)
        if path_parts[0] == "task":
            full_node_path = "task." + ".".join(potential_node_path_parts)
        else:
            full_node_path = ".".join(potential_node_path_parts)

        target_node = run_structure.get_node(full_node_path)
        if target_node:
            # We found a node, now access the field
            node_instance = get_node_instance(target_node, full_node_path)
            return _navigate_object_path(
                node_instance, field_name, "global tree context", full_node_path
            )

    # Full node path attempt
    if path_parts[0] == "task":
        full_node_path = path  # Use the original path
    else:
        full_node_path = path

    target_node = run_structure.get_node(full_node_path)
    if target_node:
        # Return the node instance itself
        return get_node_instance(target_node, full_node_path)

    # Pending forward reference check
    pending_targets = run_structure._pending_target_registry.pending_targets
    original_path = path  # Use the original path as provided

    if original_path in pending_targets:
        # Forward reference not yet materialized
        raise KeyError(
            f"Forward reference to '{path}' not yet resolved. "
            f"Target node will be available after tree construction completes."
        )
    else:
        # Path unknown
        raise KeyError(f"Path '{path}' not found in global tree")


def _resolve_in_target_node_context(
    run_structure: "RunStructure", path: str, target_node: "StructureTreeNode | None"
) -> bool:
    """Resolve path in target node's context.

    Params:
        run_structure: The RunStructure instance
        path: The path to resolve within the target node structure
        target_node: The target node to resolve against, may be None for pending targets

    Returns:
        bool: True if the path is valid in the target node structure, False otherwise.
              This validates that the target node has the required structure for the path.

    Raises:
        ValueError: When target_node is None (pending targets not supported)
        KeyError: When path doesn't exist in target node's structure
        AttributeError: When target node has no field_type defined

    Implementation details:
    1. Validate target node has required structure for the path
    2. Check field existence in target node's Pydantic model
    3. Support nested path validation via dot notation
    4. Return validation results for structure creation planning
    References: EXECUTION_DESIGN.md variable mapping target resolution.
    """
    # Target presence
    if target_node is None:
        raise ValueError(
            "Target node cannot be None - pending targets not supported in structure validation"
        )

    # Schema presence
    if not target_node.field_type:
        raise AttributeError(
            f"Target node '{target_node.name}' has no field type defined"
        )

    # Basic path validation
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Instantiate for shape introspection
    try:
        target_instance = target_node.field_type()
    except Exception as e:
        raise ValueError(f"Failed to instantiate target node type: {e}")

    # Structural path traversal
    try:
        # Use the same navigation logic as other context resolvers
        _navigate_object_path(
            target_instance, path, "target node context", target_node.name
        )
        return True  # Path exists and is valid
    except (KeyError, AttributeError, ValueError) as e:
        # Re-raise with more specific context for target validation
        raise KeyError(
            f"Path '{path}' not found in target node '{target_node.name}' structure: {e}"
        )


def _resolve_in_target_outputs_context(
    run_structure: "RunStructure", path: str, target_node: "StructureTreeNode | None"
):
    """Resolve path in target node's outputs context during chain assembly.

    Validates that the path can be created in the target node's outputs when
    chains execute. During assembly phase, this validates structure compatibility.

    Params:
        run_structure: The RunStructure instance
        path: The path to resolve within the target node's outputs
        target_node: The target node to resolve against, may be None

    Returns:
        dict: Validation result for the outputs path

    Raises:
        ValueError: When target_node is None or path is invalid
    """
    if target_node is None:
        raise ValueError("Target node cannot be None for outputs context validation")

    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Get the target node's full path
    target_node_path = _get_node_path(run_structure, target_node)

    # During chain assembly, outputs don't exist yet - validate structure compatibility
    return {
        "status": "valid",
        "path": path,
        "target_node": target_node_path,
        "existing_value": None,
        "validation": "Outputs context will be created during LangChain execution",
    }


def _resolve_in_target_prompt_context(
    run_structure: "RunStructure", path: str, target_node: "StructureTreeNode | None"
):
    """Resolve path in target node's prompt context.

    Params:
        run_structure: The RunStructure instance
        path: The path to resolve within the target node's prompt context
        target_node: The target node to resolve against, may be None

    Returns:
        str: Currently returns placeholder debug string.
             Should return prompt validation result.

    Raises:
        NotImplementedError: When prompt validation fails (planned).
        ValueError: When target_node is None (planned).

    TODO: Implement target prompt context resolution.
    Currently returns debug string - need to:
    1. Access target node's prompt structure and variables
    2. Validate path exists or can be created in prompt context
    3. Return prompt structure validation results
    References: EXECUTION_DESIGN.md prompt scope resolution for targets.
    """
    # TODO: Replace with prompt structure traversal
    if target_node:
        node_path = _get_node_path(run_structure, target_node)
        return f"target_prompt[{node_path}].{path}"
    return f"target_prompt[unknown].{path}"


def _resolve_in_current_prompt_context(
    run_structure: "RunStructure", path: str, node_tag: str
):
    """Resolve a path inside the node's prompt template context.

    Placeholder returning diagnostic string until prompt variable storage is implemented.

    Params:
        run_structure: Active structure (unused placeholder).
        path: Prompt variable / section path.
        node_tag: Node tag.

    Returns:
        Debug string placeholder.
    """
    # TODO: Replace with actual current prompt context validation
    return f"current_prompt[{node_tag}].{path}"


def _get_node_path(run_structure: "RunStructure", node: "StructureTreeNode") -> str:
    """Get the full path/tag of a node by traversing up to root."""
    path_parts = []
    current = node
    while current and current.parent:
        path_parts.append(current.name)
        current = current.parent
    if current:
        path_parts.append(current.name)
    return ".".join(reversed(path_parts))


def _navigate_object_path(
    obj: object, path: str, context_name: str, node_tag: str
) -> "PromptValue":
    """Navigate through an object using dot notation path.

    Params:
        obj: The object to navigate
        path: The dot-separated path to navigate
        context_name: Name of the context for error messages
        node_tag: The node tag for error messages

    Returns:
        The object at the specified path

    Raises:
        ValueError: When path is malformed
        KeyError: When path component doesn't exist
    """
    current_obj = obj
    path_parts = path.split(".")

    for i, part in enumerate(path_parts):
        if not part:  # Handle empty parts from malformed paths
            raise ValueError(
                f"Malformed path '{path}': empty component at position {i}"
            )

        try:
            if hasattr(current_obj, part):
                current_obj = getattr(current_obj, part)
            elif isinstance(current_obj, dict) and part in current_obj:
                current_obj = current_obj[part]
            elif isinstance(current_obj, list):
                # For list navigation, we have special handling based on the context
                # For @each commands, the list itself is the target for iteration
                # Return the list as-is for iteration processing
                if i == len(path_parts) - 1:  # Last part of path
                    # Check if the list items have the requested attribute
                    if (
                        current_obj and hasattr(current_obj[0], part)
                        if current_obj
                        else False
                    ):
                        # Return a list of the attribute values from all items
                        return [
                            getattr(item, part)
                            for item in current_obj
                            if hasattr(item, part)
                        ]
                    else:
                        # If list is empty or items don't have the attribute, return empty list
                        return []
                else:
                    # Not the last part, this shouldn't happen in normal usage
                    raise KeyError(
                        f"Cannot navigate deeper into list at path component '{part}'"
                    )
            else:
                # Check if it's a Pydantic model field that exists but isn't set
                try:
                    model_fields = getattr(current_obj.__class__, "model_fields", {})
                    if part in model_fields:
                        # Field exists but may not be set - get default value
                        field_info = model_fields[part]
                        if hasattr(field_info, "default"):
                            current_obj = field_info.default
                            continue
                except (AttributeError, TypeError):
                    pass  # Ignore attribute errors for non-pydantic objects

                raise KeyError(
                    f"Path component '{part}' not found in {context_name} for {type(current_obj).__name__}"
                )
        except (AttributeError, KeyError, TypeError) as e:
            remaining_path = ".".join(path_parts[i:])
            raise KeyError(
                f"Cannot resolve path '{remaining_path}' in {context_name} for node '{node_tag}': {e}"
            )

    return cast("PromptValue", current_obj)


# Runtime Variable Resolution System
# =================================


def resolve_runtime_variables(
    content: str,
    run_structure: "RunStructure",
    current_node: "StructureTreeNode",
    validate: bool = True,
) -> str:
    """
    Expand runtime variables {variable} to their double-underscore format.

    This function performs variable expansion during assembly/compilation phase,
    converting simple variable references like {variable} into their expanded
    format like {prompt__path__to__node__variable}. These expanded placeholders
    will be resolved to actual values later by LangChain when chain.invoke()
    is called with the appropriate dictionary.

    Runtime variables can reference:
    - Node fields and nested attributes
    - Scope contexts (task, value, outputs, prompt, current_node)

    Note: Assembly variables are NOT available during runtime expansion.
    They are completely separate and only used during chain construction.

    Args:
        content: String content containing runtime variables {var}
        run_structure: RunStructure for context resolution
        current_node: Current node context for variable expansion

    Returns:
        Content with runtime variables expanded to {prompt__path__var} format

    Raises:
        RuntimeVariableError: When variable expansion fails
    """
    if not content:
        return content

    from langtree.exceptions import RuntimeVariableError

    # Pattern matches {variable} - NO assembly variable syntax
    # Excludes reserved template variables PROMPT_SUBTREE and COLLECTED_CONTEXT
    runtime_var_pattern = re.compile(r"\{([^}]+)\}")

    def resolve_variable(match):
        variable_path = match.group(1).strip()

        # Skip reserved template variables
        from langtree.templates.variables import VALID_TEMPLATE_VARIABLES

        if variable_path in VALID_TEMPLATE_VARIABLES:
            return match.group(0)  # Return the original {TEMPLATE_VAR} unchanged

        try:
            # Check for malformed syntax (e.g., {{var}} instead of {var})
            if "{" in variable_path or "}" in variable_path:
                raise RuntimeVariableError(
                    f"Malformed variable syntax. Use single braces {{variable}} for runtime variables, not {{{{{variable_path}}}}}."
                )

            # Runtime variables must be single tokens (no dots, no double underscores)
            if "." in variable_path:
                raise RuntimeVariableError(
                    f"Runtime variable '{{{variable_path}}}' cannot contain dots. Use single token names only."
                )

            if "__" in variable_path:
                raise RuntimeVariableError(
                    f"Runtime variable '{{{variable_path}}}' cannot contain double underscores. Double underscores are reserved for system use."
                )

            # Validate variable exists in current node (field variables OR assembly variables)
            # Only perform existence validation when explicitly requested
            if (
                validate
                and current_node
                and hasattr(current_node, "field_type")
                and current_node.field_type
            ):
                try:
                    from typing import get_type_hints

                    field_hints = get_type_hints(current_node.field_type)

                    # Check if it's a field variable
                    is_field_variable = variable_path in field_hints

                    # Check if it's an assembly variable (available from run_structure)
                    assembly_variables = []
                    is_assembly_variable = False
                    if run_structure and hasattr(
                        run_structure, "_assembly_variable_registry"
                    ):
                        try:
                            assembly_vars_list = run_structure._assembly_variable_registry.get_variables_for_node(
                                current_node.name
                            )
                            assembly_variables = [
                                var.name for var in assembly_vars_list
                            ]
                            is_assembly_variable = variable_path in assembly_variables
                        except Exception:
                            # If assembly variable check fails, assume none available
                            pass

                    # Assembly variables are NOT allowed in runtime contexts
                    if is_assembly_variable:
                        raise RuntimeVariableError(
                            f"Variable '{{{variable_path}}}' is an assembly variable (defined with '! {variable_path}=value') "
                            f"and cannot be used in runtime contexts. Assembly variables are only valid in command arguments."
                        )

                    # If not a field variable, raise error
                    if not is_field_variable:
                        available_fields = list(field_hints.keys())

                        # Only mention assembly variables if some exist
                        if assembly_variables:
                            assembly_info = (
                                f"Available assembly variables: {assembly_variables}. "
                            )
                            assembly_suggestion = f"or verify the assembly variable is declared with '! {variable_path}=value'."
                        else:
                            assembly_info = ""
                            assembly_suggestion = ""

                        raise RuntimeVariableError(
                            f"Runtime variable '{{{variable_path}}}' is undefined. "
                            f"Available fields: {available_fields}. "
                            f"{assembly_info}"
                            f"Check variable name spelling, ensure the field is defined in the TreeNode{', ' + assembly_suggestion if assembly_suggestion else '.'}"
                        )
                except RuntimeVariableError:
                    # Always re-raise our specific validation errors
                    raise
                except Exception as validation_error:
                    # For debugging purposes, continue with expansion but warn about validation errors
                    import warnings

                    warnings.warn(
                        f"Unexpected validation error in runtime variable resolution: {validation_error}"
                    )
                    # Continue with expansion for backward compatibility
                    pass

            # Perform double underscore expansion for runtime variables
            # {variable_name} → {prompt__path__to__node__variable_name}
            if current_node:
                # Handle both StructureTreeNode objects and string paths
                if hasattr(current_node, "name"):
                    # It's a StructureTreeNode object
                    # node.name is stored with underscores: 'task__with_runtime_var'
                    node_path = current_node.name
                    # Remove 'task__' prefix if present (with underscores)
                    if node_path.startswith("task__"):
                        node_path = node_path[6:]  # Remove 'task__' prefix (6 chars)
                    elif node_path.startswith("task."):
                        node_path = node_path[5:]  # Remove 'task.' prefix (5 chars)
                else:
                    # It's a string path
                    node_path = str(current_node)
                    # Remove 'task.' prefix if present
                    if node_path.startswith("task."):
                        node_path = node_path[5:]  # Remove 'task.' prefix
                    # Convert dots to double underscores for string paths
                    node_path = node_path.replace(".", "__")

                expanded_path = "prompt__" + node_path + "__" + variable_path
                return f"{{{expanded_path}}}"
            else:
                # If no current node context, use external scope
                return f"{{prompt__{variable_path}}}"

        except RuntimeVariableError:
            # Re-raise RuntimeVariableError as-is
            raise
        except Exception as e:
            raise RuntimeVariableError(
                f"Failed to resolve runtime variable '{{{variable_path}}}': {str(e)}"
            )

    return runtime_var_pattern.sub(resolve_variable, content)


def _resolve_regular_variable(
    run_structure: "RunStructure", variable_path: str, current_node: "StructureTreeNode"
) -> str:
    """
    Resolve {variable} using standard resolution order.

    Resolution order:
    1. Scoped variables (task.field, value.field, etc.)
    2. Current node fields
    3. Parent node fields (for nested contexts)
    4. Global tree context
    """
    # Handle scoped variables (task.field, value.field, etc.)
    if "." in variable_path:
        scope_parts = variable_path.split(".", 1)
        scope_name = scope_parts[0]
        field_path = scope_parts[1]

        # Check if this is a scope context
        if scope_name in ["task", "value", "outputs", "prompt", "current_node"]:
            return _resolve_scoped_runtime_variable(
                run_structure, scope_name, field_path, current_node
            )

        # Handle nested node references (parent.field, child.field)
        return _resolve_nested_runtime_variable(
            run_structure, variable_path, current_node
        )

    # Simple field resolution on current node
    return _resolve_node_field_variable(run_structure, variable_path, current_node)


def _resolve_scoped_runtime_variable(
    run_structure: "RunStructure",
    scope_name: str,
    field_path: str,
    current_node: "StructureTreeNode",
) -> str:
    """
    Resolve scoped runtime variables like {{task.field}} or {{value.data}}.
    """
    node_tag = current_node.name  # Use node name as tag for resolution

    if scope_name == "task":
        # Resolve in current node context (task context)
        result = _resolve_in_current_node_context(run_structure, field_path, node_tag)
        return str(result) if result is not None else ""

    elif scope_name == "value":
        # Resolve in value context
        result = _resolve_in_value_context(run_structure, field_path, node_tag)
        return str(result) if result is not None else ""

    elif scope_name == "outputs":
        # Resolve in outputs context
        result = _resolve_in_outputs_context(run_structure, field_path, node_tag)
        return str(result) if result is not None else ""

    elif scope_name == "prompt":
        # Resolve in prompt context
        result = _resolve_in_current_prompt_context(run_structure, field_path, node_tag)
        return str(result) if result is not None else ""

    elif scope_name == "current_node":
        # Resolve in current node context
        result = _resolve_in_current_node_context(run_structure, field_path, node_tag)
        return str(result) if result is not None else ""

    from langtree.exceptions import RuntimeVariableError

    raise RuntimeVariableError(f"Unknown scope: {scope_name}")


def _resolve_nested_runtime_variable(
    run_structure: "RunStructure", variable_path: str, current_node: "StructureTreeNode"
) -> str:
    """
    Resolve nested runtime variables like {{parent.field}} or {{child.field}}.
    """
    path_parts = variable_path.split(".", 1)
    node_reference = path_parts[0]
    field_path = path_parts[1]

    # Handle parent reference
    if node_reference == "parent" and current_node.parent:
        parent_tag = current_node.parent.name
        result = _resolve_in_current_node_context(run_structure, field_path, parent_tag)
        return str(result) if result is not None else ""

    # Handle child reference
    target_child = None
    for child in current_node.children:
        if child.name == node_reference:
            target_child = child
            break

    if target_child:
        child_tag = target_child.name
        result = _resolve_in_current_node_context(run_structure, field_path, child_tag)
        return str(result) if result is not None else ""

    # Handle sibling or other tree references
    try:
        result = _resolve_in_global_tree_context(run_structure, variable_path)
        return str(result) if result is not None else ""
    except Exception:
        pass

    from langtree.exceptions import RuntimeVariableError

    raise RuntimeVariableError(f"Could not resolve nested variable: {variable_path}")


def _resolve_node_field_variable(
    run_structure: "RunStructure", field_name: str, current_node: "StructureTreeNode"
) -> str:
    """
    Resolve simple field variables like {{field_name}} on the current node.
    """
    node_tag = current_node.name

    try:
        result = _resolve_in_current_node_context(run_structure, field_name, node_tag)
        return str(result) if result is not None else ""
    except Exception:
        # Try parent context if field not found on current node
        if current_node.parent:
            try:
                parent_tag = current_node.parent.name
                result = _resolve_in_current_node_context(
                    run_structure, field_name, parent_tag
                )
                return str(result) if result is not None else ""
            except Exception:
                pass

    from langtree.exceptions import RuntimeVariableError

    raise RuntimeVariableError(
        f"Failed to resolve runtime variable '{{{field_name}}}': field not found in current or parent context"
    )


def resolve_runtime_variables_in_commands(
    command: "ParsedCommand",
    run_structure: "RunStructure",
    current_node: "StructureTreeNode",
) -> "ParsedCommand":
    """
    Resolve runtime variables in LangTree DSL command arguments.

    This processes command arguments that contain {{variable}} syntax
    and resolves them before command execution.
    """
    # TODO: Implement runtime variable resolution in command arguments
    # This will process command.parameters and resolve any runtime variables
    # contained within string parameters
    return command


def cache_runtime_variable_resolution(
    variable_path: str, resolved_value: str, current_node: "StructureTreeNode"
) -> None:
    """
    Cache runtime variable resolution for performance optimization.

    This helps avoid repeated resolution of the same variables
    within the same node context.
    """
    # TODO: Implement caching system for runtime variable resolution
    # This could use a simple dict cache keyed by (node_tag, variable_path)
    pass


def bulk_resolve_runtime_variables(
    variable_list: list[str],
    run_structure: "RunStructure",
    current_node: "StructureTreeNode",
) -> dict[str, str]:
    """
    Resolve multiple runtime variables in a single operation.

    This is more efficient than resolving variables one by one
    when processing large amounts of content.
    """
    # TODO: Implement bulk resolution for performance optimization
    results = {}
    for variable in variable_list:
        try:
            results[variable] = resolve_runtime_variables(
                f"{{{variable}}}", run_structure, current_node
            ).strip("{}")
        except Exception as e:
            results[variable] = f"ERROR: {str(e)}"

    return results
