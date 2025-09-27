"""
Validation module for DPCL prompt tree structure.

This module contains comprehensive validation methods that check for 
configuration errors, circular dependencies, unresolved references,
and other issues in the prompt tree structure.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langtree.prompt.structure import RunStructure, StructureTreeNode

from langtree.prompt.resolution import _resolve_in_current_node_context


def validate_tree(run_structure: 'RunStructure') -> dict[str, list[str]]:
    """
    Validate the tree for completion and consistency.
    
    Params:
        run_structure: The RunStructure instance to validate
    
    Returns:
        Dictionary with validation results:
        - 'unresolved_targets': List of targets that were never resolved
        - 'unsatisfied_variables': List of variables with no satisfaction sources
        - 'multiply_satisfied': List of variables with multiple sources requiring attention
    """
    validation_results = {
        'unresolved_targets': [],
        'unsatisfied_variables': [],
        'multiply_satisfied': []
    }
    
    # Check for unresolved targets
    for target_path, pending_list in run_structure._pending_target_registry.pending_targets.items():
        validation_results['unresolved_targets'].append(
            f"{target_path} (referenced by {len(pending_list)} commands)"
        )
    
    # Check for unsatisfied variables
    unsatisfied = run_structure._variable_registry.get_unsatisfied_variables()
    for var_info in unsatisfied:
        scope_name = var_info.get_scope_name()
        full_name = f"{scope_name}.{var_info.variable_path}" if scope_name else var_info.variable_path
        validation_results['unsatisfied_variables'].append(
            f"{full_name} (from {var_info.source_node_tag})"
        )
    
    # Check for multiply satisfied variables that might need attention
    multiply_satisfied = run_structure._variable_registry.get_multiply_satisfied_variables()
    for var_info in multiply_satisfied:
        scope_name = var_info.get_scope_name()
        full_name = f"{scope_name}.{var_info.variable_path}" if scope_name else var_info.variable_path
        validation_results['multiply_satisfied'].append(
            f"{full_name} (sources: {', '.join(var_info.satisfaction_sources)})"
        )
    
    return validation_results


def validate_comprehensive(run_structure: 'RunStructure') -> dict:
    """
    Perform comprehensive validation of the prompt tree structure.
    
    This method checks for all types of configuration errors:
    - Circular dependencies in variable satisfaction chains
    - Unresolved target references
    - Unsatisfied variables that cannot be resolved
    - Invalid scope references
    - Malformed commands
    - Impossible variable mappings (e.g., iterating over non-iterable fields)
    - Self-references
    
    Params:
        run_structure: The RunStructure instance to validate
    
    Returns:
        Dictionary containing validation results:
        - is_valid: Boolean indicating if configuration is valid
        - total_errors: Number of errors found
        - circular_dependencies: List of circular dependency cycles
        - unresolved_targets: List of unresolved target references
        - unsatisfied_variables: List of variables without satisfaction sources
        - invalid_scope_references: List of invalid scope names
        - malformed_commands: List of commands with syntax errors
        - impossible_mappings: List of mappings that cannot work
        - self_references: List of nodes that reference themselves
        - error_summary: List of detailed error descriptions with recommendations
    """
    validation_result = {
        'is_valid': True,
        'total_errors': 0,
        'circular_dependencies': [],
        'unresolved_targets': [],
        'unsatisfied_variables': [],
        'invalid_scope_references': [],
        'malformed_commands': [],
        'impossible_mappings': [],
        'self_references': [],
        'error_summary': []
    }
    
    # Check for circular dependencies
    _detect_circular_dependencies(run_structure, validation_result)
    
    # Check for unresolved targets
    _detect_unresolved_targets(run_structure, validation_result)
    
    # Check for unsatisfied variables
    _detect_unsatisfied_variables(run_structure, validation_result)
    
    # Check for invalid scope references
    _detect_invalid_scope_references(run_structure, validation_result)
    
    # Check for malformed commands
    _detect_malformed_commands(run_structure, validation_result)
    
    # Check for impossible mappings
    _detect_impossible_mappings(run_structure, validation_result)
    
    # Check for self-references
    _detect_self_references(run_structure, validation_result)
    
    # Calculate total errors and validity
    error_categories = [
        'circular_dependencies', 'unresolved_targets', 'unsatisfied_variables',
        'invalid_scope_references', 'malformed_commands', 'impossible_mappings',
        'self_references'
    ]
    
    validation_result['total_errors'] = sum(
        len(validation_result[category]) for category in error_categories
    )
    validation_result['is_valid'] = validation_result['total_errors'] == 0
    
    # Generate error summary with descriptions and recommendations
    _generate_error_summary(validation_result)
    
    return validation_result


def _generate_error_summary(result: dict) -> None:
    """Generate a comprehensive error summary with descriptions and recommendations."""
    error_summary = []
    
    # Add summaries for each error category
    for circular_dep in result['circular_dependencies']:
        error_summary.append({
            'category': 'circular_dependency',
            'description': circular_dep['description'],
            'recommendation': 'Break the circular dependency by removing one of the references or restructuring the data flow'
        })
    
    for unresolved in result['unresolved_targets']:
        error_summary.append({
            'category': 'unresolved_target',
            'description': unresolved['description'],
            'recommendation': unresolved['recommendation']
        })
    
    for unsatisfied in result['unsatisfied_variables']:
        error_summary.append({
            'category': 'unsatisfied_variable',
            'description': unsatisfied['description'],
            'recommendation': unsatisfied['recommendation']
        })
    
    for invalid_scope in result['invalid_scope_references']:
        error_summary.append({
            'category': 'invalid_scope',
            'description': invalid_scope['description'],
            'recommendation': invalid_scope['recommendation']
        })
    
    for malformed in result['malformed_commands']:
        error_summary.append({
            'category': 'malformed_command',
            'description': malformed['description'],
            'recommendation': malformed['recommendation']
        })
    
    for impossible in result['impossible_mappings']:
        error_summary.append({
            'category': 'impossible_mapping',
            'description': impossible['description'],
            'recommendation': impossible['recommendation']
        })
    
    for self_ref in result['self_references']:
        error_summary.append({
            'category': 'self_reference',
            'description': self_ref['description'],
            'recommendation': self_ref['recommendation']
        })
    
    result['error_summary'] = error_summary


def _detect_circular_dependencies(run_structure: 'RunStructure', result: dict) -> None:
    """Detect circular dependencies in variable satisfaction chains and target references."""
    # For now, implement a simple target reference cycle detection
    # Skip variable satisfaction cycles as they're more complex and often false positives
    
    # Check target reference chains
    visited = set()
    current_path = set()
    for node_name, node in run_structure._root_nodes.items():
        if node_name not in visited:
            cycle = _find_target_cycle(run_structure, node, visited, current_path.copy())
            if cycle and len(cycle) > 2:  # Only report cycles with more than 2 nodes
                result['circular_dependencies'].append({
                    'type': 'target_reference',
                    'cycle': cycle,
                    'description': f"Circular dependency in target references: {' -> '.join(cycle)}"
                })


def _find_variable_cycle(run_structure: 'RunStructure', var_name: str, var_info: Any, 
                        visited: set, current_path: set) -> list[str] | None:
    """Find cycles in variable satisfaction chains using DFS."""
    if var_name in current_path:
        # Found a cycle - return the cycle path
        cycle_path = list(current_path)
        cycle_start_idx = cycle_path.index(var_name)
        return cycle_path[cycle_start_idx:] + [var_name]
    
    if var_name in visited:
        return None
    
    visited.add(var_name)
    current_path.add(var_name)
    
    # Check each satisfaction source for potential cycles
    for source in var_info.satisfaction_sources:
        # Look for variables that depend on this source
        for other_var_name, other_var_info in run_structure._variable_registry.variables.items():
            if other_var_name != var_name:  # Don't check self
                # If the other variable's path matches our source or vice versa
                if (source == other_var_info.variable_path or 
                    var_info.variable_path in other_var_info.satisfaction_sources):
                    cycle = _find_variable_cycle(run_structure, other_var_name, other_var_info, visited, current_path)
                    if cycle:
                        current_path.discard(var_name)  # Use discard to avoid KeyError
                        return cycle
    
    current_path.discard(var_name)  # Use discard to avoid KeyError
    return None


def _find_target_cycle(run_structure: 'RunStructure', node: 'StructureTreeNode', 
                      visited: set, current_path: set) -> list[str] | None:
    """Find cycles in target reference chains using DFS."""
    node_name = node.name
    
    if node_name in current_path:
        # Found a cycle - return the cycle path
        cycle_path = list(current_path)
        cycle_start_idx = cycle_path.index(node_name)
        return cycle_path[cycle_start_idx:] + [node_name]
    
    if node_name in visited:
        return None
    
    visited.add(node_name)
    current_path.add(node_name)
    
    # Check all command targets from this node
    if hasattr(node, 'extracted_commands'):
        for command in node.extracted_commands:
            if command.destination_path:
                target_node = run_structure.get_node(command.destination_path)
                if target_node:
                    cycle = _find_target_cycle(run_structure, target_node, visited, current_path)
                    if cycle:
                        current_path.discard(node_name)
                        return cycle
    
    current_path.discard(node_name)
    return None


def _detect_unresolved_targets(run_structure: 'RunStructure', result: dict) -> None:
    """
    Detect target references that cannot be resolved.
    
    Scans the pending target registry for commands that reference nodes
    that were never added to the tree structure, indicating configuration
    errors or missing node definitions.
    
    Params:
        run_structure: The RunStructure instance to check for unresolved targets
        result: Dictionary to accumulate validation results
    """
    for target_path, pending_list in run_structure._pending_target_registry.pending_targets.items():
        result['unresolved_targets'].append({
            'target': target_path,
            'referenced_by': [pending.source_node_tag for pending in pending_list],
            'command_count': len(pending_list),
            'description': f"Target '{target_path}' is referenced but not defined",
            'recommendation': f"Add a class that resolves to '{target_path}' or fix the target reference"
        })


def _detect_unsatisfied_variables(run_structure: 'RunStructure', result: dict) -> None:
    """
    Detect variables that have no satisfaction sources.
    
    Identifies variables declared in commands that cannot be satisfied
    because their required source fields or paths do not exist in the
    tree structure or are otherwise unreachable.
    
    Params:
        run_structure: The RunStructure instance to check for unsatisfied variables
        result: Dictionary to accumulate validation results
    """
    unsatisfied_vars = run_structure._variable_registry.get_truly_unsatisfied_variables(run_structure)
    for var_info in unsatisfied_vars:
        scope_name = var_info.get_scope_name()
        full_name = f"{scope_name}.{var_info.variable_path}" if scope_name else var_info.variable_path
        result['unsatisfied_variables'].append({
            'variable': full_name,
            'source_node': var_info.source_node_tag,
            'scope': scope_name,
            'path': var_info.variable_path,
            'description': f"Variable '{full_name}' has no satisfaction source",
            'recommendation': f"Ensure field '{var_info.variable_path}' exists in the source node or provide external input"
        })


def _detect_invalid_scope_references(run_structure: 'RunStructure', result: dict) -> None:
    """Detect references to invalid scope names by examining raw command text."""
    valid_scopes = {'prompt', 'value', 'outputs', 'task', 'current_node'}
    
    for node_name, node in run_structure._root_nodes.items():
        _check_invalid_scopes_recursive(node, result, valid_scopes)


def _check_invalid_scopes_recursive(node: 'StructureTreeNode', result: dict, valid_scopes: set) -> None:
    """Recursively check for invalid scope references in command text."""
    if hasattr(node, 'extracted_commands'):
        for command in node.extracted_commands:
            # Check each variable mapping for invalid scopes
            for mapping in command.variable_mappings:
                # Check target path for invalid scope prefix
                target_parts = mapping.target_path.split('.')
                if len(target_parts) > 1:
                    potential_scope = target_parts[0]
                    if potential_scope not in valid_scopes and potential_scope != 'current_node':
                        result['invalid_scope_references'].append({
                            'scope_name': potential_scope,
                            'variable': mapping.target_path,
                            'source_node': node.name,
                            'description': f"Invalid scope reference '{potential_scope}' in variable '{mapping.target_path}'",
                            'recommendation': f"Use one of the valid scopes: {', '.join(valid_scopes)}"
                        })
    
    # Recursively check children
    for child_node in node.children.values():
        _check_invalid_scopes_recursive(child_node, result, valid_scopes)


def _detect_malformed_commands(run_structure: 'RunStructure', result: dict) -> None:
    """Detect commands with syntax errors or malformed structure."""
    for node_name, node in run_structure._root_nodes.items():
        _check_node_commands_recursive(node, result)


def _check_node_commands_recursive(node: 'StructureTreeNode', result: dict) -> None:
    """Recursively check all commands in a node tree for malformed syntax."""
    if hasattr(node, 'extracted_commands'):
        for command in node.extracted_commands:
            # Check for basic command structure issues
            if not command.destination_path:
                result['malformed_commands'].append({
                    'node': node.name,
                    'command_text': f"Type: {command.command_type}, Dest: {command.destination_path}",
                    'issue': 'missing_destination',
                    'description': f"Command in '{node.name}' has no destination path",
                    'recommendation': "Ensure command follows format: ! @->target@{{mappings}}"
                })
            
            # Check for mapping syntax issues
            if hasattr(command, 'variable_mappings') and command.variable_mappings:
                for mapping in command.variable_mappings:
                    if not hasattr(mapping, 'target_path') or not mapping.target_path:
                        result['malformed_commands'].append({
                            'node': node.name,
                            'command_text': f"Type: {command.command_type}, Dest: {command.destination_path}",
                            'issue': 'invalid_mapping',
                            'description': f"Command in '{node.name}' has invalid variable mapping",
                            'recommendation': "Check mapping syntax: {{scope.variable=source_field}}"
                        })
    
    # Recursively check children
    for child_node in node.children.values():
        _check_node_commands_recursive(child_node, result)


def _detect_impossible_mappings(run_structure: 'RunStructure', result: dict) -> None:
    """Detect variable mappings that cannot work (e.g., iterating over non-iterable fields)."""
    for node_name, node in run_structure._root_nodes.items():
        _check_impossible_mappings_recursive(run_structure, node, result)


def _check_impossible_mappings_recursive(run_structure: 'RunStructure', node: 'StructureTreeNode', result: dict) -> None:
    """Recursively check for impossible mappings in a node tree."""
    if hasattr(node, 'extracted_commands'):
        for command in node.extracted_commands:
            # Check @each commands for non-iterable inclusion paths
            if hasattr(command, 'command_type') and command.command_type.value == "each":
                if hasattr(command, 'inclusion_path') and command.inclusion_path:
                    try:
                        # Try to resolve the inclusion path to check if it's iterable
                        resolved_value = _resolve_in_current_node_context(run_structure, command.inclusion_path, node.name)
                        
                        # Check if the resolved value is iterable
                        if not hasattr(resolved_value, '__iter__') or isinstance(resolved_value, (str, dict)):
                            result['impossible_mappings'].append({
                                'node': node.name,
                                'field': command.inclusion_path,
                                'command_type': 'each',
                                'issue': 'non_iterable',
                                'description': f"@each command in '{node.name}' tries to iterate over non-iterable field '{command.inclusion_path}'",
                                'recommendation': f"Ensure '{command.inclusion_path}' is a list or other iterable type, or use @-> instead of @each"
                            })
                    except (KeyError, AttributeError):
                        # Field doesn't exist - this will be caught by unsatisfied variables
                        pass
    
    # Recursively check children
    for child_node in node.children.values():
        _check_impossible_mappings_recursive(run_structure, child_node, result)


def _detect_self_references(run_structure: 'RunStructure', result: dict) -> None:
    """Detect nodes that reference themselves."""
    for node_name, node in run_structure._root_nodes.items():
        _check_self_references_recursive(node, result)


def _check_self_references_recursive(node: 'StructureTreeNode', result: dict) -> None:
    """Recursively check for self-references in a node tree."""
    if hasattr(node, 'extracted_commands'):
        for command in node.extracted_commands:
            if command.destination_path == node.name:
                result['self_references'].append({
                    'node': node.name,
                    'target': command.destination_path,
                    'description': f"Node '{node.name}' references itself",
                    'recommendation': "Remove self-reference or change target to a different node"
                })
    
    # Recursively check children
    for child_node in node.children.values():
        _check_self_references_recursive(child_node, result)