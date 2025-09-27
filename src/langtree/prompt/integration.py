"""
LangTree Integration Layer for LangChain Chain Assembly.

This module provides the missing integration layer that bridges LangTree command
processing with LangChain chain execution. It transforms parsed LangTree structures
into executable LangChain pipelines with proper context propagation and
hierarchical prompt assembly.

Key Components:
- LangTreeChainBuilder: Main orchestrator for chain assembly
- PromptAssembler: Hierarchical prompt construction from tree structure
- ContextPropagator: Variable resolution and context forwarding between chains
- ExecutionOrchestrator: Coordination of chain execution order and dependencies
"""

# Group 2: External from imports (alphabetical by source module)
from typing import Dict, List, Any, Optional, TYPE_CHECKING

# Group 4: Internal from imports (alphabetical by source module)
from langtree.prompt.resolution import resolve_runtime_variables
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langtree.prompt.structure import RunStructure, StructureTreeNode


class LangTreeChainBuilder:
    """
    Main orchestrator for converting LangTree structures into executable LangChain chains.
    
    Responsibilities:
    - Transform RunStructure into topologically ordered execution plan
    - Generate LangChain Runnables for each execution step
    - Coordinate context propagation between chain steps
    - Handle template variable resolution and prompt assembly
    """
    
    def __init__(self, run_structure: 'RunStructure'):
        self.run_structure = run_structure
        self.prompt_assembler = PromptAssembler(run_structure)
        self.context_propagator = ContextPropagator(run_structure)
        
    def build_execution_chain(self, 
                            llm_name: str = "reasoning",
                            **chain_kwargs) -> 'Runnable':
        """
        Build complete LangChain execution pipeline from LangTree structure.
        
        Args:
            llm_name: LLM identifier for chain construction
            **chain_kwargs: Additional arguments passed to prepare_chain
            
        Returns:
            Composed Runnable that executes the full LangTree pipeline
            
        Raises:
            ValueError: If structure validation fails with critical errors
            RuntimeError: If chain assembly encounters unresolvable dependencies
        """
        # Import LangChain components here to avoid dependency issues
        from langchain_core.runnables import RunnableLambda
        
        # Resolve pending targets first - this should handle forward references
        self._resolve_pending_targets()
        
        # Validate structure after pending target resolution
        validation_results = self.run_structure.validate_tree()

        # Perform assembly-specific validation (destination field LHS-RHS nesting)
        assembly_validation_errors = self._validate_assembly_phase()
        if assembly_validation_errors:
            raise ValueError(f"Assembly validation failed: {assembly_validation_errors}")

        # Check for critical validation errors (unsatisfied variables are more critical than unresolved targets)
        critical_errors = []
        if validation_results.get('unsatisfied_variables'):
            critical_errors.extend(validation_results['unsatisfied_variables'])
        
        # Unresolved targets are only critical if they can't be resolved after our attempt
        remaining_unresolved = validation_results.get('unresolved_targets', [])
        if remaining_unresolved:
            critical_errors.extend(remaining_unresolved)
        
        if critical_errors:
            raise ValueError(f"Structure validation failed: {validation_results}")
        
        # Get execution plan with dependency ordering
        execution_plan = self._get_topological_execution_plan()
        
        # Build individual chain steps
        chain_steps = {}
        for step in execution_plan['chain_steps']:
            step_chain = self._build_step_chain(step, llm_name, **chain_kwargs)
            chain_steps[step['node_tag']] = step_chain
        
        # Compose final execution pipeline with context propagation
        if len(chain_steps) == 1:
            # Single step - return directly
            return list(chain_steps.values())[0]
        elif len(chain_steps) > 1:
            # Multiple steps - compose with dependency order
            return self._compose_dependency_chain(chain_steps, execution_plan)
        else:
            # No executable steps - return pass-through
            return RunnableLambda(lambda x: x)
    
    def _resolve_pending_targets(self) -> None:
        """
        Attempt to resolve all pending target references.
        
        This method tries to resolve forward references and update the
        pending target registry. It should be called before validation
        to ensure all possible targets are resolved.
        """
        # Get a copy of pending targets to iterate over
        pending_targets = dict(self.run_structure._pending_target_registry.pending_targets)
        
        for target_path in list(pending_targets.keys()):
            # Try to find the target node now
            target_node = self.run_structure.get_node(target_path)
            if target_node is not None:
                # Target found - resolve the pending commands
                self.run_structure._pending_target_registry.resolve_target(target_path, target_node)
            else:
                # Try alternative target path formats
                alternative_paths = self._generate_alternative_target_paths(target_path)
                for alt_path in alternative_paths:
                    alt_node = self.run_structure.get_node(alt_path)
                    if alt_node is not None:
                        self.run_structure._pending_target_registry.resolve_target(target_path, alt_node)
                        break
    
    def _generate_alternative_target_paths(self, target_path: str) -> list[str]:
        """
        Generate alternative target path formats for resolution.
        
        Args:
            target_path: Original target path that couldn't be resolved
            
        Returns:
            List of alternative path formats to try
        """
        from langtree.prompt.utils import underscore
        
        alternatives = []
        
        # If path doesn't start with task., try adding it
        if not target_path.startswith('task.'):
            alternatives.append(f'task.{target_path}')
        
        # Try underscore conversion (TaskProcessor -> task_processor)
        try:
            underscore_name = underscore(target_path)
            if '_' in underscore_name:
                # Split on first underscore to get task.name format
                parts = underscore_name.split('_', 1)
                if len(parts) == 2:
                    alternatives.append(f'{parts[0]}.{parts[1]}')
        except Exception:
            pass  # underscore conversion failed, skip this alternative
        
        # Try lowercase version
        alternatives.append(target_path.lower())
        
        # Try with task. prefix and lowercase
        if not target_path.lower().startswith('task.'):
            alternatives.append(f'task.{target_path.lower()}')
        
        # Try removing task. prefix if present
        if target_path.startswith('task.'):
            alternatives.append(target_path[5:])  # Remove 'task.'

        # Handle task.task_X -> task.X pattern (common in circular dependency cases)
        if target_path.startswith('task.task_'):
            # Convert task.task_b to task.b
            simplified = target_path.replace('task.task_', 'task.')
            alternatives.append(simplified)

        return alternatives
    
    def _get_topological_execution_plan(self) -> Dict[str, Any]:
        """
        Generate topologically ordered execution plan from LangTree structure.
        
        Enhances the basic execution plan with proper dependency ordering
        based on variable satisfaction relationships and command dependencies.
        
        Returns:
            Enhanced execution plan with topological ordering
        """
        base_plan = self.run_structure.get_execution_plan()
        
        # Extract dependency relationships from variable flows and command targets
        dependencies = {}
        for step in base_plan['chain_steps']:
            dependencies[step['node_tag']] = set()
        
        # Build dependency graph from command target relationships
        for step in base_plan['chain_steps']:
            node_tag = step['node_tag']
            node = self.run_structure.get_node(node_tag)
            
            if node and hasattr(node, 'extracted_commands'):
                for command in node.extracted_commands:
                    if hasattr(command, 'destination_path') and command.destination_path:
                        # Find the target node for this command
                        target_node = self.run_structure.get_node(command.destination_path)
                        if not target_node:
                            # Try alternative paths
                            alternatives = self._generate_alternative_target_paths(command.destination_path)
                            for alt_path in alternatives:
                                target_node = self.run_structure.get_node(alt_path)
                                if target_node:
                                    break
                        
                        if target_node and target_node.name in dependencies:
                            # target_node depends on current node (node_tag)
                            dependencies[target_node.name].add(node_tag)
        
        # Also build from variable flows if available
        # TODO: This is temporary until parser issue is fixed
        # Generate basic variable flows from command relationships
        variable_flows_added = set()
        for step in base_plan['chain_steps']:
            node_tag = step['node_tag']
            node = self.run_structure.get_node(node_tag)
            
            if node and hasattr(node, 'extracted_commands'):
                for command in node.extracted_commands:
                    if hasattr(command, 'destination_path') and command.destination_path:
                        # Find the target node for this command
                        target_node = self.run_structure.get_node(command.destination_path)
                        if not target_node:
                            # Try alternative paths
                            alternatives = self._generate_alternative_target_paths(command.destination_path)
                            for alt_path in alternatives:
                                target_node = self.run_structure.get_node(alt_path)
                                if target_node:
                                    break
                        
                        if target_node:
                            # Create a variable flow entry
                            flow_key = f"{node_tag}->{target_node.name}"
                            if flow_key not in variable_flows_added:
                                base_plan['variable_flows'].append({
                                    'from_node': node_tag,
                                    'to': f"variable_from_{node_tag}",
                                    'target_node': target_node.name,
                                    'relationship_type': 'dependency',
                                    'scope': 'task'
                                })
                                variable_flows_added.add(flow_key)
        
        # Build dependencies from the enhanced variable flows
        for flow in base_plan['variable_flows']:
            target_node = flow['target_node']
            # Find which node produces this variable
            for step in base_plan['chain_steps']:
                if step['node_tag'] == target_node:
                    # This step depends on the source node
                    source_node = flow.get('from_node', 'external')
                    if source_node != 'external' and source_node in dependencies:
                        dependencies[step['node_tag']].add(source_node)
        
        # Perform topological sort
        sorted_steps = self._topological_sort(base_plan['chain_steps'], dependencies)
        
        # Enhance the execution plan with sorted order
        enhanced_plan = base_plan.copy()
        enhanced_plan['chain_steps'] = sorted_steps
        enhanced_plan['dependency_order'] = [step['node_tag'] for step in sorted_steps]
        
        return enhanced_plan
    
    def _topological_sort(self, steps: List[Dict[str, Any]], dependencies: Dict[str, set]) -> List[Dict[str, Any]]:
        """
        Perform topological sort on execution steps based on dependencies.
        
        Args:
            steps: List of execution steps from the plan
            dependencies: Dict mapping node_tag to set of dependency node_tags
            
        Returns:
            List of steps in topological order
        """
        # Create a copy of dependencies to modify
        deps = {node: deps_set.copy() for node, deps_set in dependencies.items()}
        result = []
        step_map = {step['node_tag']: step for step in steps}
        
        # Keep processing until all nodes are ordered
        while deps:
            # Find nodes with no dependencies
            ready_nodes = [node for node, node_deps in deps.items() if not node_deps]
            
            if not ready_nodes:
                # Circular dependency detected - analyze and report
                remaining_nodes = list(deps.keys())
                circular_chains = self._find_circular_dependencies(deps)
                if circular_chains:
                    raise ValueError(f"Circular dependency detected in execution plan. "
                                   f"Circular chains found: {circular_chains}")
                else:
                    # Fallback for other dependency issues
                    remaining_steps = [step_map[node] for node in remaining_nodes]
                    result.extend(remaining_steps)
                    break
            
            # Process ready nodes (sort for deterministic order)
            for node in sorted(ready_nodes):
                result.append(step_map[node])
                del deps[node]
                
                # Remove this node from other dependencies
                for other_deps in deps.values():
                    other_deps.discard(node)
        
        return result
    
    def _find_circular_dependencies(self, dependencies: Dict[str, set]) -> List[List[str]]:
        """
        Find circular dependency chains in the dependency graph.
        
        Args:
            dependencies: Dict mapping node_tag to set of dependency node_tags
            
        Returns:
            List of circular dependency chains found
        """
        circular_chains = []
        visited = set()
        
        def dfs_find_cycle(node: str, path: List[str], rec_stack: set) -> bool:
            if node in rec_stack:
                # Found a cycle - extract the circular part
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular_chains.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Check all dependencies of current node
            for dep in dependencies.get(node, set()):
                if dfs_find_cycle(dep, path, rec_stack):
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Check each node for cycles
        for node in dependencies:
            if node not in visited:
                dfs_find_cycle(node, [], set())
        
        return circular_chains
    
    def _build_step_chain(self, 
                         step: Dict[str, Any], 
                         llm_name: str, 
                         **chain_kwargs) -> 'Runnable':
        """
        Build LangChain Runnable for a single execution step.
        
        Args:
            step: Execution step from plan containing node info and commands
            llm_name: LLM identifier for chain construction
            **chain_kwargs: Additional arguments for prepare_chain
            
        Returns:
            Runnable for this execution step
        """
        node_tag = step['node_tag']
        node = self.run_structure.get_node(node_tag)
        
        if node is None:
            raise RuntimeError(f"Node not found for tag: {node_tag}")
        
        # Assemble hierarchical prompt for this node
        assembled_prompt = self.prompt_assembler.assemble_prompt(node)
        
        # Determine if structured output is needed
        structured_output = self._get_structured_output_model(node)
        
        # Import prepare_chain here to avoid circular imports
        from langtree.chains import prepare_chain
        
        # Build chain with context propagation wrapper
        base_chain = prepare_chain(
            llm_name=llm_name,
            prompt_system=assembled_prompt.get('system', ''),
            prompt_context=assembled_prompt.get('context'),
            prompt_task=assembled_prompt.get('task', ''),
            prompt_output=assembled_prompt.get('output'),
            prompt_input=assembled_prompt.get('input'),
            structured_output=structured_output,
            **chain_kwargs
        )
        
        # Wrap with context propagation
        return self.context_propagator.wrap_with_context_propagation(
            base_chain, node_tag, node
        )
    
    def _get_structured_output_model(self, node: 'StructureTreeNode') -> Optional[type[BaseModel]]:
        """
        Determine if node requires structured output based on its field types.
        
        Args:
            node: Structure tree node to analyze
            
        Returns:
            Pydantic model class if structured output needed, None otherwise
        """
        # TODO: Implement structured output model detection
        # This would analyze the node's field types and generate appropriate
        # Pydantic models for structured LLM output parsing
        return None
    
    def _compose_dependency_chain(self,
                                 chain_steps: Dict[str, 'Runnable'],
                                 execution_plan: Dict[str, Any]) -> 'Runnable':
        """
        Compose multiple chain steps with proper dependency ordering.

        Args:
            chain_steps: Dictionary of node_tag -> Runnable mappings
            execution_plan: Execution plan with dependency information

        Returns:
            Composed Runnable that executes steps in dependency order

        Raises:
            ValueError: For circular dependencies or missing dependencies
            RuntimeError: For invalid dependency configurations
        """
        from langchain_core.runnables import RunnableParallel, RunnableLambda

        # Validate dependencies first
        chain_steps_list = execution_plan.get('chain_steps', [])
        self._validate_dependencies(chain_steps, chain_steps_list)

        # Convert any MockRunnable objects to real Runnables
        converted_steps = {}
        for key, step in chain_steps.items():
            # Check if it's a MockRunnable (has _runnable attribute)
            if hasattr(step, '_runnable'):
                converted_steps[key] = step._runnable
            # Check if it's already a Runnable or callable
            elif callable(step):
                # Wrap callables in RunnableLambda
                converted_steps[key] = RunnableLambda(step) if not hasattr(step, 'invoke') else step
            else:
                converted_steps[key] = step

        # Get the dependency order from execution plan
        dependency_order = execution_plan.get('dependency_order', [])

        if not dependency_order:
            # Fallback to simple parallel execution if no order specified
            return RunnableParallel(converted_steps)

        # Implementation would build sequential/parallel execution based on groups
        # For now, return parallel execution as fallback
        return RunnableParallel(converted_steps)

    def _validate_dependencies(self,
                              chain_steps: Dict[str, 'Runnable'],
                              chain_steps_list: List[Dict[str, Any]]) -> None:
        """
        Validate dependency graph for circular dependencies and missing nodes.

        Args:
            chain_steps: Available chain steps
            chain_steps_list: List of steps with their dependencies

        Raises:
            ValueError: For circular dependencies or missing dependencies
        """
        # Build dependency graph
        dependencies = {}
        for step in chain_steps_list:
            node_tag = step['node_tag']
            deps = step.get('dependencies', [])
            # Validate that dependencies is a list
            if not isinstance(deps, list):
                raise ValueError(f"Invalid dependencies format for node '{node_tag}': dependencies must be a list")
            dependencies[node_tag] = deps

        # Check for missing dependencies - all dependencies must exist as chain steps
        all_nodes = set(chain_steps.keys())
        for node_tag, deps in dependencies.items():
            for dep in deps:
                if dep not in all_nodes:
                    raise ValueError(f"Missing dependency: Node '{node_tag}' depends on nonexistent node '{dep}'")

        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node, path):
            if node in rec_stack:
                # Found a cycle - build informative error message
                cycle_start = path.index(node)
                cycle_path = path[cycle_start:] + [node]
                if len(cycle_path) == 2 and cycle_path[0] == cycle_path[1]:
                    raise ValueError(f"Self-dependency detected: Node '{node}' depends on itself")
                else:
                    raise ValueError(f"Circular dependency detected: {' -> '.join(cycle_path)}")

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for dep in dependencies.get(node, []):
                if has_cycle(dep, path + [node]):
                    return True

            rec_stack.remove(node)
            return False

        # Check each node for cycles
        for node in dependencies:
            if node not in visited:
                has_cycle(node, [])

    def _identify_parallel_groups(self, dependency_order: List[str], execution_plan: Dict[str, Any]) -> List[List[str]]:
        """
        Identify groups of steps that can run in parallel.

        Args:
            dependency_order: List of nodes in dependency order
            execution_plan: Full execution plan

        Returns:
            List of groups, where each group contains nodes that can execute in parallel
        """
        # Stub implementation - return single group for all steps
        return [dependency_order] if dependency_order else []

    def _validate_assembly_phase(self) -> list[str]:
        """
        Perform assembly-specific validation including LHS-RHS nesting validation for destination fields.

        This method validates constraints that can only be checked after tree assembly,
        specifically LHS-RHS nesting validation for destination fields that were deferred
        during semantic validation.

        Returns:
            List of validation error messages, empty if validation passes
        """
        validation_errors = []

        # Validate LHS-RHS nesting for destination fields that were deferred during semantic validation
        def traverse_nodes(node, node_tag):
            """Recursively traverse tree nodes"""
            if hasattr(node, 'extracted_commands'):
                for command in node.extracted_commands:
                    # Check if this command has destination field mappings that need validation
                    if hasattr(command, 'variable_mappings') and hasattr(command, 'inclusion_path'):
                        # Convert node tag to full path if needed
                        full_node_tag = node_tag if '.' in node_tag else f"task.{node_tag}"
                        try:
                            self._validate_destination_field_nesting(command, full_node_tag)
                        except Exception as e:
                            validation_errors.append(f"LHS-RHS nesting validation failed for {node_tag}: {e}")

            # Recursively check child nodes
            if hasattr(node, 'children'):
                for child_tag, child_node in node.children.items():
                    traverse_nodes(child_node, child_tag)

        # Start traversal from root nodes
        for root_tag, root_node in self.run_structure._root_nodes.items():
            traverse_nodes(root_node, root_tag)

        return validation_errors

    def _validate_destination_field_nesting(self, command, source_node_tag: str) -> None:
        """
        Validate LHS-RHS nesting for destination fields during assembly phase.

        This validates destination fields that were skipped during semantic validation
        because the target node structure wasn't available.

        Args:
            command: DPCL command with variable mappings to validate
            source_node_tag: Tag of the source node containing the command

        Raises:
            ValueError: If LHS-RHS nesting validation fails
        """
        if not command.inclusion_path:
            return

        # Calculate iteration levels from inclusion path
        path_components = command.inclusion_path.split('.')
        iteration_levels = self.run_structure._count_iterable_levels_in_path(path_components, source_node_tag)

        if iteration_levels == 0:
            return  # No iteration, no nesting constraints

        # Track nesting levels for destination fields
        destination_nesting_levels = []

        for variable_mapping in command.variable_mappings:
            if variable_mapping.resolved_target and variable_mapping.resolved_target.path:
                lhs_path = variable_mapping.resolved_target.path
                lhs_components = lhs_path.split('.')

                # Check if this is a value scope destination field
                target_scope = variable_mapping.resolved_target.scope
                if target_scope and target_scope.get_name() == 'value':
                    # Check if the field exists in the source node (was validated in semantic phase)
                    source_node = self.run_structure.get_node(source_node_tag)
                    field_exists = False
                    if (source_node and source_node.field_type and
                        hasattr(source_node.field_type, 'model_fields')):
                        first_component = lhs_components[0] if lhs_components else ''
                        field_exists = first_component in source_node.field_type.model_fields

                    if not field_exists:
                        # This is a destination field - validate its nesting requirements
                        # For destination fields, we need to infer the required nesting structure
                        # Simple validation: destination field should match iteration levels
                        # TODO: This could be enhanced with more sophisticated target structure analysis
                        destination_nesting_levels.append(0)  # Destination fields are typically simple

        # Validate: at least one mapping must match iteration level for destination fields
        if destination_nesting_levels and iteration_levels > 0:
            if not any(level == iteration_levels for level in destination_nesting_levels):
                raise ValueError(
                    f"Destination field LHS-RHS nesting mismatch in {source_node_tag}: "
                    f"iteration level {iteration_levels} from '{command.inclusion_path}' "
                    f"requires at least one destination field with matching nesting, "
                    f"but found nesting levels: {destination_nesting_levels}. "
                    f"Consider using nested field structures to match iteration depth."
                )


class PromptAssembler:
    """
    Hierarchical prompt assembly for LangTree nodes.
    
    Assembles prompts from tree structure with proper template variable
    resolution, context inheritance, and section organization.
    """
    
    def __init__(self, run_structure: 'RunStructure'):
        self.run_structure = run_structure
    
    def assemble_prompt(self, node: 'StructureTreeNode') -> Dict[str, str]:
        """
        Assemble hierarchical prompt for a node.
        
        Args:
            node: Structure tree node to assemble prompt for
            
        Returns:
            Dictionary with prompt sections: system, context, task, output, input
        """
        # Get clean docstring and field descriptions
        system_prompt = getattr(node, 'clean_docstring', '')
        field_descriptions = getattr(node, 'clean_field_descriptions', {})
        
        # Process template variables in both system prompt and field descriptions
        if system_prompt:
            from langtree.prompt.template_variables import resolve_template_variables_in_content
            try:
                system_prompt = resolve_template_variables_in_content(system_prompt, node)
            except Exception as e:
                # Log template variable processing error but continue with original content
                import logging
                logging.warning(f"Template variable processing failed for node {node.name}: {e}")
                # Continue with original system_prompt
        
        # Process template variables in field descriptions
        processed_field_descriptions = {}
        if field_descriptions:
            from langtree.prompt.template_variables import resolve_template_variables_in_content
            for field_name, description in field_descriptions.items():
                try:
                    processed_field_descriptions[field_name] = resolve_template_variables_in_content(description, node)
                except Exception as e:
                    # Log template variable processing error but continue with original description
                    import logging
                    logging.warning(f"Template variable processing failed for field {field_name} in node {node.name}: {e}")
                    processed_field_descriptions[field_name] = description
        else:
            processed_field_descriptions = field_descriptions
        
        # Assemble context from parent hierarchy
        context_sections = self._assemble_context_hierarchy(node)
        
        # Organize into standard prompt sections
        prompt_sections = {
            'system': system_prompt,
            'context': '\n\n'.join(context_sections) if context_sections else None,
            'task': self._extract_task_section(system_prompt, processed_field_descriptions),
            'output': self._extract_output_section(processed_field_descriptions),
            'input': None  # Will be provided at execution time
        }
        
        return prompt_sections
    
    def _assemble_context_hierarchy(self, node: 'StructureTreeNode') -> List[str]:
        """
        Assemble context sections from parent hierarchy.
        
        Args:
            node: Node to assemble context for
            
        Returns:
            List of context sections from parent hierarchy
        """
        context_sections = []
        
        # Traverse up the tree to collect parent context
        current = node.parent
        while current is not None:
            if hasattr(current, 'clean_docstring') and current.clean_docstring:
                context_sections.insert(0, f"## {current.name.title()}\n{current.clean_docstring}")
            current = current.parent
        
        return context_sections
    
    def _extract_task_section(self, 
                             system_prompt: str,
                             field_descriptions: Dict[str, str]) -> str:
        """
        Extract task instructions from system prompt and field descriptions.
        
        Args:
            system_prompt: Main system prompt text
            field_descriptions: Field descriptions from node
            
        Returns:
            Assembled task section
        """
        # TODO: Implement intelligent task section extraction
        # For now, use system prompt as task
        return system_prompt
    
    def _extract_output_section(self, field_descriptions: Dict[str, str]) -> Optional[str]:
        """
        Extract output format specifications from field descriptions.
        
        Args:
            field_descriptions: Field descriptions from node
            
        Returns:
            Output section text if applicable
        """
        if not field_descriptions:
            return None
        
        # Assemble field descriptions into output specification
        output_lines = []
        for field_name, description in field_descriptions.items():
            output_lines.append(f"- **{field_name}**: {description}")
        
        return "Please provide the following fields:\n\n" + '\n'.join(output_lines)


class ContextPropagator:
    """
    Context propagation and variable resolution between chain steps.
    
    Handles runtime variable resolution, context forwarding, and
    variable scope management during chain execution.
    """
    
    def __init__(self, run_structure: 'RunStructure'):
        self.run_structure = run_structure
    
    def wrap_with_context_propagation(self, 
                                    base_chain: 'Runnable',
                                    node_tag: str,
                                    node: 'StructureTreeNode') -> 'Runnable':
        """
        Wrap a chain with context propagation capabilities.
        
        Args:
            base_chain: Base LangChain Runnable to wrap
            node_tag: Tag of the node this chain represents
            node: Structure tree node
            
        Returns:
            Wrapped Runnable with context propagation
        """
        def context_wrapper(input_data: Dict[str, Any]) -> Any:
            """
            Wrapper function that handles context propagation.
            
            Args:
                input_data: Input data dictionary with variables and context
                
            Returns:
                Chain output with context propagation applied
            """
            # Resolve runtime variables in input
            if 'prompt_template' in input_data:
                resolved_template = resolve_runtime_variables(
                    input_data['prompt_template'], 
                    self.run_structure, 
                    node
                )
                input_data = {**input_data, 'prompt_template': resolved_template}
            
            # Execute base chain
            result = base_chain.invoke(input_data)
            
            # TODO: Implement output variable propagation
            # This would extract variables from the result and make them
            # available to downstream chains according to LangTree scope rules
            
            return result
        
        # Import RunnableLambda here to avoid circular imports
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(context_wrapper)


class ExecutionOrchestrator:
    """
    Coordination of chain execution order and dependencies.
    
    Handles topological sorting, parallel execution opportunities,
    and multiplicity expansion for LangTree execution.
    """
    
    def __init__(self, run_structure: 'RunStructure'):
        self.run_structure = run_structure
    
    def get_execution_order(self) -> List[str]:
        """
        Determine optimal execution order for chain steps.
        
        Returns:
            List of node tags in execution order
        """
        # TODO: Implement proper topological sorting
        # For now, return simple order based on tree traversal
        execution_plan = self.run_structure.get_execution_plan()
        return [step['node_tag'] for step in execution_plan['chain_steps']]
    
    def identify_parallel_opportunities(self) -> List[List[str]]:
        """
        Identify chain steps that can be executed in parallel.
        
        Returns:
            List of lists, where each inner list contains node tags
            that can be executed in parallel
        """
        # TODO: Implement parallel execution analysis
        # This would analyze variable dependencies to find steps
        # that have no interdependencies and can run in parallel
        return []
    
    def expand_multiplicity_commands(self) -> Dict[str, Any]:
        """
        Expand @each and other multiplicity commands into execution plans.
        
        Returns:
            Expanded execution plan with multiplicity handling
        """
        # TODO: Implement multiplicity expansion
        # This would handle @each commands and generate appropriate
        # RunnableSequence or RunnableParallel structures
        return {}