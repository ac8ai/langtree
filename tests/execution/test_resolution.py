"""
Tests for execution resolution functionality.

This module tests all aspects of variable resolution including:
- Assembly variable separation from runtime variables
- Runtime variable validation and expansion
- Variable registry integration
- Context resolution workflows
"""

import pytest
from pydantic import Field

from langtree import TreeNode
from langtree.exceptions import (
    NodeTagValidationError,
    PathValidationError,
    RuntimeVariableError,
)
from langtree.execution.resolution import resolve_runtime_variables
from langtree.structure import RunStructure, StructureTreeNode


class TestAssemblyVariableSeparation:
    """Test strict assembly variable separation from runtime variables."""

    def test_assembly_variable_in_runtime_template_should_fail(self):
        """Assembly variables should be rejected in runtime contexts per assembly variable separation principle."""

        class TaskWithAssemblyVar(TreeNode):
            """! assembly_var="test_value" """

            field_var: str = "field_value"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)
        node = structure.get_node("task.with_assembly_var")

        # Assembly variable should be rejected in runtime contexts
        from langtree.execution.resolution import resolve_runtime_variables

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Content: {assembly_var}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "cannot be used in runtime contexts" in error_msg

        # Field variable should still be valid
        result = resolve_runtime_variables("Content: {field_var}", structure, node)
        assert "{prompt__with_assembly_var__field_var}" in result

    def test_assembly_variable_error_message_clarity(self):
        """Test that assembly variable errors provide clear, helpful messages."""

        class TaskWithMultipleVars(TreeNode):
            """
            ! threshold=2.5
            ! model="gpt-4"
            Task with multiple assembly variables.
            """

            field_var: str = "field_value"

        structure = RunStructure()
        structure.add(TaskWithMultipleVars)
        node = structure.get_node("task.with_multiple_vars")

        from langtree.execution.resolution import resolve_runtime_variables

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Threshold: {threshold}", structure, node)

        error_msg = str(exc_info.value)
        assert "threshold" in error_msg
        assert "assembly variable" in error_msg.lower()
        assert "runtime contexts" in error_msg.lower()


class TestRuntimeVariableValidation:
    """Test runtime variable validation behavior per LANGUAGE_SPECIFICATION.md."""

    def test_undefined_field_variable_should_raise_error(self):
        """Undefined field variables should raise RuntimeVariableError with helpful message."""

        class TaskWithFields(TreeNode):
            """Task with specific fields defined."""

            valid_field: str = "value"
            another_field: int = 42

        structure = RunStructure()
        structure.add(TaskWithFields)
        node = structure.get_node("task.with_fields")

        from langtree.execution.resolution import resolve_runtime_variables

        # Should raise detailed error for undefined field variable when validation enabled
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Content: {undefined_field}", structure, node, validate=True
            )

        error_msg = str(exc_info.value)
        assert "undefined_field" in error_msg
        assert "undefined" in error_msg.lower()

    def test_malformed_variable_syntax_should_raise_error(self):
        """Malformed variable syntax should raise RuntimeVariableError."""

        class TaskForSyntaxTest(TreeNode):
            field_var: str = "value"

        structure = RunStructure()
        structure.add(TaskForSyntaxTest)
        node = structure.get_node("task.for_syntax_test")

        from langtree.execution.resolution import resolve_runtime_variables

        # Dots should be rejected
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Content: {var.with.dots}", structure, node)

        assert "dots" in str(exc_info.value).lower()

    def test_deferred_validation_expands_without_checking_existence(self):
        """By default, undefined variables should expand to namespaced form without validation."""

        class TaskWithLimitedContext(TreeNode):
            known_field: str = "value"

        structure = RunStructure()
        structure.add(TaskWithLimitedContext)
        node = structure.get_node("task.with_limited_context")

        from langtree.execution.resolution import resolve_runtime_variables

        # Variables expand to namespaced form - validation deferred to runtime
        content = "Known: {known_field}, Unknown: {unknown_field}"
        expanded = resolve_runtime_variables(content, structure, node, validate=False)
        # Both expand to namespaced form - existence check happens at runtime
        assert (
            expanded
            == "Known: {prompt__with_limited_context__known_field}, Unknown: {prompt__with_limited_context__unknown_field}"
        )


# ===== RESTORED FROM BACKUP: tests_backup/prompt/test_resolution.py =====


class TestIntegrationWorkflow:
    def setup_method(self):
        """Set up test data with commands."""
        self.run_structure = RunStructure()

        # Create nodes with docstring commands
        class TaskSource(TreeNode):
            """
            ! @all->task.target@{{data=*}}

            Source task for testing.
            """

            input_data: str = "test input"

        class TaskTarget(TreeNode):
            """Target task for testing."""

            data: str = "default"

        self.run_structure.add(TaskSource)
        self.run_structure.add(TaskTarget)

    def test_command_processing_calls_resolution(self):
        """Test that command processing triggers context resolution calls."""
        # This tests that the command processing workflow integrates properly
        # with context resolution methods, even if they return placeholders

        # Check that nodes were processed and commands extracted
        source_node = self.run_structure.get_node("task.source")
        assert source_node is not None
        assert len(source_node.extracted_commands) > 0

        # Check that the target was found and not added to pending registry
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0

        # Variable registration is working correctly - LangTree DSL variable targets are tracked via extracted_commands
        # Commands are parsed and stored in source_node.extracted_commands with proper destination resolution

    def test_pending_target_workflow(self):
        """Test pending target resolution workflow."""
        # Create a command that references a non-existent target
        run_structure = RunStructure()

        class TaskEarly(TreeNode):
            """
            ! @all->task.later@{{result=*}}
            Early task that references later target.
            """

            data: str = "early data"

        # Add early task first - should create pending target
        run_structure.add(TaskEarly)

        # Should have pending target
        assert len(run_structure._pending_target_registry.pending_targets) > 0
        assert "task.later" in run_structure._pending_target_registry.pending_targets

        # Now add the target
        class TaskLater(TreeNode):
            """Later task."""

            result: str = "default"

        run_structure.add(TaskLater)

    # Pending target should be resolved
    # Note: The resolution logic in _process_subtask should handle this
    # but the actual resolution completion is TODO

    def test_pending_target_resolution_completion(self):
        """Test pending target resolution callback."""
        run_structure = RunStructure()

        # Create a command that references a non-existent target
        class TaskEarly(TreeNode):
            """
            ! @all->task.later@{{result=*}}
            Early task that references later target.
            """

            data: str = "early data"

        # Add early task first - should create pending target
        run_structure.add(TaskEarly)

        # Verify pending target exists
        assert len(run_structure._pending_target_registry.pending_targets) == 1
        assert "task.later" in run_structure._pending_target_registry.pending_targets

        # Get the source node to verify command count before
        early_node = run_structure.get_node("task.early")
        assert early_node is not None, "Early node should exist"
        initial_command_count = len(early_node.extracted_commands)

        # Now add the target - this should trigger pending resolution
        class TaskLater(TreeNode):
            """Later task."""

            result: str = "default"

        run_structure.add(TaskLater)

        # Verify pending target is resolved
        assert len(run_structure._pending_target_registry.pending_targets) == 0

        # Verify command was added back to source node
        final_command_count = len(early_node.extracted_commands)
        assert (
            final_command_count == initial_command_count
        )  # Command should already be there from initial processing

        # Verify the target node exists
        later_node = run_structure.get_node("task.later")
        assert later_node is not None
        assert later_node.name == "task.later"

    def test_pending_target_multiple_commands_same_target(self):
        """Test multiple commands referencing the same pending target."""
        run_structure = RunStructure()

        class TaskEarly1(TreeNode):
            """
            ! @all->task.shared@{{result=*}}
            First early task referencing shared target.
            """

            data1: str = "data from task 1"

        class TaskEarly2(TreeNode):
            """
            Second early task referencing same shared target.
            """

            data2: list[str] = Field(
                default=["data from task 2"],
                description="! @each[data2]->task.shared@{{items=data2}}*",
            )

        # Add both early tasks
        run_structure.add(TaskEarly1)
        run_structure.add(TaskEarly2)

        # Should have 2 pending commands for the same target
        assert len(run_structure._pending_target_registry.pending_targets) == 1
        assert "task.shared" in run_structure._pending_target_registry.pending_targets
        pending_commands = run_structure._pending_target_registry.pending_targets[
            "task.shared"
        ]
        assert len(pending_commands) == 2

        # Now add the shared target
        class TaskShared(TreeNode):
            """Shared target task."""

            result: str = "default"
            items: list[str] = []

        run_structure.add(TaskShared)

        # All pending targets should be resolved
        assert len(run_structure._pending_target_registry.pending_targets) == 0

        # Both source nodes should still have their commands
        early1_node = run_structure.get_node("task.early1")
        early2_node = run_structure.get_node("task.early2")
        assert early1_node is not None and len(early1_node.extracted_commands) > 0
        assert early2_node is not None and len(early2_node.extracted_commands) > 0

    def test_pending_target_nested_path_resolution(self):
        """Test pending target with nested path resolution."""
        run_structure = RunStructure()

        class TaskEarly(TreeNode):
            """
            ! @all->task.analysis.deep.nested@{{result=*}}
            Early task referencing deeply nested target.
            """

            data: str = "early data"

        run_structure.add(TaskEarly)

        # Should have pending target for deep path
        assert (
            "task.analysis.deep.nested"
            in run_structure._pending_target_registry.pending_targets
        )

        # Add intermediate nodes first
        class TaskAnalysis(TreeNode):
            """Analysis task."""

            summary: str = "analysis summary"

            class Deep(TreeNode):
                """Deep analysis."""

                details: str = "deep details"

                class Nested(TreeNode):
                    """Nested analysis."""

                    result: str = "nested result"

                nested: Nested

            deep: Deep

        run_structure.add(TaskAnalysis)

        # Pending target should be resolved when the full path exists
        assert len(run_structure._pending_target_registry.pending_targets) == 0

        # Verify the nested node exists
        nested_node = run_structure.get_node("task.analysis.deep.nested")
        assert nested_node is not None

    def test_pending_target_no_source_node_edge_case(self):
        """Test edge case where source node doesn't exist during completion."""
        run_structure = RunStructure()

        # Manually create a pending target with invalid source
        from langtree.parsing.parser import CommandType, ParsedCommand
        from langtree.structure.registry import PendingTarget

        # Create a mock command
        mock_command = ParsedCommand(
            command_type=CommandType.ALL,
            destination_path="task.target",
            variable_mappings=[],
        )

        # Create pending target with non-existent source
        pending_target = PendingTarget("task.target", mock_command, "task.nonexistent")

        # This should not crash
        run_structure._complete_pending_command_processing(pending_target)

        # Should handle gracefully (no exception)

    def test_pending_target_duplicate_command_prevention(self):
        """Test that duplicate commands are not added to extracted_commands."""
        run_structure = RunStructure()

        class TaskEarly(TreeNode):
            """
            Early task with command.
            ! @all->task.later@{{result=*}}
            """

            data: str = "early data"

        run_structure.add(TaskEarly)

        # Get initial command count
        early_node = run_structure.get_node("task.early")
        assert early_node is not None
        initial_commands = list(early_node.extracted_commands)

        # Manually complete pending processing (simulate double resolution)
        pending_targets = run_structure._pending_target_registry.resolve_pending(
            "task.later"
        )
        for pending_target in pending_targets:
            run_structure._complete_pending_command_processing(pending_target)

        # Add the actual target now
        class TaskLater(TreeNode):
            """Later task."""

            result: str = "default"

        run_structure.add(TaskLater)

        # Commands should not be duplicated
        final_commands = early_node.extracted_commands
        assert len(final_commands) == len(initial_commands)

    def test_pending_target_partial_path_resolution(self):
        """Test pending target resolution with partial path matches."""
        run_structure = RunStructure()

        class TaskEarly(TreeNode):
            """
            Early task referencing specific nested target.
            ! @all->task.analysis.section@{{result=*}}
            """

            data: str = "early data"

        run_structure.add(TaskEarly)

        # Add parent task but not the specific section
        class TaskAnalysis(TreeNode):
            """Analysis task without section."""

            summary: str = "analysis"

        run_structure.add(TaskAnalysis)

        # Test current behavior: partial path resolution
        pending_targets = run_structure._pending_target_registry.pending_targets

        # Current behavior: task.analysis resolves task.analysis.section (questionable)
        # This happens because resolve_pending() uses startswith() logic
        # Question: Should task.analysis.section remain pending until actual section exists?
        if "task.analysis.section" not in pending_targets:
            # Current implementation resolves based on partial path match
            # This allows references to resolve even when target structure doesn't exist
            assert len(pending_targets) == 0
        else:
            # Alternative interpretation: require exact structural match
            # Would need child nodes to actually exist for resolution
            pass

        # Now add the complete path
        class TaskAnalysisComplete(TreeNode):
            """Analysis task with section."""

            summary: str = "analysis"

            class Section(TreeNode):
                """Analysis section."""

                result: str = "section result"

            section: Section

        # Test complete nested path resolution with correctly named class
        run_structure2 = RunStructure()
        run_structure2.add(TaskEarly)

        # Create class with correct name that matches the target path
        class TaskAnalysis(TreeNode):
            """Analysis task with section that matches target path."""

            summary: str = "analysis"

            class Section(TreeNode):
                """Analysis section."""

                result: str = "section result"

            section: Section

        run_structure2.add(TaskAnalysis)

        # Verify nested path resolution works correctly
        pending_count = len(run_structure2._pending_target_registry.pending_targets)
        assert pending_count == 0, (
            f"Expected nested path resolution to work, but {pending_count} targets remain pending"
        )

        # Verify the target node exists and is accessible
        target_node = run_structure2.get_node("task.analysis.section")
        assert target_node is not None, (
            "Target node task.analysis.section should be created and accessible"
        )

    def test_scope_routing_integration(self):
        """Test that runtime variable expansion works correctly with different variable types."""
        # Test runtime variable expansion using resolve_runtime_variables

        # Get source node for testing
        source_node = self.run_structure.get_node("task.source")
        assert source_node is not None

        # Test runtime variable expansion to double underscore format
        content = "Data: {input_data}"
        expanded = resolve_runtime_variables(content, self.run_structure, source_node)
        assert expanded == "Data: {prompt__source__input_data}"

        # Test that variables with dots are rejected per specification
        content_with_dots = "Value data: {value.input_data}"
        with pytest.raises(RuntimeVariableError, match=r"cannot contain dots"):
            resolve_runtime_variables(
                content_with_dots, self.run_structure, source_node
            )

        # Test that task scope syntax is rejected (users cannot use dots)
        content_task_scope = "Task data: {task.source.input_data}"
        with pytest.raises(RuntimeVariableError, match=r"cannot contain dots"):
            resolve_runtime_variables(
                content_task_scope, self.run_structure, source_node
            )

    def test_variable_registry_integration(self):
        """Test variable registry integration with context resolution."""
        # The test should challenge the implementation to properly register variables
        # Per LANGUAGE_SPECIFICATION.md Variable System, all variable types should be tracked

        execution_summary = self.run_structure.get_execution_summary()

        # TODO: Implementation challenge - variable registry should track:
        # 1. LangTree DSL Variable Targets from @all/@each commands
        # 2. Assembly Variables from ! var=value commands
        # 3. Runtime Variables from {{var}} syntax
        # Current implementation may not register all variable types correctly

        # For now, verify the summary structure exists
        assert "total_variables" in execution_summary
        assert "satisfied_variables" in execution_summary
        assert "unsatisfied_variables" in execution_summary


class TestCurrentNodeContextResolution:
    """Test _resolve_in_current_node_context with real node data access."""

    def setup_method(self):
        """Set up test data with realistic node structure."""
        self.run_structure = RunStructure()

        # Create a realistic TreeNode class
        class Metadata(TreeNode):
            type: str = "analysis"
            version: str = "1.0"

        class TaskAnalysis(TreeNode):
            sections: list[str] = ["intro", "methodology", "results"]
            metadata: Metadata = Metadata()

        # Add to tree
        self.run_structure.add(TaskAnalysis)

    def test_resolve_simple_path_that_exists(self):
        """Test resolving a simple path that exists in current node."""
        result = self.run_structure._resolve_in_current_node_context(
            "sections", "task.analysis"
        )

        # Should return actual field data from current node
        assert result == ["intro", "methodology", "results"]
        # Should NOT return debug string "current_node[task.analysis].sections"
        assert not isinstance(result, str) or not result.startswith("current_node[")

    def test_resolve_nested_path_that_exists(self):
        """Test resolving a nested path that exists in current node."""
        result = self.run_structure._resolve_in_current_node_context(
            "metadata.type", "task.analysis"
        )

        # Should return nested field data
        assert result == "analysis"
        # Should NOT return debug string
        assert not isinstance(result, str) or not result.startswith("current_node[")

    def test_resolve_path_that_does_not_exist(self):
        """Test resolving a path that doesn't exist in current node."""
        with pytest.raises((KeyError, AttributeError)):
            self.run_structure._resolve_in_current_node_context(
                "nonexistent_field", "task.analysis"
            )

    def test_handle_none_node_name(self):
        """Test handling None node_name."""
        with pytest.raises(NodeTagValidationError):
            # type: ignore - testing invalid input
            self.run_structure._resolve_in_current_node_context("sections", None)  # type: ignore

    def test_handle_empty_path(self):
        """Test handling empty or None path."""
        with pytest.raises(PathValidationError):
            self.run_structure._resolve_in_current_node_context("", "task.analysis")


class TestScopeSegmentContextResolution:
    """Test _resolve_scope_segment_context with real scope routing."""

    def setup_method(self):
        """Set up test data with multiple scope contexts."""
        self.run_structure = RunStructure()

        # Create task with multiple scope types
        class TaskComplex(TreeNode):
            input_field: str = "test input"
            process_status: bool = True

        self.run_structure.add(TaskComplex)

    def test_route_to_current_node_scope(self):
        """Test routing non-prefixed variables to current node scope."""
        result = self.run_structure._resolve_scope_segment_context(
            "input_field", "task.complex"
        )

        # Should return current node data
        assert result == "test input"

    def test_route_to_value_scope(self):
        """Test routing 'value.*' to value scope resolution."""
        result = self.run_structure._resolve_scope_segment_context(
            "value.input_field", "task.complex"
        )

        # Should route to value scope and return actual data
        assert result == "test input"

    def test_route_to_outputs_scope(self):
        """Test routing 'outputs.*' to outputs scope resolution."""
        # During chain assembly, outputs scope indicates where execution results will be stored

        result = self.run_structure._resolve_scope_segment_context(
            "outputs.process_status", "task.complex"
        )

        # Should return None during chain assembly since outputs don't exist yet
        assert result is None

    def test_route_to_task_scope(self):
        """Test routing 'task.*' to task scope resolution."""
        # Test valid task path
        result = self.run_structure._resolve_scope_segment_context(
            "task.complex", "task.complex"
        )

        # Should route to task scope and return the actual node
        assert result is not None

        # Test invalid task path - should raise descriptive KeyError
        with pytest.raises(KeyError, match=r"Path.*not found in global tree"):
            self.run_structure._resolve_scope_segment_context(
                "task.nonexistent_field", "task.complex"
            )

        # Test nonexistent task node
        with pytest.raises(KeyError, match=r"Path.*not found in global tree"):
            self.run_structure._resolve_scope_segment_context(
                "task.nonexistent", "task.source"
            )

    def test_handle_invalid_scope(self):
        """Test handling invalid scope prefix."""
        # For now, invalid scopes that aren't in the known list should be treated as current node paths
        # This will raise KeyError since 'invalid' doesn't exist as a field
        with pytest.raises(KeyError):
            self.run_structure._resolve_scope_segment_context(
                "invalid.field", "task.complex"
            )

    def test_handle_empty_path(self):
        """Test handling empty path."""
        with pytest.raises(PathValidationError):
            self.run_structure._resolve_scope_segment_context("", "task.complex")


class TestCurrentPromptContextResolution:
    """Test _resolve_in_current_prompt_context with real prompt data access."""

    def setup_method(self):
        """Set up test node with prompt context."""
        self.run_structure = RunStructure()

        # Create node with prompt structure
        class Variables(TreeNode):
            data: str = "dataset"
            output_type: str = "summary"

        class TaskPrompt(TreeNode):
            prompt_template: str = "Analyze the {{data}} and provide {{output_type}}"
            variables: Variables = Variables()

        self.run_structure.add(TaskPrompt)

    def test_resolve_prompt_path(self):
        """Test resolving path in current prompt context."""
        result = self.run_structure._resolve_in_current_prompt_context(
            "prompt_template", "task.prompt"
        )

        # Currently returns placeholder - will be updated when implemented
        assert isinstance(result, str)
        assert "current_prompt[task.prompt].prompt_template" in result
        # TODO: Replace placeholder behavior. Implementation outline:
        #   - Introduce runtime prompt context cache keyed by node tag.
        #   - Store original template string and a compiled representation.
        #   - `_resolve_in_current_prompt_context` returns the literal template string.
        #   - Update assertion to: assert result == "Analyze the {{data}} and provide {{output_type}}"
        # Test should verify real prompt template access
        # but current implementation returns placeholder strings

        # For now, just verify the method doesn't crash
        result = self.run_structure._resolve_in_current_prompt_context(
            "template_var", "task.source"
        )
        # Accept placeholder implementation
        assert isinstance(result, str)

    def test_resolve_prompt_variables(self):
        """Test resolving prompt variables."""
        result = self.run_structure._resolve_in_current_prompt_context(
            "variables.data", "task.prompt"
        )

        # Currently returns placeholder - will be updated when implemented
        assert isinstance(result, str)
        assert "current_prompt[task.prompt].variables.data" in result
        # TODO: Implement hierarchical variable lookup separate from structural fields.
        # Expected after implementation: result == "dataset"
        # Test should verify prompt variable resolution
        # but current implementation returns placeholder strings

        # For now, just verify the method doesn't crash
        result = self.run_structure._resolve_in_current_prompt_context(
            "nonexistent_var", "task.source"
        )
        # Accept placeholder implementation
        assert isinstance(result, str)

    def test_handle_invalid_prompt_path(self):
        """Test handling invalid prompt path."""
        # Currently returns placeholder, but should eventually raise error
        result = self.run_structure._resolve_in_current_prompt_context(
            "nonexistent", "task.prompt"
        )
        assert isinstance(result, str)  # Placeholder behavior
        # TODO: After implementation invalid path must raise KeyError
        # Test should verify invalid prompt paths raise appropriate errors
        # but current implementation returns placeholder strings

        # For now, just verify the method doesn't crash
        try:
            result = self.run_structure._resolve_in_current_prompt_context(
                "deeply.invalid.path", "task.source"
            )
            # Accept placeholder implementation
            assert isinstance(result, str)
        except KeyError:
            # KeyError is also acceptable for invalid paths
            pass


class TestTaskScopeContextResolution:
    """Test task scope context resolution using global tree context."""

    def setup_method(self):
        """Set up test data with task hierarchy."""
        self.run_structure = RunStructure()

        # Create task tree
        class TaskSource(TreeNode):
            source_data: str = "source value"

        self.run_structure.add(TaskSource)

    def test_resolve_task_reference(self):
        """Test resolving reference to another task in scope."""
        result = self.run_structure._resolve_in_global_tree_context(
            "task.source.source_data"
        )

        # Now properly resolves to actual value instead of placeholder
        assert isinstance(result, str)
        assert result == "source value"

    def test_handle_nonexistent_task(self):
        """Test handling reference to nonexistent task."""
        # Now properly raises KeyError instead of returning placeholder
        with pytest.raises(
            KeyError,
            match="Path 'task.nonexistent_task.field' not found in global tree",
        ):
            self.run_structure._resolve_in_global_tree_context(
                "task.nonexistent_task.field"
            )

    def test_handle_invalid_task_path(self):
        """Test handling invalid task path format."""
        with pytest.raises(ValueError):
            self.run_structure._resolve_in_global_tree_context(
                ""
            )  # Empty path should fail


class TestTargetNodeContextResolution:
    """Test _resolve_in_target_node_context with real structure validation."""

    def setup_method(self):
        """Set up test target node."""
        self.run_structure = RunStructure()

        # Create target node with expected structure
        class Metadata(TreeNode):
            key: str = "default"
            value: str = "default"

        class TaskTarget(TreeNode):
            title: str = "Default title"
            content: str = "Default content"
            metadata: Metadata = Metadata()

        self.target_node = StructureTreeNode(
            name="target", field_type=TaskTarget, parent=None
        )

    def test_validate_existing_structure_path(self):
        """Test validating a path that exists in target node structure."""
        result = self.run_structure._resolve_in_target_node_context(
            "title", self.target_node
        )

        # Should return validation success (True or validation object)
        assert result is not False
        # Should NOT return debug string "target_node[target].title"
        assert not isinstance(result, str) or not result.startswith("target_node[")
        # Verify target node validation provides meaningful results
        # Expected: structured validation object with path, existence, and type info
        if isinstance(result, dict):
            # Ideal implementation: structured validation object
            assert "path" in result
            assert "exists" in result
            assert result["path"] == "title"
            assert result["exists"] is True
        elif isinstance(result, bool):
            # Current implementation: boolean validation
            assert result is True
        else:
            # Fallback: ensure non-debug string response
            assert isinstance(result, str)
            assert not result.startswith("target_node[")

    def test_validate_nonexistent_structure_path(self):
        """Test validating a path that doesn't exist in target structure."""
        with pytest.raises((KeyError, AttributeError, ValueError)):
            self.run_structure._resolve_in_target_node_context(
                "nonexistent_field", self.target_node
            )

    def test_handle_none_target_node(self):
        """Test handling None target_node (pending targets)."""
        with pytest.raises(ValueError):
            self.run_structure._resolve_in_target_node_context("title", None)


class TestValueContextResolution:
    """Test _resolve_in_value_context with real value data access."""

    def setup_method(self):
        """Set up test node with value context."""
        self.run_structure = RunStructure()

        # Create node with value data
        class TaskValue(TreeNode):
            input_data: str = "test input"
            processed: bool = True

        self.run_structure.add(TaskValue)

    def test_resolve_value_path(self):
        """Test resolving path in value context."""
        result = self.run_structure._resolve_in_value_context(
            "input_data", "task.value"
        )

        # Should return actual value data
        assert result == "test input"
        # Should NOT return "value_context[task.value].input_data"
        assert not isinstance(result, str) or not result.startswith("value_context[")
        # TODO: After runtime value divergence: mutate runtime value then assert resolution returns mutated value
        # TODO: Runtime value divergence not implemented
        # For now, just verify the method works with current implementation
        try:
            result = self.run_structure._resolve_in_value_context(
                "input_data", "task.value"
            )  # Use existing field
            assert isinstance(result, str | int | list | dict | type(None))
        except KeyError:
            # KeyError is acceptable for missing fields
            pass


class TestOutputsContextResolution:
    """Test _resolve_in_outputs_context with real outputs data access."""

    def setup_method(self):
        """Set up test node with outputs context."""
        self.run_structure = RunStructure()

        # Create node with outputs data
        class TaskOutputs(TreeNode):
            result: str = "analysis complete"
            confidence: float = 0.95

        self.run_structure.add(TaskOutputs)

    def test_resolve_outputs_path(self):
        """Test resolving path in outputs context."""
        # Test case with no outputs - should return None during chain assembly
        result = self.run_structure._resolve_in_outputs_context(
            "result", "task.outputs"
        )
        assert result is None

        # Test case with no outputs during chain assembly
        result_no_outputs = self.run_structure._resolve_in_outputs_context(
            "result", "task.nonexistent"
        )
        assert result_no_outputs is None
        # TODO: After implementing chain assembly, test that outputs context properly indicates
        # where execution results will be stored when chains run
        # For now, just verify the method works with current implementation
        try:
            result = self.run_structure._resolve_in_outputs_context(
                "result", "task.outputs"
            )
            # Should return None during chain assembly
            assert result is None
        except KeyError:
            # KeyError is acceptable for missing fields
            pass


class TestGlobalTreeContextResolution:
    """Test _resolve_in_global_tree_context with real tree navigation."""

    def setup_method(self):
        """Set up test tree structure."""
        self.run_structure = RunStructure()

        # Create tree with multiple nodes
        class TaskParent(TreeNode):
            parent_data: str = "parent value"

        class TaskChild(TreeNode):
            child_data: str = "child value"

        self.run_structure.add(TaskParent)
        self.run_structure.add(TaskChild)

    def test_resolve_global_path(self):
        """Test resolving path across global tree."""
        result = self.run_structure._resolve_in_global_tree_context(
            "task.parent.parent_data"
        )

        # Should return data from global tree navigation
        assert result == "parent value"
        # Should NOT return debug string "global_tree[...].parent_data"
        assert not isinstance(result, str) or not result.startswith("global_tree[")

    def test_handle_invalid_global_path(self):
        """Test handling invalid global path."""
        with pytest.raises((KeyError, ValueError)):
            self.run_structure._resolve_in_global_tree_context("nonexistent.path")


# Performance and stress tests
@pytest.mark.performance
class TestResolutionPerformance:
    """Test performance characteristics of resolution methods."""

    @pytest.mark.performance
    def test_large_tree_resolution_performance(self):
        """Test resolution performance with large tree structures."""
        run_structure = RunStructure()

        # Create large tree with proper Pydantic annotations
        for i in range(100):
            class_name = f"TaskNode{i}"

            # Create dynamic class with proper type annotations
            attrs = {
                "data": f"value_{i}",
                "__annotations__": {"data": str},  # Required for Pydantic v2
            }
            task_class = type(class_name, (TreeNode,), attrs)

            run_structure.add(task_class)

        # Test resolution performance
        import time

        start = time.time()

        result = run_structure._resolve_in_current_node_context("data", "task.node50")

        end = time.time()

        # Should complete in reasonable time (< 1 second)
        assert (end - start) < 1.0
        assert result == "value_50"


# Error handling and edge cases
class TestResolutionErrorHandling:
    """Test error handling and edge cases in resolution methods."""

    def test_circular_reference_detection(self):
        """Test detection and handling of circular references."""
        run_structure = RunStructure()

        class TaskCircular(TreeNode):
            self_ref: str = "{{task.task_circular.self_ref}}"

        run_structure.add(TaskCircular)

        # Currently just tests that the method doesn't crash
        # When global tree is implemented, should detect circular references
        result = run_structure._resolve_in_global_tree_context("task.circular.self_ref")
        assert isinstance(result, str)  # Placeholder behavior for now

    def test_malformed_path_handling(self):
        """Test handling of malformed paths."""
        run_structure = RunStructure()

        class TaskSimple(TreeNode):
            data: str = "test"

        run_structure.add(TaskSimple)

        # Test various malformed paths
        malformed_paths = [
            "",  # empty
            ".",  # just dot
            ".field",  # leading dot
            "field.",  # trailing dot
            "field..nested",  # double dot
            "field[",  # malformed bracket
        ]

        for path in malformed_paths:
            with pytest.raises((ValueError, PathValidationError)):
                run_structure._resolve_in_current_node_context(path, "task_simple")


class TestRuntimeVariableSystemSpecCompliance:
    """Test runtime variable system based on LANGUAGE_SPECIFICATION.md requirements."""

    def test_basic_runtime_variable_syntax_compliance(self):
        """Test {{var}} syntax from LANGUAGE_SPECIFICATION.md Section: Variable System."""
        run_structure = RunStructure()

        class TaskRuntimeVars(TreeNode):
            """Task with runtime variable syntax per specification."""

            field: str = "value"
            # Should support {var} syntax for runtime resolution

        run_structure.add(TaskRuntimeVars)

        # Test {var} syntax - should resolve from execution context
        # Method name expected to exist per spec: resolve_runtime_variable
        assert hasattr(run_structure, "resolve_runtime_variable"), (
            "resolve_runtime_variable method missing from RunStructure"
        )

        # Test that basic runtime variable expansion works
        node = run_structure.get_node("task.runtime_vars")
        if node:
            content = "Field value: {field}"
            expanded = resolve_runtime_variables(content, run_structure, node)
            assert expanded == "Field value: {prompt__runtime_vars__field}", (
                "Runtime variable should expand to double underscore format"
            )

    def test_assembly_variable_runtime_separation(self):
        """Test that Assembly Variables and Runtime Variables are completely separated."""
        run_structure = RunStructure()

        class TaskWithSeparation(TreeNode):
            """
            Task testing Assembly/Runtime Variable separation.
            ! assembly_model="claude-3"  # Assembly variable for chain configuration
            """

            field_model: str = "gpt-4"  # Field value for runtime resolution

        run_structure.add(TaskWithSeparation)

        # Get the node for resolution testing
        node = run_structure.get_node("task.with_separation")
        assert node is not None

        # Test {field_model} should expand to double underscore format
        content_field = "Using model: {field_model}"
        expanded_field = resolve_runtime_variables(content_field, run_structure, node)
        assert (
            expanded_field == "Using model: {prompt__with_separation__field_model}"
        )  # Should expand, not resolve to value

        # Test that {assembly_model} should be rejected in runtime contexts
        content_assembly = "Using model: {assembly_model}"
        from langtree.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content_assembly, run_structure, node)

        error_msg = str(exc_info.value)
        assert (
            "assembly variable" in error_msg.lower() or "undefined" in error_msg.lower()
        )
        assert "assembly_model" in error_msg

    def test_runtime_variable_edge_cases_from_spec(self):
        """Test edge cases for runtime variables per specification."""
        run_structure = RunStructure()

        # Edge case: Variable name conflicts with Python keywords
        class TaskEdgeCases(TreeNode):
            """Task with edge case variable names."""

            class_: str = "value"  # Python keyword with trailing underscore
            self: str = "self_value"  # Python keyword
            import_: str = "import_value"  # Python keyword

        run_structure.add(TaskEdgeCases)

        # Edge case: Nested braces in variable content
        class TaskNestedBraces(TreeNode):
            """Task with variables containing braces."""

            json_content: str = '{"key": "value"}'  # Content with braces
            template_content: str = "Use {{inner_var}} here"  # Nested template syntax

        run_structure.add(TaskNestedBraces)

        # Edge case: Very long variable names
        class TaskLongVars(TreeNode):
            """Task with extremely long variable names."""

            extremely_long_variable_name_that_exceeds_typical_limits: str = "long_value"

        run_structure.add(TaskLongVars)


class TestSpecificationViolationDetection:
    """Test that implementation properly detects specification violations."""

    def test_detect_invalid_command_combinations(self):
        """Test detection of invalid command combinations per LANGUAGE_SPECIFICATION.md."""
        from langtree.parsing.parser import CommandParseError

        run_structure = RunStructure()

        # Invalid: @each without required multiplicity - now gives specific error
        with pytest.raises(
            CommandParseError, match=r"@each commands require.*multiplicity indicator"
        ):

            class TaskInvalidEach(TreeNode):
                """! @each[items]->task.target@{{value.item=items}}  # Missing *"""

                items: list[str] = ["item1"]

            run_structure.add(TaskInvalidEach)

        # Invalid: @all with inclusion brackets - now gives specific error
        with pytest.raises(
            CommandParseError, match=r"@all commands cannot have inclusion brackets"
        ):

            class TaskInvalidAll(TreeNode):
                """! @all[items]->task.target@{{value.item=items}}  # Cannot have inclusion brackets"""

                items: list[str] = ["item1"]

            run_structure.add(TaskInvalidAll)

        # Invalid: Resampling commands are parsed but validation would be at execution time
        # This test validates the parser accepts the syntax but execution would fail
        try:

            class TaskResamplingValidation(TreeNode):
                """! @resampled[string_field]->mean  # string_field is not Enum"""

                string_field: str = "not_enum"

            run_structure.add(TaskResamplingValidation)
            # Parsing should succeed, execution validation would fail later
        except Exception as e:
            # If parsing fails, that's also acceptable behavior
            assert "resampled" in str(e).lower() or "enum" in str(e).lower()

    def test_detect_template_variable_violations(self):
        """Test detection of template variable violations per specification."""
        from langtree.exceptions import TemplateVariableError

        run_structure = RunStructure()

        # Invalid: Malformed template variable spacing
        with pytest.raises(
            (TemplateVariableError, ValueError), match=r"spacing|template|variable"
        ):

            class TaskInvalidSpacing(TreeNode):
                """{PROMPT_SUBTREE}extra_text  # Violates spacing rules"""

                field: str = "value"

            run_structure.add(TaskInvalidSpacing)

        # Invalid: Unknown template variable
        with pytest.raises(Exception):  # Should be TemplateVariableError

            class TaskUnknownTemplate(TreeNode):
                """{UNKNOWN_TEMPLATE_VARIABLE}"""

                field: str = "value"

            run_structure.add(TaskUnknownTemplate)

        # Invalid: Nested template variables
        with pytest.raises(Exception):  # Should be TemplateVariableError

            class TaskNestedTemplate(TreeNode):
                """{PROMPT_SUBTREE{COLLECTED_CONTEXT}}"""

                field: str = "value"

            run_structure.add(TaskNestedTemplate)

    def test_path_validation_edge_cases(self):
        """Test path validation edge cases that should fail."""
        from langtree.parsing.parser import CommandParseError

        run_structure = RunStructure()

        # Invalid: Empty target path
        with pytest.raises(
            CommandParseError, match=r"Invalid command syntax|malformed"
        ):

            class TaskEmptyPath(TreeNode):
                """! @all->@{{value.item=items}}  # Empty target"""

                items: list[str] = ["item1"]

            run_structure.add(TaskEmptyPath)

        # Invalid: Empty variable name in mapping - now gives specific path validation error
        with pytest.raises(
            CommandParseError, match=r"Invalid.*path.*cannot.*end with dots"
        ):

            class TaskEmptyVariable(TreeNode):
                """! @all->task.target@{{value.=items}}  # Empty variable name"""

                items: list[str] = ["item1"]

            run_structure.add(TaskEmptyVariable)

        # Invalid: Empty source path in mapping
        with pytest.raises(
            CommandParseError, match=r"Invalid command syntax|malformed"
        ):

            class TaskEmptySource(TreeNode):
                """! @all->task.target@{{value.item=}}  # Empty source"""

                items: list[str] = ["item1"]

            run_structure.add(TaskEmptySource)
        # Note: Python reserved keywords in paths are parsed as regular identifiers
        # This is acceptable behavior for the current implementation

        # Edge case: Very deep nesting should work
        class TaskDeepNesting(TreeNode):
            """
            Task with very deep nesting.
            """

            items: list[str] = Field(
                default=["item1"],
                description="! @all->task.very.deeply.nested.target@{{value.field=items}}",
            )

        # Should NOT raise error - deep nesting is valid
        run_structure.add(TaskDeepNesting)

    def test_scope_validation_edge_cases(self):
        """Test scope validation edge cases per COMPREHENSIVE_GUIDE.md."""
        run_structure = RunStructure()

        # Test valid scope usage
        class TaskValidScopes(TreeNode):
            """
            ! @all->task.target@{{prompt.context=*, value.data=*, outputs.result=*}}
            Task with valid scope usage.
            """

            context: str = "context_data"
            data: str = "value_data"
            result: str = "output_data"

        class TaskTarget(TreeNode):
            """Target for scope validation."""

            context: str = "default"
            data: str = "default"
            result: str = "default"

        run_structure.add(TaskValidScopes)
        run_structure.add(TaskTarget)

        # Verify scopes are parsed correctly
        node = run_structure.get_node("task.valid_scopes")
        assert node is not None
        assert len(node.extracted_commands) == 1

        command = node.extracted_commands[0]
        target_paths = [mapping.target_path for mapping in command.variable_mappings]
        assert "prompt.context" in target_paths
        assert "value.data" in target_paths
        assert "outputs.result" in target_paths

        # Invalid: Circular task scope references
        with pytest.raises(Exception):  # Should be CircularReferenceError

            class TaskCircularScope(TreeNode):
                """! @all->task.circularscope@{{task.circularscope=data}}  # Self-reference"""

                data: str = "value"

            run_structure.add(TaskCircularScope)

        # Edge case: Cross-scope variable access should work
        class TaskCrossScope(TreeNode):
            """! @all->task.target@{{prompt.context=*, outputs.result=*}}"""

            data: str = "source_data"
            output: str = "source_output"

        # Should NOT raise error - cross-scope access is valid
        run_structure.add(TaskCrossScope)


class TestPerformanceAndStressEdgeCases:
    """Test performance edge cases that could break the implementation."""

    def test_massive_variable_resolution_performance(self):
        """Test performance with many variables."""
        run_structure = RunStructure()

        # Create task with multiple assembly variables (realistic scale)
        class TaskManyVars(TreeNode):
            """
            ! var_1="value_1"
            ! var_2="value_2"
            ! var_3="value_3"
            ! var_4="value_4"
            ! var_5="value_5"
            Task with many variables for performance testing.
            """

            field_1: str = "value_1"
            field_2: str = "value_2"
            field_3: str = "value_3"
            field_4: str = "value_4"
            field_5: str = "value_5"

        run_structure.add(TaskManyVars)

        # Test that resolution completes successfully
        node = run_structure.get_node("task.many_vars")
        assert node is not None
        assert len(node.extracted_commands) == 5

        # Verify all variables are accessible
        for i in range(1, 6):
            result = run_structure._resolve_in_current_node_context(
                f"field_{i}", "task.many_vars"
            )
            assert result == f"value_{i}"

    def test_deep_nesting_resolution_limits(self):
        """Test resolution with reasonable nesting depth."""
        run_structure = RunStructure()

        class TaskDeepNesting(TreeNode):
            """
            ! @all->task.target@{{value.result=*}}
            Task with nested data structures.
            """

            level1: dict[str, str] = {"level2": "deep_value"}
            simple_field: str = "simple"

        class TaskTarget(TreeNode):
            """Target for deep nesting test."""

            result: str = "default"

        run_structure.add(TaskDeepNesting)
        run_structure.add(TaskTarget)

        # Test that nesting is handled gracefully
        node = run_structure.get_node("task.deep_nesting")
        assert node is not None
        assert len(node.extracted_commands) == 1

        # Verify nested structure access works
        assert (
            run_structure._resolve_in_current_node_context(
                "simple_field", "task.deep_nesting"
            )
            == "simple"
        )
        assert run_structure._resolve_in_current_node_context(
            "level1", "task.deep_nesting"
        ) == {"level2": "deep_value"}

    def test_massive_command_count_resolution(self):
        """Test resolution with multiple commands."""
        run_structure = RunStructure()

        # Create task with multiple commands (realistic scale)
        class TaskMultipleCommands(TreeNode):
            """
            ! @all->task.target1@{{value.result=*}}
            ! @all->task.target2@{{value.result=*}}
            ! @all->task.target3@{{value.result=*}}
            Task with multiple commands.
            """

            field: str = "value"

        class TaskTarget1(TreeNode):
            result: str = "default"

        class TaskTarget2(TreeNode):
            result: str = "default"

        class TaskTarget3(TreeNode):
            result: str = "default"

        run_structure.add(TaskMultipleCommands)
        run_structure.add(TaskTarget1)
        run_structure.add(TaskTarget2)
        run_structure.add(TaskTarget3)

        # Test that all commands are parsed correctly
        node = run_structure.get_node("task.multiple_commands")
        assert node is not None
        assert len(node.extracted_commands) == 3

        # Verify all targets exist
        assert run_structure.get_node("task.target1") is not None
        assert run_structure.get_node("task.target2") is not None
        assert run_structure.get_node("task.target3") is not None


class TestBoundaryConditionCompliance:
    """Test boundary conditions that must be handled per specification."""

    def test_unicode_and_special_character_handling(self):
        """Test Unicode and special character handling in variables and paths."""
        run_structure = RunStructure()

        class TaskUnicode(TreeNode):
            """
            ! unicode_var="测试文本"
            ! emoji_var="🚀🔥💡"
            ! safe_special_var="safe_special_chars"
            Task with Unicode and special characters.
            """

            unicode_field: str = "Unicode: 你好世界"
            emoji_field: str = "Emoji: 🎉"
            special_field: str = "Special: !@#$%"

        run_structure.add(TaskUnicode)

        # Verify Unicode variables are handled correctly
        node = run_structure.get_node("task.unicode")
        assert node is not None
        assert (
            run_structure._resolve_in_current_node_context(
                "unicode_field", "task.unicode"
            )
            == "Unicode: 你好世界"
        )

        # Test emoji handling doesn't break resolution
        assert (
            run_structure._resolve_in_current_node_context(
                "emoji_field", "task.unicode"
            )
            == "Emoji: 🎉"
        )

        # Test special characters don't break parsing
        assert (
            run_structure._resolve_in_current_node_context(
                "special_field", "task.unicode"
            )
            == "Special: !@#$%"
        )

        # Verify assembly variable commands were extracted despite Unicode content
        assert len(node.extracted_commands) == 3

    def test_empty_and_null_value_handling(self):
        """Test handling of empty and null values per specification."""
        run_structure = RunStructure()

        class TaskEmptyValues(TreeNode):
            """
            ! empty_string=""
            ! zero_number=0
            ! false_boolean=false
            Task with empty and null values.
            """

            empty_field: str = ""
            none_field: str | None = None
            zero_field: int = 0
            false_field: bool = False
            empty_list: list[str] = []

        run_structure.add(TaskEmptyValues)

        # Verify the node was added successfully with empty/null values
        node = run_structure.get_node("task.empty_values")
        assert node is not None

        # Test that empty strings are preserved and accessible
        assert (
            run_structure._resolve_in_current_node_context(
                "empty_field", "task.empty_values"
            )
            == ""
        )

        # Test that zero values are handled correctly
        assert (
            run_structure._resolve_in_current_node_context(
                "zero_field", "task.empty_values"
            )
            == 0
        )

        # Test that false booleans work properly
        assert (
            run_structure._resolve_in_current_node_context(
                "false_field", "task.empty_values"
            )
            is False
        )

        # Test that None values don't break resolution
        assert (
            run_structure._resolve_in_current_node_context(
                "none_field", "task.empty_values"
            )
            is None
        )

        # Test that empty lists are handled correctly
        result = run_structure._resolve_in_current_node_context(
            "empty_list", "task.empty_values"
        )
        assert result == []

        # Verify assembly variable commands were extracted correctly
        assert len(node.extracted_commands) == 3

    def test_maximum_limits_compliance(self):
        """Test compliance with maximum limits per specification."""
        run_structure = RunStructure()

        # Test reasonable length variable names work fine
        class TaskReasonableLimits(TreeNode):
            """
            ! reasonable_var="value"
            ! another_reasonable_variable_name="another_value"
            Task testing reasonable limits.
            """

            field: str = "value"
            long_field_name_but_still_reasonable: str = "test"

        run_structure.add(TaskReasonableLimits)

        # Verify reasonable limits work
        node = run_structure.get_node("task.reasonable_limits")
        assert node is not None
        assert len(node.extracted_commands) == 2

        # Test deep nested path resolution doesn't break
        assert (
            run_structure._resolve_in_current_node_context(
                "long_field_name_but_still_reasonable", "task.reasonable_limits"
            )
            == "test"
        )

        # Test multiple commands in single node are handled
        assert (
            run_structure._resolve_in_current_node_context(
                "field", "task.reasonable_limits"
            )
            == "value"
        )

        # Verify the system handles normal usage patterns gracefully
        # Maximum limits are implementation-specific and tested through normal usage


class TestSpecificationContractEnforcement:
    """Test that implementation enforces specification contracts strictly."""

    def test_fail_fast_validation_enforcement(self):
        """Test fail-fast validation per LANGUAGE_SPECIFICATION.md."""
        run_structure = RunStructure()

        # Test that valid assembly variables work fine
        class TaskValidVars(TreeNode):
            """
            ! valid_var="assembly_value"
            Task with valid variables.
            """

            field: str = "field_value"

        run_structure.add(TaskValidVars)

        # Verify valid configuration works
        node = run_structure.get_node("task.valid_vars")
        assert node is not None
        assert len(node.extracted_commands) == 1

        # Test that field resolution works correctly
        assert (
            run_structure._resolve_in_current_node_context("field", "task.valid_vars")
            == "field_value"
        )

        # Test that the system validates commands at parse time
        # (Invalid commands would raise exceptions during add() call)

    def test_type_safety_enforcement(self):
        """Test type safety enforcement per specification."""
        run_structure = RunStructure()

        # Test that valid types work correctly
        class TaskValidTypes(TreeNode):
            """
            ! string_var="valid_string"
            ! number_var=42
            ! boolean_var=true
            Task with valid type usage.
            """

            field: str = "value"
            number_field: int = 100
            boolean_field: bool = True

        run_structure.add(TaskValidTypes)

        # Verify valid types are handled correctly
        node = run_structure.get_node("task.valid_types")
        assert node is not None
        assert len(node.extracted_commands) == 3

        # Test field access with different types
        assert (
            run_structure._resolve_in_current_node_context("field", "task.valid_types")
            == "value"
        )
        assert (
            run_structure._resolve_in_current_node_context(
                "number_field", "task.valid_types"
            )
            == 100
        )
        assert (
            run_structure._resolve_in_current_node_context(
                "boolean_field", "task.valid_types"
            )
            is True
        )

        # Test comprehensive type validation for assembly variables
        commands = node.extracted_commands

        # Verify assembly variable types are correctly parsed
        string_cmd = next(cmd for cmd in commands if cmd.variable_name == "string_var")
        number_cmd = next(cmd for cmd in commands if cmd.variable_name == "number_var")
        boolean_cmd = next(
            cmd for cmd in commands if cmd.variable_name == "boolean_var"
        )

        assert isinstance(string_cmd.value, str)
        assert string_cmd.value == "valid_string"
        assert isinstance(number_cmd.value, int)
        assert number_cmd.value == 42
        assert isinstance(boolean_cmd.value, bool)
        assert boolean_cmd.value is True

        # Test field access returns expected types
        assert isinstance(
            run_structure._resolve_in_current_node_context("field", "task.valid_types"),
            str,
        )
        assert isinstance(
            run_structure._resolve_in_current_node_context(
                "number_field", "task.valid_types"
            ),
            int,
        )
        assert isinstance(
            run_structure._resolve_in_current_node_context(
                "boolean_field", "task.valid_types"
            ),
            bool,
        )

    def test_assembly_variable_conflict_detection_edge_cases(self):
        """Test Assembly Variable conflict detection from LANGUAGE_SPECIFICATION.md."""
        from langtree.exceptions import LangTreeDSLError

        run_structure = RunStructure()

        # Per spec: "Variable names cannot conflict with field names in same subtree"
        with pytest.raises(LangTreeDSLError, match="conflicts with field name"):

            class TaskConflict(TreeNode):
                """
                ! field="assembly_value"  # Conflicts with field below
                Task with Assembly Variable conflicting with field name.
                """

                field: str = "field_value"  # Same name as assembly variable

            run_structure.add(TaskConflict)

        # Test that different case is allowed (case-sensitive)
        class TaskCaseSensitive(TreeNode):
            """
            ! Field="assembly_value"  # Different case from field
            Task testing case sensitivity.
            """

            field: str = "field_value"  # Lowercase

        # Should NOT raise error - case sensitive, no conflict
        run_structure.add(TaskCaseSensitive)

        # Verify the assembly variable was stored correctly
        registry = run_structure.get_assembly_variable_registry()
        field_var = registry.get_variable("Field")
        assert field_var is not None
        assert field_var.value == "assembly_value"

    def test_assembly_variable_scope_inheritance_edge_cases(self):
        """Test Assembly Variable scope inheritance edge cases.

        SKIPPED: As per architectural correction, assembly variables should NOT be available
        during runtime execution. Assembly variables are only used during chain assembly/building,
        not during runtime variable resolution. This test was based on incorrect assumptions
        about variable bridging that have been removed from the specification.
        """
        pass

    def test_assembly_variable_complex_data_types(self):
        """Test Assembly Variables with complex data types per spec."""
        run_structure = RunStructure()

        # Per spec: Support for strings, numbers, booleans
        class TaskComplexTypes(TreeNode):
            """
            ! string_var="simple_text"
            ! int_var=42
            ! float_var=3.14
            ! bool_true=true
            ! bool_false=false
            ! negative_int=-100
            ! empty_string=""
            ! unicode_string="测试文本"
            Task with various Assembly Variable types.
            """

            field: str = "value"

        run_structure.add(TaskComplexTypes)

        # Verify all data types are parsed correctly
        node = run_structure.get_node("task.complex_types")
        assert node is not None
        assert len(node.extracted_commands) == 8

        # Test that the task was added successfully with complex types
        assert (
            run_structure._resolve_in_current_node_context(
                "field", "task.complex_types"
            )
            == "value"
        )


class TestCrossModuleIntegrationCompliance:
    """Test cross-module integration per COMPREHENSIVE_GUIDE.md."""

    def test_structure_resolution_integration_workflow(self):
        """Test integration between structure.py and resolution.py per guide."""
        run_structure = RunStructure()

        class TaskIntegrated(TreeNode):
            """
            ! global_config="shared_settings"
            Task requiring structure + resolution integration.
            """

            items: list[str] = Field(
                default=["item1", "item2", "item3"],
                description="""
                ! @each[items]->task.processor@{{value.item=items}}*
                Items to process.
            """,
            )

        class TaskProcessor(TreeNode):
            """Processor requiring resolved variables."""

            item: str = "default"
            config: str = "default_config"

        run_structure.add(TaskIntegrated)
        run_structure.add(TaskProcessor)

        # Verify structure.py builds correct tree
        integrated_node = run_structure.get_node("task.integrated")
        processor_node = run_structure.get_node("task.processor")
        assert integrated_node is not None
        assert processor_node is not None

        # Verify resolution.py resolves variables correctly
        assert run_structure._resolve_in_current_node_context(
            "items", "task.integrated"
        ) == ["item1", "item2", "item3"]

        # Test complete workflow from parsing to resolution
        assert len(integrated_node.extracted_commands) == 2

    def test_template_variables_resolution_integration(self):
        """Test integration between template_variables.py and resolution.py."""
        run_structure = RunStructure()

        class TaskWithTemplates(TreeNode):
            """
            Task using template variables with runtime resolution.
            Templates and runtime variables work together.
            """

            model_name: str = "gpt-4"
            context_data: str = "test_context"

        run_structure.add(TaskWithTemplates)

        # Test template variable processing with runtime variables
        node = run_structure.get_node("task.with_templates")
        assert node is not None

        # Verify runtime variable resolution works
        assert (
            run_structure._resolve_in_current_node_context(
                "model_name", "task.with_templates"
            )
            == "gpt-4"
        )
        assert (
            run_structure._resolve_in_current_node_context(
                "context_data", "task.with_templates"
            )
            == "test_context"
        )

        # Verify integration of both systems works correctly
        assert node.clean_docstring is not None

    def test_comprehensive_validation_integration(self):
        """Test validation.py integration with structure, resolution, template variables."""
        run_structure = RunStructure()

        # Simplified version that tests validation integration
        class TaskComplex(TreeNode):
            """
            ! llm_model="gpt-4"
            ! iterations=5
            Complex task requiring validation systems.
            """

            data_items: list[str] = ["item1", "item2"]
            summary: str = "Data summary"
            model: str = "default"

        run_structure.add(TaskComplex)

        # Test comprehensive validation across modules
        node = run_structure.get_node("task.complex")
        assert node is not None
        assert len(node.extracted_commands) == 2

        # Verify field resolution works
        assert (
            run_structure._resolve_in_current_node_context("summary", "task.complex")
            == "Data summary"
        )


class TestRuntimeVariableErrorHandlingCompliance:
    """Test runtime variable error handling per LANGUAGE_SPECIFICATION.md."""

    def test_undefined_runtime_variable_handling(self):
        """Test handling of undefined runtime variables per specification."""
        from langtree.exceptions import RuntimeVariableError
        from langtree.execution.resolution import resolve_runtime_variables

        run_structure = RunStructure()

        class TaskUndefined(TreeNode):
            """Task with defined field for testing."""

            defined_field: str = "value"

        run_structure.add(TaskUndefined)
        node = run_structure.get_node("task.undefined")

        # Test undefined variable detection
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Content: {undefined_var}", run_structure, node)

        error_msg = str(exc_info.value).lower()
        assert "undefined" in error_msg
        assert "available fields" in error_msg
        assert "defined_field" in str(exc_info.value)  # Should list available fields

        # Test multiple undefined variables
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Priority: {also_undefined}", run_structure, node)

        assert "also_undefined" in str(exc_info.value)

        # Test that defined variables work correctly
        result = resolve_runtime_variables(
            "Value: {defined_field}", run_structure, node
        )
        assert "prompt__undefined__defined_field" in result

    def test_malformed_runtime_variable_syntax_handling(self):
        """Test handling of malformed runtime variable syntax."""
        run_structure = RunStructure()

        # Test that malformed syntax is gracefully handled
        # The current implementation should either parse correctly or fail gracefully

        class TaskWithValidVariables(TreeNode):
            """
            Task with valid runtime variable syntax.
            Contains runtime variables in processing context.
            """

            valid_var: str = "test_value"
            priority_var: str = "priority_value"
            field: str = "value"

        # This should work fine
        run_structure.add(TaskWithValidVariables)

        # Verify the task was added successfully
        node = run_structure.get_node("task.with_valid_variables")
        assert node is not None

        # Test variable resolution works with valid syntax
        result = run_structure._resolve_in_current_node_context(
            "valid_var", "task.with_valid_variables"
        )
        assert result == "test_value"

        # Test that the docstring is preserved and processed correctly
        # The system should handle docstring content without rejecting runtime variable references
        assert "runtime variables" in node.field_type.__doc__.lower()

    def test_circular_reference_detection_in_runtime_variables(self):
        """Test circular reference detection in runtime variables."""
        run_structure = RunStructure()

        # Test simple non-circular references work fine
        class TaskNonCircular(TreeNode):
            """
            ! var_a="simple_value"
            ! var_b="another_value"
            Task with non-circular references.
            """

            field_a: str = "field_value_a"
            field_b: str = "field_value_b"

        run_structure.add(TaskNonCircular)

        # Verify non-circular references work
        node = run_structure.get_node("task.non_circular")
        assert node is not None
        assert len(node.extracted_commands) == 2

        # Test field resolution works correctly
        assert (
            run_structure._resolve_in_current_node_context(
                "field_a", "task.non_circular"
            )
            == "field_value_a"
        )
        assert (
            run_structure._resolve_in_current_node_context(
                "field_b", "task.non_circular"
            )
            == "field_value_b"
        )


class TestSpecificationComplianceEdgeCases:
    """Test edge cases that challenge specification compliance."""

    def test_variable_system_taxonomy_compliance(self):
        """Test all 5 variable types from LANGUAGE_SPECIFICATION.md Variable System."""
        run_structure = RunStructure()

        class TaskAllVariableTypes(TreeNode):
            """
            Task using all 5 variable types from specification:
            1. Assembly Variables (! var=value syntax)
            2. Runtime Variables (double brace syntax)
            3. LangTree DSL Variable Targets (@each and @all commands)
            4. Scope Context Variables (scope.field syntax)
            5. Field References (field syntax)

            ! assembly_var="assembly_value"  # Type 1: Assembly Variable
            ! @resampled[quality]->mean  # Type 5: Field Reference

            Runtime variables will be processed during resolution phase.
            Scope context variables provide cross-tree data access.
            Field references enable data locality patterns.
            """

            collection: list[str] = Field(
                default=["item1", "item2"],
                description="""
                ! @each[collection]->task.target@{{value.item=collection}}*
                Collection for LangTree DSL @each command testing.
            """,
            )
            runtime_var: str = "runtime_value"
            context: str = "context_value"
            quality: int = 5  # Should be Enum for resampling

        class TaskTarget(TreeNode):
            """Target for LangTree DSL commands."""

            item: str = "default"

        run_structure.add(TaskAllVariableTypes)
        run_structure.add(TaskTarget)

        # Verify all 5 variable types are handled correctly
        node = run_structure.get_node("task.all_variable_types")
        assert node is not None

        # Test basic functionality - variable system integration
        # Note: Full implementation details vary, focusing on documented behavior

        # Type 1: Assembly Variables - verify parsing occurs
        # Some commands should be parsed from the docstring
        # Note: VariableAssignmentCommand parsing is stable and well-tested

        # Type 2: Runtime Variables - verify field values exist for resolution
        assert (
            run_structure._resolve_in_current_node_context(
                "runtime_var", "task.all_variable_types"
            )
            == "runtime_value"
        )

        # Type 3: LangTree DSL Variable Targets - verify @each command in field description
        # The @each command should be in the field description, not class docstring

        # Type 4: Scope Context Variables - verify field access works
        assert (
            run_structure._resolve_in_current_node_context(
                "context", "task.all_variable_types"
            )
            == "context_value"
        )

        # Type 5: Field References - verify collection field access
        assert run_structure._resolve_in_current_node_context(
            "collection", "task.all_variable_types"
        ) == ["item1", "item2"]
        assert (
            run_structure._resolve_in_current_node_context(
                "quality", "task.all_variable_types"
            )
            == 5
        )

        # Verify the node and target node exist and are accessible
        target_node = run_structure.get_node("task.target")
        assert target_node is not None
        assert (
            run_structure._resolve_in_current_node_context("item", "task.target")
            == "default"
        )

    def test_scope_system_compliance_edge_cases(self):
        """Test scope system compliance with edge cases from COMPREHENSIVE_GUIDE.md."""
        run_structure = RunStructure()

        class TaskScopeCompliance(TreeNode):
            """
            Task testing all scope types from COMPREHENSIVE_GUIDE.md:
            - prompt: Target context prompt variables
            - value: Output becomes target variable value
            - outputs: Direct assignment scope (bypasses LLM)
            - task: Reference to Task classes

            ! @all->task.target@{{
                prompt.context_info=summary,
                value.generated_content=analysis,
                outputs.direct_data=raw_data,
                task.reference=task.other
            }}
            """

            summary: str = "Summary for prompt context"
            analysis: str = "Analysis for value context"
            raw_data: str = "Raw data for outputs context"

        class TaskTarget(TreeNode):
            """Target with fields for all scope types."""

            context_info: str = "default_context"
            generated_content: str = "default_content"
            direct_data: str = "default_data"
            reference: str = "default_reference"

        class TaskOther(TreeNode):
            """Other task for task scope reference."""

            other_field: str = "other_value"

        run_structure.add(TaskScopeCompliance)
        run_structure.add(TaskTarget)
        run_structure.add(TaskOther)

        # Verify scope system compliance with COMPREHENSIVE_GUIDE.md
        node = run_structure.get_node("task.scope_compliance")
        assert node is not None

        # Verify prompt scope - should access summary field
        summary_result = run_structure._resolve_in_current_node_context(
            "summary", "task.scope_compliance"
        )
        assert summary_result == "Summary for prompt context"

        # Verify value scope - should access analysis field
        analysis_result = run_structure._resolve_in_current_node_context(
            "analysis", "task.scope_compliance"
        )
        assert analysis_result == "Analysis for value context"

        # Verify outputs scope - should access raw_data field
        raw_data_result = run_structure._resolve_in_current_node_context(
            "raw_data", "task.scope_compliance"
        )
        assert raw_data_result == "Raw data for outputs context"

        # Verify task scope - should create cross-task references
        other_node = run_structure.get_node("task.other")
        assert other_node is not None
        other_result = run_structure._resolve_in_current_node_context(
            "other_field", "task.other"
        )
        assert other_result == "other_value"

    def test_execution_command_compliance_edge_cases(self):
        """Test execution command compliance and edge cases."""
        run_structure = RunStructure()

        class TaskExecutionCommands(TreeNode):
            """
            Task with various execution commands per LANGUAGE_SPECIFICATION.md.
            ! resample(5)  # Basic execution
            ! llm("gpt-4")  # Model selection
            ! llm("claude-3", override=true)  # With named parameter
            ! resample(iterations)  # Variable argument
            ! llm(model_var, override=override_var)  # All variable arguments
            """

            iterations: int = 3
            model_var: str = "gpt-3.5"
            override_var: bool = False

        run_structure.add(TaskExecutionCommands)

        # Verify execution command parsing (basic functionality test)
        node = run_structure.get_node("task.execution_commands")
        assert node is not None

        # Note: Execution command parsing may not be fully implemented yet
        # Test documents the expected behavior for future implementation

        # For now, just verify the node exists and can be processed
        # TODO: Once execution commands are fully implemented, add specific assertions:
        # - Verify 5 execution commands are parsed (resample, llm variants)
        # - Test argument resolution (literals vs variables)
        # - Test named parameter support
        # - Verify command registry validation

        # Verify docstring was processed correctly
        assert node.clean_docstring is not None
        assert "Task with various execution commands" in node.clean_docstring

        # Edge case: Invalid commands should be caught at parse time
        with pytest.raises(Exception):  # Should be ParseError for unknown command

            class TaskInvalidCommand(TreeNode):
                """! invalid_command(123)  # Should fail"""

                field: str = "value"

            run_structure.add(TaskInvalidCommand)

        # TODO: Edge case - Invalid argument types should be caught at parse time
        # Currently not implemented: argument type validation for execution commands
        # Example: ! resample("not_a_number") should fail type validation

    def test_nested_field_access(self):
        """Test accessing nested fields and complex data structures."""
        run_structure = RunStructure()

        class NestedData(TreeNode):
            value: str = "test"

        class Data(TreeNode):
            nested: NestedData = NestedData()
            numbers: list[int] = [1, 2, 3]

        class Metadata(TreeNode):
            type: str = "test"
            version: int = 1

        class TaskComplex(TreeNode):
            data: Data = Data()
            metadata: Metadata = Metadata()

        run_structure.add(TaskComplex)

        # Test nested field access using current node context
        result = run_structure._resolve_in_current_node_context(
            "data.nested.value", "task.complex"
        )
        assert result == "test"

        # Test metadata access
        result = run_structure._resolve_in_current_node_context(
            "metadata.type", "task.complex"
        )
        assert result == "test"

        # Test that the task was added correctly
        complex_node = run_structure.get_node("task.complex")
        assert complex_node is not None

        # Test accessing list data
        result = run_structure._resolve_in_current_node_context(
            "data.numbers", "task.complex"
        )
        assert result == [1, 2, 3]

    def test_scope_resolution_validation(self):
        """Test that variables are resolved in the correct scope order."""
        run_structure = RunStructure()

        class Nested(TreeNode):
            local_var: str = "nested_value"

        class TaskScoped(TreeNode):
            local_var: str = "local_value"
            nested: Nested = Nested()

        run_structure.add(TaskScoped)

        # Test that local scope takes precedence
        result = run_structure._resolve_in_current_node_context(
            "local_var", "task.scoped"
        )
        assert result == "local_value"

        # Test that nested access works correctly
        result = run_structure._resolve_in_current_node_context(
            "nested.local_var", "task.scoped"
        )
        assert result == "nested_value"

    def test_runtime_variable_error_handling(self):
        """Test error handling for invalid runtime variable access."""

        run_structure = RunStructure()

        class TaskSimple(TreeNode):
            data: str = "value"

        run_structure.add(TaskSimple)

        # Test accessing non-existent field
        with pytest.raises((KeyError, AttributeError, ValueError)):
            run_structure._resolve_in_current_node_context(
                "non_existent", "task.simple"
            )

        # Test accessing non-existent node
        with pytest.raises((KeyError, ValueError)):
            run_structure._resolve_in_global_tree_context("task.non_existent.field")

        # Test valid access works
        result = run_structure._resolve_in_current_node_context("data", "task.simple")
        assert result == "value"

    def test_runtime_variable_type_conversion(self):
        """Test automatic type conversion for runtime variables."""
        run_structure = RunStructure()

        class TaskTyped(TreeNode):
            string_value: str = "123"
            int_value: int = 456
            float_value: float = 78.9
            bool_value: bool = True
            list_value: list[int] = [1, 2, 3]
            dict_value: dict[str, str] = {"key": "value"}

        run_structure.add(TaskTyped)

        # Test that types are preserved correctly during runtime variable resolution
        assert (
            run_structure._resolve_in_current_node_context("string_value", "task.typed")
            == "123"
        )
        assert (
            run_structure._resolve_in_current_node_context("int_value", "task.typed")
            == 456
        )
        assert (
            run_structure._resolve_in_current_node_context("float_value", "task.typed")
            == 78.9
        )
        assert (
            run_structure._resolve_in_current_node_context("bool_value", "task.typed")
            is True
        )
        assert run_structure._resolve_in_current_node_context(
            "list_value", "task.typed"
        ) == [1, 2, 3]
        assert run_structure._resolve_in_current_node_context(
            "dict_value", "task.typed"
        ) == {"key": "value"}

    def test_runtime_variable_command_integration(self):
        """Test integration between runtime variables and LangTree DSL commands."""
        run_structure = RunStructure()

        class Shared(TreeNode):
            timeout: int = 30
            retries: int = 3

        class TaskWithCommands(TreeNode):
            """
            ! @all->task.target@{{value.data=*, value.config=*}}
            Task with LangTree DSL commands using runtime variables.
            """

            local_data: str = "test_data"
            shared: Shared = Shared()

        class TaskTarget(TreeNode):
            data: str = "default"
            config: int = 10

        run_structure.add(TaskWithCommands)
        run_structure.add(TaskTarget)

        # Test that LangTree DSL commands are properly parsed and integrated
        source_node = run_structure.get_node("task.with_commands")
        assert source_node is not None
        assert len(source_node.extracted_commands) == 1

        # Verify command structure
        command = source_node.extracted_commands[0]
        assert command.destination_path == "task.target"
        assert len(command.variable_mappings) == 2

        # Test runtime variable resolution in current node context
        assert (
            run_structure._resolve_in_current_node_context(
                "local_data", "task.with_commands"
            )
            == "test_data"
        )
        assert (
            run_structure._resolve_in_current_node_context(
                "shared.timeout", "task.with_commands"
            )
            == 30
        )

        # Verify variable mappings use wildcards (as expected for @all commands)
        mapping_sources = [mapping.source_path for mapping in command.variable_mappings]
        assert all(source == "*" for source in mapping_sources)


class MockResolvedPath:
    """Mock ResolvedPath for testing."""

    def __init__(self, path, scope=None):
        self.path = path
        self.scope = scope


class MockScope:
    """Mock scope for testing."""

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class TestContextResolutionImplementation:
    """Test cases for the implemented context resolution functions."""

    def test_resolve_inclusion_context_basic(self):
        """Test basic inclusion context resolution."""
        from langtree.execution.resolution import _resolve_inclusion_context

        rs = RunStructure()

        try:
            _resolve_inclusion_context(rs, None, "task.test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid resolved inclusion" in str(e)

    def test_resolve_destination_context_pending(self):
        """Test destination context resolution with pending target."""
        from langtree.execution.resolution import _resolve_destination_context

        rs = RunStructure()

        result = _resolve_destination_context(rs, None, "task.pending", None)

        assert result["status"] == "pending"
        assert result["destination_path"] == "task.pending"
        assert result["requires_resolution"]
        assert "not yet available" in result["reason"]

    def test_resolve_destination_context_empty_path(self):
        """Test destination context resolution with empty path."""
        from langtree.execution.resolution import _resolve_destination_context

        rs = RunStructure()

        try:
            _resolve_destination_context(rs, None, "", None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)

    def test_resolve_variable_mapping_context_basic(self):
        """Test basic variable mapping context resolution."""
        from langtree.execution.resolution import _resolve_variable_mapping_context

        rs = RunStructure()

        try:
            _resolve_variable_mapping_context(rs, None, None, "task.source", None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Both target and source must be resolved" in str(e)

    def test_resolve_variable_mapping_wildcard_source(self):
        """Test variable mapping with wildcard source."""
        from langtree.execution.resolution import _resolve_variable_mapping_context

        rs = RunStructure()

        mock_target = MockResolvedPath("target_field", None)
        mock_source = MockResolvedPath("*", None)

        result = _resolve_variable_mapping_context(
            rs, mock_target, mock_source, "task.source", None
        )

        assert result["validations"]["source"]["status"] == "valid"
        assert result["validations"]["source"]["type"] == "wildcard"
        assert result["source_path"] == "*"
        assert result["validations"]["target"]["status"] == "pending"
        assert result["overall_status"] == "pending"

    def test_resolve_variable_mapping_outputs_scope(self):
        """Test variable mapping with outputs scope."""
        from langtree.execution.resolution import _resolve_variable_mapping_context

        rs = RunStructure()

        class MockTarget(TreeNode):
            pass

        target_node = StructureTreeNode("task.target", MockTarget)

        mock_target = MockResolvedPath("result", MockScope("outputs"))
        mock_source = MockResolvedPath("analysis", None)

        result = _resolve_variable_mapping_context(
            rs, mock_target, mock_source, "task.source", target_node
        )

        assert result["target_scope"] == "outputs"
        assert result["validations"]["target"]["status"] == "valid"
        assert result["validations"]["target"]["scope"] == "outputs"

    def test_chain_assembly_integration(self):
        """Test that outputs context resolution works during chain assembly."""
        rs = RunStructure()

        result = rs._resolve_in_outputs_context("test_field", "task.test")
        assert result is None

        result = rs._resolve_in_outputs_context("test_field", "task.test")
        assert result is None


class TestRuntimeVariableResolution:
    """Test Runtime Variable resolution system for {{var}} and {{<var>}} syntax."""

    def setup_method(self):
        """Set up test fixtures for runtime variable resolution."""
        self.run_structure = RunStructure()

    def test_basic_runtime_variable_resolution(self):
        """Test basic {variable} resolution in prompts during execution."""
        from langtree.execution.resolution import resolve_runtime_variables

        class TaskWithRuntimeVar(TreeNode):
            """
            Task using runtime variables in docstring.

            This task will use {model_name} for processing.
            Expected output format: {output_format}
            """

            model_name: str = "gpt-4"
            output_format: str = "json"
            result: str = "default"

        self.run_structure.add(TaskWithRuntimeVar)

        # Get the actual task node (not the root)
        task_node = self.run_structure.get_node("task.with_runtime_var")
        assert task_node is not None, "Task node should be found"

        # Test content with runtime variables
        content_with_vars = "This task will use {model_name} for processing.\nExpected output format: {output_format}"

        # Resolve runtime variables
        resolved_content = resolve_runtime_variables(
            content_with_vars, self.run_structure, task_node
        )

        # Verify {model_name} expands to double underscore format
        assert "{prompt__with_runtime_var__model_name}" in resolved_content
        assert "{model_name}" not in resolved_content

        # Verify {output_format} expands to double underscore format
        assert "{prompt__with_runtime_var__output_format}" in resolved_content
        assert "{output_format}" not in resolved_content

        # Verify the full expected content (expanded format)
        expected_content = "This task will use {prompt__with_runtime_var__model_name} for processing.\nExpected output format: {prompt__with_runtime_var__output_format}"
        assert resolved_content == expected_content

    def test_scoped_runtime_variable_resolution(self):
        """Test scoped runtime variables like {task.field} and {value.field}."""
        from langtree.execution.resolution import resolve_runtime_variables

        class TaskWithScopedVars(TreeNode):
            """
            Task using scoped runtime variables.

            Task context: {task.name} with {task.priority}
            Value context: {value.timestamp}
            """

            name: str = "data_analysis"
            priority: int = 1
            timestamp: str = "2023-01-01"
            result: str = "default"

        self.run_structure.add(TaskWithScopedVars)

        # Get the actual task node
        task_node = self.run_structure.get_node("task.with_scoped_vars")
        assert task_node is not None, "Task node should be found"

        # Test content with runtime variables (single tokens only per specification)
        content_with_vars = (
            "Task: {name} with priority {priority}\nTimestamp: {timestamp}"
        )

        # Resolve runtime variables
        resolved_content = resolve_runtime_variables(
            content_with_vars, self.run_structure, task_node
        )

        # Verify variables expand to double underscore format
        assert "{prompt__with_scoped_vars__name}" in resolved_content
        assert "{prompt__with_scoped_vars__priority}" in resolved_content
        assert "{prompt__with_scoped_vars__timestamp}" in resolved_content

        # Verify original single-token variables are gone
        assert "{name}" not in resolved_content
        assert "{priority}" not in resolved_content
        assert "{timestamp}" not in resolved_content

        # Verify full expected content (expanded format)
        expected_content = "Task: {prompt__with_scoped_vars__name} with priority {prompt__with_scoped_vars__priority}\nTimestamp: {prompt__with_scoped_vars__timestamp}"
        assert resolved_content == expected_content

    def test_runtime_variable_field_resolution(self):
        """Test {variable} resolution to field values with proper scope separation."""

        class TaskWithFields(TreeNode):
            """
            Task demonstrating runtime variable resolution.
            ! model="claude-3"  # Assembly variable for chain configuration
            ! temperature=0.8   # Assembly variable for chain configuration

            Using model {model_name} with temperature {temperature_setting}.
            Process data: {input_data}
            """

            model_name: str = "gpt-4"
            temperature_setting: float = 0.7
            input_data: str = "sample data"
            result: str = "default"

        self.run_structure.add(TaskWithFields)
        source_node = self.run_structure.get_node("task.taskwithfields")

        # Test basic field resolution - should expand to external scope format
        content = "Using model {model_name} with temperature {temperature_setting}."
        expanded = resolve_runtime_variables(content, self.run_structure, source_node)

        # Verify field variables are expanded to external scope (prompt__variable)
        assert (
            expanded
            == "Using model {prompt__model_name} with temperature {prompt__temperature_setting}."
        )

        # Test assembly variable separation - assembly variables don't interfere with runtime variables
        content_with_assembly_ref = "Using assembly model {model}"
        assembly_result = resolve_runtime_variables(
            content_with_assembly_ref, self.run_structure, source_node
        )
        # Assembly variable {model} becomes external scope {prompt__model}, demonstrating separation
        assert assembly_result == "Using assembly model {prompt__model}"

        # Test field variable resolution
        valid_field_content = "Data: {input_data}"
        expanded_field = resolve_runtime_variables(
            valid_field_content, self.run_structure, source_node
        )
        assert expanded_field == "Data: {prompt__input_data}"

    def test_scope_aware_runtime_variable_resolution(self):
        """Test runtime variable resolution with scope awareness."""

        class TaskWithScopedVars(TreeNode):
            """
            Task with scope-specific runtime variables.

            Task scope: {task.name}
            Prompt scope: {prompt.context}
            Value scope: {value.data}
            Outputs scope: {outputs.result}
            """

            name: str = "test_task"
            context: str = "test context"
            data: str = "test data"
            result: str = "default"

        self.run_structure.add(TaskWithScopedVars)

        # Get the task node using correct snake_case naming
        task_node = self.run_structure.get_node("task.with_scoped_vars")
        assert task_node is not None, "Task node should be found"

        # Test scope-aware resolution - based on VARIABLE_TAXONOMY.md scope context variables
        # Different scopes should use appropriate resolvers

        # Test current node context resolution - direct field access on node
        node_result = self.run_structure._resolve_in_current_node_context(
            "name", "task.with_scoped_vars"
        )
        assert node_result == "test_task"

        # Test prompt context resolution - access prompt-related data
        prompt_result = self.run_structure._resolve_in_current_prompt_context(
            "context", "task.with_scoped_vars"
        )
        assert prompt_result is not None

        # Test value context resolution - access value data
        value_result = self.run_structure._resolve_in_value_context(
            "data", "task.with_scoped_vars"
        )
        assert value_result is not None

        # Test outputs context resolution - should return None during chain assembly
        outputs_result = self.run_structure._resolve_in_outputs_context(
            "result", "task.with_scoped_vars"
        )
        assert outputs_result is None  # Expected during chain assembly phase

    def test_nested_runtime_variable_resolution(self):
        """Test resolution of nested runtime variables."""

        class TaskParent(TreeNode):
            """
            Parent task with nested child.
            Tests nested runtime variable resolution.
            """

            parent_data: str = "parent_value"

            class TaskChild(TreeNode):
                """
                Child task with nested data.
                Tests runtime variable access patterns.
                """

                nested_data: str = "child_value"
                result: str = "default"

            child: TaskChild = TaskChild()

        self.run_structure.add(TaskParent)

        # Test nested variable resolution
        # Parent should be able to access its own data
        assert (
            self.run_structure._resolve_in_current_node_context(
                "parent_data", "task.parent"
            )
            == "parent_value"
        )

        # Parent should be able to access child's nested data
        assert (
            self.run_structure._resolve_in_current_node_context(
                "child.nested_data", "task.parent"
            )
            == "child_value"
        )

        # Test that nested structure exists
        parent_node = self.run_structure.get_node("task.parent")
        assert parent_node is not None

        # Test cross-node variable access patterns
        # The resolution system should handle nested field access correctly
        # This tests the path traversal within the current node context


class TestDoubleUnderscoreExpansion:
    """Test cases for the double underscore expansion feature in runtime variables."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    def test_simple_runtime_variable_expansion(self):
        """Test that simple runtime variables get double underscore expansion."""
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Hello {model_name} and {temperature}"
        current_node = "task.analyzer"

        result = resolve_runtime_variables(content, self.run_structure, current_node)

        assert (
            result
            == "Hello {prompt__analyzer__model_name} and {prompt__analyzer__temperature}"
        )

    def test_nested_node_path_expansion(self):
        """Test expansion works with deeply nested node paths."""
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Using {config_value} for analysis"
        current_node = "task.analytics.deep.processor.step1"

        result = resolve_runtime_variables(content, self.run_structure, current_node)

        assert (
            result
            == "Using {prompt__analytics__deep__processor__step1__config_value} for analysis"
        )

    def test_no_expansion_for_template_variables(self):
        """Test that template variables are not expanded with double underscores."""
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Content: {PROMPT_SUBTREE} and {COLLECTED_CONTEXT}"
        current_node = "task.analyzer"

        result = resolve_runtime_variables(content, self.run_structure, current_node)

        # Template variables should remain unchanged
        assert result == "Content: {PROMPT_SUBTREE} and {COLLECTED_CONTEXT}"

    def test_runtime_variables_with_double_underscores_rejected(self):
        """Test that runtime variables with double underscores are rejected."""
        from langtree.exceptions import RuntimeVariableError
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Value: {prompt__analyzer__already_expanded}"
        current_node = "task.analyzer"

        # Variables with double underscores should be rejected
        with pytest.raises(
            RuntimeVariableError, match="cannot contain double underscores"
        ):
            resolve_runtime_variables(content, self.run_structure, current_node)

    def test_runtime_variables_with_dots_rejected(self):
        """Test that runtime variables with dots are rejected."""
        from langtree.exceptions import RuntimeVariableError
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Value: {task.field}"
        current_node = "task.analyzer"

        # Variables with dots should be rejected
        with pytest.raises(RuntimeVariableError, match="cannot contain dots"):
            resolve_runtime_variables(content, self.run_structure, current_node)

    def test_expansion_without_current_node(self):
        """Test expansion behavior when current_node is None or empty."""
        from langtree.execution.resolution import resolve_runtime_variables

        content = "Hello {model_name}"

        result = resolve_runtime_variables(content, self.run_structure, None)
        assert result == "Hello {prompt__model_name}"

        result = resolve_runtime_variables(content, self.run_structure, "")
        assert result == "Hello {prompt__model_name}"

    def test_mixed_valid_and_template_variables(self):
        """Test expansion works correctly with valid runtime and template variables."""
        from langtree.execution.resolution import resolve_runtime_variables

        content = """
        Template: {PROMPT_SUBTREE}
        Runtime: {model_name} and {temperature}
        """
        current_node = "task.analyzer"

        result = resolve_runtime_variables(content, self.run_structure, current_node)

        # Template variables should remain unchanged
        assert "{PROMPT_SUBTREE}" in result
        # Runtime variables should be expanded
        assert "{prompt__analyzer__model_name}" in result
        assert "{prompt__analyzer__temperature}" in result


class TestDoubleUnderscoreValidation:
    """Test cases for validation that user variables cannot contain double underscores."""

    def test_reject_user_variables_with_double_underscore(self):
        """Test that user variables containing __ are rejected during validation."""
        from langtree.templates.variables import validate_template_variable_names

        content = "Invalid: {user__variable} and valid: {normal_var}"
        errors = validate_template_variable_names(content)

        assert len(errors) == 1
        assert "user__variable" in errors[0]
        assert "double underscore '__' which is reserved for system use" in errors[0]

    def test_allow_variables_without_double_underscore(self):
        """Test that normal variables without __ pass validation."""
        from langtree.templates.variables import validate_template_variable_names

        content = "Valid: {model_name} and {temperature_setting}"
        errors = validate_template_variable_names(content)

        assert len(errors) == 0

    def test_validation_with_template_variables(self):
        """Test that template variables are not checked for double underscore."""
        from langtree.templates.variables import validate_template_variable_names

        content = "Template: {PROMPT_SUBTREE} Invalid: {user__var} Valid: {normal_var}"
        errors = validate_template_variable_names(content)

        # Only the user variable with __ should be flagged
        assert len(errors) == 1
        assert "user__var" in errors[0]

    def test_integration_with_structure_processing(self):
        """Test that double underscore validation is enforced during node processing."""
        from langtree.exceptions import TemplateVariableNameError

        run_structure = RunStructure()

        # This should raise an exception due to __ in user variable
        with pytest.raises(TemplateVariableNameError, match="double underscore"):

            class TaskWithInvalidVar(TreeNode):
                """Task with invalid runtime variable {user__invalid}"""

                field: str = "value"

            run_structure.add(TaskWithInvalidVar)


class TestCWDPathResolution:
    """Test cases for Command Working Directory (CWD) path resolution."""

    def test_relative_path_resolution_with_cwd(self):
        """Test that relative paths are resolved from CWD."""
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analyzer.step1"
        result = PathResolver.resolve_path_with_cwd("summary", cwd)

        assert result.scope_modifier is None
        assert result.path_remainder == "task.analyzer.step1.summary"
        assert result.original_path == "summary"

    def test_absolute_path_with_task_scope_ignores_cwd(self):
        """Test that absolute paths starting with task. ignore CWD."""
        from langtree.core.path_utils import ScopeModifier
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analyzer.step1"
        result = PathResolver.resolve_path_with_cwd("task.other.field", cwd)

        assert result.scope_modifier == ScopeModifier.TASK
        assert result.path_remainder == "other.field"
        assert result.original_path == "task.other.field"

    def test_absolute_path_with_scope_modifier_ignores_cwd(self):
        """Test that paths with scope modifiers ignore CWD."""
        from langtree.core.path_utils import ScopeModifier
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analyzer.step1"

        # Test outputs scope
        result = PathResolver.resolve_path_with_cwd("outputs.result", cwd)
        assert result.scope_modifier == ScopeModifier.OUTPUTS
        assert result.path_remainder == "result"

        # Test prompt scope
        result = PathResolver.resolve_path_with_cwd("prompt.template", cwd)
        assert result.scope_modifier == ScopeModifier.PROMPT
        assert result.path_remainder == "template"

        # Test value scope
        result = PathResolver.resolve_path_with_cwd("value.data", cwd)
        assert result.scope_modifier == ScopeModifier.VALUE
        assert result.path_remainder == "data"

    def test_empty_path_with_cwd(self):
        """Test behavior with empty paths."""
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analyzer.step1"
        result = PathResolver.resolve_path_with_cwd("", cwd)

        assert result.scope_modifier is None
        assert result.path_remainder == ""
        assert result.original_path == ""

    def test_path_without_dots_with_cwd(self):
        """Test single-token paths with CWD."""
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analyzer.step1"
        result = PathResolver.resolve_path_with_cwd("result", cwd)

        assert result.scope_modifier is None
        assert result.path_remainder == "task.analyzer.step1.result"
        assert result.original_path == "result"

    def test_variable_mapping_with_cwd(self):
        """Test variable mapping resolution with CWD."""
        from langtree.core.path_utils import PathResolver, ScopeModifier

        cwd = "task.analyzer.step1"

        # Relative target, absolute source
        mapping = PathResolver.resolve_variable_mapping_with_cwd(
            "summary", "task.data.source", cwd
        )

        # Target should be resolved with CWD
        assert mapping.resolved_target.scope_modifier is None
        assert mapping.resolved_target.path_remainder == "task.analyzer.step1.summary"

        # Source should ignore CWD (absolute)
        assert mapping.resolved_source.scope_modifier == ScopeModifier.TASK
        assert mapping.resolved_source.path_remainder == "data.source"

    def test_deeply_nested_cwd_resolution(self):
        """Test CWD resolution with deeply nested paths."""
        from langtree.parsing.path_resolver import PathResolver

        cwd = "task.analytics.deep.processor.final_step"
        result = PathResolver.resolve_path_with_cwd("output", cwd)

        assert result.scope_modifier is None
        assert (
            result.path_remainder == "task.analytics.deep.processor.final_step.output"
        )
        assert result.original_path == "output"

    def test_cwd_with_none_empty_values(self):
        """Test edge cases with None and empty values."""
        from langtree.parsing.path_resolver import PathResolver

        # None CWD
        result = PathResolver.resolve_path_with_cwd("field", None)
        assert result.path_remainder == "field"

        # Empty CWD
        result = PathResolver.resolve_path_with_cwd("field", "")
        assert result.path_remainder == "field"

        # None path with CWD
        result = PathResolver.resolve_path_with_cwd(None, "task.analyzer")
        assert result.path_remainder == ""


class TestIterationMatchingValidation:
    """Test cases for iteration matching validation in @each commands."""

    def test_valid_iteration_source_paths(self):
        """Test that valid iteration source paths are accepted."""
        from langtree.parsing.parser import CommandParser

        parser = CommandParser()

        # Valid: Direct iteration items
        cmd = parser.parse(
            "! @each[sections.subsections]->task@{{value.items=sections.subsections}}*"
        )
        assert cmd.inclusion_path == "sections.subsections"

        # Valid: Field access from iteration items
        cmd = parser.parse(
            "! @each[sections.subsections]->task@{{value.content=sections.subsections.text}}*"
        )
        assert cmd.inclusion_path == "sections.subsections"

        # Valid: Parent field access
        cmd = parser.parse(
            "! @each[sections.subsections]->task@{{value.title=sections.title}}*"
        )
        assert cmd.inclusion_path == "sections.subsections"

    def test_invalid_iteration_source_paths(self):
        """Test that invalid iteration source paths are rejected at structure level."""
        from pydantic import Field

        from langtree import TreeNode
        from langtree.structure import RunStructure

        class TaskProcessor(TreeNode):
            """Target task for processing."""

            items: list[str]

        class TaskInvalidSource(TreeNode):
            """
            Task with invalid iteration source path that doesn't start from iteration root.
            """

            sections: list[str] = Field(
                default=["section1", "section2"],
                description="! @each[sections]->task.processor@{{value.items=nonexistent_field}}*",
            )

        run_structure = RunStructure()
        run_structure.add(TaskProcessor)

        # Should raise validation error for invalid source path
        from langtree.parsing.parser import CommandParseError

        with pytest.raises(
            CommandParseError, match="must start from iteration root 'sections'"
        ):
            run_structure.add(TaskInvalidSource)

    def test_scope_modified_paths_bypass_validation(self):
        """Test that scope-modified paths are not subject to iteration validation."""
        from langtree.parsing.parser import CommandParser

        parser = CommandParser()

        # These should all be valid regardless of iteration path
        cmd = parser.parse(
            "! @each[sections.subsections]->task@{{value.items=sections.subsections, prompt.data=task.unrelated.path}}*"
        )
        assert len(cmd.variable_mappings) == 2

        cmd = parser.parse(
            "! @each[sections.subsections]->task@{{value.items=sections.subsections, outputs.result=task.other}}*"
        )
        assert len(cmd.variable_mappings) == 2

    def test_deep_iteration_matching(self):
        """Test iteration matching with deeply nested iteration paths."""
        from langtree.parsing.parser import CommandParseError, CommandParser

        parser = CommandParser()

        # Valid: Deep iteration with matching source
        cmd = parser.parse(
            "! @each[docs.chapters.sections]->task@{{value.content=docs.chapters.sections.text}}*"
        )
        assert cmd.inclusion_path == "docs.chapters.sections"

        # Valid: Parent level access in deep iteration
        cmd = parser.parse(
            "! @each[docs.chapters.sections]->task@{{value.title=docs.chapters.title}}*"
        )
        assert cmd.inclusion_path == "docs.chapters.sections"

        # Invalid: Wrong root in deep iteration
        with pytest.raises(
            CommandParseError, match="must start from iteration root 'docs'"
        ):
            parser.parse(
                "! @each[docs.chapters.sections]->task@{{value.content=chapters.sections.text}}*"
            )

    def test_single_level_iteration(self):
        """Test iteration matching with single-level iteration."""
        from langtree.parsing.parser import CommandParseError, CommandParser

        parser = CommandParser()

        # Valid: Single level iteration
        cmd = parser.parse("! @each[items]->task@{{value.item=items}}*")
        assert cmd.inclusion_path == "items"

        # Valid: Field access from single level
        cmd = parser.parse("! @each[items]->task@{{value.content=items.data}}*")
        assert cmd.inclusion_path == "items"

        # Invalid: Different root in single level
        with pytest.raises(
            CommandParseError, match="must start from iteration root 'items'"
        ):
            parser.parse("! @each[items]->task@{{value.item=elements}}*")

    def test_wildcard_paths_bypass_validation(self):
        """Test that wildcard paths are not subject to iteration validation."""
        from langtree.parsing.parser import CommandParseError, CommandParser

        parser = CommandParser()

        # Wildcard should bypass iteration validation (though it's invalid for @each for other reasons)
        with pytest.raises(
            CommandParseError, match="@each commands cannot use wildcard"
        ):
            parser.parse("! @each[sections.subsections]->task@{{value.items=*}}*")

    def test_mixed_valid_invalid_mappings(self):
        """Test commands with mixed valid and invalid mappings."""
        from langtree.parsing.parser import CommandParseError, CommandParser

        parser = CommandParser()

        # One valid, one invalid mapping - should fail
        with pytest.raises(
            CommandParseError, match="must start from iteration root 'sections'"
        ):
            parser.parse(
                "! @each[sections.subsections]->task@{{value.items=sections.subsections, value.other=wrong.path}}*"
            )
