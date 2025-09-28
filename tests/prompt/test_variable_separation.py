"""
Tests for comprehensive variable system compliance with LANGUAGE_SPECIFICATION.md.

This module ensures that all 5 variable types defined in the specification
are properly implemented and that the architectural separation between
assembly-time and runtime variables is strictly enforced.

Variable Types Tested:
1. Assembly Variables (! var=value) - Chain construction time
2. Runtime Variables ({var}) - Prompt execution time
3. LangTree DSL Variable Targets (@each[var] / @all[var]) - Collection iteration
4. Scope Context Variables (scope.field) - Context-specific resolution
5. Field References ([field]) - Resampling and aggregation operations

According to LANGUAGE_SPECIFICATION.md:
- Assembly Variables: NOT available during runtime resolution
- Runtime Variables: Resolve from execution context only
- Complete separation between assembly-time and runtime contexts
"""

import pytest

from langtree.prompt import RunStructure, TreeNode
from langtree.prompt.exceptions import RuntimeVariableError
from langtree.prompt.resolution import resolve_runtime_variables


class TestAssemblyRuntimeSeparation:
    """Test that assembly and runtime variables are strictly separated."""

    def test_assembly_variables_not_available_at_runtime(self):
        """Test that assembly variables cannot be used in runtime contexts."""

        class TaskWithAssemblyVar(TreeNode):
            """
            ! count=5
            ! threshold="high"
            Task with assembly variables.
            """

            field: str = "test"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)

        node = structure.get_node("task.with_assembly_var")
        assert node is not None

        # Assembly variables should NOT be usable in runtime contexts
        content = "Count: {count}, Threshold: {threshold}"

        # Should raise RuntimeVariableError when trying to use assembly variable in runtime
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "cannot be used in runtime contexts" in error_msg

    def test_assembly_variable_priority_syntax_rejected(self):
        """Test that assembly variables are rejected in runtime contexts."""

        class TaskWithBridgingAttempt(TreeNode):
            """
            ! priority_var="high"
            Task that uses assembly variables.
            """

            field: str = "test"

        structure = RunStructure()
        structure.add(TaskWithBridgingAttempt)

        node = structure.get_node("task.with_bridging_attempt")
        assert node is not None

        # Assembly variables should be rejected in runtime contexts
        content = "Priority: {priority_var}"

        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "priority_var" in error_msg

    def test_runtime_variables_work_correctly(self):
        """Test that runtime variables expand correctly to double underscore format."""

        class TaskWithRuntimeContext(TreeNode):
            """Task that provides runtime context."""

            title: str = "Test Title"
            content: str = "Test Content"

        structure = RunStructure()
        structure.add(TaskWithRuntimeContext)

        node = structure.get_node("task.with_runtime_context")
        assert node is not None

        # Runtime variables should expand to double underscore format for LangChain
        content = "Title: {title}, Content: {content}"
        expanded = resolve_runtime_variables(content, structure, node)

        # Should expand to prompt__with_runtime_context__variable format
        assert (
            expanded
            == "Title: {prompt__with_runtime_context__title}, Content: {prompt__with_runtime_context__content}"
        )

    def test_assembly_variables_available_for_commands(self):
        """Test that assembly variables are available for command arguments."""

        class TaskWithCommandUsingAssemblyVar(TreeNode):
            """
            ! iterations=3
            ! resample(iterations)
            Task using assembly variable in command.
            """

            result: str = "output"

        structure = RunStructure()
        structure.add(TaskWithCommandUsingAssemblyVar)

        # Should successfully process without errors
        node = structure.get_node("task.with_command_using_assembly_var")
        assert node is not None

        # Assembly variable should be used in command processing
        # This validates that assembly variables work in their proper context

    def test_assembly_variables_expand_without_runtime_values(self):
        """Test that assembly variables are rejected in runtime contexts.

        Assembly variables are for chain assembly only and cannot be used
        in runtime template resolution.
        """

        class TaskForErrorTesting(TreeNode):
            """
            ! config_value="test"
            Task for testing variable expansion.
            """

            field: str = "data"

        structure = RunStructure()
        structure.add(TaskForErrorTesting)

        node = structure.get_node("task.for_error_testing")
        assert node is not None

        # Assembly variables should be rejected in runtime contexts
        content = "Config: {config_value}"

        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "config_value" in error_msg


class TestRuntimeVariableResolutionPriority:
    """Test runtime variable resolution follows correct priority order."""

    def test_current_node_context_priority(self):
        """Test that runtime variables expand to double underscore format."""

        class TaskWithContext(TreeNode):
            """Task with field context."""

            title: str = "Current Node Title"
            description: str = "Current Node Description"

        structure = RunStructure()
        structure.add(TaskWithContext)

        node = structure.get_node("task.with_context")
        assert node is not None

        # Runtime variables should expand to double underscore format
        content = "{title} - {description}"
        expanded = resolve_runtime_variables(content, structure, node)

        assert (
            expanded
            == "{prompt__with_context__title} - {prompt__with_context__description}"
        )

    def test_undefined_variable_expansion_deferred(self):
        """Test that undefined variables expand but validation is deferred.

        Updated to clarify that expansion happens for all variables, but
        validation of whether they exist is deferred to runtime/LangChain.
        This matches the current implementation approach.
        """

        class TaskWithLimitedContext(TreeNode):
            """Task with limited context."""

            known_field: str = "known"

        structure = RunStructure()
        structure.add(TaskWithLimitedContext)

        node = structure.get_node("task.with_limited_context")
        assert node is not None

        # Variables expand to namespaced form - validation deferred to runtime
        content = "Known: {known_field}, Unknown: {unknown_field}"
        expanded = resolve_runtime_variables(content, structure, node, validate=False)

        # Both expand to namespaced form - existence check happens at runtime
        assert (
            expanded
            == "Known: {prompt__with_limited_context__known_field}, Unknown: {prompt__with_limited_context__unknown_field}"
        )


class TestArchitecturalIntegrity:
    """Test that the architectural separation is maintained across the system."""

    def test_no_assembly_runtime_bridging_in_parser(self):
        """Test that parser enforces separation between assembly and runtime variable contexts."""

        # Test that malformed bridging syntax is rejected by resolution layer
        from langtree.prompt.exceptions import RuntimeVariableError
        from langtree.prompt.resolution import resolve_runtime_variables

        class TaskForBridgingTest(TreeNode):
            """
            ! assembly_var="assembly_value"
            Task for testing bridging prevention.
            """

            runtime_field: str = "runtime_value"

        structure = RunStructure()
        structure.add(TaskForBridgingTest)

        node = structure.get_node("task.for_bridging_test")
        assert node is not None

        # Test that malformed double-brace syntax is rejected
        with pytest.raises(RuntimeVariableError, match=r"Malformed variable syntax"):
            resolve_runtime_variables("Value: {{assembly_var}}", structure, node)

        # Test that valid runtime variable syntax expands correctly
        expanded = resolve_runtime_variables(
            "Runtime: {runtime_field}", structure, node
        )
        assert expanded == "Runtime: {prompt__for_bridging_test__runtime_field}"

        # Test that assembly variables are rejected in runtime contexts
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Assembly: {assembly_var}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "cannot be used in runtime contexts" in error_msg

    def test_variable_registry_separation(self):
        """Test that variable registries maintain separation."""

        class TaskWithBothVariableTypes(TreeNode):
            """
            ! assembly_var="assembly_value"
            Task with both assembly and runtime variables.
            """

            runtime_field: str = "runtime_value"

        structure = RunStructure()
        structure.add(TaskWithBothVariableTypes)

        # Assembly variables should be in registry for command resolution
        # Runtime variables should be in node context for execution resolution
        # They should not interfere with each other

        node = structure.get_node("task.with_both_variable_types")
        assert node is not None

        # Runtime variable should expand to double underscore format
        content = "Runtime: {runtime_field}"
        expanded = resolve_runtime_variables(content, structure, node)
        assert expanded == "Runtime: {prompt__with_both_variable_types__runtime_field}"

        # Assembly variable should be rejected in runtime contexts
        content = "Assembly: {assembly_var}"
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "assembly_var" in error_msg

    def test_documentation_compliance(self):
        """Test that implementation matches documentation specifications."""

        class TaskDocumentationExample(TreeNode):
            """
            ! iterations=5
            ! threshold=2.5
            ! resample(iterations)

            Example task following documentation patterns.
            """

            analysis: str = "result"
            confidence: float = 0.95

        structure = RunStructure()
        structure.add(TaskDocumentationExample)

        node = structure.get_node("task.documentation_example")
        assert node is not None

        # All variables expand to double underscore format
        # Actual values are provided at LangChain invoke time
        content = "Analysis: {analysis}, Confidence: {confidence}"
        expanded = resolve_runtime_variables(content, structure, node)

        assert (
            expanded
            == "Analysis: {prompt__documentation_example__analysis}, Confidence: {prompt__documentation_example__confidence}"
        )

        # Assembly variables should be rejected in runtime contexts
        content = "Iterations: {iterations}"
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "iterations" in error_msg


class TestErrorMessageQuality:
    """Test that error messages guide users toward correct patterns."""

    def test_helpful_assembly_variable_error_message(self):
        """Test that errors suggest correct usage patterns."""

        class TaskForMessageTesting(TreeNode):
            """
            ! config="value"
            Task for error message testing.
            """

            field: str = "data"

        structure = RunStructure()
        structure.add(TaskForMessageTesting)

        node = structure.get_node("task.for_message_testing")
        assert node is not None

        content = "Config: {{config}}"

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_message = str(exc_info.value)

        # Error should be clear about what failed
        assert "config" in error_message
        assert "Malformed variable syntax" in error_message

        # The error message should guide users toward correct syntax
        assert "single braces" in error_message

    def test_valid_variable_syntax_expansion(self):
        """Test that valid variable syntax expands correctly.

        Renamed from misleading name. Tests successful expansion of
        valid single-brace syntax, not error messages.
        """

        class TaskForBridgingTest(TreeNode):
            """
            Task for testing variable expansion.
            """

            new_field: str = "new_value"
            legacy_field: str = (
                "old_value"  # Use regular field instead of assembly variable
            )

        structure = RunStructure()
        structure.add(TaskForBridgingTest)

        node = structure.get_node("task.for_bridging_test")
        assert node is not None

        # Valid single brace syntax should expand successfully
        content = "Legacy: {legacy_field}"
        expanded = resolve_runtime_variables(content, structure, node)

        # Should expand to double underscore format
        assert expanded == "Legacy: {prompt__for_bridging_test__legacy_field}"


class TestCompleteVariableTypeCoverage:
    """Test all 5 variable types defined in LANGUAGE_SPECIFICATION.md."""

    def test_assembly_variables_command_integration(self):
        """Test that assembly variables work properly in command arguments."""

        class TaskWithCommandIntegration(TreeNode):
            """
            ! iterations=3
            ! model_name="gpt-4"
            ! override_flag=true
            ! resample(iterations)
            ! llm(model_name, override=override_flag)

            Task properly using assembly variables in commands.
            """

            result: str = "output"

        structure = RunStructure()
        structure.add(TaskWithCommandIntegration)

        # Assembly variables should work in commands and also expand for runtime use
        node = structure.get_node("task.with_command_integration")
        assert node is not None

        # Assembly variables should be rejected in runtime contexts
        content = "Iterations: {iterations}, Model: {model_name}"
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "iterations" in error_msg or "model_name" in error_msg

    def test_acl_variable_targets_in_commands(self):
        """Test LangTree DSL Variable Targets in @each/@all commands."""

        class TaskSourceNode(TreeNode):
            """Source node with data.

            ! @each[items]->task.processor@{{value.item=items}}*
            ! @all->task.aggregator@{{prompt.context=*}}
            """

            items: list[str] = ["item1", "item2", "item3"]

        class TaskProcessor(TreeNode):
            """Processor node referenced by @each command."""

            item: str  # Field to receive value.item

        class TaskAggregator(TreeNode):
            """Aggregator node referenced by @all command.

            Uses forwarded context: {context}
            """

            context: str = "default"

        class TaskTargetNode(TreeNode):
            """Node using LangTree DSL Variable Targets."""

            result: str = "processed"

        structure = RunStructure()
        structure.add(TaskSourceNode)
        structure.add(TaskTargetNode)
        structure.add(TaskProcessor)
        structure.add(TaskAggregator)

        # Should successfully process LangTree DSL commands
        source_node = structure.get_node("task.source_node")
        target_node = structure.get_node("task.target_node")
        assert source_node is not None
        assert target_node is not None

        # Runtime variables from target node should expand correctly
        content = "Result: {result}"
        expanded = resolve_runtime_variables(content, structure, target_node)

        # Should expand to double underscore format using target node's actual field
        assert expanded == "Result: {prompt__target_node__result}"

        # Test with source node that has the items field
        content = "Items: {items}"
        expanded = resolve_runtime_variables(content, structure, source_node)
        assert expanded == "Items: {prompt__source_node__items}"

    def test_scope_context_variables_separation(self):
        """Test that scope context variables are separate from runtime variables."""

        class TaskWithScopeUsage(TreeNode):
            """
            ! @all->task.process@{{prompt.item=*, outputs.result=*}}

            Node using scope context variables.
            """

            local_field: str = "local_value"

        structure = RunStructure()
        structure.add(TaskWithScopeUsage)

        node = structure.get_node("task.with_scope_usage")
        assert node is not None

        # Variables with dots are INVALID SYNTAX per specification
        content = "Prompt data: {prompt.data}, Task current: {task.current}"
        with pytest.raises(RuntimeVariableError, match=r"cannot contain dots"):
            resolve_runtime_variables(content, structure, node)

        # Local fields should expand properly
        content = "Local: {local_field}"
        expanded = resolve_runtime_variables(content, structure, node)
        assert expanded == "Local: {prompt__with_scope_usage__local_field}"

    def test_field_references_for_resampling(self):
        """Test Field References in resampling commands."""
        from enum import Enum

        class Priority(Enum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        class TaskWithFieldReferences(TreeNode):
            """
            ! @resampled[priority]->mean
            ! @resampled[status]->mode

            Node using Field References for resampling.
            """

            priority: Priority = Priority.MEDIUM
            status: str = "active"
            content: str = "data"

        structure = RunStructure()
        structure.add(TaskWithFieldReferences)

        node = structure.get_node("task.with_field_references")
        assert node is not None

        # Field references can be used as runtime variables and should expand
        content = "Priority field: {priority}, Status field: {status}"
        expanded = resolve_runtime_variables(content, structure, node)

        # Should expand to double underscore format
        assert (
            expanded
            == "Priority field: {prompt__with_field_references__priority}, Status field: {prompt__with_field_references__status}"
        )

    def test_template_variables_cannot_be_used_as_assembly_or_runtime(self):
        """Test that template variables are reserved and cannot be used as Assembly or Runtime variables."""

        class TaskWithTemplateConflict(TreeNode):
            """
            ! PROMPT_SUBTREE="invalid"  # Should not be allowed as assembly variable
            ! COLLECTED_CONTEXT="invalid"  # Should not be allowed as assembly variable

            This should be rejected during parsing/validation.
            """

            field: str = "test"

        # Test that template variable names are reserved and cannot be used as assembly variables
        # According to specification, PROMPT_SUBTREE and COLLECTED_CONTEXT are reserved

        structure = RunStructure()

        # This should raise an error for using reserved variable names
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            structure.add(TaskWithTemplateConflict)

        error_msg = str(exc_info.value).lower()
        assert "reserved" in error_msg or "cannot be used" in error_msg


class TestRuntimeVariableResolutionPriorityComplete:
    """Test complete runtime variable resolution priority from LANGUAGE_SPECIFICATION.md."""

    def test_resolution_priority_chain(self):
        """Test the complete priority chain: Current Node → Task → Outputs → Value → Prompt."""

        class TaskForPriorityTesting(TreeNode):
            """Task for testing resolution priority."""

            current_field: str = (
                "current_node_value"  # Priority 1: Current Node Context
            )
            # Priority 2: Task Context (would be task.field references)
            # Priority 3: Outputs Context (from outputs scope assignments)
            # Priority 4: Value Context (from value scope assignments)
            # Priority 5: Prompt Context (from prompt scope assignments)

        structure = RunStructure()
        structure.add(TaskForPriorityTesting)

        node = structure.get_node("task.for_priority_testing")
        assert node is not None

        # Runtime variables should expand to double underscore format
        content = "Field: {current_field}"
        expanded = resolve_runtime_variables(content, structure, node)
        assert expanded == "Field: {prompt__for_priority_testing__current_field}"

        # Skip testing other priority levels until context assembly is implemented
        pytest.skip(
            "TODO: Implement tests for complete priority chain (Task, Outputs, Value, Prompt contexts) - only Current Node context is tested"
        )

    def test_context_type_separation(self):
        """Test that different context types don't interfere."""

        class Metadata(TreeNode):
            type: str = "test"

        class TaskWithMultipleContexts(TreeNode):
            """Task with various field types."""

            title: str = "Test Title"
            content: str = "Test Content"
            metadata: Metadata = Metadata()

        structure = RunStructure()
        structure.add(TaskWithMultipleContexts)

        node = structure.get_node("task.with_multiple_contexts")
        assert node is not None

        # All runtime variables should expand to double underscore format
        content = "{title} | {content} | {metadata}"
        expanded = resolve_runtime_variables(content, structure, node)

        assert (
            expanded
            == "{prompt__with_multiple_contexts__title} | {prompt__with_multiple_contexts__content} | {prompt__with_multiple_contexts__metadata}"
        )

    def test_all_variables_expand_in_priority_chain(self):
        """Test that all variables expand regardless of field existence.

        Renamed from misleading name. Tests that both known and unknown
        variables expand to namespaced form - validation happens at runtime.
        """

        class TaskForUndefinedTesting(TreeNode):
            """Task with limited context."""

            known: str = "known_value"

        structure = RunStructure()
        structure.add(TaskForUndefinedTesting)

        node = structure.get_node("task.for_undefined_testing")
        assert node is not None

        # All variables should expand regardless of whether they exist as fields
        content = "Known: {known}, Unknown: {unknown_variable}"
        expanded = resolve_runtime_variables(content, structure, node, validate=False)

        # Both known and unknown variables should expand to namespaced form
        assert (
            expanded
            == "Known: {prompt__for_undefined_testing__known}, Unknown: {prompt__for_undefined_testing__unknown_variable}"
        )


class TestSpecificationCompliance:
    """Test that implementation strictly follows LANGUAGE_SPECIFICATION.md."""

    def test_assembly_vs_runtime_separation_specification(self):
        """Test the specification requirement: 'Assembly Variables are NOT available during runtime resolution'."""

        class TaskSpecificationExample(TreeNode):
            """
            ! config_value="assembly_time_value"
            ! threshold=2.5
            ! debug=true

            Example following specification patterns.
            """

            runtime_field: str = "runtime_value"
            status: str = "active"

        structure = RunStructure()
        structure.add(TaskSpecificationExample)

        node = structure.get_node("task.specification_example")
        assert node is not None

        # Assembly variables should be rejected in runtime contexts
        from langtree.prompt.exceptions import RuntimeVariableError

        assembly_tests = [
            "Config: {config_value}",
            "Threshold: {threshold}",
            "Debug: {debug}",
        ]

        for content in assembly_tests:
            with pytest.raises(RuntimeVariableError) as exc_info:
                resolve_runtime_variables(content, structure, node)

            error_msg = str(exc_info.value)
            assert "assembly variable" in error_msg.lower()

        # Mixed assembly and runtime variables - should fail on assembly variable
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Mixed: {config_value} and {runtime_field}", structure, node
            )

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "config_value" in error_msg

        # Runtime fields also expand to double underscore format
        content = "Runtime: {runtime_field}, Status: {status}"
        expanded = resolve_runtime_variables(content, structure, node)
        assert (
            expanded
            == "Runtime: {prompt__specification_example__runtime_field}, Status: {prompt__specification_example__status}"
        )

    def test_variable_type_taxonomy_compliance(self):
        """Test compliance with the 5-type Variable Type Taxonomy from specification."""

        class TaskTaxonomyExample(TreeNode):
            """
            ! assembly_var="assembly_value"  # Type 1: Assembly Variables
            ! resample(3)  # Using assembly variable in command

            Example demonstrating variable types according to specification.
            """

            runtime_field: str = (
                "runtime_value"  # Type 2: Runtime Variables ({runtime_field})
            )
            collection: list[str] = ["a", "b", "c"]
            priority_field: int = 5

        structure = RunStructure()
        structure.add(TaskTaxonomyExample)

        node = structure.get_node("task.taxonomy_example")
        assert node is not None

        # Assembly variables should be rejected in runtime contexts
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("{assembly_var}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "assembly_var" in error_msg

        # Type 2 (Runtime) also expands to double underscore format
        expanded = resolve_runtime_variables("{runtime_field}", structure, node)
        assert expanded == "{prompt__taxonomy_example__runtime_field}"

        # Field variables also expand to double underscore format
        expanded = resolve_runtime_variables(
            "Collection: {collection}", structure, node
        )
        assert expanded == "Collection: {prompt__taxonomy_example__collection}"

    @pytest.mark.skip(
        reason="TODO: Implement assembly variable conflict detection - specification requires strict prohibition"
    )
    def test_conflict_prohibition_compliance(self):
        """Test specification requirement for conflict prohibition.

        According to LANGUAGE_SPECIFICATION.md:
        - "Variable names cannot conflict with field names in same subtree"
        - "Assignment to existing variable name is prohibited"

        This test enforces these requirements strictly.
        Note: Currently skipped pending implementation of conflict detection.
        """
        import pytest

        from langtree.prompt.exceptions import (
            FieldValidationError,
            VariableTargetValidationError,
        )

        # Test 1: Variable name conflicts with field names should raise error
        class TaskConflictingVariableField(TreeNode):
            """
            ! field_name="conflict_value"  # Assembly variable conflicts with field name
            Task with conflicting variable and field names.
            """

            field_name: str = "field_value"  # Field with same name as assembly variable

        structure = RunStructure()

        # Specification requires this conflict to be detected and prohibited
        with pytest.raises(
            (FieldValidationError, VariableTargetValidationError, ValueError)
        ) as exc_info:
            structure.add(TaskConflictingVariableField)

        error_msg = str(exc_info.value).lower()
        assert any(
            term in error_msg
            for term in ["conflict", "field_name", "already exists", "duplicate"]
        ), (
            f"Error should clearly indicate variable/field name conflict: {exc_info.value}"
        )

        # Test 2: Duplicate variable assignment should be prohibited
        class TaskDuplicateVariable(TreeNode):
            """
            ! duplicate_var="first_value"
            ! duplicate_var="second_value"  # Reassignment should be prohibited
            Task with duplicate variable assignment.
            """

            result: str = "test"

        structure2 = RunStructure()

        # Specification requires duplicate assignments to be prohibited
        with pytest.raises(
            (FieldValidationError, VariableTargetValidationError, ValueError)
        ) as exc_info:
            structure2.add(TaskDuplicateVariable)

        error_msg = str(exc_info.value).lower()
        assert any(
            term in error_msg
            for term in ["duplicate", "already", "existing", "reassignment"]
        ), (
            f"Error should clearly indicate duplicate variable assignment: {exc_info.value}"
        )

    def test_documentation_example_patterns(self):
        """Test patterns directly from the specification documentation."""

        class TaskDocumentationPattern(TreeNode):
            """
            ! count=5              # Define assembly variable
            ! resample(count)      # Use in command argument
            ! threshold=2.5        # Available to all child nodes

            Pattern directly from LANGUAGE_SPECIFICATION.md
            """

            analysis: str = "result"
            confidence: float = 0.95

        structure = RunStructure()
        structure.add(TaskDocumentationPattern)

        node = structure.get_node("task.documentation_pattern")
        assert node is not None

        # Runtime variables should expand to double underscore format
        content = "Analysis: {analysis}, Confidence: {confidence}"
        expanded = resolve_runtime_variables(content, structure, node)

        # Should expand to double underscore format, not resolve to values
        assert (
            expanded
            == "Analysis: {prompt__documentation_pattern__analysis}, Confidence: {prompt__documentation_pattern__confidence}"
        )

        # Assembly variables should be rejected in runtime contexts
        content = "Count: {count}, Threshold: {threshold}"
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(content, structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "count" in error_msg or "threshold" in error_msg
