"""
TDD Tests for Runtime Variable Validation - Define Expected Behavior

These tests define the correct runtime variable validation behavior BEFORE implementation.
Tests should FAIL initially, then we implement code to make them PASS.

Focus Areas:
1. Undefined variable detection with proper error messages
2. Assembly variable integration validation
3. Field variable validation
4. Reserved variable handling
"""

import pytest

from langtree.prompt import RunStructure, TreeNode
from langtree.prompt.exceptions import RuntimeVariableError
from langtree.prompt.resolution import resolve_runtime_variables


class TestRuntimeVariableValidationBehavior:
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

        # Should raise detailed error for undefined field variable when validation enabled
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Content: {undefined_field}", structure, node, validate=True
            )

        error_msg = str(exc_info.value)
        assert "undefined_field" in error_msg
        assert "undefined" in error_msg.lower()
        assert "available" in error_msg.lower()
        # Should list available fields in error message
        assert "valid_field" in error_msg
        assert "another_field" in error_msg

    def test_assembly_variable_should_be_rejected_in_runtime_validation(self):
        """Assembly variables should be rejected in runtime contexts per assembly variable separation principle."""

        class TaskWithAssemblyVar(TreeNode):
            """! assembly_var="test_value" """

            field_var: str = "field_value"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)
        node = structure.get_node("task.with_assembly_var")

        # Assembly variable should be rejected in runtime contexts
        from langtree.prompt.exceptions import RuntimeVariableError

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Content: {assembly_var}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert "cannot be used in runtime contexts" in error_msg

        # Field variable should still be valid
        result = resolve_runtime_variables("Content: {field_var}", structure, node)
        assert "{prompt__with_assembly_var__field_var}" in result

    def test_reserved_template_variables_should_be_excluded_from_validation(self):
        """Reserved template variables should not trigger validation errors."""

        class TaskWithReserved(TreeNode):
            """
            Template with reserved variables:

            {PROMPT_SUBTREE}

            And also:

            {COLLECTED_CONTEXT}

            """

            field_var: str = "value"

        structure = RunStructure()
        structure.add(TaskWithReserved)
        node = structure.get_node("task.with_reserved")

        # Reserved variables should be left unchanged (no validation error)
        result = resolve_runtime_variables(
            "Content: {PROMPT_SUBTREE} and {COLLECTED_CONTEXT}", structure, node
        )
        assert "{PROMPT_SUBTREE}" in result
        assert "{COLLECTED_CONTEXT}" in result

    def test_malformed_variable_syntax_should_raise_error(self):
        """Malformed variable syntax should raise RuntimeVariableError."""

        class TaskForSyntaxTest(TreeNode):
            field_var: str = "value"

        structure = RunStructure()
        structure.add(TaskForSyntaxTest)
        node = structure.get_node("task.for_syntax_test")

        # Double underscores should be rejected
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Content: {var__with__underscores}", structure, node
            )

        assert "double underscores" in str(exc_info.value).lower()

        # Dots should be rejected
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Content: {var.with.dots}", structure, node)

        assert "dots" in str(exc_info.value).lower()

    def test_mixed_valid_and_invalid_variables(self):
        """Test content with both valid and invalid variables - should fail on first invalid."""

        class TaskMixed(TreeNode):
            valid_field: str = "value"

        structure = RunStructure()
        structure.add(TaskMixed)
        node = structure.get_node("task.mixed")

        # Should fail on first invalid variable encountered when validation enabled
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Valid: {valid_field}, Invalid: {invalid_field}",
                structure,
                node,
                validate=True,
            )

        # Should mention the invalid variable
        assert "invalid_field" in str(exc_info.value)

    def test_deferred_validation_expands_without_checking_existence(self):
        """By default, undefined variables should expand to namespaced form without validation."""

        class TaskWithLimitedContext(TreeNode):
            known_field: str = "value"

        structure = RunStructure()
        structure.add(TaskWithLimitedContext)
        node = structure.get_node("task.with_limited_context")

        # Variables expand to namespaced form - validation deferred to runtime
        content = "Known: {known_field}, Unknown: {unknown_field}"
        expanded = resolve_runtime_variables(content, structure, node, validate=False)
        # Both expand to namespaced form - existence check happens at runtime
        assert (
            expanded
            == "Known: {prompt__with_limited_context__known_field}, Unknown: {prompt__with_limited_context__unknown_field}"
        )
