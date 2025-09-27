"""
Tests for strict assembly variable separation from runtime variables.

Assembly variables are ONLY for use in `! command` arguments and must NEVER
be accessible in runtime template contexts. This enforces the separation
principle outlined in LANGUAGE_SPECIFICATION.md.

Expected Behavior:
- Assembly variables (defined with `! var=value`) can only be used in command arguments
- Runtime template usage like `{assembly_var}` should raise RuntimeVariableError
- Assembly variables are never passed to chain.invoke() runtime execution
"""

import pytest

from langtree.prompt import PromptTreeNode, RunStructure
from langtree.prompt.exceptions import RuntimeVariableError
from langtree.prompt.resolution import resolve_runtime_variables


class TestAssemblyVariableSeparation:
    """Test that assembly variables are strictly separated from runtime context."""

    def test_assembly_variable_in_runtime_template_should_fail(self):
        """Assembly variables used in runtime templates should raise errors."""

        class TaskWithAssemblyVar(PromptTreeNode):
            """! count=5
            ! threshold=10.5

            Template with assembly variables: {count} items above {threshold}
            """

            field_var: str = "default"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)
        node = structure.get_node("task.with_assembly_var")

        # Assembly variable 'count' in runtime template should fail
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Count: {count}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert (
            "cannot be used in runtime contexts" in error_msg
            or "only valid in command arguments" in error_msg
        )

    def test_assembly_variable_in_runtime_template_should_fail_threshold(self):
        """Second assembly variable should also fail in runtime template."""

        class TaskWithAssemblyVar(PromptTreeNode):
            """! count=5
            ! threshold=10.5

            Template with assembly variables: {count} items above {threshold}
            """

            field_var: str = "default"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)
        node = structure.get_node("task.with_assembly_var")

        # Assembly variable 'threshold' in runtime template should fail
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Threshold: {threshold}", structure, node)

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()
        assert (
            "cannot be used in runtime contexts" in error_msg
            or "only valid in command arguments" in error_msg
        )

    def test_multiple_assembly_variables_in_runtime_template_should_fail(self):
        """Multiple assembly variables in single template should fail."""

        class TaskWithAssemblyVars(PromptTreeNode):
            """! iterations=3
            ! model_name="gpt-4"
            ! debug_mode=true

            Template: Run {iterations} times with {model_name} in {debug_mode}
            """

            field_var: str = "default"

        structure = RunStructure()
        structure.add(TaskWithAssemblyVars)
        node = structure.get_node("task.with_assembly_vars")

        # All assembly variables should fail - test first one encountered
        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables(
                "Run {iterations} times with {model_name}", structure, node
            )

        error_msg = str(exc_info.value)
        assert "assembly variable" in error_msg.lower()

    def test_field_variables_should_still_work(self):
        """Field variables should continue to work normally."""

        class TaskWithMixedVars(PromptTreeNode):
            """! assembly_var=42

            Template with field variable: {field_var}
            """

            field_var: str = "default"

        structure = RunStructure()
        structure.add(TaskWithMixedVars)
        node = structure.get_node("task.with_mixed_vars")

        # Field variable should expand normally
        expanded = resolve_runtime_variables("Field: {field_var}", structure, node)
        assert expanded == "Field: {prompt__with_mixed_vars__field_var}"

    def test_assembly_variable_error_message_clarity(self):
        """Assembly variable error should provide clear guidance."""

        class TaskWithAssemblyVar(PromptTreeNode):
            """! config_value=123

            Template: Configuration: {config_value}
            """

            pass

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)
        node = structure.get_node("task.with_assembly_var")

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("{config_value}", structure, node)

        error_msg = str(exc_info.value)
        # Error should mention the specific variable name
        assert "config_value" in error_msg
        # Error should explain assembly variables are for commands only
        assert "assembly variable" in error_msg.lower() and (
            "command" in error_msg.lower() or "!" in error_msg
        )

    def test_assembly_variables_in_commands_should_still_work(self):
        """Assembly variables should continue to work in command contexts."""

        class TaskWithCommandUsage(PromptTreeNode):
            """! iterations=5
            ! resample(iterations)

            This is the template content.
            """

            result: str = "default"

        structure = RunStructure()
        # This should not raise any errors - assembly variables in commands are valid
        structure.add(TaskWithCommandUsage)

        node = structure.get_node("task.with_command_usage")
        assert node is not None

        # But using assembly variable in template should still fail
        with pytest.raises(RuntimeVariableError):
            resolve_runtime_variables("Iterations: {iterations}", structure, node)

    def test_nonexistent_variable_still_gives_appropriate_error(self):
        """Non-existent variables should still give the usual error."""

        class TaskWithoutVariables(PromptTreeNode):
            """Simple template."""

            field_var: str = "default"

        structure = RunStructure()
        structure.add(TaskWithoutVariables)
        node = structure.get_node("task.without_variables")

        with pytest.raises(RuntimeVariableError) as exc_info:
            resolve_runtime_variables("Unknown: {unknown_var}", structure, node)

        error_msg = str(exc_info.value)
        assert "undefined" in error_msg.lower()
        # Should not mention assembly variables since none exist
        assert "assembly variable" not in error_msg.lower()


class TestAssemblyVariableValidUsage:
    """Test that assembly variables work correctly in their valid contexts."""

    def test_assembly_variables_available_for_commands(self):
        """Assembly variables should be accessible for command validation."""

        class TaskWithAssemblyVars(PromptTreeNode):
            """! batch_size=10
            ! timeout=30
            ! resample(batch_size)

            Process items in batches.
            """

            items: list[str] = ["item1", "item2"]

        structure = RunStructure()
        # Should not raise errors - assembly variables used in commands
        structure.add(TaskWithAssemblyVars)

        node = structure.get_node("task.with_assembly_vars")
        assert node is not None

    def test_assembly_variable_registry_integration(self):
        """Assembly variables should be properly tracked in registry."""

        class TaskWithAssemblyVar(PromptTreeNode):
            """! setting="value"

            Template content.
            """

            pass

        structure = RunStructure()
        structure.add(TaskWithAssemblyVar)

        # Assembly variable should be in registry
        node_name = "task.with_assembly_var"
        if hasattr(structure, "_assembly_variable_registry"):
            assembly_vars = (
                structure._assembly_variable_registry.get_variables_for_node(node_name)
            )
            assembly_var_names = [var.name for var in assembly_vars]
            assert "setting" in assembly_var_names
