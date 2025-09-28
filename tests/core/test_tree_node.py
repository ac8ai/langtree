"""
Tests for TreeNode core functionality.

Focus Areas:
1. Naming convention validation for TreeNode classes
2. TreeNode base class behavior
"""

import pytest

from langtree import TreeNode
from langtree.templates.utils import get_root_tag


class TestNamingConventionValidation:
    """Test that TreeNode classes follow strict CamelCase and Task prefix naming conventions."""

    def test_non_camelcase_nodes_should_fail(self):
        """Non-CamelCase class names should be rejected."""

        # Snake case should fail
        class task_early(TreeNode):  # noqa: N801
            """Task with snake_case name - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(task_early)

        error_msg = str(exc_info.value).lower()
        assert (
            "camelcase" in error_msg
            or "naming convention" in error_msg
            or "invalid" in error_msg
        )

    def test_camelcase_without_capital_first_letter_should_fail(self):
        """camelCase (lowercase first letter) should be rejected."""

        class taskEarly(TreeNode):  # noqa: N801
            """Task with camelCase name - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(taskEarly)

        error_msg = str(exc_info.value).lower()
        assert (
            "camelcase" in error_msg
            or "naming convention" in error_msg
            or "invalid" in error_msg
        )

    def test_mixed_case_with_underscores_should_fail(self):
        """Mixed case with underscores should be rejected."""

        class Task_early(TreeNode):  # noqa: N801
            """Task with mixed case - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(Task_early)

        error_msg = str(exc_info.value).lower()
        assert (
            "camelcase" in error_msg
            or "naming convention" in error_msg
            or "invalid" in error_msg
        )

    def test_invalid_task_prefix_tasks_should_fail(self):
        """Task prefix ending with 's' should be rejected."""

        class Tasks(TreeNode):
            """Task class with invalid 'Tasks' prefix - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(Tasks)

        error_msg = str(exc_info.value).lower()
        assert "task" in error_msg and ("prefix" in error_msg or "invalid" in error_msg)

    def test_invalid_task_prefix_lowercase_after_task_should_fail(self):
        """Task prefix with lowercase letter after 'Task' should be rejected."""

        class Taskx(TreeNode):
            """Task class with lowercase 'x' after Task - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(Taskx)

        error_msg = str(exc_info.value).lower()
        assert "task" in error_msg and (
            "prefix" in error_msg or "invalid" in error_msg or "capital" in error_msg
        )

    def test_lowercase_task_should_fail(self):
        """All lowercase 'task' should be rejected."""

        class task(TreeNode):  # noqa: N801
            """All lowercase task - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(task)

        error_msg = str(exc_info.value).lower()
        assert (
            "camelcase" in error_msg
            or "naming convention" in error_msg
            or "invalid" in error_msg
        )

    def test_valid_task_classes_should_pass(self):
        """Properly formatted Task classes should be accepted."""

        class TaskEarly(TreeNode):
            """Properly named Task class."""

            pass

        class TaskDocumentProcessor(TreeNode):
            """Another properly named Task class."""

            pass

        class TaskProcessor(TreeNode):
            """Third properly named Task class."""

            pass

        # Should not raise errors for valid naming
        result1 = get_root_tag(TaskEarly)
        result2 = get_root_tag(TaskDocumentProcessor)
        result3 = get_root_tag(TaskProcessor)

        # Verify correct tag generation
        assert result1 == "task.early"
        assert result2 == "task.document_processor"
        assert result3 == "task.processor"

    def test_edge_case_task_with_single_letter_should_pass(self):
        """Task with single capital letter after 'Task' should be accepted."""

        class TaskA(TreeNode):
            """Task with single capital letter."""

            pass

        result = get_root_tag(TaskA)
        assert result == "task.a"

    def test_edge_case_task_with_numbers_should_pass(self):
        """Task with numbers in CamelCase should be accepted."""

        class Task2Analysis(TreeNode):
            """Task with numbers in name."""

            pass

        result = get_root_tag(Task2Analysis)
        assert result == "task.2_analysis"
