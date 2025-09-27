"""
Tests for DPCL prompt utility functions.

Focus Areas:
1. Naming convention validation for PromptTreeNode classes
2. Command extraction and text processing
"""

import pytest

from langtree.prompt import PromptTreeNode
from langtree.prompt.utils import extract_commands, get_root_tag


class TestNamingConventionValidation:
    """Test that PromptTreeNode classes follow strict CamelCase and Task prefix naming conventions."""

    def test_non_camelcase_nodes_should_fail(self):
        """Non-CamelCase class names should be rejected."""

        # Snake case should fail
        class task_early(PromptTreeNode):  # noqa: N801
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

        class taskEarly(PromptTreeNode):  # noqa: N801
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

        class Task_early(PromptTreeNode):  # noqa: N801
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

        class Tasks(PromptTreeNode):
            """Task class with invalid 'Tasks' prefix - should be rejected."""

            pass

        with pytest.raises(ValueError) as exc_info:
            get_root_tag(Tasks)

        error_msg = str(exc_info.value).lower()
        assert "task" in error_msg and ("prefix" in error_msg or "invalid" in error_msg)

    def test_invalid_task_prefix_lowercase_after_task_should_fail(self):
        """Task prefix with lowercase letter after 'Task' should be rejected."""

        class Taskx(PromptTreeNode):
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

        class task(PromptTreeNode):  # noqa: N801
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

        class TaskEarly(PromptTreeNode):
            """Properly named Task class."""

            pass

        class TaskDocumentProcessor(PromptTreeNode):
            """Another properly named Task class."""

            pass

        class TaskProcessor(PromptTreeNode):
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

        class TaskA(PromptTreeNode):
            """Task with single capital letter."""

            pass

        result = get_root_tag(TaskA)
        assert result == "task.a"

    def test_edge_case_task_with_numbers_should_pass(self):
        """Task with numbers in CamelCase should be accepted."""

        class Task2Analysis(PromptTreeNode):
            """Task with numbers in name."""

            pass

        result = get_root_tag(Task2Analysis)
        assert result == "task.2_analysis"


class TestCommandExtraction:
    """Test command extraction functionality."""

    def test_extract_commands_basic(self):
        """Test basic command extraction."""
        content = """! command1
        ! command2

        Regular text here"""

        commands, clean_content = extract_commands(content)

        assert len(commands) == 2
        assert "command1" in commands[0]
        assert "command2" in commands[1]
        assert clean_content.strip() == "Regular text here"

    def test_extract_commands_empty_content(self):
        """Test command extraction with empty content."""
        commands, clean_content = extract_commands("")

        assert commands == []
        assert clean_content == ""

    def test_extract_commands_no_commands(self):
        """Test command extraction with no commands."""
        content = "Just regular text"

        commands, clean_content = extract_commands(content)

        assert commands == []
        assert clean_content.strip() == "Just regular text"
