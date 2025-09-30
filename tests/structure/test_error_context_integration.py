"""
Simple test to verify ErrorContext is wired up.

Test one specific error path end-to-end.
"""

import pytest

from langtree import TreeNode
from langtree.exceptions import FieldValidationError
from langtree.exceptions.core import ErrorLevel
from langtree.structure import RunStructure


def test_task_target_completeness_error_has_context():
    """Test that incomplete task target error includes context."""

    class TaskSource(TreeNode):
        """
        ! @all->task@{{value=*}}
        """

        value: str

    structure = RunStructure(error_level=ErrorLevel.DEVELOPER)

    # This should trigger _validate_task_target_completeness which we updated
    with pytest.raises(FieldValidationError) as exc_info:
        structure.add(TaskSource)

    error_msg = str(exc_info.value)

    print(f"\n\nError message:\n{error_msg}\n\n")

    # Should contain error details
    assert "task" in error_msg
    assert "incomplete" in error_msg.lower()

    # Should contain line number (the command is on line 1 of docstring)
    assert "line" in error_msg.lower()

    # Developer level should show file information
    assert ".py" in error_msg
