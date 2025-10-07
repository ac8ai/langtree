"""
Test for COLLECTED_CONTEXT bug - previous_values not appearing in resolved prompts.

Bug: When calling node.get_prompt(previous_values={...}), the COLLECTED_CONTEXT
template variable resolves to an empty string instead of showing the previous values.
"""

from pydantic import Field

from langtree.core.tree_node import TreeNode
from langtree.structure.builder import RunStructure


class TaskWithContext(TreeNode):
    """
    Task that should show collected context.

    {COLLECTED_CONTEXT}

    ## Instructions
    Process the data based on previous results.

    {PROMPT_SUBTREE}
    """

    input_data: str = Field(description="Input data to process")
    result: str = Field(description="Processing result")


def test_collected_context_shows_previous_values():
    """Test that COLLECTED_CONTEXT resolves to show tag references for previous fields."""
    structure = RunStructure()
    structure.add(TaskWithContext)

    node = structure.get_node("task.with_context")
    assert node is not None

    # Call get_prompt with previous_values
    prompt = node.get_prompt(
        previous_values={"input_data": "Sample input data from previous step"}
    )

    # COLLECTED_CONTEXT should show FULL tag reference, not the actual value
    assert "# Input Data" in prompt, (
        "Should have Input Data heading in COLLECTED_CONTEXT"
    )
    assert "{task.with_context.input_data}" in prompt, (
        "Should show the FULL tag reference"
    )
    assert "Sample input data from previous step" not in prompt, (
        "Should NOT show actual value"
    )

    # Also verify PROMPT_SUBTREE still shows remaining fields
    assert "# Result" in prompt, "Should show Result field in PROMPT_SUBTREE"
    assert "Processing result" in prompt, "Should show Result description"


def test_collected_context_empty_when_no_previous_values():
    """Test that COLLECTED_CONTEXT is empty when no previous values provided."""
    structure = RunStructure()
    structure.add(TaskWithContext)

    node = structure.get_node("task.with_context")

    # Call without previous_values
    prompt = node.get_prompt()

    # COLLECTED_CONTEXT should be empty
    # PROMPT_SUBTREE should show ONLY the first field (current leaf)
    assert "# Input Data" in prompt, (
        "Should show Input Data (first field) in PROMPT_SUBTREE"
    )
    assert "# Result" not in prompt, (
        "Should NOT show Result (sibling field) in PROMPT_SUBTREE"
    )


def test_collected_context_with_multiple_previous_values():
    """Test COLLECTED_CONTEXT with multiple previous field tag references."""
    structure = RunStructure()
    structure.add(TaskWithContext)

    node = structure.get_node("task.with_context")

    # Both fields have previous values
    prompt = node.get_prompt(
        previous_values={
            "input_data": "Initial data",
            "result": "Partial result from first pass",
        }
    )

    # Both fields should appear in COLLECTED_CONTEXT as FULL tag references
    assert "# Input Data" in prompt
    assert "{task.with_context.input_data}" in prompt
    assert "# Result" in prompt
    assert "{task.with_context.result}" in prompt

    # Actual values should NOT appear
    assert "Initial data" not in prompt
    assert "Partial result from first pass" not in prompt
