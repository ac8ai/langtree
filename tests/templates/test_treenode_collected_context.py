"""
Tests for COLLECTED_CONTEXT with nested TreeNode tag references.

Verifies that when previous_values contains nested TreeNode results,
COLLECTED_CONTEXT generates tag references (like {field.subfield}) instead
of actual values. Values will be filled in at runtime.
"""

from pydantic import Field

from langtree.core.tree_node import TreeNode
from langtree.structure.builder import RunStructure
from langtree.templates.element_resolution import resolve_collected_context_elements


class TaskSourceAnalysis(TreeNode):
    """Source analysis task."""

    credible_sources: list[str] = Field(description="List of credible sources")
    quality_assessment: str = Field(description="Quality assessment")


class TaskLiteratureReview(TreeNode):
    """
    Literature review task.

    {COLLECTED_CONTEXT}

    ## Instructions
    Review the literature based on source analysis.

    {PROMPT_SUBTREE}
    """

    source_analysis: TaskSourceAnalysis = Field(description="Source analysis")
    key_findings: str = Field(description="Key findings")


def test_collected_context_with_nested_treenode_tags():
    """
    Test that nested TreeNode fields generate tag references with dot notation.

    When a field is a TreeNode type, COLLECTED_CONTEXT should show tags like
    {field.subfield} for nested fields, not actual values.
    """
    structure = RunStructure()
    structure.add(TaskLiteratureReview)
    structure.add(TaskSourceAnalysis)

    node = structure.get_node("task.literature_review")
    assert node is not None

    # Simulate a nested TreeNode result in previous_values
    previous_values = {
        "source_analysis": {
            "credible_sources": ["Source 1", "Source 2", "Source 3"],
            "quality_assessment": "High quality sources",
        }
    }

    # Resolve COLLECTED_CONTEXT
    elements = resolve_collected_context_elements(
        node, previous_values, base_heading_level=1
    )

    # Should have proper structure with FULL tag references:
    # - Title: "Source Analysis" (level 1)
    # - Title: "Credible Sources" (level 2)
    # - Text: {task.literature_review.source_analysis.credible_sources} tag (level 2)
    # - Title: "Quality Assessment" (level 2)
    # - Text: {task.literature_review.source_analysis.quality_assessment} tag (level 2)

    assert len(elements) == 5, f"Expected 5 elements, got {len(elements)}: {elements}"

    # Check Source Analysis title
    assert elements[0].content == "Source Analysis"
    assert elements[0].level == 1

    # Check Credible Sources title
    assert elements[1].content == "Credible Sources"
    assert elements[1].level == 2

    # Check FULL tag reference for credible_sources
    assert (
        elements[2].content
        == "{task.literature_review.source_analysis.credible_sources}"
    )
    assert elements[2].level == 2

    # Check Quality Assessment title
    assert elements[3].content == "Quality Assessment"
    assert elements[3].level == 2

    # Check FULL tag reference for quality_assessment
    assert (
        elements[4].content
        == "{task.literature_review.source_analysis.quality_assessment}"
    )
    assert elements[4].level == 2


def test_collected_context_with_multiple_nested_treenode_tags():
    """
    Test COLLECTED_CONTEXT with multiple nested TreeNode fields.

    Ensures that when multiple fields are TreeNode types, each generates
    proper tag references.
    """
    structure = RunStructure()
    structure.add(TaskLiteratureReview)
    structure.add(TaskSourceAnalysis)

    node = structure.get_node("task.literature_review")
    assert node is not None

    # Simulate multiple fields with nested TreeNode results
    previous_values = {
        "source_analysis": {
            "credible_sources": ["Source A", "Source B"],
            "quality_assessment": "Good",
        },
        "key_findings": "Important findings here",
    }

    # Resolve COLLECTED_CONTEXT
    elements = resolve_collected_context_elements(
        node, previous_values, base_heading_level=1
    )

    # Should have:
    # - Source Analysis section with nested tag references
    # - Key Findings section with simple tag reference

    # Find the source_analysis section
    source_analysis_idx = next(
        i
        for i, e in enumerate(elements)
        if hasattr(e, "content") and e.content == "Source Analysis"
    )
    assert elements[source_analysis_idx].level == 1

    # Next should be Credible Sources at level 2
    assert elements[source_analysis_idx + 1].content == "Credible Sources"
    assert elements[source_analysis_idx + 1].level == 2

    # Should have FULL tag reference
    assert (
        elements[source_analysis_idx + 2].content
        == "{task.literature_review.source_analysis.credible_sources}"
    )

    # Find the key_findings section
    key_findings_idx = next(
        i
        for i, e in enumerate(elements)
        if hasattr(e, "content") and e.content == "Key Findings"
    )
    assert elements[key_findings_idx].level == 1
    assert (
        elements[key_findings_idx + 1].content
        == "{task.literature_review.key_findings}"
    )


def test_collected_context_regular_dict_vs_treenode_dict():
    """
    Test that regular dict fields generate simple tags, not nested tags.

    Regular (non-TreeNode) dict fields should just get a single tag reference,
    not recursive nested tags.
    """

    class TaskWithMixedFields(TreeNode):
        """Task with both TreeNode and regular dict fields."""

        nested_task: TaskSourceAnalysis = Field(description="Nested task")
        metadata: dict[str, str] = Field(description="Metadata dict")

    structure = RunStructure()
    structure.add(TaskWithMixedFields)
    structure.add(TaskSourceAnalysis)

    node = structure.get_node("task.with_mixed_fields")
    assert node is not None

    previous_values = {
        "nested_task": {"credible_sources": ["Source 1"], "quality_assessment": "Good"},
        "metadata": {"author": "John Doe", "version": "1.0"},
    }

    # Resolve COLLECTED_CONTEXT
    elements = resolve_collected_context_elements(
        node, previous_values, base_heading_level=1
    )

    # Check nested_task generates recursive tags
    nested_task_idx = next(
        i
        for i, e in enumerate(elements)
        if hasattr(e, "content") and e.content == "Nested Task"
    )
    assert elements[nested_task_idx].level == 1

    # Next should be "Credible Sources" at level 2 (recursive tag generation)
    assert elements[nested_task_idx + 1].content == "Credible Sources"
    assert elements[nested_task_idx + 1].level == 2
    assert (
        elements[nested_task_idx + 2].content
        == "{task.with_mixed_fields.nested_task.credible_sources}"
    )

    # Check metadata generates simple tag (not recursive)
    metadata_idx = next(
        i
        for i, e in enumerate(elements)
        if hasattr(e, "content") and e.content == "Metadata"
    )
    assert elements[metadata_idx].level == 1

    # Regular dict should just have a single FULL tag reference
    assert elements[metadata_idx + 1].content == "{task.with_mixed_fields.metadata}"


def test_collected_context_full_prompt_integration():
    """
    Integration test: verify nested TreeNode tag references in actual prompts.

    Uses get_prompt() to ensure the full resolution chain works correctly.
    """
    structure = RunStructure()
    structure.add(TaskLiteratureReview)
    structure.add(TaskSourceAnalysis)

    node = structure.get_node("task.literature_review")
    assert node is not None

    previous_values = {
        "source_analysis": {
            "credible_sources": ["GitHub Study 2023", "ACM Survey"],
            "quality_assessment": "High quality recent research",
        }
    }

    # Generate full prompt
    prompt = node.get_prompt(previous_values=previous_values)

    # Verify nested structure appears in prompt with FULL tag references
    assert "# Source Analysis" in prompt
    assert "## Credible Sources" in prompt
    assert "{task.literature_review.source_analysis.credible_sources}" in prompt
    assert "## Quality Assessment" in prompt
    assert "{task.literature_review.source_analysis.quality_assessment}" in prompt

    # Verify actual values do NOT appear
    assert "GitHub Study 2023" not in prompt
    assert "ACM Survey" not in prompt
    assert "High quality recent research" not in prompt
