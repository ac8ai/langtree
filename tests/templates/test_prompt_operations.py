"""
Tests for prompt element list operations.

This module tests utility operations on structured prompt element lists,
including level alignment, template checks, and insertions.
"""

import pytest

from langtree.templates.prompt_operations import (
    adjust_element_levels,
    has_template_variable,
    insert_elements_at_template,
)
from langtree.templates.prompt_structure import (
    PromptTemplate,
    PromptText,
    PromptTitle,
)


class TestAdjustElementLevels:
    """Test level adjustment operations."""

    def test_adjust_levels_increase(self):
        """Test increasing all element levels."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Text", level=1),
            PromptTitle(content="Subtitle", level=2),
        ]

        result = adjust_element_levels(elements, align_by=1)

        assert result[0].level == 2
        assert result[1].level == 2
        assert result[2].level == 3

    def test_adjust_levels_decrease(self):
        """Test decreasing element levels."""
        elements = [
            PromptTitle(content="Title", level=3),
            PromptText(content="Text", level=3),
        ]

        result = adjust_element_levels(elements, align_by=-2)

        assert result[0].level == 1
        assert result[1].level == 1

    def test_adjust_levels_no_change(self):
        """Test with zero offset."""
        elements = [
            PromptTitle(content="Title", level=2),
        ]

        result = adjust_element_levels(elements, align_by=0)

        assert result[0].level == 2

    def test_adjust_levels_preserves_content(self):
        """Test that adjustment preserves element content."""
        elements = [
            PromptTitle(content="Title", level=1, optional=True),
            PromptText(content="Text content", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
        ]

        result = adjust_element_levels(elements, align_by=2)

        assert result[0].content == "Title"
        assert result[0].optional is True
        assert result[1].content == "Text content"
        assert result[2].variable_name == "PROMPT_SUBTREE"

    def test_adjust_levels_empty_list(self):
        """Test with empty list."""
        result = adjust_element_levels([], align_by=5)

        assert result == []

    def test_adjust_levels_minimum_clamp(self):
        """Test that levels are clamped to minimum of 1."""
        elements = [
            PromptTitle(content="Title", level=2),
        ]

        result = adjust_element_levels(elements, align_by=-5)

        assert result[0].level == 1  # Should not go below 1

    def test_adjust_levels_no_maximum_clamp(self):
        """Test that levels can exceed 6 for deep nesting."""
        elements = [
            PromptTitle(content="Title", level=5),
        ]

        result = adjust_element_levels(elements, align_by=5)

        assert result[0].level == 10  # Can go above 6 for deep hierarchies

    def test_adjust_levels_align_to(self):
        """Test aligning minimum level to target."""
        elements = [
            PromptTitle(content="Title", level=3),
            PromptText(content="Text", level=4),
            PromptTitle(content="Subtitle", level=5),
        ]

        result = adjust_element_levels(elements, align_to=1)

        # Min was 3, now should be 1, so offset is -2
        assert result[0].level == 1  # 3 - 2
        assert result[1].level == 2  # 4 - 2
        assert result[2].level == 3  # 5 - 2

    def test_adjust_levels_align_to_increase(self):
        """Test aligning can increase levels."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Text", level=2),
        ]

        result = adjust_element_levels(elements, align_to=3)

        assert result[0].level == 3  # 1 + 2
        assert result[1].level == 4  # 2 + 2

    def test_adjust_levels_both_params_raises_error(self):
        """Test that providing both by and align_to raises error."""
        elements = [PromptTitle(content="Title", level=1)]

        with pytest.raises(ValueError, match="Exactly one"):
            adjust_element_levels(elements, align_by=1, align_to=2)

    def test_adjust_levels_neither_param_raises_error(self):
        """Test that providing neither by nor align_to raises error."""
        elements = [PromptTitle(content="Title", level=1)]

        with pytest.raises(ValueError, match="Exactly one"):
            adjust_element_levels(elements)


class TestHasTemplateVariable:
    """Test template variable detection."""

    def test_has_prompt_subtree(self):
        """Test detection of PROMPT_SUBTREE."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
        ]

        assert has_template_variable(elements, "PROMPT_SUBTREE") is True

    def test_has_collected_context(self):
        """Test detection of COLLECTED_CONTEXT."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2),
        ]

        assert has_template_variable(elements, "COLLECTED_CONTEXT") is True

    def test_no_template_variable(self):
        """Test when template variable is not present."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Text", level=1),
        ]

        assert has_template_variable(elements, "PROMPT_SUBTREE") is False

    def test_has_different_template(self):
        """Test when different template is present."""
        elements = [
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
        ]

        assert has_template_variable(elements, "COLLECTED_CONTEXT") is False

    def test_empty_list(self):
        """Test with empty list."""
        assert has_template_variable([], "PROMPT_SUBTREE") is False

    def test_has_template_include_optional_default(self):
        """Test that optional templates are found by default."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1, optional=True),
        ]

        assert has_template_variable(elements, "PROMPT_SUBTREE") is True

    def test_has_template_exclude_optional(self):
        """Test excluding optional templates."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1, optional=True),
        ]

        assert (
            has_template_variable(elements, "PROMPT_SUBTREE", include_optional=False)
            is False
        )

    def test_has_template_non_optional_found_when_excluding_optional(self):
        """Test that non-optional templates are found even when excluding optional."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1, optional=False),
        ]

        assert (
            has_template_variable(elements, "PROMPT_SUBTREE", include_optional=False)
            is True
        )


class TestInsertElementsAtTemplate:
    """Test template replacement operations."""

    def test_insert_at_prompt_subtree(self):
        """Test inserting elements at PROMPT_SUBTREE location."""
        elements = [
            PromptTitle(content="Main", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
            PromptText(content="End", level=1),
        ]

        to_insert = [
            PromptTitle(content="Inserted", level=1),
            PromptText(content="Inserted text", level=1),
        ]

        result = insert_elements_at_template(elements, "PROMPT_SUBTREE", to_insert)

        assert len(result) == 4
        assert result[0].content == "Main"
        assert result[1].content == "Inserted"
        assert result[2].content == "Inserted text"
        assert result[3].content == "End"

    def test_insert_with_level_adjustment(self):
        """Test that inserted elements inherit template level."""
        elements = [
            PromptTitle(content="Section", level=2),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=2),
        ]

        to_insert = [
            PromptTitle(content="Subsection", level=1),
            PromptText(content="Content", level=1),
        ]

        result = insert_elements_at_template(elements, "PROMPT_SUBTREE", to_insert)

        # Inserted elements should be adjusted to match template level
        assert result[1].level == 2  # Subsection adjusted from 1 to 2
        assert result[2].level == 2  # Content adjusted from 1 to 2

    def test_insert_preserves_other_elements(self):
        """Test that non-template elements are preserved."""
        elements = [
            PromptTitle(content="Before", level=1),
            PromptText(content="Text", level=1),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2),
            PromptTitle(content="After", level=1),
        ]

        to_insert = [PromptText(content="Context", level=1)]

        result = insert_elements_at_template(elements, "COLLECTED_CONTEXT", to_insert)

        assert len(result) == 4
        assert result[0].content == "Before"
        assert result[1].content == "Text"
        assert result[2].content == "Context"
        assert result[3].content == "After"

    def test_insert_when_template_not_found(self):
        """Test insertion when template variable not found."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Text", level=1),
        ]

        to_insert = [PromptText(content="Should not appear", level=1)]

        result = insert_elements_at_template(elements, "PROMPT_SUBTREE", to_insert)

        # Should return original elements unchanged
        assert len(result) == 2
        assert result[0].content == "Title"
        assert result[1].content == "Text"

    def test_insert_empty_list(self):
        """Test inserting empty list."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
        ]

        result = insert_elements_at_template(elements, "PROMPT_SUBTREE", [])

        # Template should be removed, but nothing inserted
        assert len(result) == 1
        assert result[0].content == "Title"

    def test_insert_multiple_occurrences(self):
        """Test that only first occurrence is replaced."""
        elements = [
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
            PromptText(content="Middle", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=2),
        ]

        to_insert = [PromptText(content="Inserted", level=1)]

        result = insert_elements_at_template(elements, "PROMPT_SUBTREE", to_insert)

        # Only first template should be replaced
        assert len(result) == 3
        assert isinstance(result[0], PromptText)
        assert result[0].content == "Inserted"
        assert result[1].content == "Middle"
        assert isinstance(result[2], PromptTemplate)
        assert result[2].variable_name == "PROMPT_SUBTREE"
