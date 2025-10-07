"""
Tests for prompt assembly with template variables.

This module tests the assembly of prompts with COLLECTED_CONTEXT and
PROMPT_SUBTREE template variables, including automatic addition and
customization options.
"""

from langtree.templates.prompt_assembly import (
    assemble_field_prompt,
    create_field_title,
    ensure_template_variables,
)
from langtree.templates.prompt_structure import (
    PromptTemplate,
    PromptText,
    PromptTitle,
)


class TestEnsureTemplateVariables:
    """Test automatic addition of template variables."""

    def test_empty_list_adds_both_templates(self):
        """Test that empty list gets both template variables added."""
        elements = []

        result = ensure_template_variables(elements)

        # Should have 4 elements: 2 titles + 2 templates
        assert len(result) == 4

        # First pair: COLLECTED_CONTEXT
        assert isinstance(result[0], PromptTitle)
        assert result[0].content == "Context"
        assert isinstance(result[1], PromptTemplate)
        assert result[1].variable_name == "COLLECTED_CONTEXT"

        # Second pair: PROMPT_SUBTREE
        assert isinstance(result[2], PromptTitle)
        assert result[2].content == "Task"
        assert isinstance(result[3], PromptTemplate)
        assert result[3].variable_name == "PROMPT_SUBTREE"

    def test_existing_collected_context_only_adds_prompt_subtree(self):
        """Test that only missing template is added."""
        elements = [
            PromptTitle(content="My Context", level=1),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2),
        ]

        result = ensure_template_variables(elements)

        # Should add 2 elements for PROMPT_SUBTREE
        assert len(result) == 4
        assert result[2].content == "Task"
        assert result[3].variable_name == "PROMPT_SUBTREE"

    def test_both_exist_no_changes(self):
        """Test that nothing is added if both exist."""
        elements = [
            PromptText(content="Some intro"),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=2),
        ]

        result = ensure_template_variables(elements)

        # Should be unchanged
        assert len(result) == 3
        assert result == elements

    def test_preserves_user_placement(self):
        """Test that user's placement of templates is preserved."""
        elements = [
            PromptTitle(content="Intro", level=1),
            PromptText(content="Description"),
            PromptTemplate(
                variable_name="PROMPT_SUBTREE", level=2
            ),  # User put this first
            PromptText(content="Middle section"),
            PromptTemplate(
                variable_name="COLLECTED_CONTEXT", level=2
            ),  # And this second
        ]

        result = ensure_template_variables(elements)

        # Should be unchanged - respects user ordering
        assert result == elements
        # PROMPT_SUBTREE still comes before COLLECTED_CONTEXT
        subtree_idx = next(
            i
            for i, e in enumerate(result)
            if isinstance(e, PromptTemplate) and e.variable_name == "PROMPT_SUBTREE"
        )
        context_idx = next(
            i
            for i, e in enumerate(result)
            if isinstance(e, PromptTemplate) and e.variable_name == "COLLECTED_CONTEXT"
        )
        assert subtree_idx < context_idx

    def test_adds_after_existing_content(self):
        """Test templates are added at end after existing content."""
        elements = [
            PromptTitle(content="Main Title", level=1),
            PromptText(content="Main description of the task"),
        ]

        result = ensure_template_variables(elements)

        assert len(result) == 6  # Original 2 + 4 new
        # Original content preserved
        assert result[0].content == "Main Title"
        assert result[1].content == "Main description of the task"
        # Templates added at end
        assert result[2].content == "Context"
        assert result[4].content == "Task"

    def test_custom_title_configuration(self):
        """Test custom titles for template sections."""
        elements = []
        config = {
            "collected_context_title": "Generated So Far",
            "prompt_subtree_title": "To Generate",
        }

        result = ensure_template_variables(elements, config=config)

        assert result[0].content == "Generated So Far"
        assert result[2].content == "To Generate"


class TestFieldTitle:
    """Test field title generation with customization."""

    def test_basic_field_title(self):
        """Test basic field name to title conversion."""
        title = create_field_title("analysis_result")

        assert isinstance(title, PromptTitle)
        assert title.content == "Analysis Result"
        assert title.level == 1

    def test_field_title_with_prefix_for_leaf(self):
        """Test title prefix for leaf node generation."""
        title = create_field_title(
            "analysis_result", is_leaf=True, title_prefix="Next Task"
        )

        assert title.content == "Next Task: Analysis Result"

    def test_no_prefix_for_non_leaf(self):
        """Test that non-leaf nodes don't get prefix."""
        title = create_field_title(
            "analysis_result",
            is_leaf=False,
            title_prefix="Next Task",  # Should be ignored
        )

        assert title.content == "Analysis Result"  # No prefix

    def test_camel_case_conversion(self):
        """Test camelCase field name conversion."""
        title = create_field_title("myComplexFieldName")

        assert title.content == "My Complex Field Name"

    def test_number_separation(self):
        """Test that numbers are separated."""
        title = create_field_title("field2value")

        assert title.content == "Field 2 Value"

    def test_custom_level(self):
        """Test custom heading level."""
        title = create_field_title("field_name", level=3)

        assert title.level == 3


class TestAssembleFieldPrompt:
    """Test full field prompt assembly."""

    def test_leaf_field_assembly(self):
        """Test assembling prompt for leaf field."""
        field_description = "Calculate the average value."
        field_config = {
            "title_prefix": "To Generate Next",
            "output_format": "int",
        }

        elements = assemble_field_prompt(
            field_name="average_score",
            field_description=field_description,
            is_leaf=True,
            config=field_config,
        )

        # Should have title with prefix for leaf
        assert elements[0].content == "To Generate Next: Average Score"

        # Should have description
        assert any(e.content == field_description for e in elements)

        # Should have output formatting section
        assert any(e.content == "Output Format" for e in elements)

    def test_non_leaf_field_assembly(self):
        """Test assembling prompt for non-leaf field."""
        field_description = "Process all data points."

        elements = assemble_field_prompt(
            field_name="data_processor",
            field_description=field_description,
            is_leaf=False,
            config={"title_prefix": "To Generate Next"},  # Should be ignored
        )

        # No prefix for non-leaf
        assert elements[0].content == "Data Processor"

        # Should have PROMPT_SUBTREE for children
        assert any(
            isinstance(e, PromptTemplate) and e.variable_name == "PROMPT_SUBTREE"
            for e in elements
        )

    def test_markdown_field_with_tags(self):
        """Test markdown field gets tag instructions."""
        field_config = {
            "output_format": "markdown",
            "use_tags": True,
        }

        elements = assemble_field_prompt(
            field_name="analysis",
            field_description="Provide detailed analysis.",
            is_leaf=True,
            config=field_config,
        )

        # Should have tag instructions
        format_elements = [e for e in elements if "langtree" in str(e.content).lower()]
        assert len(format_elements) > 0
        assert any(
            "langtree-markdown-output" in str(e.content) for e in format_elements
        )

    def test_skip_output_formatting(self):
        """Test skipping output format section."""
        field_config = {
            "skip_formatting": True,
        }

        elements = assemble_field_prompt(
            field_name="custom_field",
            field_description="User handles formatting.",
            is_leaf=True,
            config=field_config,
        )

        # Should NOT have output formatting section
        assert not any(
            e.content == "Output Format" for e in elements if isinstance(e, PromptTitle)
        )
