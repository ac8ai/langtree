"""
Tests for the @output_format command functionality.
"""

import pytest
from pydantic import Field

from langtree.core.tree_node import TreeNode
from langtree.exceptions.core import (
    ComprehensiveStructuralValidationError,
    TemplateVariableConflictError,
)
from langtree.templates.prompt_assembly import (
    assemble_field_prompt_with_format,
    get_output_format,
    get_output_format_for_field,
    validate_output_format_usage,
)
from langtree.templates.utils import extract_commands
from langtree.templates.variables import (
    process_class_docstring,
)


class TestOutputFormatParsing:
    """Test parsing and extraction of @output_format commands."""

    def test_extract_output_format_command_markdown(self):
        """Test extracting markdown output format command."""
        content = """! @output_format("markdown")
Some description text.
More content here."""
        commands, clean = extract_commands(content)
        assert len(commands) == 1
        assert commands[0].text == '! @output_format("markdown")'
        assert commands[0].line == 0

    def test_extract_output_format_command_plain(self):
        """Test extracting plain output format command."""
        content = """! @output_format("plain")"""
        commands, clean = extract_commands(content)
        assert len(commands) == 1
        assert commands[0].text == '! @output_format("plain")'

    def test_multiple_output_formats_detected(self):
        """Test that multiple output_format commands are detected."""
        content = """! @output_format("markdown")
! @output_format("plain")"""
        commands, clean = extract_commands(content)
        assert len(commands) == 2
        assert commands[0].text == '! @output_format("markdown")'
        assert commands[1].text == '! @output_format("plain")'

    def test_output_format_with_other_commands(self):
        """Test output_format mixed with other commands."""
        content = """! @output_format("markdown")
! @sequential
! llm("gpt-4")"""
        commands, clean = extract_commands(content)
        assert len(commands) == 3
        assert any('@output_format("markdown")' in cmd.text for cmd in commands)


class TestOutputFormatExtraction:
    """Test extracting output format from processed docstrings."""

    def test_get_output_format_from_processed_docstring(self):
        """Test extracting output format from ProcessedDocstring."""
        # Test with markdown format
        docstring = """! @output_format("markdown")
Analysis with markdown output."""
        processed = process_class_docstring(docstring, "TestClass")
        output_format = get_output_format(processed)
        assert output_format == "markdown"

    def test_get_output_format_default_plain(self):
        """Test default output format is plain when not specified."""
        docstring = """
        Regular docstring without format command.
        """
        processed = process_class_docstring(docstring, "TestClass")
        output_format = get_output_format(processed)
        assert output_format == "plain"

    def test_get_output_format_invalid_format(self):
        """Test error handling for invalid output formats."""
        docstring = """! @output_format("invalid")"""
        processed = process_class_docstring(docstring, "TestClass")
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_output_format(processed, validate=True)

    def test_multiple_output_formats_raises_error(self):
        """Test that multiple output_format commands raise an error."""
        docstring = """! @output_format("markdown")
! @output_format("plain")"""
        processed = process_class_docstring(docstring, "TestClass")
        with pytest.raises(
            TemplateVariableConflictError, match="Multiple output_format commands"
        ):
            get_output_format(processed, validate=True)


class TestOutputFormatValidation:
    """Test validation of output_format usage constraints."""

    def test_validate_output_format_on_str_leaf(self):
        """Test output_format is valid on str-typed leaf fields."""

        class ValidMarkdownNode(TreeNode):
            """
            ! @output_format("markdown")
            """

            content: str = Field(description="Markdown content")
            analysis: str = Field(description="Analysis text")

        # Should not raise any errors
        validate_output_format_usage(ValidMarkdownNode)

    def test_validate_output_format_on_non_str_raises_error(self):
        """Test output_format on non-string fields raises error."""

        class InvalidNode(TreeNode):
            """
            ! @output_format("markdown")
            """

            count: int = Field(description="Number field")
            values: list[str] = Field(description="List field")

        with pytest.raises(
            ComprehensiveStructuralValidationError,
            match="output_format.*only valid on str-typed",
        ):
            validate_output_format_usage(InvalidNode)

    def test_validate_output_format_on_non_leaf_raises_error(self):
        """Test output_format on non-leaf nodes raises error."""

        class ChildNode(TreeNode):
            value: str

        class ParentNode(TreeNode):
            """
            ! @output_format("markdown")
            """

            child: ChildNode = Field(description="Has nested node")

        with pytest.raises(
            ComprehensiveStructuralValidationError,
            match="output_format.*only valid on.*leaf",
        ):
            validate_output_format_usage(ParentNode)

    def test_validate_mixed_fields_with_output_format(self):
        """Test node with both str and non-str fields with output_format."""

        class MixedNode(TreeNode):
            """
            ! @output_format("markdown")
            """

            text: str = Field(description="Valid for markdown")
            number: int = Field(description="Invalid for markdown")
            items: list[str] = Field(description="Invalid for markdown")

        # Should detect the non-str fields
        with pytest.raises(
            ComprehensiveStructuralValidationError,
            match="output_format.*only valid on str-typed",
        ):
            validate_output_format_usage(MixedNode)


class TestOutputFormatInPromptAssembly:
    """Test output_format integration with prompt assembly."""

    def test_assemble_prompt_with_markdown_format(self):
        """Test prompt assembly with markdown output format."""

        class MarkdownNode(TreeNode):
            """! @output_format("markdown")
            Generate markdown-formatted analysis."""

            summary: str = Field(description="Executive summary")
            details: str = Field(description="Detailed findings")

        # Test field prompt assembly
        prompt = assemble_field_prompt_with_format(
            field_name="summary",
            field_info=MarkdownNode.model_fields["summary"],
            node_class=MarkdownNode,
            is_leaf=True,
            heading_level=2,
        )

        # Should include langtree-output tags
        assert "<langtree-output>" in prompt
        assert "</langtree-output>" in prompt
        assert "## Summary" in prompt

    def test_assemble_prompt_with_plain_format(self):
        """Test prompt assembly with plain output format."""

        class PlainNode(TreeNode):
            """
            Regular text output.
            """

            content: str = Field(description="Plain text content")

        prompt = assemble_field_prompt_with_format(
            field_name="content",
            field_info=PlainNode.model_fields["content"],
            node_class=PlainNode,
            is_leaf=True,
            heading_level=2,
        )

        # Should NOT include langtree-output tags
        assert "<langtree-output>" not in prompt
        assert "</langtree-output>" not in prompt
        assert "## Content" in prompt

    def test_output_format_preserved_in_metadata(self):
        """Test output_format is preserved in field metadata for structured generation."""

        class MarkdownNode(TreeNode):
            """
            ! @output_format("markdown")
            """

            content: str = Field(description="Content")

        # The output format should be accessible for structured generation
        output_format = get_output_format_for_field(MarkdownNode, "content")
        assert output_format == "markdown"


class TestOutputFormatEdgeCases:
    """Test edge cases and error conditions for output_format."""

    def test_output_format_with_whitespace_variations(self):
        """Test output_format parsing with various whitespace."""
        variations = [
            '! @output_format("markdown")',
            '!@output_format("markdown")',
            '! @output_format( "markdown" )',
            '!  @output_format("markdown")',
        ]

        for variant in variations:
            content = f"{variant}\nDescription\n"
            commands, clean = extract_commands(content)
            assert any("@output_format" in cmd.text for cmd in commands)

    def test_output_format_case_sensitivity(self):
        """Test output format values are case-sensitive."""
        docstring = """! @output_format("Markdown")"""
        processed = process_class_docstring(docstring, "TestClass")
        # Should treat "Markdown" as invalid (not "markdown")
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_output_format(processed, validate=True)

    def test_empty_output_format_value(self):
        """Test empty output format value handling."""
        docstring = """! @output_format("")"""
        processed = process_class_docstring(docstring, "TestClass")
        with pytest.raises(ValueError, match="Empty output format"):
            get_output_format(processed, validate=True)
