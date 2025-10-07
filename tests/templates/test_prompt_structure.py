"""
Tests for prompt structure data classes.

This module tests the core PromptElement base class and its subclasses
(PromptTitle, PromptText, PromptTemplate) that form the structured
representation of prompts.
"""

from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)


class TestPromptElementBaseClass:
    """Test PromptElement base class and shared attributes."""

    def test_prompt_title_has_base_attributes(self):
        """Test that PromptTitle inherits level and optional from base class."""
        title = PromptTitle(content="Test", level=1)

        assert hasattr(title, "level")
        assert hasattr(title, "optional")
        assert title.level == 1
        assert title.optional is False

    def test_prompt_text_has_base_attributes(self):
        """Test that PromptText inherits level and optional from base class."""
        text = PromptText(content="Content", level=1)

        assert hasattr(text, "level")
        assert hasattr(text, "optional")
        assert text.level == 1
        assert text.optional is False

    def test_prompt_template_has_base_attributes(self):
        """Test that PromptTemplate inherits level and optional from base class."""
        template = PromptTemplate(variable_name="PROMPT_SUBTREE", level=2)

        assert hasattr(template, "level")
        assert hasattr(template, "optional")
        assert template.level == 2
        assert template.optional is False

    def test_optional_flag_on_all_types(self):
        """Test that all element types can be marked as optional."""
        title = PromptTitle(content="Title", level=1, optional=True)
        text = PromptText(content="Text", level=1, optional=True)
        template = PromptTemplate(
            variable_name="PROMPT_SUBTREE", level=2, optional=True
        )

        assert title.optional is True
        assert text.optional is True
        assert template.optional is True

    def test_all_elements_are_prompt_elements(self):
        """Test that all element types inherit from PromptElement."""
        title = PromptTitle("Title", 1)
        text = PromptText("Content", 1)
        template = PromptTemplate("PROMPT_SUBTREE", 2)

        assert isinstance(title, PromptElement)
        assert isinstance(text, PromptElement)
        assert isinstance(template, PromptElement)


class TestPromptTitle:
    """Test PromptTitle specific functionality."""

    def test_create_title_with_level(self):
        """Test creating title with specific heading level."""
        title = PromptTitle(content="Section Title", level=2)

        assert title.content == "Section Title"
        assert title.level == 2
        assert title.optional is False

    def test_title_optional_flag(self):
        """Test creating optional title."""
        title = PromptTitle(content="Optional Section", level=3, optional=True)

        assert title.content == "Optional Section"
        assert title.level == 3
        assert title.optional is True


class TestPromptText:
    """Test PromptText specific functionality."""

    def test_create_text_with_level(self):
        """Test creating text with inherited level."""
        text = PromptText(content="Paragraph content", level=1)

        assert text.content == "Paragraph content"
        assert text.level == 1
        assert text.optional is False

    def test_text_optional_flag(self):
        """Test creating optional text."""
        text = PromptText(content="Optional content", level=2, optional=True)

        assert text.content == "Optional content"
        assert text.level == 2
        assert text.optional is True


class TestPromptTemplate:
    """Test PromptTemplate specific functionality."""

    def test_create_prompt_subtree_template(self):
        """Test creating PROMPT_SUBTREE template."""
        template = PromptTemplate(variable_name="PROMPT_SUBTREE", level=1)

        assert template.variable_name == "PROMPT_SUBTREE"
        assert template.level == 1
        assert template.optional is False
        assert template.resolved_content is None

    def test_create_collected_context_template(self):
        """Test creating COLLECTED_CONTEXT template."""
        template = PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2)

        assert template.variable_name == "COLLECTED_CONTEXT"
        assert template.level == 2
        assert template.optional is False
        assert template.resolved_content is None

    def test_template_optional_flag(self):
        """Test creating optional template."""
        template = PromptTemplate(
            variable_name="PROMPT_SUBTREE", level=1, optional=True
        )

        assert template.optional is True

    def test_template_with_resolved_content(self):
        """Test template with resolved content."""
        resolved = [
            PromptTitle("Child", 2),
            PromptText("Child content", 2),
        ]
        template = PromptTemplate(
            variable_name="PROMPT_SUBTREE", level=1, resolved_content=resolved
        )

        assert template.resolved_content == resolved
        assert len(template.resolved_content) == 2
