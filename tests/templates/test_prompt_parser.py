"""
Tests for prompt parsing and serialization.

This module tests parsing markdown to structured prompt lists and
serializing structured lists back to markdown.
"""

from langtree.templates.prompt_parser import (
    parse_prompt_to_list,
    prompt_list_to_string,
)
from langtree.templates.prompt_structure import (
    PromptTemplate,
    PromptText,
    PromptTitle,
)


class TestParseMarkdownToList:
    """Test parsing markdown content to structured prompt lists."""

    def test_parse_simple_title_and_text(self):
        """Test parsing simple markdown with title and text."""
        content = "# Title\n\nSome text"
        elements = parse_prompt_to_list(content)

        assert len(elements) == 2
        assert isinstance(elements[0], PromptTitle)
        assert elements[0].content == "Title"
        assert elements[0].level == 1
        assert isinstance(elements[1], PromptText)
        assert elements[1].content == "Some text"
        assert elements[1].level == 1

    def test_parse_multiple_heading_levels(self):
        """Test parsing with multiple heading levels."""
        content = "# Level 1\n\nText\n\n## Level 2\n\nMore text"
        elements = parse_prompt_to_list(content)

        assert len(elements) == 4
        assert elements[0].level == 1
        assert elements[0].content == "Level 1"
        assert elements[1].level == 1  # Inherits from h1
        assert elements[2].level == 2
        assert elements[2].content == "Level 2"
        assert elements[3].level == 2  # Inherits from h2

    def test_parse_template_variables(self):
        """Test parsing template variables."""
        content = "{PROMPT_SUBTREE}\n\n{COLLECTED_CONTEXT}"
        elements = parse_prompt_to_list(content)

        assert len(elements) == 2
        assert isinstance(elements[0], PromptTemplate)
        assert elements[0].variable_name == "PROMPT_SUBTREE"
        assert isinstance(elements[1], PromptTemplate)
        assert elements[1].variable_name == "COLLECTED_CONTEXT"

    def test_parse_mixed_content(self):
        """Test parsing mixed content with titles, text, and templates."""
        content = "# Task\n\nProcess data\n\n{PROMPT_SUBTREE}\n\n## Context\n\n{COLLECTED_CONTEXT}"
        elements = parse_prompt_to_list(content)

        assert len(elements) == 5
        assert isinstance(elements[0], PromptTitle)
        assert isinstance(elements[1], PromptText)
        assert isinstance(elements[2], PromptTemplate)
        assert isinstance(elements[3], PromptTitle)
        assert isinstance(elements[4], PromptTemplate)

    def test_parse_defaults_optional_false(self):
        """Test that parsed elements default optional to False."""
        content = "# Title\n\nText"
        elements = parse_prompt_to_list(content)

        assert all(not e.optional for e in elements)

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        elements = parse_prompt_to_list("")

        assert elements == []

    def test_parse_only_whitespace(self):
        """Test parsing content with only whitespace."""
        elements = parse_prompt_to_list("\n\n  \n\n")

        assert elements == []

    def test_parse_heading_levels_1_to_6(self):
        """Test parsing all markdown heading levels."""
        content = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        elements = parse_prompt_to_list(content)

        assert len(elements) == 6
        for i, elem in enumerate(elements):
            assert isinstance(elem, PromptTitle)
            assert elem.level == i + 1

    def test_parse_text_inherits_level_from_last_title(self):
        """Test that text elements inherit level from last title."""
        content = "# H1\n\nText1\n\n## H2\n\nText2\n\nText3"
        elements = parse_prompt_to_list(content)

        # H1
        assert elements[0].level == 1
        # Text1 (inherits from H1)
        assert elements[1].level == 1
        # H2
        assert elements[2].level == 2
        # Text2 (inherits from H2)
        assert elements[3].level == 2
        # Text3 (still inherits from H2)
        assert elements[4].level == 2

    def test_parse_template_inherits_level(self):
        """Test that template variables inherit level from context."""
        content = "## Heading\n\n{PROMPT_SUBTREE}"
        elements = parse_prompt_to_list(content, normalize=False)

        assert elements[0].level == 2
        assert elements[1].level == 2  # Template inherits from heading

    def test_parse_with_normalization_default(self):
        """Test that normalization happens by default."""
        content = "### Title\n\nText\n\n#### Subtitle"
        elements = parse_prompt_to_list(content)

        # Original: level 3 and 4, normalized: level 1 and 2
        assert elements[0].level == 1  # ### → #
        assert elements[1].level == 1
        assert elements[2].level == 2  # #### → ##

    def test_parse_without_normalization(self):
        """Test parsing with normalization disabled."""
        content = "### Title\n\nText\n\n#### Subtitle"
        elements = parse_prompt_to_list(content, normalize=False)

        # Should preserve original levels
        assert elements[0].level == 3
        assert elements[1].level == 3
        assert elements[2].level == 4

    def test_parse_normalization_preserves_relative_structure(self):
        """Test that normalization preserves relative heading structure."""
        content = "## L2\n\n### L3\n\n##### L5\n\n###### L6"
        elements = parse_prompt_to_list(content)

        # Min is 2, normalized to 1, so: 2→1, 3→2, 5→4, 6→5
        assert elements[0].level == 1  # 2-1
        assert elements[1].level == 2  # 3-1
        assert elements[2].level == 4  # 5-1
        assert elements[3].level == 5  # 6-1


class TestSerializeListToMarkdown:
    """Test serializing prompt lists to markdown."""

    def test_render_title_and_text(self):
        """Test rendering simple title and text."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Content", level=1),
        ]
        result = prompt_list_to_string(elements)

        assert result == "# Title\n\nContent"

    def test_render_multiple_levels(self):
        """Test rendering multiple heading levels."""
        elements = [
            PromptTitle(content="H1", level=1),
            PromptText(content="Text1", level=1),
            PromptTitle(content="H2", level=2),
            PromptText(content="Text2", level=2),
        ]
        result = prompt_list_to_string(elements)

        expected = "# H1\n\nText1\n\n## H2\n\nText2"
        assert result == expected

    def test_render_template_variables(self):
        """Test rendering unresolved template variables."""
        elements = [
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=2),
        ]
        result = prompt_list_to_string(elements)

        assert "{PROMPT_SUBTREE}" in result
        assert "{COLLECTED_CONTEXT}" in result

    def test_render_with_optional_included(self):
        """Test rendering with optional elements included."""
        elements = [
            PromptTitle(content="Required", level=1, optional=False),
            PromptTitle(content="Optional", level=2, optional=True),
            PromptText(content="Optional content", level=2, optional=True),
        ]

        result = prompt_list_to_string(elements, include_optional=True)

        assert "Required" in result
        assert "Optional" in result
        assert "Optional content" in result

    def test_render_with_optional_excluded(self):
        """Test rendering with optional elements excluded."""
        elements = [
            PromptTitle(content="Required", level=1, optional=False),
            PromptTitle(content="Optional", level=2, optional=True),
            PromptText(content="Optional content", level=2, optional=True),
        ]

        result = prompt_list_to_string(elements, include_optional=False)

        assert "Required" in result
        assert "Optional" not in result
        assert "Optional content" not in result

    def test_render_empty_list(self):
        """Test rendering empty list."""
        result = prompt_list_to_string([])

        assert result == ""

    def test_render_all_heading_levels(self):
        """Test rendering all heading levels."""
        elements = [
            PromptTitle(content="H1", level=1),
            PromptTitle(content="H2", level=2),
            PromptTitle(content="H3", level=3),
            PromptTitle(content="H4", level=4),
            PromptTitle(content="H5", level=5),
            PromptTitle(content="H6", level=6),
        ]
        result = prompt_list_to_string(elements)

        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result
        assert "#### H4" in result
        assert "##### H5" in result
        assert "###### H6" in result


class TestRoundTripConversion:
    """Test parsing and serialization round-trip."""

    def test_round_trip_simple_content(self):
        """Test round-trip with simple content."""
        original = "# Title\n\nContent"
        elements = parse_prompt_to_list(original)
        result = prompt_list_to_string(elements)

        assert result == original

    def test_round_trip_complex_content(self):
        """Test round-trip with complex nested content."""
        original = "# Main\n\nIntro\n\n## Section\n\nDetails\n\n{PROMPT_SUBTREE}"
        elements = parse_prompt_to_list(original)
        result = prompt_list_to_string(elements)

        assert result == original


class TestLineNumberTracking:
    """Test line number tracking during parsing."""

    def test_line_numbers_simple(self):
        """Test line numbers for simple content."""
        lines = [
            "# Title",  # Line 0
            "",  # Line 1 (empty, skipped)
            "Text content.",  # Line 2
        ]
        content = "\n".join(lines)

        elements = parse_prompt_to_list(content, normalize=False)

        assert len(elements) == 2
        assert elements[0].line_number == 0  # Title at line 0
        assert elements[1].line_number == 2  # Text at line 2

    def test_line_numbers_with_template(self):
        """Test line numbers with template variables."""
        lines = [
            "# Section",  # Line 0
            "",  # Line 1
            "Content before.",  # Line 2
            "",  # Line 3
            "{PROMPT_SUBTREE}",  # Line 4
            "",  # Line 5
            "Content after.",  # Line 6
        ]
        content = "\n".join(lines)

        elements = parse_prompt_to_list(content, normalize=False)

        assert len(elements) == 4
        assert elements[0].line_number == 0  # Section title
        assert elements[1].line_number == 2  # Content before
        assert elements[2].line_number == 4  # Template
        assert elements[3].line_number == 6  # Content after

    def test_line_numbers_multiple_headings(self):
        """Test line numbers with multiple heading levels."""
        lines = [
            "# Main",  # Line 0
            "",  # Line 1
            "Main text.",  # Line 2
            "",  # Line 3
            "## Subsection",  # Line 4
            "",  # Line 5
            "Sub text.",  # Line 6
            "",  # Line 7
            "### Subsubsection",  # Line 8
        ]
        content = "\n".join(lines)

        elements = parse_prompt_to_list(content, normalize=False)

        assert len(elements) == 5
        assert elements[0].line_number == 0  # # Main
        assert elements[1].line_number == 2  # Main text
        assert elements[2].line_number == 4  # ## Subsection
        assert elements[3].line_number == 6  # Sub text
        assert elements[4].line_number == 8  # ### Subsubsection

    def test_line_numbers_multi_line_text(self):
        """Test line number for multi-line text blocks."""
        lines = [
            "# Title",  # Line 0
            "",  # Line 1
            "Line 1 of text.",  # Line 2
            "Line 2 of text.",  # Line 3
            "Line 3 of text.",  # Line 4
        ]
        content = "\n".join(lines)

        elements = parse_prompt_to_list(content, normalize=False)

        assert len(elements) == 2
        assert elements[0].line_number == 0  # Title
        # Multi-line text should have line number of first line
        assert elements[1].line_number == 2

    def test_line_numbers_preserved_after_normalization(self):
        """Test that line numbers are preserved after level normalization."""
        lines = [
            "## Title",  # Line 0 (level 2)
            "",  # Line 1
            "Text.",  # Line 2
        ]
        content = "\n".join(lines)

        # Parse with normalization (normalizes ## to #)
        elements = parse_prompt_to_list(content, normalize=True)

        assert len(elements) == 2
        # Line numbers should be preserved even though levels changed
        assert elements[0].line_number == 0
        assert elements[1].line_number == 2
        # But level should be normalized
        assert elements[0].level == 1  # Normalized from 2 to 1

    def test_line_numbers_irregular_multiple_empty_lines(self):
        """Test line number tracking with irregular numbers of empty lines (>1)."""
        lines = [
            "",  # Line 0
            "",  # Line 1
            "",  # Line 2
            "# Main Title",  # Line 3
            "",  # Line 4
            "",  # Line 5
            "",  # Line 6
            "",  # Line 7
            "Paragraph text.",  # Line 8
            "",  # Line 9
            "",  # Line 10
            "",  # Line 11
            "{PROMPT_SUBTREE}",  # Line 12
            "",  # Line 13
            "",  # Line 14
            "",  # Line 15
            "## Subsection",  # Line 16
            "",  # Line 17
            "",  # Line 18
            "",  # Line 19
            "More text.",  # Line 20
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        elements = parse_prompt_to_list(content, normalize=False)

        assert len(elements) == 5
        # Verify line numbers track correctly despite irregular spacing
        assert elements[0].line_number == 3  # # Main Title
        assert elements[1].line_number == 8  # Paragraph text
        assert elements[2].line_number == 12  # {PROMPT_SUBTREE}
        assert elements[3].line_number == 16  # ## Subsection
        assert elements[4].line_number == 20  # More text
