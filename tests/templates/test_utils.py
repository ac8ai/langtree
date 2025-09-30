"""
Tests for template utility functions.

Focus Areas:
1. Command extraction and text processing
2. Template content manipulation
"""

from langtree.templates.utils import extract_commands


class TestCommandExtraction:
    """Test command extraction functionality."""

    def test_extract_commands_basic(self):
        """Test basic command extraction."""
        content = """! command1
        ! command2

        Regular text here"""

        commands, clean_content = extract_commands(content)

        assert len(commands) == 2
        assert "command1" in commands[0].text
        assert "command2" in commands[1].text
        assert isinstance(commands[0].line, int)
        assert isinstance(commands[1].line, int)
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


class TestExtractCommandsLineNumbers:
    """Test that extract_commands tracks line numbers."""

    def test_single_command_at_line_0(self):
        """Test single command on first line."""
        content = "! @all->target@{{value=*}}\nThis is content"
        commands, clean = extract_commands(content)

        assert len(commands) == 1
        assert commands[0].text == "! @all->target@{{value=*}}"
        assert commands[0].line == 0
        assert clean == "This is content"

    def test_command_after_blank_line(self):
        """Test command with blank line before it."""
        content = "\n\n! @all->target@{{value=*}}\nContent here"
        commands, clean = extract_commands(content)

        assert len(commands) == 1
        assert commands[0].line == 2  # Third line (0-indexed)

    def test_multiple_commands_track_individual_lines(self):
        """Test multiple commands each track their starting line."""
        content = """! model=gpt-4
! @all->target@{{value=*}}

! @each[items]->processor@{{data=*}}
Content"""
        commands, clean = extract_commands(content)

        assert len(commands) == 3
        assert commands[0].line == 0  # First line
        assert commands[1].line == 1  # Second line
        assert commands[2].line == 3  # Fourth line (after blank)

    def test_multiline_command_reports_start_line(self):
        """Test multiline command reports its starting line number."""
        content = """! @all->target@{{
    value.field=source.data,
    other=*
}}
Content here"""
        commands, clean = extract_commands(content)

        assert len(commands) == 1
        assert commands[0].line == 0  # Starts on first line
        assert "value.field" in commands[0].text

    def test_command_in_docstring_with_prefix_text(self):
        """Test command line number relative to full docstring."""
        content = """Task description here.

Some more text.

! model=gpt-4
! @all->target@{{value=*}}

Final content."""
        commands, clean = extract_commands(content)

        # Commands won't be found because they appear after regular text
        # This tests the "commands only at start" rule
        assert len(commands) == 0

    def test_commands_at_start_with_content_after(self):
        """Test commands at start of docstring."""
        content = """! model=gpt-4
! @all->target@{{value=*}}

Task description here.

Some more text."""
        commands, clean = extract_commands(content)

        assert len(commands) == 2
        assert commands[0].line == 0
        assert commands[1].line == 1
        assert "Task description" in clean

    def test_empty_content_returns_empty_list(self):
        """Test empty content returns empty list."""
        commands, clean = extract_commands(None)
        assert commands == []

        commands, clean = extract_commands("")
        assert commands == []

    def test_extracted_command_structure(self):
        """Test return structure is list of ExtractedCommand instances."""
        from langtree.templates.utils import ExtractedCommand

        content = "! model=gpt-4\nContent"
        commands, clean = extract_commands(content)

        # Should be list of ExtractedCommand instances
        assert isinstance(commands, list)
        assert isinstance(commands[0], ExtractedCommand)
        assert hasattr(commands[0], "text")
        assert hasattr(commands[0], "line")
        assert isinstance(commands[0].text, str)
        assert isinstance(commands[0].line, int)
