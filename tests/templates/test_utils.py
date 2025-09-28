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
