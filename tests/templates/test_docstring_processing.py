"""
Tests for docstring processing with line tracking.

This module tests the processing of docstrings from inheritance chains,
including line number tracking, command extraction, and template variable
validation across multiple classes.
"""

from langtree.exceptions import TemplateVariableNameError
from langtree.templates.variables import (
    merge_processed_docstrings,
    process_class_docstring,
)


class TestProcessClassDocstring:
    """Test processing single class docstrings with line tracking."""

    def test_empty_docstring(self):
        """Test processing empty docstring."""
        # Use list of line strings (no \n inside each string)
        lines = []
        content = "\n".join(lines)

        # Confirm no \n in individual strings
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "TestClass")

        assert result.commands == ()
        assert result.clean == ""
        assert result.line_offset == 0
        assert result.source_class == "TestClass"
        assert result.merged_start == 0

    def test_simple_content_no_commands(self):
        """Test content with no commands."""
        lines = [
            "Simple docstring content.",
            "Second line of content.",
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "MyClass")

        assert result.commands == ()
        assert result.clean == "Simple docstring content.\nSecond line of content."
        assert result.line_offset == 0
        assert result.source_class == "MyClass"

    def test_commands_at_top_with_empty_lines(self):
        """Test commands with empty lines before and between them."""
        lines = [
            "",  # Empty line before first command
            "",
            "! @all->task.analyzer@{{value=*}}",
            "",  # Empty line between commands
            "! @each[items]->task.processor@{{value.item=items}}*",
            "",  # Empty line after commands
            "Process items in the collection.",
            "Return results.",
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "ProcessorClass")

        # Commands extracted
        assert len(result.commands) == 2
        assert result.commands[0].text == "! @all->task.analyzer@{{value=*}}"
        assert result.commands[0].line == 2  # Line index where command starts
        assert (
            result.commands[1].text
            == "! @each[items]->task.processor@{{value.item=items}}*"
        )
        assert result.commands[1].line == 4

        # Clean content (stripped)
        assert result.clean == "Process items in the collection.\nReturn results."

        # Line offset: lines 0-6 were removed (2 empty + 2 commands + 1 empty + 1 empty after second cmd)
        # Clean starts at line 6
        assert result.line_offset == 6

    def test_multiline_command_with_brackets(self):
        """Test multiline command with brackets spanning multiple lines."""
        lines = [
            "! @each[items.nested]->task.handler@{{",
            "    value.x=items.nested.a,",
            "    value.y=items.nested.b",
            "}}*",
            "",
            "Handle nested items.",
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "HandlerClass")

        assert len(result.commands) == 1
        assert result.commands[0].line == 0  # Starts at line 0
        # Multiline command preserved
        assert "items.nested" in result.commands[0].text
        assert "value.x=items.nested.a" in result.commands[0].text

        assert result.clean == "Handle nested items."
        assert (
            result.line_offset == 5
        )  # Lines 0-4 removed (command spans lines 0-3, then empty line 4)

    def test_content_with_leading_trailing_whitespace(self):
        """Test that clean content is stripped."""
        lines = [
            "! @all->task@{{value=*}}",
            "",
            "   ",  # Whitespace-only line
            "Content line.",
            "",
            "   ",  # Trailing whitespace
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "TestClass")

        # Stripped clean content
        assert result.clean == "Content line."
        assert result.line_offset == 3  # Command + empty + whitespace removed

    def test_template_variable_in_clean_content(self):
        """Test docstring with template variable in content."""
        lines = [
            "! @all->outputs.result@{{value=*}}",
            "",
            "Main content.",
            "",
            "{PROMPT_SUBTREE}",
            "",
            "Footer content.",
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "NodeClass")

        assert len(result.commands) == 1
        assert "{PROMPT_SUBTREE}" in result.clean
        assert result.line_offset == 2  # Command + empty line


class TestDuplicateTemplateValidation:
    """Test duplicate template variable detection."""

    def test_no_template_variables(self):
        """Test content with no template variables passes validation."""
        lines = [
            "Simple content without templates.",
            "More content.",
        ]
        content = "\n".join(lines)

        # Should not raise
        result = process_class_docstring(content, "TestClass")
        assert result.clean == content

    def test_single_template_variable(self):
        """Test content with single template variable passes validation."""
        lines = [
            "Content before template.",
            "",
            "{PROMPT_SUBTREE}",
            "",
            "Content after template.",
        ]
        content = "\n".join(lines)

        # Should not raise
        result = process_class_docstring(content, "TestClass")
        assert "{PROMPT_SUBTREE}" in result.clean

    def test_duplicate_template_variable_raises_error(self):
        """Test that duplicate template variable in same docstring raises error."""
        lines = [
            "Content.",
            "",
            "{PROMPT_SUBTREE}",
            "",
            "More content.",
            "",
            "{PROMPT_SUBTREE}",
            "",
        ]
        content = "\n".join(lines)

        # Should raise error
        try:
            process_class_docstring(content, "DuplicateClass")
            assert False, "Expected TemplateVariableNameError"
        except TemplateVariableNameError as e:
            assert "PROMPT_SUBTREE" in str(e)
            assert "2 times" in str(e) or "appears multiple times" in str(e).lower()

    def test_multiple_different_template_variables_allowed(self):
        """Test that different template variables are allowed."""
        lines = [
            "Content.",
            "",
            "{PROMPT_SUBTREE}",
            "",
            "Middle content.",
            "",
            "{COLLECTED_CONTEXT}",
            "",
        ]
        content = "\n".join(lines)

        # Should not raise
        result = process_class_docstring(content, "TestClass")
        assert "{PROMPT_SUBTREE}" in result.clean
        assert "{COLLECTED_CONTEXT}" in result.clean


class TestMergeProcessedDocstrings:
    """Test merging multiple processed docstrings with line mapping."""

    def test_merge_single_docstring(self):
        """Test merging a single docstring."""
        lines = [
            "Simple content.",
            "Second line.",
        ]
        content = "\n".join(lines)

        processed = process_class_docstring(content, "SingleClass")
        merged, mappings = merge_processed_docstrings([processed])

        assert merged == "Simple content.\nSecond line."
        assert len(mappings) == 1
        assert mappings[0].class_name == "SingleClass"
        assert mappings[0].merged_start == 0
        assert mappings[0].merged_end == 2  # 2 lines
        assert mappings[0].original_offset == 0

    def test_merge_two_docstrings(self):
        """Test merging two docstrings from parent and child classes."""
        # Parent class docstring
        parent_lines = [
            "! @all->task@{{value=*}}",
            "",
            "Parent content line 1.",
            "Parent content line 2.",
        ]
        parent_content = "\n".join(parent_lines)

        # Child class docstring
        child_lines = [
            "Child content line 1.",
            "Child content line 2.",
        ]
        child_content = "\n".join(child_lines)

        parent_processed = process_class_docstring(parent_content, "ParentClass")
        child_processed = process_class_docstring(child_content, "ChildClass")

        merged, mappings = merge_processed_docstrings(
            [parent_processed, child_processed]
        )

        # Check merged content
        expected_merged = "Parent content line 1.\nParent content line 2.\n\nChild content line 1.\nChild content line 2."
        assert merged == expected_merged

        # Check line mappings
        assert len(mappings) == 2

        # Parent mapping
        assert mappings[0].class_name == "ParentClass"
        assert mappings[0].merged_start == 0
        assert mappings[0].merged_end == 2  # Lines 0-1 in merged
        assert mappings[0].original_offset == 2  # Command + empty line removed

        # Child mapping
        assert mappings[1].class_name == "ChildClass"
        assert (
            mappings[1].merged_start == 3
        )  # After parent (2 lines) + separator (1 line)
        assert mappings[1].merged_end == 5  # Lines 3-4 in merged
        assert mappings[1].original_offset == 0  # No lines removed

    def test_merge_with_commands_and_templates(self):
        """Test merging with commands and template variables."""
        # Grandparent with commands
        grandparent_lines = [
            "",
            "! @all->task.processor@{{value=*}}",
            "",
            "! @each[items]->task.handler@{{value.item=items}}*",
            "",
            "Grandparent content.",
            "",
            "{PROMPT_SUBTREE}",
        ]
        grandparent_content = "\n".join(grandparent_lines)

        # Parent with content
        parent_lines = [
            "Parent content line 1.",
            "Parent content line 2.",
        ]
        parent_content = "\n".join(parent_lines)

        # Child with template
        child_lines = [
            "Child content.",
            "",
            "{COLLECTED_CONTEXT}",
        ]
        child_content = "\n".join(child_lines)

        gp = process_class_docstring(grandparent_content, "Grandparent")
        p = process_class_docstring(parent_content, "Parent")
        c = process_class_docstring(child_content, "Child")

        merged, mappings = merge_processed_docstrings([gp, p, c])

        # Verify merged content contains all parts
        assert "Grandparent content." in merged
        assert "{PROMPT_SUBTREE}" in merged
        assert "Parent content line 1." in merged
        assert "Child content." in merged
        assert "{COLLECTED_CONTEXT}" in merged

        # Verify line mappings
        assert len(mappings) == 3
        assert mappings[0].class_name == "Grandparent"
        assert mappings[1].class_name == "Parent"
        assert mappings[2].class_name == "Child"

        # Check line offset for grandparent (5 lines removed: empty + 2 commands + 2 empty)
        assert gp.line_offset == 5

    def test_merge_empty_list(self):
        """Test merging empty list of docstrings."""
        merged, mappings = merge_processed_docstrings([])

        assert merged == ""
        assert mappings == []

    def test_line_number_reverse_lookup(self):
        """Test finding original location from merged line number."""
        # Create multiple docstrings with different offsets
        doc1_lines = ["! command1", "", "Doc1 line 1.", "Doc1 line 2."]
        doc2_lines = ["Doc2 line 1.", "Doc2 line 2.", "Doc2 line 3."]

        doc1 = process_class_docstring("\n".join(doc1_lines), "Class1")
        doc2 = process_class_docstring("\n".join(doc2_lines), "Class2")

        merged, mappings = merge_processed_docstrings([doc1, doc2])

        # Helper function to find original location
        def find_original_line(merged_line: int) -> tuple[str, int]:
            for mapping in mappings:
                if mapping.merged_start <= merged_line < mapping.merged_end:
                    # Calculate line within segment
                    segment_line = merged_line - mapping.merged_start
                    # Add back the offset to get original line
                    original_line = mapping.original_offset + segment_line
                    return (mapping.class_name, original_line)
            raise ValueError(f"Line {merged_line} not found in mappings")

        # Test lookups
        # Merged line 0 → Class1, original line 2 (after "! command1" and "")
        class_name, original_line = find_original_line(0)
        assert class_name == "Class1"
        assert original_line == 2  # Line offset (2) + segment line (0)

        # Merged line 1 → Class1, original line 3
        class_name, original_line = find_original_line(1)
        assert class_name == "Class1"
        assert original_line == 3

        # Merged line 2 is the BLANK separator line - skip testing it
        # Merged line 3 → Class2, original line 0
        class_name, original_line = find_original_line(3)
        assert class_name == "Class2"
        assert original_line == 0  # No offset + segment line 0

        # Merged line 4 → Class2, original line 1
        class_name, original_line = find_original_line(4)
        assert class_name == "Class2"
        assert original_line == 1

    def test_irregular_multiple_empty_lines(self):
        """Test processing with irregular numbers of empty lines (>1)."""
        lines = [
            "",  # Line 0
            "",  # Line 1
            "",  # Line 2
            "! @all->task@{{value=*}}",  # Line 3
            "",  # Line 4
            "",  # Line 5
            "",  # Line 6
            "! @each[items]->handler@{{value.item=items}}*",  # Line 7
            "",  # Line 8
            "",  # Line 9
            "",  # Line 10
            "",  # Line 11
            "Content line 1.",  # Line 12
            "",  # Line 13
            "",  # Line 14
            "Content line 2.",  # Line 15
            "",  # Line 16
            "",  # Line 17
            "",  # Line 18
        ]
        content = "\n".join(lines)
        assert all("\n" not in line for line in lines)

        result = process_class_docstring(content, "IrregularClass")

        # Commands extracted
        assert len(result.commands) == 2
        assert result.commands[0].line == 3
        assert result.commands[1].line == 7

        # Clean content stripped (preserves internal empty lines between content)
        assert result.clean == "Content line 1.\n\n\nContent line 2."

        # Line offset: everything up to line 12 removed
        assert result.line_offset == 12
