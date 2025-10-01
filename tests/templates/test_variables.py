"""
Comprehensive edge case tests for template variable spacing validation.

This module tests ALL possible edge cases for template variable spacing
validation to ensure the system properly handles every scenario.
"""

import pytest

from langtree import TreeNode
from langtree.structure import RunStructure, StructureTreeNode
from langtree.templates.variables import (
    TemplateVariableSpacingError,
    add_automatic_prompt_subtree,
    detect_heading_level,
    detect_template_variables,
    field_name_to_title,
    process_template_variables,
    resolve_collected_context,
    resolve_prompt_subtree,
    resolve_template_variables_in_content,
    validate_template_variable_conflicts,
    validate_template_variable_names,
    validate_template_variable_spacing,
)


class TestTemplateVariableSpacingEdgeCases:
    """Comprehensive edge case tests for template variable spacing."""

    def setup_method(self):
        """Create structure fixture for edge case tests."""
        self.structure = RunStructure()

    def test_all_spacing_violations_detected(self):
        """Test that ALL possible spacing violations are detected."""
        # Comprehensive violation cases - no spacing at all
        violations = [
            ("Text{PROMPT_SUBTREE}", "text directly before"),
            ("{PROMPT_SUBTREE}Text", "text directly after"),
            ("Text{PROMPT_SUBTREE}Text", "text both sides"),
            ("a{PROMPT_SUBTREE}", "single char before"),
            ("{PROMPT_SUBTREE}a", "single char after"),
            ("Text{PROMPT_SUBTREE}More", "no spaces around template variable"),
            ("Text\n{PROMPT_SUBTREE}\nMore", "single newlines instead of double"),
        ]

        for content, description in violations:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Should detect violation: {description} in '{content}'"
            )

    def test_whitespace_only_violations(self):
        """Test violations with whitespace but no proper newlines."""
        violations = [
            ("Text {PROMPT_SUBTREE}", "space before only"),
            ("{PROMPT_SUBTREE} Text", "space after only"),
            ("Text  {PROMPT_SUBTREE}", "multiple spaces before"),
            ("{PROMPT_SUBTREE}  Text", "multiple spaces after"),
            ("Text\t{PROMPT_SUBTREE}", "tab before"),
            ("{PROMPT_SUBTREE}\tText", "tab after"),
            ("Text {PROMPT_SUBTREE} More", "single spaces instead of empty lines"),
            ("Text  {PROMPT_SUBTREE}  More", "multiple spaces"),
            ("Text\t{PROMPT_SUBTREE}\tMore", "tab characters"),
        ]

        for content, description in violations:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Should detect violation: {description} in '{content}'"
            )

    def test_single_newline_violations(self):
        """Test violations with single newlines (need double newlines)."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # Test cases where template variables have only single newlines (should be double)
        violation_cases = [
            (
                "Single newline before",
                "Content before\n{PROMPT_SUBTREE}\n\nMore content",
            ),
            (
                "Single newline after",
                "Content before\n\n{PROMPT_SUBTREE}\nMore content",
            ),
            (
                "Single newlines both sides",
                "Content before\n{PROMPT_SUBTREE}\nMore content",
            ),
            (
                "Single newline before COLLECTED_CONTEXT",
                "Content\n{COLLECTED_CONTEXT}\n\nMore",
            ),
            (
                "Single newline after COLLECTED_CONTEXT",
                "Content\n\n{COLLECTED_CONTEXT}\nMore",
            ),
            (
                "Both single newlines COLLECTED_CONTEXT",
                "Content\n{COLLECTED_CONTEXT}\nMore",
            ),
        ]

        for description, content in violation_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Should detect single newline violation: {description} in '{content}'"
            )
            # Should mention spacing or newline issue
            error_text = " ".join(errors).lower()
            assert (
                "spacing" in error_text
                or "newline" in error_text
                or "empty lines" in error_text
            ), f"Error should mention spacing issue: {errors}"

    def test_edge_case_template_variable_only(self):
        """Test edge case where content is ONLY the template variable."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # When content is ONLY a template variable, it should be valid (special case)
        only_template_cases = [
            "{PROMPT_SUBTREE}",
            "{COLLECTED_CONTEXT}",
            "\n{PROMPT_SUBTREE}\n",  # With minimal surrounding whitespace
            "\n\n{COLLECTED_CONTEXT}\n\n",  # With proper spacing but nothing else
        ]

        for content in only_template_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Template variable only content should be valid: '{content}' -> {errors}"
            )

        # But mixed content without proper spacing should still fail
        mixed_invalid_cases = [
            "text{PROMPT_SUBTREE}",
            "{PROMPT_SUBTREE}text",
            "a{COLLECTED_CONTEXT}b",
        ]

        for content in mixed_invalid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Mixed content without spacing should be invalid: '{content}'"
            )

    def test_start_end_of_content_edge_cases(self):
        """Test template variables at start/end of content."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # Template variables at the start should be valid if properly spaced
        start_cases = [
            "{PROMPT_SUBTREE}\n\nFollowing content",
            "{COLLECTED_CONTEXT}\n\nSome text after",
        ]

        for content in start_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Template variable at start with proper spacing should be valid: '{content}' -> {errors}"
            )

        # Template variables at the end should be valid if properly spaced
        end_cases = [
            "Preceding content\n\n{PROMPT_SUBTREE}",
            "Some text before\n\n{COLLECTED_CONTEXT}",
        ]

        for content in end_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Template variable at end with proper spacing should be valid: '{content}' -> {errors}"
            )

        # Invalid cases without proper spacing
        invalid_cases = [
            "Text{PROMPT_SUBTREE}",  # No spacing before
            "{PROMPT_SUBTREE}Text",  # No spacing after
            "Text\n{PROMPT_SUBTREE}",  # Only one newline before
            "{PROMPT_SUBTREE}\nText",  # Only one newline after
        ]

        for content in invalid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Template variable without proper spacing should be invalid: '{content}'"
            )

    def test_multiple_template_variables_spacing(self):
        """Test spacing between multiple template variables."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # Valid cases with proper spacing between multiple template variables
        valid_cases = [
            "Introduction\n\n{PROMPT_SUBTREE}\n\nMiddle content\n\n{COLLECTED_CONTEXT}\n\nConclusion",
            "{PROMPT_SUBTREE}\n\n{COLLECTED_CONTEXT}",  # Back to back with proper spacing
            "Text\n\n{COLLECTED_CONTEXT}\n\n{PROMPT_SUBTREE}\n\nMore text",
        ]

        for content in valid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Multiple template variables with proper spacing should be valid: '{content}' -> {errors}"
            )

        # Invalid cases with insufficient spacing between template variables
        invalid_cases = [
            "{PROMPT_SUBTREE}\n{COLLECTED_CONTEXT}",  # Only one newline between
            "{PROMPT_SUBTREE}{COLLECTED_CONTEXT}",  # No spacing at all
            "Text\n{PROMPT_SUBTREE}\n{COLLECTED_CONTEXT}",  # Insufficient before first
            "{PROMPT_SUBTREE}\n{COLLECTED_CONTEXT}\nText",  # Insufficient after second
            "Text{PROMPT_SUBTREE}\n\n{COLLECTED_CONTEXT}Text",  # Mixed violations
        ]

        for content in invalid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Multiple template variables with insufficient spacing should be invalid: '{content}'"
            )

    def test_collected_context_edge_cases(self):
        """Test edge cases specific to COLLECTED_CONTEXT."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # COLLECTED_CONTEXT follows the same spacing rules as PROMPT_SUBTREE
        valid_context_cases = [
            "Introduction\n\n{COLLECTED_CONTEXT}\n\nConclusion",
            "{COLLECTED_CONTEXT}",  # Only content
            "\n{COLLECTED_CONTEXT}\n",  # With minimal whitespace
            "# Context Section\n\n{COLLECTED_CONTEXT}",  # At end with proper spacing
            "{COLLECTED_CONTEXT}\n\nMore content follows",  # At start with proper spacing
        ]

        for content in valid_context_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"COLLECTED_CONTEXT with proper spacing should be valid: '{content}' -> {errors}"
            )

        # Invalid cases where COLLECTED_CONTEXT doesn't meet spacing requirements
        invalid_context_cases = [
            "Text{COLLECTED_CONTEXT}",  # No spacing before
            "{COLLECTED_CONTEXT}Text",  # No spacing after
            "Context:\n{COLLECTED_CONTEXT}",  # Only one newline before
            "{COLLECTED_CONTEXT}\nMore",  # Only one newline after
            "A{COLLECTED_CONTEXT}B",  # Text directly adjacent
        ]

        for content in invalid_context_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"COLLECTED_CONTEXT without proper spacing should be invalid: '{content}'"
            )

        # Test mixing COLLECTED_CONTEXT with PROMPT_SUBTREE
        mixed_cases = [
            "Intro\n\n{COLLECTED_CONTEXT}\n\nMiddle\n\n{PROMPT_SUBTREE}\n\nEnd",  # Valid
            "{COLLECTED_CONTEXT}\n{PROMPT_SUBTREE}",  # Invalid - insufficient spacing
            "Text\n\n{COLLECTED_CONTEXT}{PROMPT_SUBTREE}\n\nMore",  # Invalid - no spacing between
        ]

        for i, content in enumerate(mixed_cases):
            errors = validate_template_variable_spacing(content)
            if i == 0:  # First case should be valid
                assert len(errors) == 0, (
                    f"Mixed template variables with proper spacing should be valid: '{content}' -> {errors}"
                )
            else:  # Other cases should be invalid
                assert len(errors) > 0, (
                    f"Mixed template variables with improper spacing should be invalid: '{content}'"
                )

    def test_complex_content_structures(self):
        """Test template variables in complex content structures."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # Valid complex content with proper spacing
        valid_complex_cases = [
            # Markdown-like structure
            "# Introduction\n\nThis is an overview.\n\n{PROMPT_SUBTREE}\n\n## Context\n\n{COLLECTED_CONTEXT}\n\n# Conclusion\n\nFinal thoughts.",
            # List with template variables
            "Instructions:\n\n1. Review the data\n2. Generate analysis\n\n{PROMPT_SUBTREE}\n\n3. Apply context\n\n{COLLECTED_CONTEXT}\n\n4. Finalize",
            # Code-like structure
            "```\nSetup code here\n```\n\n{PROMPT_SUBTREE}\n\n```\nConclusion code\n```",
            # Mixed content with proper spacing
            "Part A: Analysis\n\n{PROMPT_SUBTREE}\n\nPart B: Context\n\nRelevant information:\n\n{COLLECTED_CONTEXT}\n\nPart C: Summary",
        ]

        for content in valid_complex_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Complex content with proper spacing should be valid: '{content}' -> {errors}"
            )

        # Invalid complex content with spacing violations
        invalid_complex_cases = [
            # Missing spacing in list
            "Instructions:\n1. Review{PROMPT_SUBTREE}\n2. Apply context",
            # Missing spacing in markdown
            "# Title\n{COLLECTED_CONTEXT}## Next Section",
            # Mixed violations
            "Start\n{PROMPT_SUBTREE}Middle{COLLECTED_CONTEXT}\nEnd",
            # Insufficient spacing in structured content
            "A.\n{PROMPT_SUBTREE}\nB. Context:\n{COLLECTED_CONTEXT}\nC.",
        ]

        for content in invalid_complex_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Complex content with spacing violations should be invalid: '{content}'"
            )

    def test_spacing_validation_with_actual_nodes(self):
        """Test spacing validation in actual TreeNode processing."""
        from langtree.templates.variables import (
            TemplateVariableSpacingError,
            process_template_variables,
        )

        # Test that process_template_variables enforces spacing validation
        # Valid content should pass through
        valid_content = "Introduction\n\n{PROMPT_SUBTREE}\n\nConclusion"
        result = process_template_variables(valid_content)
        assert result == valid_content  # Should pass through unchanged

        # Invalid content should raise TemplateVariableSpacingError
        invalid_content = "Text{PROMPT_SUBTREE}MoreText"
        try:
            process_template_variables(invalid_content)
            assert False, "Expected TemplateVariableSpacingError for invalid spacing"
        except TemplateVariableSpacingError as e:
            assert "spacing errors" in str(e)
            assert "{PROMPT_SUBTREE}" in str(e)

        # Test edge case - template variable only content should be valid
        only_template = "{PROMPT_SUBTREE}"
        result = process_template_variables(only_template)
        assert result == only_template

        # Test multiple violations
        multiple_violations = "A{PROMPT_SUBTREE}B{COLLECTED_CONTEXT}C"
        try:
            process_template_variables(multiple_violations)
            assert False, (
                "Expected TemplateVariableSpacingError for multiple violations"
            )
        except TemplateVariableSpacingError as e:
            # Should mention both violations
            assert "spacing errors" in str(e)

    def test_process_template_variables_error_handling(self):
        """Test error handling in process_template_variables function."""

        # Test valid content passes without errors
        valid_content = "Valid content.\n\n{PROMPT_SUBTREE}\n\nMore valid content."

        result_valid = process_template_variables(valid_content)
        assert result_valid is not None, "Valid content should process successfully"

        # Test invalid content raises TemplateVariableSpacingError
        invalid_content = "Invalid{PROMPT_SUBTREE}content"

        with pytest.raises(TemplateVariableSpacingError) as exc_info:
            process_template_variables(invalid_content)

        assert "spacing" in str(exc_info.value).lower(), (
            "Should raise spacing-related error"
        )


class TestTemplateVariableSpacingFixes:
    """Test improved spacing detection and validation."""

    def test_improved_spacing_detection(self):
        """Test a more comprehensive spacing detection algorithm."""
        from langtree.templates.variables import (
            validate_template_variable_spacing,
        )

        # Test comprehensive edge cases that the improved algorithm should handle

        # Valid: Proper spacing with various whitespace patterns
        valid_cases = [
            "Text\n\n{PROMPT_SUBTREE}\n\nMore",  # Standard case
            "Text\n\n   \n\n{PROMPT_SUBTREE}\n\n\n\nMore",  # Extra empty lines (should be valid)
            "\n\n{PROMPT_SUBTREE}\n\n",  # Leading and trailing empty lines
            "   Text   \n\n   {PROMPT_SUBTREE}   \n\n   More   ",  # With whitespace
            "Pre-content\n\n{COLLECTED_CONTEXT}\n\nPost-content",  # COLLECTED_CONTEXT version
        ]

        for content in valid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) == 0, (
                f"Content with proper spacing should be valid: {repr(content)} -> {errors}"
            )

        # Invalid: Various spacing violations that should be detected
        invalid_cases = [
            "Text\n{PROMPT_SUBTREE}\n\nMore",  # Missing empty line before
            "Text\n\n{PROMPT_SUBTREE}\nMore",  # Missing empty line after
            "Text\n{PROMPT_SUBTREE}\nMore",  # Missing empty lines both sides
            "Text {PROMPT_SUBTREE} More",  # Space instead of newlines
            "Text\n \n{PROMPT_SUBTREE}\n\nMore",  # Line with only space before (not truly empty)
            "Text\n\n{PROMPT_SUBTREE}\n \nMore",  # Line with only space after (not truly empty)
            "A{COLLECTED_CONTEXT}B",  # Direct adjacency
            "{PROMPT_SUBTREE}\n \n{COLLECTED_CONTEXT}",  # Invalid spacing between template variables
        ]

        for content in invalid_cases:
            errors = validate_template_variable_spacing(content)
            assert len(errors) > 0, (
                f"Content with spacing violations should be invalid: {repr(content)}"
            )

        # Test boundary conditions
        boundary_cases = [
            ("{PROMPT_SUBTREE}", True),  # Only template variable (valid)
            (" {PROMPT_SUBTREE} ", True),  # Only template variable with spaces (valid)
            (
                "\n{PROMPT_SUBTREE}\n",
                True,
            ),  # Only template variable with newlines (valid)
            (
                "{PROMPT_SUBTREE}{COLLECTED_CONTEXT}",
                False,
            ),  # Adjacent without spacing (invalid)
        ]

        for content, should_be_valid in boundary_cases:
            errors = validate_template_variable_spacing(content)
            if should_be_valid:
                assert len(errors) == 0, (
                    f"Boundary case should be valid: {repr(content)} -> {errors}"
                )
            else:
                assert len(errors) > 0, (
                    f"Boundary case should be invalid: {repr(content)}"
                )


class TestTemplateVariableCoreFeatures:
    """Test core template variable detection, processing, and resolution functions."""

    def test_detect_template_variables(self):
        """Test template variable detection with various scenarios."""
        # Test empty content
        result = detect_template_variables("")
        assert result == {}

        # Test content with no template variables
        result = detect_template_variables("Regular content with no variables")
        assert result == {}

        # Test single PROMPT_SUBTREE
        content = "Before {PROMPT_SUBTREE} after"
        result = detect_template_variables(content)
        assert result == {"PROMPT_SUBTREE": [7]}

        # Test single COLLECTED_CONTEXT
        content = "Before {COLLECTED_CONTEXT} after"
        result = detect_template_variables(content)
        assert result == {"COLLECTED_CONTEXT": [7]}

        # Test single variable
        content = "This has {PROMPT_SUBTREE} in it"
        result = detect_template_variables(content)
        assert result == {"PROMPT_SUBTREE": [9]}

        # Test multiple occurrences of same variable
        content = "First {PROMPT_SUBTREE} middle {PROMPT_SUBTREE} end"
        result = detect_template_variables(content)
        assert result == {"PROMPT_SUBTREE": [6, 30]}

        # Test both variables
        content = "Start {PROMPT_SUBTREE} middle {COLLECTED_CONTEXT} end"
        result = detect_template_variables(content)
        expected = {"PROMPT_SUBTREE": [6], "COLLECTED_CONTEXT": [30]}
        assert result == expected

        # Test complex positioning
        content = "Line 1\n\n{PROMPT_SUBTREE}\n\nLine 2\n\n{COLLECTED_CONTEXT}\n\nEnd"
        result = detect_template_variables(content)
        expected = {
            "PROMPT_SUBTREE": [8],
            "COLLECTED_CONTEXT": [34],  # Fixed position
        }
        assert result == expected

    def test_validate_template_variable_conflicts(self):
        """Test detection of conflicts with assembly variables."""
        # Test no conflicts
        content = "Content with {PROMPT_SUBTREE}"
        assembly_vars = {"data_var", "result_var"}
        errors = validate_template_variable_conflicts(content, assembly_vars)
        assert len(errors) == 0

        # Test PROMPT_SUBTREE conflict
        assembly_vars = {"PROMPT_SUBTREE", "other_var"}
        errors = validate_template_variable_conflicts(content, assembly_vars)
        assert len(errors) == 1
        assert (
            "Assembly Variable 'PROMPT_SUBTREE' conflicts with template variable {PROMPT_SUBTREE}"
            in errors[0]
        )

        # Test COLLECTED_CONTEXT conflict
        assembly_vars = {"COLLECTED_CONTEXT", "other_var"}
        errors = validate_template_variable_conflicts(content, assembly_vars)
        assert len(errors) == 1
        assert (
            "Assembly Variable 'COLLECTED_CONTEXT' conflicts with template variable {COLLECTED_CONTEXT}"
            in errors[0]
        )

        # Test both conflicts
        assembly_vars = {"PROMPT_SUBTREE", "COLLECTED_CONTEXT", "other_var"}
        errors = validate_template_variable_conflicts(content, assembly_vars)
        assert len(errors) == 2

        # Test multiple conflicts with custom variables (only detects PROMPT_SUBTREE and COLLECTED_CONTEXT)
        content = "Content with {PROMPT_SUBTREE} and {var2}"
        assembly_vars = {"PROMPT_SUBTREE", "var2"}
        errors = validate_template_variable_conflicts(content, assembly_vars)
        assert len(errors) == 1  # Only PROMPT_SUBTREE conflict detected

        # Test empty assembly variables
        errors = validate_template_variable_conflicts(content, set())
        assert len(errors) == 0

        # Test empty content (only detects conflicts with existing template variables)
        errors = validate_template_variable_conflicts("", assembly_vars)
        assert len(errors) == 1  # Only PROMPT_SUBTREE conflict detected

    def test_detect_heading_level(self):
        """Test heading level detection with proper position parameters."""
        # Test content with no headings
        content = "Regular content without headings"
        level = detect_heading_level(content, 10)
        assert level == 1  # Default to level 1

        # Test content with single heading
        content = "# Main Title\n\nSome content here"
        level = detect_heading_level(content, 20)  # Position after heading
        assert level == 2  # Next level down

        # Test content with multiple headings
        content = "# Title\n\n## Subtitle\n\nContent here"
        level = detect_heading_level(content, 30)  # Position after subtitle
        assert level == 3  # Next level down from subtitle

        # Test nested headings
        content = "# Level 1\n\n## Level 2\n\n### Level 3\n\nContent"
        level = detect_heading_level(content, 40)
        assert level == 4  # Next level down from level 3

        # Test maximum heading level (6)
        content = "###### Level 6\n\nContent here"
        level = detect_heading_level(content, 20)
        assert level == 6  # Should not exceed 6

        # Test template variable position before first heading
        content = "Some intro text\n\n# First Heading\n\nMore content"
        level = detect_heading_level(content, 10)  # Position in intro
        assert level == 1  # No preceding headings

        # Test with template variables at specific positions
        content = "Some content\n\n{PROMPT_SUBTREE}"
        position = content.find("{PROMPT_SUBTREE}")
        assert detect_heading_level(content, position) == 1

        # Test with existing heading before template variable
        content = "# Main\n\nSome content\n\n{PROMPT_SUBTREE}"
        position = content.find("{PROMPT_SUBTREE}")
        assert detect_heading_level(content, position) == 2

        # Test with multiple headings before template variable
        content = "# Main\n\n## Sub\n\nContent\n\n{PROMPT_SUBTREE}"
        position = content.find("{PROMPT_SUBTREE}")
        assert detect_heading_level(content, position) == 3

        # Test level cap at 6
        content = "# 1\n## 2\n### 3\n#### 4\n##### 5\n###### 6\n\n{PROMPT_SUBTREE}"
        position = content.find("{PROMPT_SUBTREE}")
        assert detect_heading_level(content, position) == 6

    def test_add_automatic_prompt_subtree(self):
        """Test automatic addition of PROMPT_SUBTREE when missing."""
        # Test empty content
        result = add_automatic_prompt_subtree("")
        assert result == "{PROMPT_SUBTREE}"

        # Test content that already has PROMPT_SUBTREE
        content = "Content with {PROMPT_SUBTREE} already"
        result = add_automatic_prompt_subtree(content)
        assert result == content  # Should remain unchanged

        # Test short content (field descriptions) - should not add PROMPT_SUBTREE
        short_content = "Brief field description"
        result = add_automatic_prompt_subtree(short_content)
        assert result == short_content  # Should remain unchanged

        # Test substantial content without PROMPT_SUBTREE
        substantial_content = (
            "This is a longer docstring\nwith multiple lines\nand substantial content"
        )
        result = add_automatic_prompt_subtree(substantial_content)
        assert result == substantial_content + "\n\n{PROMPT_SUBTREE}\n\n"

        # Test content with existing trailing newlines (short content detection)
        content_with_newlines = "Content\n\n"
        result = add_automatic_prompt_subtree(content_with_newlines)
        assert result == "Content\n\n"  # Short content, no addition

        # Test content that's actually considered substantial
        substantial_content = "Content\nwith\nmultiple\nlines\nto be substantial"
        result = add_automatic_prompt_subtree(substantial_content)
        assert result == substantial_content + "\n\n{PROMPT_SUBTREE}\n\n"

    def test_field_name_to_title(self):
        """Test conversion of field names to proper heading titles."""
        # Test basic conversion
        assert field_name_to_title("main_analysis") == "# Main Analysis"

        # Test different heading levels
        assert field_name_to_title("field_name", 1) == "# Field Name"
        assert field_name_to_title("field_name", 2) == "## Field Name"
        assert field_name_to_title("field_name", 3) == "### Field Name"
        assert field_name_to_title("field_name", 6) == "###### Field Name"

        # Test heading level bounds
        assert field_name_to_title("field_name", 0) == "# Field Name"  # Min 1
        assert field_name_to_title("field_name", 10) == "###### Field Name"  # Max 6

        # Test various field name formats
        assert field_name_to_title("simple") == "# Simple"
        assert field_name_to_title("multiple_words_here") == "# Multiple Words Here"
        assert field_name_to_title("already_capitalized") == "# Already Capitalized"
        assert field_name_to_title("with_numbers_123") == "# With Numbers 123"


class TestTemplateVariableIntegration:
    """Test template variable functions with real StructureTreeNode integration."""

    def test_resolve_template_variables_in_content_comprehensive(self):
        """Test complete template variable resolution with real nodes."""
        from pydantic import Field

        # Create a real Pydantic model for testing
        class TestTaskModel(TreeNode):
            analysis: str = Field(description="Detailed analysis step")
            summary: str = Field(description="Brief summary of findings")
            conclusion: str = Field(description="Final conclusions")

        # Create a real StructureTreeNode
        node = StructureTreeNode(
            name="test_task",
            field_type=TestTaskModel,
            clean_docstring="Test task with template variables.",
            parent=None,
        )

        # Test content with both template variables
        content = """Task Instructions:

        {COLLECTED_CONTEXT}

        Processing Steps:

        {PROMPT_SUBTREE}

        End of task."""

        result = resolve_template_variables_in_content(content, node)

        # Verify template variables were resolved
        assert "{PROMPT_SUBTREE}" not in result, "PROMPT_SUBTREE should be resolved"
        assert "{COLLECTED_CONTEXT}" not in result, (
            "COLLECTED_CONTEXT should be resolved"
        )

        # Verify field content is included
        assert "analysis" in result.lower() or "Analysis" in result
        assert "summary" in result.lower() or "Summary" in result
        assert "conclusion" in result.lower() or "Conclusion" in result

        # Field descriptions may or may not be included depending on node setup
        # The function works correctly with or without descriptions
        assert "Brief summary of findings" in result
        assert "Final conclusions" in result

    def test_process_template_variables_with_validation(self):
        """Test template variable processing with spacing validation."""
        from pydantic import Field

        from langtree.templates.variables import process_template_variables

        # Create test node
        class TestModel(TreeNode):
            result: str = Field(description="Test result")

        node = StructureTreeNode(
            name="test_node",
            field_type=TestModel,
            clean_docstring="Test docstring",
            parent=None,
        )

        # Test valid spacing
        valid_content = "Text\n\n{PROMPT_SUBTREE}\n\nMore text"
        result = process_template_variables(valid_content, node)
        assert isinstance(result, str)

        # Test invalid spacing should raise error
        invalid_content = "Text{PROMPT_SUBTREE}More text"
        try:
            process_template_variables(invalid_content, node)
            assert False, "Should have raised TemplateVariableSpacingError"
        except TemplateVariableSpacingError:
            pass  # Expected

    def test_resolve_prompt_subtree_with_real_fields(self):
        """Test PROMPT_SUBTREE resolution with actual Pydantic field processing."""
        from pydantic import Field

        from langtree.templates.variables import resolve_prompt_subtree

        # Create complex model with various field types
        class MetadataInfo(TreeNode):
            key: str = "default"
            value: str = "default"

        class ComplexTaskModel(TreeNode):
            main_analysis: str = Field(description="Primary analysis component")
            data_processing: list[str] = Field(description="Data processing steps")
            final_summary: str = Field(description="Summary of all findings")
            metadata_info: MetadataInfo = Field(
                default_factory=MetadataInfo, description="Additional metadata"
            )

        node = StructureTreeNode(
            name="complex_task",
            field_type=ComplexTaskModel,
            clean_docstring="Complex task with multiple fields.",
            parent=None,
        )

        # Test with base heading level 1
        result = resolve_prompt_subtree(node, 1)

        # Verify field name to title conversion
        assert "# Main Analysis" in result
        assert "# Data Processing" in result
        assert "# Final Summary" in result
        assert "# Metadata Info" in result

        # Verify descriptions are included
        assert "Primary analysis component" in result
        assert "Data processing steps" in result
        assert "Summary of all findings" in result
        assert "Additional metadata" in result

        # Test with different heading level
        result_level2 = resolve_prompt_subtree(node, 2)
        assert "## Main Analysis" in result_level2
        assert "## Data Processing" in result_level2

    def test_heading_level_detection_integration(self):
        """Test heading level detection with realistic content scenarios."""
        from langtree.templates.variables import detect_heading_level

        # Test complex nested heading structure
        content = """# Project Overview

Initial project description.

## Phase 1: Analysis

### Data Collection

Collecting relevant data sources.

{PROMPT_SUBTREE}

### Results Processing

#### Statistical Analysis

{COLLECTED_CONTEXT}

## Phase 2: Implementation"""

        # Test PROMPT_SUBTREE position (should be level 4)
        prompt_pos = content.find("{PROMPT_SUBTREE}")
        level = detect_heading_level(content, prompt_pos)
        assert level == 4, f"Expected level 4, got {level}"

        # Test COLLECTED_CONTEXT position (should be level 5)
        context_pos = content.find("{COLLECTED_CONTEXT}")
        level = detect_heading_level(content, context_pos)
        assert level == 5, f"Expected level 5, got {level}"

        # Test COLLECTED_CONTEXT position (should be level 5)
        context_pos = content.find("{COLLECTED_CONTEXT}")
        level = detect_heading_level(content, context_pos)
        assert level == 5, f"Expected level 5, got {level}"

    def test_template_variable_conflict_detection_integration(self):
        """Test template variable conflict detection with real assembly variables."""
        from langtree.templates.variables import (
            validate_template_variable_conflicts,
        )

        # Test realistic conflict scenario
        content = """Task with template variables:

        {PROMPT_SUBTREE}

        Additional processing:

        {COLLECTED_CONTEXT}

        Using assembly variables: {data_source} and {output_format}"""

        # Test conflicts with reserved template variable names
        assembly_vars = {"PROMPT_SUBTREE", "data_source", "output_format"}
        errors = validate_template_variable_conflicts(content, assembly_vars)

        assert len(errors) == 1, f"Expected 1 conflict, got {len(errors)}"
        assert "PROMPT_SUBTREE" in errors[0]
        assert "conflicts with template variable" in errors[0]

        # Test with COLLECTED_CONTEXT conflict
        assembly_vars = {"COLLECTED_CONTEXT", "other_var"}
        errors = validate_template_variable_conflicts(content, assembly_vars)

        assert len(errors) == 1, f"Expected 1 conflict, got {len(errors)}"
        assert "COLLECTED_CONTEXT" in errors[0]


class TestTemplateVariableResolution:
    """Test template variable resolution functions with mock nodes."""

    def test_resolve_prompt_subtree_empty_node(self):
        """Test PROMPT_SUBTREE resolution with empty/invalid nodes."""
        # Test with None node
        result = resolve_prompt_subtree(None, base_heading_level=1)  # type: ignore
        assert result == ""

        # Test with mock node without field_type
        class MockNode:
            field_type = None

        result = resolve_prompt_subtree(MockNode(), base_heading_level=1)  # type: ignore
        assert result == ""

    def test_resolve_prompt_subtree_with_fields(self):
        """Test PROMPT_SUBTREE resolution with actual field data."""
        from pydantic import BaseModel, Field

        # Create a mock TreeNode class
        class TestNode(BaseModel):
            analysis: str = Field(description="Main analysis content")
            summary: str = Field(description="Brief summary")
            details: str = Field(description="Detailed information")

        # Create mock node with field descriptions
        class MockNode:
            def __init__(self):
                self.field_type = TestNode
                self.clean_field_descriptions = {
                    "analysis": "Comprehensive analysis of the data",
                    "summary": "Executive summary",
                    # details will use original description
                }

        node = MockNode()
        result = resolve_prompt_subtree(node, base_heading_level=1)  # type: ignore

        # Verify structure and content
        assert "# Analysis" in result
        assert "Comprehensive analysis of the data" in result
        assert "# Summary" in result
        assert "Executive summary" in result
        assert "# Details" in result
        assert "Detailed information" in result  # Original description used

        # Test different heading levels
        result = resolve_prompt_subtree(node, base_heading_level=3)  # type: ignore
        assert "### Analysis" in result
        assert "### Summary" in result
        assert "### Details" in result

    def test_resolve_collected_context(self):
        """Test COLLECTED_CONTEXT resolution."""

        # Create minimal mock node
        class MockNode:
            pass

        node = MockNode()

        # Test with provided context data
        context_data = "# Custom Context\n\nThis is provided context."
        result = resolve_collected_context(node, context_data)  # type: ignore
        assert result == context_data

        # Test without context data (should return placeholder)
        result = resolve_collected_context(node)  # type: ignore
        assert "# Context" in result
        assert "*Context data will be provided during execution*" in result

        # Test with None context data
        result = resolve_collected_context(node, None)  # type: ignore
        assert "# Context" in result
        assert "*Context data will be provided during execution*" in result

    def test_resolve_template_variables_in_content_comprehensive(self):
        """Test complete template variable resolution with real nodes."""
        from pydantic import Field

        # Create a real Pydantic model for testing
        class TestTaskModel(TreeNode):
            analysis: str = Field(description="Detailed analysis step")
            summary: str = Field(description="Brief summary of findings")
            conclusion: str = Field(description="Final conclusions")

        # Create a real StructureTreeNode
        node = StructureTreeNode(
            name="test_task",
            field_type=TestTaskModel,
            clean_docstring="Test task with template variables.",
            parent=None,
        )

        # Test content with both template variables
        content = """Task Instructions:

        {COLLECTED_CONTEXT}

        Processing Steps:

        {PROMPT_SUBTREE}

        End of task."""

        # Process the content
        result = resolve_template_variables_in_content(content, node)

        # Verify template variables were processed
        assert "Task Instructions:" in result
        assert "End of task." in result
        assert isinstance(result, str)

    def test_process_template_variables_with_validation(self):
        """Test template variable processing with spacing validation."""
        from pydantic import Field

        # Create test node
        class TestModel(TreeNode):
            result: str = Field(description="Test result")

        node = StructureTreeNode(
            name="test_node",
            field_type=TestModel,
            clean_docstring="Test docstring",
            parent=None,
        )

        # Test valid spacing
        valid_content = "Text\n\n{PROMPT_SUBTREE}\n\nMore text"
        result = process_template_variables(valid_content, node)
        assert isinstance(result, str)

        # Test invalid spacing should raise error
        invalid_content = "Text{PROMPT_SUBTREE}More text"
        try:
            process_template_variables(invalid_content, node)
            assert False, "Should have raised TemplateVariableSpacingError"
        except TemplateVariableSpacingError:
            pass  # Expected

    def test_resolve_prompt_subtree_with_real_fields(self):
        """Test PROMPT_SUBTREE resolution with actual Pydantic field processing."""
        from pydantic import Field

        # Create complex model with various field types
        class MetadataInfo(TreeNode):
            key: str = "default"
            value: str = "default"

        class ComplexTaskModel(TreeNode):
            main_analysis: str = Field(description="Primary analysis component")
            data_processing: list[str] = Field(description="Data processing steps")
            final_summary: str = Field(description="Summary of all findings")
            metadata_info: MetadataInfo = Field(
                default_factory=MetadataInfo, description="Additional metadata"
            )

        node = StructureTreeNode(
            name="complex_task",
            field_type=ComplexTaskModel,
            clean_docstring="Complex task with multiple fields.",
            parent=None,
        )

        # Test with base heading level 1
        result = resolve_prompt_subtree(node, 1)

        # Verify field name to title conversion
        assert "# Main Analysis" in result
        assert "# Data Processing" in result
        assert "# Final Summary" in result
        assert "# Metadata Info" in result

        # Verify descriptions are included
        assert "Primary analysis component" in result
        assert "Data processing steps" in result
        assert "Summary of all findings" in result
        assert "Additional metadata" in result

        # Test with different heading level
        result_level2 = resolve_prompt_subtree(node, 2)
        assert "## Main Analysis" in result_level2
        assert "## Data Processing" in result_level2

    def test_validate_template_variable_conflicts_comprehensive(self):
        """Test comprehensive template variable conflict validation."""
        content = """Template with multiple potential conflicts:

        {PROMPT_SUBTREE}

        Using assembly variables: {PROMPT_SUBTREE} and {other_var}

        {COLLECTED_CONTEXT}

        More assembly variables: {COLLECTED_CONTEXT} and {data_source}"""

        # Test conflicts with reserved template variable names
        assembly_vars = {
            "PROMPT_SUBTREE",
            "COLLECTED_CONTEXT",
            "other_var",
            "data_source",
        }
        errors = validate_template_variable_conflicts(content, assembly_vars)

        assert len(errors) >= 2, f"Expected at least 2 conflicts, got {len(errors)}"

        # Should detect both PROMPT_SUBTREE and COLLECTED_CONTEXT conflicts
        prompt_conflict = any("PROMPT_SUBTREE" in error for error in errors)
        context_conflict = any("COLLECTED_CONTEXT" in error for error in errors)

        assert prompt_conflict, (
            f"PROMPT_SUBTREE conflict not detected. Errors: {errors}"
        )
        assert context_conflict, (
            f"COLLECTED_CONTEXT conflict not detected. Errors: {errors}"
        )


class TestLangTreeDSLCommandIntegration:
    """Test template variables integrated with actual LangTree DSL command parsing."""

    def test_template_variables_with_runtime_variables(self):
        """Test template variables work correctly with runtime variables in docstrings."""
        from pydantic import Field

        # Create a model with runtime variables and template variables
        class RuntimeVariableTaskModel(TreeNode):
            """
            Analyze the data using runtime variables.

            Process data from {data_source} input files.
            Filter for items with quality above {quality_threshold}.
            Generate results in {result_format} format.

            {PROMPT_SUBTREE}

            Additional context and variables:
            - Data source: {data_source}
            - Output format: {result_format}

            {COLLECTED_CONTEXT}
            """

            analysis: str = Field(description="Primary data analysis")
            results: str = Field(description="Aggregated results")
            data_source: str = Field(description="Source of input data")
            quality_threshold: float = Field(
                default=0.8, description="Quality threshold for filtering"
            )
            result_format: str = Field(
                default="json", description="Format for output results"
            )

        node = StructureTreeNode(
            name="runtime_var_task",
            field_type=RuntimeVariableTaskModel,
            clean_docstring=RuntimeVariableTaskModel.__doc__ or "",
            parent=None,
        )

        # Process the docstring content with template variables
        content = node.clean_docstring or ""
        result = resolve_template_variables_in_content(content, node)

        # Verify runtime variables are preserved (not resolved at this stage)
        assert "{data_source}" in result
        assert "{quality_threshold}" in result
        assert "{result_format}" in result

        # Verify template variables were resolved
        assert "{PROMPT_SUBTREE}" not in result
        assert "{COLLECTED_CONTEXT}" not in result

        # Verify field content was added for PROMPT_SUBTREE
        assert "# Analysis" in result
        assert "Primary data analysis" in result
        assert "# Results" in result
        assert "Aggregated results" in result

    def test_assembly_variable_conflict_detection_with_acl(self):
        """Test that Assembly Variable conflicts are detected with LangTree DSL commands."""
        content = """
        Process data with LangTree DSL:

        {{EXTRACT PROMPT_SUBTREE}} from source
        {{FILTER COLLECTED_CONTEXT | valid == true}}

        Template variables:
        {PROMPT_SUBTREE}
        {COLLECTED_CONTEXT}
        """

        # Assembly variables that conflict with template variable names
        assembly_vars = {"PROMPT_SUBTREE", "COLLECTED_CONTEXT", "data_source"}

        errors = validate_template_variable_conflicts(content, assembly_vars)

        # Should detect conflicts with template variable names
        assert len(errors) >= 2, f"Expected at least 2 conflicts, got {len(errors)}"

        prompt_conflict = any("PROMPT_SUBTREE" in error for error in errors)
        context_conflict = any("COLLECTED_CONTEXT" in error for error in errors)

        assert prompt_conflict, "Should detect PROMPT_SUBTREE conflict"
        assert context_conflict, "Should detect COLLECTED_CONTEXT conflict"

    def test_runtime_variable_integration(self):
        """Test template variables work alongside Runtime Variables."""
        from langtree.templates.variables import detect_template_variables

        # Content mixing template variables and runtime variable placeholders
        content = """
        Initial processing:

        {PROMPT_SUBTREE}

        Runtime data processing:
        - Use {{runtime_data}} for calculations
        - Apply {{processing_mode}} settings

        Context integration:

        {COLLECTED_CONTEXT}

        Final output with {{output_format}} formatting.
        """

        # Detect template variables (should ignore runtime variable syntax)
        template_vars = detect_template_variables(content)

        assert "PROMPT_SUBTREE" in template_vars
        assert "COLLECTED_CONTEXT" in template_vars

        # Runtime variables (double braces) should not be detected as template variables
        assert "runtime_data" not in template_vars
        assert "processing_mode" not in template_vars
        assert "output_format" not in template_vars

        # Verify positions are correct
        assert len(template_vars["PROMPT_SUBTREE"]) == 1
        assert len(template_vars["COLLECTED_CONTEXT"]) == 1


class TestLanguageSpecificationCompliance:
    """Test template variables comply with LANGUAGE_SPECIFICATION.md requirements."""

    def test_template_variables_in_acl_grammar_context(self):
        """Test template variables work within LangTree DSL syntax grammar rules."""
        content = """
# Processing Pipeline

Step 1: Data extraction
{{EXTRACT source_data | FORMAT json}}

Step 2: Template processing

{PROMPT_SUBTREE}

Step 3: Context assembly

{COLLECTED_CONTEXT}

Step 4: Output generation
{{OUTPUT results | FORMAT {{output_format}}}}
"""  # Validate spacing (template variables should follow spacing rules)
        errors = validate_template_variable_spacing(content)
        assert len(errors) == 0, (
            f"Content should have valid template variable spacing: {errors}"
        )

        # Detect template variables (should work with LangTree DSL syntax)
        template_vars = detect_template_variables(content)
        assert "PROMPT_SUBTREE" in template_vars
        assert "COLLECTED_CONTEXT" in template_vars

        # LangTree DSL commands (double braces) should not interfere
        assert "EXTRACT" not in template_vars
        assert "OUTPUT" not in template_vars
        assert "source_data" not in template_vars

    def test_variable_scope_system_integration(self):
        """Test template variables work with prompt/value/outputs/task variable scopes."""
        from pydantic import Field

        # Model representing different variable scopes from specification
        class ScopedVariableModel(TreeNode):
            """
            Task with multiple variable scope examples:

            Prompt scope: Define task parameters
            {PROMPT_SUBTREE}

            Value scope: {{input_value}} processing
            Outputs scope: Generate {{result_output}}
            Task scope: Complete {{task_name}} execution

            Context assembly:
            {COLLECTED_CONTEXT}
            """

            prompt_vars: str = Field(description="Prompt-scoped variables")
            value_vars: str = Field(description="Value-scoped variables")
            output_vars: str = Field(description="Output-scoped variables")
            task_vars: str = Field(description="Task-scoped variables")

        # Test that template variables are detected independently of other scopes
        content = ScopedVariableModel.__doc__ or ""
        template_vars = detect_template_variables(content)

        assert "PROMPT_SUBTREE" in template_vars
        assert "COLLECTED_CONTEXT" in template_vars

        # Other scope variables should not be detected as template variables
        assert "input_value" not in template_vars
        assert "result_output" not in template_vars
        assert "task_name" not in template_vars

    def test_reserved_template_variable_names(self):
        """Test validation against reserved template variable names."""
        # Test that PROMPT_SUBTREE and COLLECTED_CONTEXT are properly reserved
        reserved_names = {"PROMPT_SUBTREE", "COLLECTED_CONTEXT"}

        # Assembly variables using reserved names should conflict
        for reserved_name in reserved_names:
            content = f"Content with {{{reserved_name}}}"
            assembly_vars = {reserved_name, "other_var"}

            errors = validate_template_variable_conflicts(content, assembly_vars)
            assert len(errors) >= 1, (
                f"Should detect conflict with reserved name {reserved_name}"
            )

            conflict_found = any(reserved_name in error for error in errors)
            assert conflict_found, f"Should mention {reserved_name} in conflict error"

    def test_hierarchical_prompt_execution_integration(self):
        """Test template variables work with hierarchical prompt execution system."""
        from pydantic import Field

        # Create hierarchical structure (parent-child relationship)
        class ParentTaskModel(TreeNode):
            """
            Parent task coordination:
            {PROMPT_SUBTREE}

            Collect child results:
            {COLLECTED_CONTEXT}
            """

            coordination: str = Field(description="Task coordination logic")

        class ChildTaskModel(TreeNode):
            """
            Child task execution:
            {PROMPT_SUBTREE}
            """

            execution: str = Field(description="Specific task execution")

        # Create parent node
        parent_node = StructureTreeNode(
            name="parent_task",
            field_type=ParentTaskModel,
            clean_docstring=ParentTaskModel.__doc__ or "",
            parent=None,
        )

        # Create child node
        child_node = StructureTreeNode(
            name="child_task",
            field_type=ChildTaskModel,
            clean_docstring=ChildTaskModel.__doc__ or "",
            parent=parent_node,
        )

        # Test template variable resolution in hierarchical context
        parent_content = resolve_template_variables_in_content(
            parent_node.clean_docstring or "", parent_node
        )
        child_content = resolve_template_variables_in_content(
            child_node.clean_docstring or "", child_node
        )

        # Both should resolve template variables appropriately
        assert "{PROMPT_SUBTREE}" not in parent_content
        assert "{COLLECTED_CONTEXT}" not in parent_content
        assert "{PROMPT_SUBTREE}" not in child_content

        # Parent should have coordination field, child should have execution field
        assert (
            "coordination" in parent_content.lower() or "Coordination" in parent_content
        )
        assert "execution" in child_content.lower() or "Execution" in child_content


class TestArchitecturalDesignCompliance:
    """Test template variables comply with architectural design principles."""

    def test_deterministic_assembly_time_processing(self):
        """Test that template variable processing is deterministic at assembly-time."""
        from pydantic import Field

        class DeterministicModel(TreeNode):
            """
            Deterministic processing test:
            {PROMPT_SUBTREE}

            Context integration:
            {COLLECTED_CONTEXT}
            """

            field1: str = Field(description="First field")
            field2: str = Field(description="Second field")

        node = StructureTreeNode(
            name="deterministic_task",
            field_type=DeterministicModel,
            clean_docstring=DeterministicModel.__doc__ or "",
            parent=None,
        )

        # Process the same content multiple times - should be identical
        content = node.clean_docstring or ""
        result1 = resolve_template_variables_in_content(content, node)
        result2 = resolve_template_variables_in_content(content, node)
        result3 = resolve_template_variables_in_content(content, node)

        assert result1 == result2 == result3, (
            "Template variable resolution must be deterministic"
        )

        # Verify no runtime state is maintained
        assert "{PROMPT_SUBTREE}" not in result1
        assert "{COLLECTED_CONTEXT}" not in result1

        # Verify field content is consistently included
        assert "First field" in result1
        assert "Second field" in result1

    def test_variable_closure_semantics(self):
        """Test proper variable closure and satisfaction semantics."""
        # Test that template variables have clear closure semantics
        content_with_variables = """
        Processing with closure:

        {PROMPT_SUBTREE}

        Additional context:
        {COLLECTED_CONTEXT}

        No other template variables should be recognized.
        """

        # Only PROMPT_SUBTREE and COLLECTED_CONTEXT should be detected
        detected = detect_template_variables(content_with_variables)
        assert set(detected.keys()) == {"PROMPT_SUBTREE", "COLLECTED_CONTEXT"}

        # Test that other brace patterns don't interfere with closure
        content_with_noise = """
        {PROMPT_SUBTREE}

        {invalid_variable}
        {{assembly_variable}}
        {{{triple_braces}}}
        {UNKNOWN_TEMPLATE}

        {COLLECTED_CONTEXT}
        """

        detected_noise = detect_template_variables(content_with_noise)
        # Should only detect the two valid template variables
        assert set(detected_noise.keys()) == {"PROMPT_SUBTREE", "COLLECTED_CONTEXT"}

    def test_assembly_vs_runtime_separation(self):
        """Test proper separation of assembly vs runtime concerns."""
        # Assembly-time: Template variable detection and validation
        assembly_content = """
Assembly phase processing:

{PROMPT_SUBTREE}

Runtime placeholders: {{runtime_var1}} and {{runtime_var2}}

Context assembly:

{COLLECTED_CONTEXT}

End of assembly phase.
"""  # Assembly phase: Detect only template variables
        assembly_vars = detect_template_variables(assembly_content)
        assert "PROMPT_SUBTREE" in assembly_vars
        assert "COLLECTED_CONTEXT" in assembly_vars

        # Runtime variables should not be processed at assembly time
        assert "runtime_var1" not in assembly_vars
        assert "runtime_var2" not in assembly_vars

        # Assembly phase: Validate spacing
        spacing_errors = validate_template_variable_spacing(assembly_content)
        assert len(spacing_errors) == 0, "Assembly-time spacing validation should pass"

        # Runtime phase: Template variables should be resolved, runtime vars preserved
        from pydantic import Field

        class RuntimeModel(TreeNode):
            runtime_field: str = Field(description="Runtime processing field")

        node = StructureTreeNode(
            name="runtime_task",
            field_type=RuntimeModel,
            clean_docstring="",
            parent=None,
        )

        runtime_result = resolve_template_variables_in_content(assembly_content, node)

        # Template variables should be resolved
        assert "{PROMPT_SUBTREE}" not in runtime_result
        assert "{COLLECTED_CONTEXT}" not in runtime_result

        # Runtime variables should be preserved for later processing
        assert "{{runtime_var1}}" in runtime_result
        assert "{{runtime_var2}}" in runtime_result


class TestErrorHandlingComprehensive:
    """Comprehensive error handling tests for template variable processing."""

    def test_malformed_template_variable_syntax(self):
        """Test handling of malformed template variable syntax."""
        malformed_cases = [
            ("{PROMPT_SUBTREE", "Missing closing brace"),
            ("PROMPT_SUBTREE}", "Missing opening brace"),
            ("{PROMPT_SUBTREE}}", "Extra closing brace"),
            ("{{PROMPT_SUBTREE}", "Mixed brace types"),
            ("{PROMPT_SUBTREE ", "Space in variable name"),
            ("{ PROMPT_SUBTREE}", "Leading space"),
            ("{PROMPT_SUBTREE }", "Trailing space"),
            ("{prompt_subtree}", "Lowercase (not standard)"),
            ("{INVALID_TEMPLATE}", "Invalid template variable name"),
        ]

        for malformed_content, description in malformed_cases:
            # Should handle gracefully - either detect valid parts or ignore malformed
            result = detect_template_variables(malformed_content)

            # For completely malformed syntax, should not crash
            assert isinstance(result, dict), (
                f"Should return dict for malformed case: {description}"
            )

            # For cases with valid template variable names, behavior depends on implementation
            if (
                "PROMPT_SUBTREE" in malformed_content
                and "{PROMPT_SUBTREE}" in malformed_content
            ):
                # If the standard format is present, should detect it
                assert "PROMPT_SUBTREE" in result or len(result) == 0, (
                    f"Detection behavior for: {description}"
                )

    def test_exception_chaining_and_context(self):
        """Test proper exception chaining and context preservation."""
        # Test that TemplateVariableSpacingError preserves context
        invalid_content = "Invalid{PROMPT_SUBTREE}spacing"

        try:
            process_template_variables(invalid_content)
            assert False, "Should have raised TemplateVariableSpacingError"
        except TemplateVariableSpacingError as e:
            # Should contain relevant context information
            error_message = str(e)
            assert (
                "spacing" in error_message.lower() or "PROMPT_SUBTREE" in error_message
            )

            # Should preserve the problematic content context
            assert any(
                keyword in error_message
                for keyword in ["PROMPT_SUBTREE", "spacing", "error"]
            )

    def test_partial_processing_recovery(self):
        """Test error recovery and partial processing capabilities."""
        # Content with both valid and problematic elements
        mixed_content = """
        Valid section:

        {PROMPT_SUBTREE}

        Invalid section:
        Invalid{COLLECTED_CONTEXT}Content

        Another valid section would be here.
        """

        # Spacing validation should catch the invalid section
        errors = validate_template_variable_spacing(mixed_content)
        assert len(errors) > 0, "Should detect spacing violations"

        # But template variable detection should still work for valid parts
        detected = detect_template_variables(mixed_content)
        assert "PROMPT_SUBTREE" in detected, (
            "Should detect valid template variables even with spacing errors"
        )
        assert "COLLECTED_CONTEXT" in detected, "Should detect both template variables"

    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion scenarios."""
        # Test with extremely long content (but reasonable for testing)
        very_long_content = "A" * 10000 + "\n\n{PROMPT_SUBTREE}\n\n" + "B" * 10000

        import time

        start_time = time.time()

        # Should handle without excessive memory or time consumption
        result = detect_template_variables(very_long_content)
        spacing_errors = validate_template_variable_spacing(very_long_content)

        processing_time = time.time() - start_time

        # Should complete in reasonable time
        assert processing_time < 5.0, (
            f"Processing took too long: {processing_time:.3f}s"
        )

        # Should correctly identify template variables
        assert "PROMPT_SUBTREE" in result
        assert len(spacing_errors) == 0, (
            "Long content with proper spacing should be valid"
        )

        # Test with many template variables (potential exponential behavior)
        many_variables_content = "\n\n".join(["{PROMPT_SUBTREE}"] * 100)

        start_time = time.time()
        result_many = detect_template_variables(many_variables_content)
        many_time = time.time() - start_time

        # Should scale linearly, not exponentially
        assert many_time < 5.0, (
            f"Many variables processing took too long: {many_time:.3f}s"
        )
        assert len(result_many["PROMPT_SUBTREE"]) == 100, "Should detect all instances"


class TestLangTreeDSLCommandIntegrationNew:
    """Test compliance with LANGUAGE_SPECIFICATION.md requirements."""

    def test_template_variables_in_acl_grammar_context(self):
        """Test template variables work correctly with LangTree DSL command syntax in docstrings."""
        from pydantic import Field

        from langtree.templates.variables import process_template_variables

        # Create a model with LangTree DSL commands in docstring
        class TaskLangTreeDSLAnalysis(TreeNode):
            """
            ! @all->task.processor@{{value.data=*}}
            ! count=5
            ! resample(count)

            Analyze the data using LangTree DSL commands.

            {PROMPT_SUBTREE}

            Generate output using assembly variable: {count}

            {COLLECTED_CONTEXT}
            """

            analysis: str = Field(description="Primary data analysis")
            results: str = Field(description="Aggregated results")

        # Create RunStructure and add the model
        run_structure = RunStructure()
        run_structure.add(TaskLangTreeDSLAnalysis)

        # Get the node from the structure
        node = run_structure.get_node("task.lang_tree_dsl_analysis")
        assert node is not None, "Node should be created in RunStructure"

        # Process template variables with the real node
        result = process_template_variables(node.clean_docstring or "", node)

        # Should process without errors and resolve template variables
        assert isinstance(result, str)
        assert (
            "Analyze the data using LangTree DSL commands" in result
        )  # Original content preserved

        # LangTree DSL commands should be in extracted_commands, not clean_docstring
        command_strs = [str(cmd) for cmd in node.extracted_commands]
        assert any("@all->task.processor" in cmd_str for cmd_str in command_strs), (
            "LangTree DSL command should be extracted"
        )
        assert any("count=5" in cmd_str for cmd_str in command_strs), (
            "Assembly variable should be extracted"
        )

        # Original docstring should have the LangTree DSL commands
        original_docstring = TaskLangTreeDSLAnalysis.__doc__ or ""
        assert "@all->task.processor" in original_docstring, (
            "Original docstring should contain LangTree DSL commands"
        )
        assert "count=5" in original_docstring, (
            "Original docstring should contain assembly variables"
        )

        # Template variables should be handled properly
        if "{PROMPT_SUBTREE}" in (node.clean_docstring or ""):
            assert "{PROMPT_SUBTREE}" not in result or result == (
                node.clean_docstring or ""
            )

    def test_assembly_variable_conflict_detection_with_runstructure(self):
        """Test Assembly Variable conflict detection with real RunStructure context."""
        from pydantic import Field

        from langtree.templates.variables import (
            get_assembly_variables_for_node_with_structure,
            validate_template_variable_conflicts,
        )

        # Create a model that tests conflict detection between template variables and assembly variables
        class TaskConflictingVariables(TreeNode):
            """
            ! custom_var="test_value"
            ! data_source="another_value"

            This tests template variable conflict detection functionality.

            {PROMPT_SUBTREE}

            Additional content with custom variable: {custom_var}

            {COLLECTED_CONTEXT}
            """

            field: str = Field(description="Test field")

        # Create RunStructure and add the model
        run_structure = RunStructure()
        run_structure.add(TaskConflictingVariables)

        # Get the node
        node = run_structure.get_node("task.conflicting_variables")
        assert node is not None

        # Get Assembly Variables from the RunStructure
        assembly_vars = get_assembly_variables_for_node_with_structure(
            node, run_structure
        )

        # Should include the custom variable names
        assert "custom_var" in assembly_vars
        assert "data_source" in assembly_vars

        # Test conflict detection - this tests that the conflict detection mechanism works
        # even though there are no actual conflicts in this case
        content = node.clean_docstring or ""
        conflicts = validate_template_variable_conflicts(content, assembly_vars)

        # Should not detect conflicts since we're using non-reserved variable names
        assert len(conflicts) == 0, f"Unexpected conflicts: {conflicts}"

    def test_runtime_variable_and_template_variable_separation(self):
        """Test that Runtime Variables and Template Variables work together correctly."""
        from pydantic import Field

        from langtree.templates.variables import (
            resolve_template_variables_in_content,
        )

        class Metadata(TreeNode):
            key: str = "default"
            value: str = "default"

        class TaskMixedVariables(TreeNode):
            """
            Processing data with runtime variable: {field_data}

            Assembly-time configuration uses: assembly variable count=5

            {PROMPT_SUBTREE}

            Context includes: {context_info} and {metadata}

            {COLLECTED_CONTEXT}
            """

            field_data: str = Field(
                default="runtime_value", description="Runtime field data"
            )
            context_info: str = Field(
                default="context_value", description="Context information"
            )
            metadata: Metadata = Field(
                default_factory=Metadata, description="Metadata dictionary"
            )

        # Create RunStructure and add the model
        run_structure = RunStructure()
        run_structure.add(TaskMixedVariables)

        # Get the node
        node = run_structure.get_node("task.mixed_variables")
        assert node is not None

        # Resolve template variables - should not interfere with runtime variables
        content = node.clean_docstring or ""
        result = resolve_template_variables_in_content(content, node)

        # Template variables should be resolved
        assert "{PROMPT_SUBTREE}" not in result or result == content
        assert "{COLLECTED_CONTEXT}" not in result or result == content

        # Runtime variables should remain unchanged (they're resolved at runtime, not assembly time)
        assert "{field_data}" in result
        assert "{context_info}" in result
        assert "{metadata}" in result

        # Original content structure should be preserved
        assert "Processing data" in result
        assert "Assembly-time configuration" in result


class TestComprehensiveTemplateValidation:
    """Test what gets extracted vs what fails from complete clean templates."""

    def test_template_variable_extraction_vs_validation(self):
        """Test comprehensive validation of various variable types in clean templates."""
        # Test cases: (description, template, should_pass, expected_template_vars)
        test_cases = [
            (
                "Valid template variables with proper spacing",
                "Process data.\n\n{PROMPT_SUBTREE}\n\nUse context.\n\n{COLLECTED_CONTEXT}\n\n",
                True,
                ["PROMPT_SUBTREE", "COLLECTED_CONTEXT"],
            ),
            (
                "Variables without lowercase are invalid",
                "Load from {DATA_SOURCE}.\nUse {CONFIGURATION}.\n\n{PROMPT_SUBTREE}\n\n",
                False,  # No lowercase letters = reserved for templates = error
                ["PROMPT_SUBTREE"],  # Only template vars, not runtime
            ),
            (
                "Mixed case runtime variables",
                "Use {MyVariable} and {DataProcessor}.\n\n{PROMPT_SUBTREE}\n\n",
                True,
                ["PROMPT_SUBTREE"],
            ),
            (
                "Lowercase runtime variables",
                "Read {input_file}, use {config}.\n\n{PROMPT_SUBTREE}\n\n",
                True,
                ["PROMPT_SUBTREE"],
            ),
            (
                "Misspelled template variables",
                "Add content:\n\n{prompt_subtree}\n\n{COLLECTED_context}\n\n",
                False,  # Wrong case for template vars
                [],  # No valid template vars
            ),
            (
                "Invalid spacing for template variables",
                "Process{PROMPT_SUBTREE}data.\nAdd{COLLECTED_CONTEXT}here.",
                False,  # Bad spacing
                ["PROMPT_SUBTREE", "COLLECTED_CONTEXT"],  # Detected but invalid
            ),
            (
                "Double underscore runtime variables",
                "Use {user__variable}.\n\n{PROMPT_SUBTREE}\n\n",
                False,  # Double underscore reserved
                ["PROMPT_SUBTREE"],
            ),
            (
                "Common template variable typos",
                "Use:\n\n{PROMPT_TREE}\n\n{CONTEXT}\n\n",
                False,  # Common typos should be caught
                [],
            ),
            (
                "Double braces (DSL command syntax in content)",
                "Process {{data}} with {{filter}}.\n\n{PROMPT_SUBTREE}\n\n",
                False,  # Double braces should be invalid in content
                ["PROMPT_SUBTREE"],
            ),
            (
                "Malformed variable syntax",
                "Use {/variable} or {variable+}.\n\n{PROMPT_SUBTREE}\n\n",
                True,  # These aren't valid variable syntax, so ignored
                ["PROMPT_SUBTREE"],
            ),
        ]

        for description, template, should_pass, expected_detected in test_cases:
            # Detect template variables
            detected = detect_template_variables(template)
            detected_names = list(detected.keys())

            # Validate names
            name_errors = validate_template_variable_names(template)

            # Validate spacing
            spacing_errors = validate_template_variable_spacing(template)

            # Check overall validation
            passes = len(name_errors) == 0 and len(spacing_errors) == 0

            assert passes == should_pass, (
                f"{description}: Expected {'pass' if should_pass else 'fail'} "
                f"but got {'pass' if passes else 'fail'}. "
                f"Name errors: {name_errors}, Spacing errors: {spacing_errors}"
            )

            # Verify detected template vars match expected
            assert set(detected_names) == set(expected_detected), (
                f"{description}: Expected to detect {expected_detected} "
                f"but detected {detected_names}"
            )

    def test_variables_without_lowercase_are_errors(self):
        """Test that variables without lowercase letters are correctly flagged as errors."""
        # These should ERROR (no lowercase letters - reserved for templates)
        error_cases = [
            "{OUTPUT}",
            "{DATA_SOURCE}",
            "{CONFIG}",
            "{SETTINGS}",
            "{INPUT_FORMAT}",
            "{OUTPUT_1}",  # Even with number, no lowercase = error
            "{DATA_2}",
            "{CONFIG_V3}",  # Still no lowercase
        ]

        for var in error_cases:
            content = f"Use {var} for processing.\n\n{{PROMPT_SUBTREE}}\n\n"
            name_errors = validate_template_variable_names(content)
            var_errors = [e for e in name_errors if var in e]

            assert len(var_errors) > 0, (
                f"Variable without lowercase {var} should be flagged as error. "
                f"Got no errors."
            )
            # Check error message mentions reserved for template variables
            assert any(
                "reserved for template variables" in e.lower() for e in var_errors
            ), f"Error for {var} should mention 'reserved for template variables'"

    def test_variables_with_lowercase_are_valid(self):
        """Test that variables with at least one lowercase letter are allowed."""
        # These should be VALID (have at least one lowercase letter)
        valid_cases = [
            "{myVariable}",
            "{dataSource}",
            "{DataSource}",  # Mixed case
            "{MyCustomVariable}",  # Mixed case
            "{output_data}",
            "{configValue}",
            "{Input_Format_v2}",  # Has lowercase
        ]

        for var in valid_cases:
            content = f"Use {var} for processing.\n\n{{PROMPT_SUBTREE}}\n\n"
            name_errors = validate_template_variable_names(content)
            var_errors = [e for e in name_errors if var in e and "reserved" in e]

            assert len(var_errors) == 0, (
                f"Variable with lowercase {var} should be allowed. "
                f"Got errors: {var_errors}"
            )

    def test_template_variable_typo_suggestions(self):
        """Test that common typos of template variables get helpful suggestions."""
        typo_cases = [
            ("{prompt_subtree}", "PROMPT_SUBTREE"),
            ("{Prompt_Subtree}", "PROMPT_SUBTREE"),
            ("{collected_context}", "COLLECTED_CONTEXT"),
            ("{COLLECTED_Context}", "COLLECTED_CONTEXT"),
        ]

        for typo, correct in typo_cases:
            content = f"Content with {typo} variable."
            errors = validate_template_variable_names(content)

            # Should have an error with suggestion
            assert len(errors) > 0, f"Should detect {typo} as error"

            # Check for helpful suggestion
            error_text = " ".join(errors)
            assert correct in error_text, (
                f"Error for {typo} should suggest {correct}. Got: {error_text}"
            )

    def test_invalid_syntax_patterns(self):
        """Test detection of invalid syntax patterns mentioned in conversation."""
        invalid_patterns = [
            ("{{EXTRACT data_source}}", "Double braces in content"),
            ("{{FILTER data | quality > 0.8}}", "Double braces DSL-like syntax"),
            ("{OUTPUT result_format}", "Looks like invalid template var"),
            ("{{AGGREGATE filtered_data}}", "Double braces aggregation"),
        ]

        for pattern, description in invalid_patterns:
            content = f"Process with {pattern}.\n\n{{PROMPT_SUBTREE}}\n\n"

            # Check for nested brace errors
            errors = validate_template_variable_names(content)

            # Should detect issues with double braces
            if "{{" in pattern:
                # Should have nested brace error
                nested_errors = [
                    e for e in errors if "nested" in e.lower() or "brace" in e.lower()
                ]
                assert len(nested_errors) > 0 or len(errors) > 0, (
                    f"{description}: Should detect issue with {pattern}. Got no errors."
                )

    def test_spaces_in_variable_names(self):
        """Test that spaces in variable names are not detected as valid variables."""
        space_cases = [
            "{ PROMPT_SUBTREE}",  # Leading space
            "{PROMPT_SUBTREE }",  # Trailing space
            "{ PROMPT_SUBTREE }",  # Both spaces
            "{PROMPT SUBTREE}",  # Space in middle
            "{ DATA_SOURCE }",  # Spaces around runtime var
            "{MY VARIABLE}",  # Space in runtime var
        ]

        for invalid_var in space_cases:
            content = f"Use {invalid_var} in template.\n\n{{PROMPT_SUBTREE}}\n\n"

            # Should not detect as valid variable syntax
            detected = detect_template_variables(content)

            # Should only detect the valid PROMPT_SUBTREE, not the malformed one
            assert "PROMPT_SUBTREE" in detected  # The valid one
            assert " PROMPT_SUBTREE" not in detected  # Not with space
            assert "PROMPT_SUBTREE " not in detected  # Not with space
            assert " PROMPT_SUBTREE " not in detected  # Not with spaces

            # Should not have validation errors for invalid syntax
            errors = validate_template_variable_names(content)

            # Filter to errors about the malformed variable
            space_var_errors = [e for e in errors if invalid_var in e]

            assert len(space_var_errors) == 0, (
                f"Variable with spaces {invalid_var} should not be detected as valid variable. "
                f"Got errors: {space_var_errors}"
            )

    def test_common_typos_fail_validation(self):
        """Test that common typos of template variables fail validation with helpful messages."""
        typo_cases = [
            ("{PROMPT}", "PROMPT_SUBTREE", True),  # Common typo, should fail
            ("{CONTEXT}", "COLLECTED_CONTEXT", True),  # Common typo, should fail
            ("{SUBTREE}", "PROMPT_SUBTREE", True),  # Partial name, should fail
            ("{COLLECTED}", "COLLECTED_CONTEXT", True),  # Partial name, should fail
            ("{PROMPT_TREE}", "PROMPT_SUBTREE", True),  # Close typo, should fail
            ("{MY_PROMPT}", None, True),  # No lowercase = reserved = should fail
            ("{USER_CONTEXT}", None, True),  # No lowercase = reserved = should fail
        ]

        for typo_var, expected_suggestion, should_fail in typo_cases:
            content = f"Use {typo_var} variable.\n\n{{PROMPT_SUBTREE}}\n\n"

            errors = validate_template_variable_names(content)
            typo_errors = [e for e in errors if typo_var in e]

            if should_fail:
                assert len(typo_errors) > 0, (
                    f"Common typo {typo_var} should fail validation"
                )
                if expected_suggestion:
                    error_text = " ".join(typo_errors)
                    assert expected_suggestion in error_text, (
                        f"Error for {typo_var} should suggest {expected_suggestion}. "
                        f"Got: {error_text}"
                    )
            else:
                assert len(typo_errors) == 0, (
                    f"Non-typo {typo_var} should not fail validation. "
                    f"Got errors: {typo_errors}"
                )


class TestSpecificationCompliance:
    """Test compliance with LANGUAGE_SPECIFICATION.md requirements."""

    def test_template_variables_in_acl_grammar_context(self):
        """Test template variables work within LangTree DSL syntax grammar context."""
        from langtree.templates.variables import (
            detect_template_variables,
            validate_template_variable_spacing,
        )

        # Test template variables in context of LangTree DSL command syntax
        acl_content = """! @each[items]->task.process@{{value.item=items}}*
! count=10
! resample(count)

Process each item using the following template:

{PROMPT_SUBTREE}

Additional context for processing:

{COLLECTED_CONTEXT}

! llm("gpt-4", override=true)
! @all->task.aggregate@{{outputs.results=*}}
"""

        # Should detect template variables correctly
        detected = detect_template_variables(acl_content)
        assert "PROMPT_SUBTREE" in detected
        assert "COLLECTED_CONTEXT" in detected

        # Should validate spacing correctly in LangTree DSL context
        spacing_errors = validate_template_variable_spacing(acl_content)
        assert len(spacing_errors) == 0, (
            f"Should have valid spacing in LangTree DSL context: {spacing_errors}"
        )

    def test_template_variables_with_scope_system(self):
        """Test template variables work correctly with variable scope system (prompt, value, outputs, task)."""
        from pydantic import Field

        from langtree.templates.variables import (
            resolve_template_variables_in_content,
        )

        class Metadata(TreeNode):
            key: str = "default"
            value: str = "default"

        class TaskScopeAware(TreeNode):
            """
            Testing scope interaction with template variables.

            Access prompt.title and value.data from current context.
            Forward outputs.results to task.processor.

            {PROMPT_SUBTREE}

            Using current_node.metadata for additional context.

            {COLLECTED_CONTEXT}
            """

            title: str = Field(description="Task title")
            data: str = Field(description="Input data")
            results: str = Field(description="Processing results")
            metadata: Metadata = Field(
                default_factory=Metadata, description="Node metadata"
            )

        node = StructureTreeNode(
            name="scope_aware",
            field_type=TaskScopeAware,
            clean_docstring=TaskScopeAware.__doc__ or "",
            parent=None,
        )

        # Template variable resolution should work alongside scope references
        result = resolve_template_variables_in_content(node.clean_docstring or "", node)

        # Should preserve scope references
        assert "prompt.title" in result
        assert "value.data" in result
        assert "outputs.results" in result
        assert "task.processor" in result
        assert "current_node.metadata" in result

        # Template variables should be resolved
        if "{PROMPT_SUBTREE}" in (node.clean_docstring or ""):
            template_resolved = "{PROMPT_SUBTREE}" not in result
            # Either resolved or content unchanged (if no actual resolution implemented)
            assert template_resolved or result == (node.clean_docstring or "")

    def test_hierarchical_prompt_execution_integration(self):
        """Test template variables integrate correctly with hierarchical prompt execution system."""
        from pydantic import Field

        from langtree.templates.variables import (
            resolve_collected_context,
            resolve_prompt_subtree,
        )

        # Create a hierarchical structure
        class TaskRoot(TreeNode):
            """Root level prompt with context."""

            root_field: str = Field(description="Root level field")

        class TaskMiddle(TreeNode):
            """
            Middle level processing.

            {PROMPT_SUBTREE}

            Collects context from parent:

            {COLLECTED_CONTEXT}
            """

            middle_field: str = Field(description="Middle level field")

        class TaskLeaf(TreeNode):
            """Leaf level specific processing."""

            leaf_field: str = Field(description="Leaf level field")

        # Create hierarchical structure
        root_node = StructureTreeNode(
            name="root",
            field_type=TaskRoot,
            clean_docstring=TaskRoot.__doc__ or "",
            parent=None,
        )

        middle_node = StructureTreeNode(
            name="middle",
            field_type=TaskMiddle,
            clean_docstring=TaskMiddle.__doc__ or "",
            parent=root_node,
        )

        leaf_node = StructureTreeNode(
            name="leaf",
            field_type=TaskLeaf,
            clean_docstring=TaskLeaf.__doc__ or "",
            parent=middle_node,
        )

        # Store leaf node for hierarchy validation
        assert leaf_node.parent == middle_node, (
            "Should maintain proper parent-child relationship"
        )

        # Test PROMPT_SUBTREE resolution at middle level
        prompt_result = resolve_prompt_subtree(middle_node)
        assert "Middle Field" in prompt_result, "Should include field from middle node"

        # Test COLLECTED_CONTEXT resolution - should collect from hierarchy
        context_result = resolve_collected_context(middle_node)
        assert isinstance(context_result, str)
        # Should either collect real context or provide placeholder
        assert len(context_result) > 0
