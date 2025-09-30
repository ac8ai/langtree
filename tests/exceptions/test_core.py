"""
Tests for error context and formatting system.

This module tests ErrorContext, ErrorLevel enum, and how exceptions
format messages based on error level (user vs developer).
"""

from langtree.exceptions.core import ErrorContext, ErrorLevel, FieldValidationError


class TestErrorLevel:
    """Tests for ErrorLevel enum."""

    def test_user_level_exists(self):
        """Test USER level is available."""
        assert ErrorLevel.USER is not None

    def test_developer_level_exists(self):
        """Test DEVELOPER level is available."""
        assert ErrorLevel.DEVELOPER is not None


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_create_minimal_context(self):
        """Test creating ErrorContext with minimal information."""
        ctx = ErrorContext(docstring_line=3, command_text="@->target@{{value=*}}")
        assert ctx.docstring_line == 3
        assert ctx.command_text == "@->target@{{value=*}}"
        assert ctx.node_tag is None

    def test_create_full_class_docstring_context(self):
        """Test creating ErrorContext for class docstring command."""
        ctx = ErrorContext(
            docstring_line=3,
            command_text="@->target@{{value=*}}",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
        )
        assert ctx.node_tag == "TaskNode"
        assert ctx.node_file == "/workspaces/langtree/examples/task.py"
        assert ctx.node_line == 42
        assert ctx.field_name is None
        assert ctx.field_line is None

    def test_create_full_field_description_context(self):
        """Test creating ErrorContext for field description command."""
        ctx = ErrorContext(
            docstring_line=2,
            command_text="! model=gpt-4",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
            field_name="result",
            field_line=47,
        )
        assert ctx.field_name == "result"
        assert ctx.field_line == 47
        assert ctx.node_tag == "TaskNode"

    def test_format_user_level_class_docstring(self):
        """Test formatting error message at USER level for class docstring."""
        ctx = ErrorContext(
            docstring_line=3,
            command_text="@->target@{{value.missing=*}}",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
        )
        formatted = ctx.format_location(ErrorLevel.USER)

        # User level should show docstring line and node tag only
        assert "line 3" in formatted
        assert "TaskNode" in formatted
        assert "examples/task.py" not in formatted
        assert ":42" not in formatted

    def test_format_developer_level_class_docstring(self):
        """Test formatting error message at DEVELOPER level for class docstring."""
        ctx = ErrorContext(
            docstring_line=3,
            command_text="@->target@{{value.missing=*}}",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
        )
        formatted = ctx.format_location(ErrorLevel.DEVELOPER)

        # Developer level should show all details
        assert "line 3" in formatted
        assert "TaskNode" in formatted
        assert "examples/task.py" in formatted or "task.py" in formatted
        assert "42" in formatted

    def test_format_user_level_field_description(self):
        """Test formatting error message at USER level for field description."""
        ctx = ErrorContext(
            docstring_line=2,
            command_text="! model=gpt-4",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
            field_name="result",
            field_line=47,
        )
        formatted = ctx.format_location(ErrorLevel.USER)

        # User level should show field name and description line
        assert "line 2" in formatted
        assert "TaskNode" in formatted
        assert "result" in formatted
        assert "examples/task.py" not in formatted

    def test_format_developer_level_field_description(self):
        """Test formatting error message at DEVELOPER level for field description."""
        ctx = ErrorContext(
            docstring_line=2,
            command_text="! model=gpt-4",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
            field_name="result",
            field_line=47,
        )
        formatted = ctx.format_location(ErrorLevel.DEVELOPER)

        # Developer level should show all details including both line numbers
        assert "line 2" in formatted
        assert "TaskNode" in formatted
        assert "result" in formatted
        assert "42" in formatted
        assert "47" in formatted

    def test_format_with_command_text(self):
        """Test that command text is included in formatted output."""
        ctx = ErrorContext(
            docstring_line=3, command_text="@->target@{{value=*}}", node_tag="TaskNode"
        )
        formatted = ctx.format_location(ErrorLevel.USER)

        # Command text should be shown
        assert "@->target@{{value=*}}" in formatted or "command:" in formatted.lower()

    def test_format_without_optional_fields(self):
        """Test formatting when optional fields are None."""
        ctx = ErrorContext(docstring_line=3, command_text="@->target@{{value=*}}")
        formatted = ctx.format_location(ErrorLevel.USER)

        # Should not crash, should show what's available
        assert "line 3" in formatted
        assert formatted is not None


class TestExceptionWithErrorContext:
    """Tests for exceptions using ErrorContext."""

    def test_field_validation_error_with_context_user_level(self):
        """Test FieldValidationError formats with ErrorContext at USER level."""
        ctx = ErrorContext(
            docstring_line=3,
            command_text="@->target@{{value.missing=*}}",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
        )

        error = FieldValidationError(
            field_path="value.missing",
            container="TargetNode",
            context=ctx,
            error_level=ErrorLevel.USER,
        )

        error_msg = str(error)

        # Should contain field validation message
        assert "value.missing" in error_msg
        assert "TargetNode" in error_msg

        # Should contain user-level context (no file paths)
        assert "line 3" in error_msg
        assert "TaskNode" in error_msg
        assert "examples/task.py" not in error_msg

    def test_field_validation_error_with_context_developer_level(self):
        """Test FieldValidationError formats with ErrorContext at DEVELOPER level."""
        ctx = ErrorContext(
            docstring_line=3,
            command_text="@->target@{{value.missing=*}}",
            node_tag="TaskNode",
            node_file="/workspaces/langtree/examples/task.py",
            node_line=42,
        )

        error = FieldValidationError(
            field_path="value.missing",
            container="TargetNode",
            context=ctx,
            error_level=ErrorLevel.DEVELOPER,
        )

        error_msg = str(error)

        # Should contain developer-level context with file paths
        assert "value.missing" in error_msg
        assert "task.py" in error_msg
        assert "42" in error_msg

    def test_field_validation_error_without_context(self):
        """Test FieldValidationError works without ErrorContext (backward compat)."""
        error = FieldValidationError(field_path="value.missing", container="TargetNode")

        error_msg = str(error)

        # Should contain basic error message
        assert "value.missing" in error_msg
        assert "TargetNode" in error_msg

        # Should not crash without context
        assert error_msg is not None
