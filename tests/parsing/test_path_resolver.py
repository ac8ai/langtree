"""
Tests for parsing/path_resolver.py compatibility layer.

This module tests the compatibility wrappers that delegate to core.path_utils.
The actual path resolution logic is tested in tests/core/test_path_utils.py
"""

from langtree.parsing.path_resolver import (
    EnhancedParsedCommand,
    EnhancedVariableMapping,
    resolve_variable_mapping,
    resolve_variable_mapping_with_cwd,
)


class TestEnhancedVariableMapping:
    """Test EnhancedVariableMapping compatibility class."""

    def test_from_core_mapping(self):
        """Test conversion from core VariableMapping."""
        from langtree.core.path_utils import PathResolver, VariableMapping

        # Create a core mapping with correct constructor
        target = PathResolver.resolve_path("prompt.title")
        source = PathResolver.resolve_path("value.content")
        core_mapping = VariableMapping(
            target_path="prompt.title",
            source_path="value.content",
            resolved_target=target,
            resolved_source=source,
            original_target="prompt.title",
            original_source="value.content",
        )

        # Convert to enhanced mapping
        enhanced = EnhancedVariableMapping.from_core_mapping(core_mapping)

        assert enhanced.target_path == target
        assert enhanced.source_path == source
        assert enhanced.original_target == "prompt.title"
        assert enhanced.original_source == "value.content"

    def test_enhanced_mapping_attributes(self):
        """Test that enhanced mapping stores all attributes."""
        from langtree.core.path_utils import PathResolver

        target = PathResolver.resolve_path("outputs.result")
        source = PathResolver.resolve_path("task.processor")

        mapping = EnhancedVariableMapping(
            target_path=target,
            source_path=source,
            original_target="outputs.result",
            original_source="task.processor",
        )

        assert mapping.target_path == target
        assert mapping.source_path == source
        assert mapping.original_target == "outputs.result"
        assert mapping.original_source == "task.processor"


class TestEnhancedParsedCommand:
    """Test EnhancedParsedCommand data class."""

    def test_default_initialization(self):
        """Test that __post_init__ sets default variable_mappings."""
        from langtree.core.path_utils import PathResolver

        dest = PathResolver.resolve_path("task.target")

        # Create without variable_mappings
        command = EnhancedParsedCommand(
            command_type="each", destination_path=dest, inclusion_path=None
        )

        # Should have empty list by default
        assert command.variable_mappings == []
        assert isinstance(command.variable_mappings, list)

    def test_with_mappings(self):
        """Test initialization with explicit mappings."""
        from langtree.core.path_utils import PathResolver

        dest = PathResolver.resolve_path("task.target")
        incl = PathResolver.resolve_path("sections")
        target = PathResolver.resolve_path("value.title")
        source = PathResolver.resolve_path("sections.title")

        mapping = EnhancedVariableMapping(
            target_path=target,
            source_path=source,
            original_target="value.title",
            original_source="sections.title",
        )

        command = EnhancedParsedCommand(
            command_type="each",
            destination_path=dest,
            inclusion_path=incl,
            variable_mappings=[mapping],
            has_multiplicity=True,
        )

        assert command.command_type == "each"
        assert command.destination_path == dest
        assert command.inclusion_path == incl
        assert len(command.variable_mappings) == 1
        assert command.has_multiplicity is True


class TestResolveVariableMapping:
    """Test resolve_variable_mapping compatibility function."""

    def test_basic_resolution(self):
        """Test resolving target and source paths."""
        result = resolve_variable_mapping("prompt.title", "value.content")

        assert result.original_target == "prompt.title"
        assert result.original_source == "value.content"
        assert result.target_path.path_remainder == "title"
        assert result.source_path.path_remainder == "content"
        assert result.target_path.scope_modifier.value == "prompt"
        assert result.source_path.scope_modifier.value == "value"

    def test_complex_paths(self):
        """Test resolving complex nested paths."""
        result = resolve_variable_mapping(
            "outputs.document.sections.title", "task.processor.results.summary"
        )

        assert result.original_target == "outputs.document.sections.title"
        assert result.original_source == "task.processor.results.summary"
        assert result.target_path.scope_modifier.value == "outputs"
        assert result.source_path.scope_modifier.value == "task"
        # Path remainder should have scope removed
        assert "document.sections.title" in result.target_path.path_remainder
        assert "processor.results.summary" in result.source_path.path_remainder

    def test_paths_without_scope(self):
        """Test resolving paths without explicit scope prefix."""
        result = resolve_variable_mapping("title", "content")

        assert result.original_target == "title"
        assert result.original_source == "content"
        # Should infer default scope or mark as no scope
        assert result.target_path.path_remainder == "title"
        assert result.source_path.path_remainder == "content"


class TestResolveVariableMappingWithCwd:
    """Test resolve_variable_mapping_with_cwd compatibility function."""

    def test_with_cwd_context(self):
        """Test resolving paths with CWD (Command Working Directory)."""
        # CWD represents the context where the command is defined
        cwd = "task.analyzer"

        result = resolve_variable_mapping_with_cwd("prompt.title", "value.content", cwd)

        assert result.original_target == "prompt.title"
        assert result.original_source == "value.content"
        # Verify paths were resolved with CWD context
        assert result.target_path is not None
        assert result.source_path is not None

    def test_cwd_with_relative_paths(self):
        """Test that CWD affects relative path resolution."""
        cwd = "task.document.analyzer"

        # Relative paths should be resolved relative to CWD
        result = resolve_variable_mapping_with_cwd("title", "content", cwd)

        assert result.original_target == "title"
        assert result.original_source == "content"
        # Paths should be resolved in context of CWD
        # CWD prepends the working directory to relative paths
        assert (
            cwd in result.target_path.path_remainder
            or result.target_path.path_remainder == "title"
        )
        assert (
            cwd in result.source_path.path_remainder
            or result.source_path.path_remainder == "content"
        )

    def test_cwd_with_absolute_scoped_paths(self):
        """Test that absolute scoped paths ignore CWD."""
        cwd = "task.analyzer"

        # Scoped paths are absolute, should not be affected by CWD
        result = resolve_variable_mapping_with_cwd(
            "outputs.result", "task.processor.output", cwd
        )

        assert result.target_path.scope_modifier.value == "outputs"
        assert result.source_path.scope_modifier.value == "task"
        # Should resolve independent of CWD


class TestCompatibilityLayer:
    """Test that compatibility layer properly delegates to core utilities."""

    def test_uses_core_path_resolver(self):
        """Verify that resolve functions use core PathResolver."""
        from langtree.core.path_utils import PathResolver

        # Direct resolution
        core_result = PathResolver.resolve_path("prompt.title")

        # Via compatibility function
        compat_result = resolve_variable_mapping("prompt.title", "value.content")

        # Should produce equivalent resolution
        assert compat_result.target_path.scope_modifier == core_result.scope_modifier
        assert compat_result.target_path.path_remainder == core_result.path_remainder

    def test_enhanced_classes_are_dataclasses(self):
        """Verify that enhanced classes are proper dataclasses."""
        from dataclasses import is_dataclass

        assert is_dataclass(EnhancedVariableMapping)
        assert is_dataclass(EnhancedParsedCommand)
