"""
Comprehensive tests for the Action Chaining Language command parser.

This module tests all aspects of command parsing including:
- Basic command syntax validation
- Scope modifier extraction from all path components
- Variable mapping resolution
- Error handling and edge cases
"""

# Test data imports
from typing import NamedTuple

import pytest

from langtree.core.tree_node import TreeNode
from langtree.parsing.parser import (
    CommandParseError,
    CommandParser,
    CommandType,
    ExecutionCommand,
    NodeModifierCommand,
    NodeModifierType,
    ParsedCommand,
    ResamplingCommand,
    VariableAssignmentCommand,
    parse_command,
)


class CommandTestCase(NamedTuple):
    """Test case for command parsing."""

    name: str
    command: str
    description: str
    expected_valid: bool = True


# Valid command test cases
VALID_COMMANDS = [
    # @each commands
    CommandTestCase(
        "basic_each_with_nested_iteration",
        "! @each[sections.subsections]->task.analyze_comparison@{{value.main_analysis.title=sections.title,value.main_analysis.subsections.title=sections.subsections}}*",
        "Basic @each with nested iteration over sections and subsections",
    ),
    CommandTestCase(
        "each_with_single_level_iteration",
        "! @each[main_analysis]->summary@{{outputs.main_analysis=main_analysis}}*",
        "@each with single-level iteration and outputs scope",
    ),
    CommandTestCase(
        "each_with_multiple_mappings",
        "! @each[sections]->task.analyze@{{value.title=sections.title,value.content=sections.content,prompt.component=sections.type}}*",
        "@each with multiple variable mappings across different scopes",
    ),
    CommandTestCase(
        "each_minimal_syntax",
        "! @each[items]->process@{{value.item=items}}*",
        "Minimal @each command syntax with inclusion",
    ),
    CommandTestCase(
        "each_without_inclusion",
        "! @each->task.rate@{{value.title=title}}*",
        "@each without inclusion brackets (iterates over annotated variable)",
    ),
    # @all commands
    CommandTestCase(
        "all_explicit_with_assignment",
        "! @all->task.output_aggregator@{{prompt.source_data=summary}}",
        "Explicit @all with variable assignment (1:1)",
    ),
    CommandTestCase(
        "all_implicit_with_assignment",
        "! @->task.output_aggregator@{{prompt.source_data=summary}}",
        "Implicit @all (using @) with variable assignment (1:1)",
    ),
    CommandTestCase(
        "all_with_wildcard_assignment",
        "! @->task.summarize_analysis@{{prompt.task_strategic_recommendations=*}}",
        "@all with wildcard assignment for complete subtree transfer",
    ),
    CommandTestCase(
        "all_implicit_variable_name",
        "! @all->task.output_aggregator@{{prompt.target_data}}",
        "@all with implicit variable assignment (variable name matches field)",
    ),
    CommandTestCase(
        "all_one_to_many_with_asterisk",
        "! @all->task.output_aggregator@{{prompt.source_data=*}}*",
        "@all with 1:n relationship (multiple outputs)",
    ),
    CommandTestCase(
        "all_implicit_one_to_many",
        "! @all->task.output_aggregator@{{prompt.source_data}}*",
        "@all with implicit assignment and 1:n relationship",
    ),
    # Variable mapping patterns
    CommandTestCase(
        "nested_path_mapping",
        "! @each[sections]->task.analyze@{{value.main_analysis.title=sections.title}}*",
        "Variable mapping with nested path on left side",
    ),
    CommandTestCase(
        "multiple_variable_assignments",
        "! @->task.process@{{prompt.title=sections.title,prompt.content=sections.content,value.metadata=sections.meta}}",
        "Multiple variable assignments in single command",
    ),
    # Scope modifier tests
    CommandTestCase(
        "scope_modifier_prompt",
        "! @all->task.destination@{{prompt.variable_name=source_field}}",
        "Using prompt scope modifier",
    ),
    CommandTestCase(
        "scope_modifier_value",
        "! @each[items]->task.destination@{{value.field_name=items.content}}*",
        "Using value scope modifier",
    ),
    CommandTestCase(
        "scope_modifier_outputs",
        "! @each[analysis]->summary@{{outputs.main_analysis=analysis.results}}*",
        "Using outputs scope modifier",
    ),
    CommandTestCase(
        "scope_modifier_task",
        "! @->task.analyze_comparison@{{prompt.component=task.current.title}}",
        "Using task scope modifier",
    ),
    # Edge cases
    CommandTestCase(
        "single_character_paths",
        "! @all->t@{{p.a=b}}",
        "Single character identifiers in paths",
    ),
    CommandTestCase(
        "invalid_scope_modifier",
        "! @->task@{{unknown.title=sections.title}}",
        "Unknown scope modifier (should be allowed)",
    ),
    # Whitespace variations
    CommandTestCase(
        "spaces_in_variable_mapping",
        "! @->task@{{ prompt.title = sections.title , value.content = sections.content }}",
        "Spaces around equals and commas in variable mapping",
    ),
    CommandTestCase(
        "spaces_in_inclusion",
        "! @each[ sections.subsections ]->task.analyzer@{{value.title=sections.subsections.title}}*",
        "Spaces inside inclusion brackets",
    ),
    CommandTestCase(
        "no_spaces",
        "!@each[sections]->task@{{value.title=sections.title}}*",
        "Command with minimal spacing",
    ),
    # Documentation examples
    CommandTestCase(
        "doc_example_each_1",
        "! @each[sections.subsections]->task.analyze_comparison@{{value.main_analysis.title=sections.title,value.main_analysis.subsections.title=sections.subsections}}*",
        "Direct example from @each documentation",
    ),
    CommandTestCase(
        "doc_example_each_2",
        "! @each[main_analysis]->summary@{{outputs.main_analysis=main_analysis}}*",
        "Second @each example from documentation",
    ),
    CommandTestCase(
        "doc_example_all_1",
        "! @->task.output_aggregator@{{prompt.source_data=summary}}",
        "First @all example from documentation",
    ),
    CommandTestCase(
        "doc_example_all_2",
        "! @->task.summarize_analysis@{{prompt.task_strategic_recommendations=*}}",
        "Second @all example from documentation",
    ),
    CommandTestCase(
        "doc_example_all_3_corrected",
        "! @all->task.output_aggregator@{{prompt.source_data}}*",
        "Third @all example from documentation",
    ),
]

# Invalid command test cases
INVALID_COMMANDS = [
    CommandTestCase(
        "each_missing_asterisk",
        "! @each[sections]->task.rate@{{value.title=sections.title}}",
        "@each command missing required asterisk multiplicity indicator",
        False,
    ),
    CommandTestCase(
        "each_empty_inclusion",
        "! @each[]->task.rate@{{value.title=title}}*",
        "@each with empty inclusion brackets",
        False,
    ),
    CommandTestCase(
        "empty_variable_mapping",
        "! @->task@{{}}",
        "Empty variable mapping should be invalid",
        False,
    ),
    CommandTestCase(
        "extra_spaces_around_arrow",
        "! @each[sections]  ->  task.rate@{{value.title=sections.title}}*",
        "Extra spaces around arrow operator (should be invalid per specification)",
        False,
    ),
    CommandTestCase(
        "missing_exclamation",
        "@each[sections]->task@{{value.title=sections.title}}*",
        "Command missing required exclamation mark",
        False,
    ),
    CommandTestCase(
        "missing_arrow",
        "! @each[sections] task@{{value.title=sections.title}}*",
        "Command missing required arrow (->)",
        False,
    ),
    CommandTestCase(
        "malformed_variable_mapping",
        "! @->task@{{prompt.title:sections.title}}",
        "Variable mapping using colon instead of equals",
        False,
    ),
    CommandTestCase(
        "unclosed_brackets",
        "! @each[sections->task@{{value.title=sections.title}}*",
        "Unclosed inclusion brackets",
        False,
    ),
    CommandTestCase(
        "unclosed_variable_braces",
        "! @->task@{{prompt.title=sections.title",
        "Unclosed variable mapping braces",
        False,
    ),
    CommandTestCase(
        "empty_variable_name",
        "! @->task@{{=sections.title}}",
        "Empty variable name in mapping",
        False,
    ),
    CommandTestCase(
        "empty_source_path",
        "! @->task@{{prompt.title=}}",
        "Empty source path in mapping",
        False,
    ),
    CommandTestCase(
        "each_with_wildcard_in_mapping",
        "! @each[items]->task.process@{{value.item=*}}*",
        "@each command with wildcard inside variable mapping",
        False,
    ),
]


class TestCommandParser:
    """Test suite for the CommandParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_valid_commands(self):
        """Test parsing of all valid commands."""
        for test_case in VALID_COMMANDS:
            try:
                result = self.parser.parse(test_case.command)
                assert isinstance(result, ParsedCommand), (
                    f"Failed to parse: {test_case.name}"
                )
            except Exception as e:
                pytest.fail(
                    f"Failed to parse valid command '{test_case.name}': {test_case.command}\nError: {e}"
                )

    def test_parse_invalid_commands(self):
        """Test that invalid commands raise CommandParseError."""
        for test_case in INVALID_COMMANDS:
            with pytest.raises(CommandParseError):
                self.parser.parse(test_case.command)

    def test_specific_each_command(self):
        """Test parsing of a specific @each command."""
        command = "! @each[sections.subsections]->task.analyze_comparison@{{value.main_analysis.title=sections.title,value.main_analysis.subsections.title=sections.subsections}}*"
        result = self.parser.parse(command)

        assert result.command_type == CommandType.EACH
        assert result.destination_path == "task.analyze_comparison"
        assert result.inclusion_path == "sections.subsections"
        assert result.has_multiplicity is True
        assert len(result.variable_mappings) == 2

        # Check first mapping
        mapping1 = result.variable_mappings[0]
        assert mapping1.target_path == "value.main_analysis.title"
        assert mapping1.source_path == "sections.title"
        assert mapping1.resolved_target is not None
        assert mapping1.resolved_target.scope.get_name() == "value"
        assert mapping1.resolved_target.path == "main_analysis.title"

        # Check second mapping
        mapping2 = result.variable_mappings[1]
        assert mapping2.target_path == "value.main_analysis.subsections.title"
        assert mapping2.source_path == "sections.subsections"
        assert mapping2.resolved_target is not None
        assert mapping2.resolved_target.scope.get_name() == "value"
        assert mapping2.resolved_target.path == "main_analysis.subsections.title"

    def test_specific_all_command(self):
        """Test parsing of a specific @all command."""
        command = "! @->task.output_aggregator@{{prompt.source_data=summary}}"
        result = self.parser.parse(command)

        assert result.command_type == CommandType.ALL
        assert result.destination_path == "task.output_aggregator"
        assert result.inclusion_path is None
        assert result.has_multiplicity is False
        assert len(result.variable_mappings) == 1

        mapping = result.variable_mappings[0]
        assert mapping.target_path == "prompt.source_data"
        assert mapping.source_path == "summary"
        assert mapping.resolved_target is not None
        assert mapping.resolved_target.scope.get_name() == "prompt"
        assert mapping.resolved_target.path == "source_data"

    def test_wildcard_assignment(self):
        """Test parsing of wildcard assignment."""
        command = (
            "! @->task.summarize_analysis@{{prompt.task_strategic_recommendations=*}}"
        )
        result = self.parser.parse(command)

        assert result.command_type == CommandType.ALL
        assert result.is_wildcard_assignment is True
        assert len(result.variable_mappings) == 1

        mapping = result.variable_mappings[0]
        assert mapping.source_path == "*"

    def test_implicit_assignment(self):
        """Test parsing of implicit variable assignment."""
        command = "! @all->task.output_aggregator@{{prompt.target_data}}"
        result = self.parser.parse(command)

        assert len(result.variable_mappings) == 1
        mapping = result.variable_mappings[0]
        assert mapping.target_path == "prompt.target_data"
        assert mapping.source_path == "target_data"  # Should infer from field name

    def test_one_to_many_relationship(self):
        """Test 1:n relationship detection."""
        command = "! @all->task.output_aggregator@{{prompt.source_data}}*"
        result = self.parser.parse(command)

        assert result.is_one_to_many() is True
        assert result.is_many_to_many() is False

    def test_many_to_many_relationship(self):
        """Test n:n relationship detection."""
        command = "! @each[sections]->task.analyze@{{value.title=sections.title}}*"
        result = self.parser.parse(command)

        assert result.is_many_to_many() is True
        assert result.is_one_to_many() is False

    def test_each_without_inclusion(self):
        """Test @each command without inclusion brackets."""
        command = "! @each->process@{{value.item=current_item}}*"
        result = self.parser.parse(command)

        assert result.command_type == CommandType.EACH
        assert result.inclusion_path is None
        assert result.has_multiplicity is True

    def test_convenience_function(self):
        """Test the convenience parse_command function."""
        command = "! @->task.test@{{prompt.field=value}}"
        result = parse_command(command)

        assert isinstance(result, ParsedCommand)
        assert result.command_type == CommandType.ALL

    def test_scope_modifiers(self):
        """Test all scope modifiers."""
        commands = [
            ("! @->task@{{prompt.field=value}}", "prompt"),
            ("! @each[items]->task@{{value.field=items}}*", "value"),
            ("! @each[items]->task@{{outputs.field=items}}*", "outputs"),
            ("! @->task@{{task.field=value}}", "task"),
        ]

        for command_str, expected_scope_name in commands:
            result = self.parser.parse(command_str)
            mapping = result.variable_mappings[0]
            assert mapping.resolved_target is not None
            assert mapping.resolved_target.scope.get_name() == expected_scope_name

    def test_parsed_output_structure(self):
        """Test that parsed output has the correct structure."""
        command = "! @each[sections]->task.analyze@{{value.title=sections.title,value.content=sections.content}}*"
        result = self.parser.parse(command)

        # Check basic structure
        assert hasattr(result, "command_type")
        assert hasattr(result, "destination_path")
        assert hasattr(result, "variable_mappings")
        assert hasattr(result, "inclusion_path")
        assert hasattr(result, "has_multiplicity")
        assert hasattr(result, "is_wildcard_assignment")
        assert hasattr(result, "resolved_destination")
        assert hasattr(result, "resolved_inclusion")

        # Check variable mapping structure
        for mapping in result.variable_mappings:
            assert hasattr(mapping, "target_path")
            assert hasattr(mapping, "source_path")
            assert hasattr(mapping, "resolved_target")
            assert hasattr(mapping, "resolved_source")

    def test_relationship_type_detection(self):
        """Test relationship type detection methods."""
        # Test 1:1 relationship (no multiplicity)
        command_1to1 = "! @all->task.analyze@{{prompt.field=value}}"
        result_1to1 = self.parser.parse(command_1to1)
        assert not result_1to1.is_one_to_many()
        assert not result_1to1.is_many_to_many()

        # Test 1:n relationship (@all with multiplicity)
        command_1ton = "! @all->task.analyze@{{prompt.field=value}}*"
        result_1ton = self.parser.parse(command_1ton)
        assert result_1ton.is_one_to_many()
        assert not result_1ton.is_many_to_many()

        # Test n:n relationship (@each with multiplicity)
        command_nton = "! @each[items]->task.analyze@{{value.field=items}}*"
        result_nton = self.parser.parse(command_nton)
        assert not result_nton.is_one_to_many()
        assert result_nton.is_many_to_many()

    def test_scope_modifier_parsing(self):
        """Test that scope modifiers are correctly parsed and assigned."""
        test_cases = [
            ("prompt.field", "prompt"),
            ("value.field", "value"),
            ("outputs.field", "outputs"),
            ("task.field", "task"),
            (
                "unknown.field",
                "current_node",
            ),  # Unknown scope modifier defaults to current_node
            (
                "p.field",
                "current_node",
            ),  # Single character (unknown) defaults to current_node
            ("field", "current_node"),  # No scope modifier defaults to current_node
        ]

        for target_path, expected_scope_name in test_cases:
            command = f"! @->task@{{{{{target_path}=source}}}}"
            result = self.parser.parse(command)

            assert len(result.variable_mappings) == 1
            mapping = result.variable_mappings[0]
            assert mapping.resolved_target is not None
            assert mapping.resolved_target.scope is not None, (
                f"Scope should never be None for path: {target_path}"
            )
            actual_scope_name = mapping.resolved_target.scope.get_name()
            assert actual_scope_name == expected_scope_name, (
                f"Failed for path '{target_path}'. Expected '{expected_scope_name}', got '{actual_scope_name}'"
            )


class TestScopeResolution:
    """Test scope modifier extraction from all path components."""

    def test_target_path_scope_extraction(self):
        """Test pathB (target_path) scope extraction.

        Note: Uses 'task.undefined' as placeholder destination to test parser's scope extraction.
        This test focuses on parsing, not semantic validation of task existence.
        """
        test_cases = [
            ("! @->task.undefined@{{prompt.var=source}}", "prompt"),
            ("! @->task.undefined@{{value.var=source}}", "value"),
            ("! @->task.undefined@{{outputs.var=source}}", "outputs"),
            ("! @->task.undefined@{{task.var=source}}", "task"),
            (
                "! @->task.undefined@{{unknown.var=source}}",
                "current_node",
            ),  # Unknown scope becomes current_node
            (
                "! @->task.undefined@{{var=source}}",
                "current_node",
            ),  # No scope becomes current_node
        ]

        for command, expected_scope_name in test_cases:
            result = parse_command(command)
            mapping = result.variable_mappings[0]
            assert mapping.resolved_target is not None
            assert mapping.resolved_target.scope is not None, (
                f"Scope should never be None for command: {command}"
            )
            actual_scope_name = mapping.resolved_target.scope.get_name()
            assert actual_scope_name == expected_scope_name, (
                f"Failed for command: {command}. Expected '{expected_scope_name}', got '{actual_scope_name}'"
            )

    def test_inclusion_path_scope_extraction(self):
        """Test pathX (inclusion_path) scope extraction.

        Note: Uses 'task.undefined' as placeholder destination to test parser's scope extraction.
        This test focuses on parsing, not semantic validation of task existence.
        """
        test_cases = [
            (
                "! @each[prompt.sections]->task.undefined@{{value.var=prompt.sections.source}}*",
                "prompt",
                "sections",
            ),
            (
                "! @each[value.items]->task.undefined@{{value.var=value.items.source}}*",
                "value",
                "items",
            ),
            (
                "! @each[outputs.results]->task.undefined@{{value.var=outputs.results.source}}*",
                "outputs",
                "results",
            ),
            (
                "! @each[task.data]->task.undefined@{{value.var=task.data.source}}*",
                "task",
                "data",
            ),
            (
                "! @each[unknown.items]->task.undefined@{{value.var=unknown.items.source}}*",
                "current_node",
                "unknown.items",
            ),
            (
                "! @each[items]->task.undefined@{{value.var=items.source}}*",
                "current_node",
                "items",
            ),
        ]

        for command, expected_scope_name, expected_path in test_cases:
            result = parse_command(command)

            assert result.resolved_inclusion is not None
            assert result.resolved_inclusion.scope is not None, (
                f"Scope should never be None for command: {command}"
            )
            actual_scope_name = result.resolved_inclusion.scope.get_name()
            assert actual_scope_name == expected_scope_name, (
                f"Failed scope for command: {command}. Expected '{expected_scope_name}', got '{actual_scope_name}'"
            )
            assert result.resolved_inclusion.path == expected_path, (
                f"Failed path for command: {command}. Expected '{expected_path}', got '{result.resolved_inclusion.path}'"
            )

    def test_destination_path_scope_extraction(self):
        """Test pathA (destination_path) scope extraction."""
        test_cases = [
            ("! @->prompt.task@{{value.var=source}}", "prompt", "task"),
            ("! @->value.destination@{{value.var=source}}", "value", "destination"),
            ("! @->outputs.processor@{{value.var=source}}", "outputs", "processor"),
            ("! @->task.analyzer@{{value.var=source}}", "task", "analyzer"),
            (
                "! @->unknown.target@{{value.var=source}}",
                "current_node",
                "unknown.target",
            ),
            ("! @->simple@{{value.var=source}}", "current_node", "simple"),
        ]

        for command, expected_scope_name, expected_path in test_cases:
            result = parse_command(command)

            assert result.resolved_destination is not None
            assert result.resolved_destination.scope is not None, (
                f"Scope should never be None for command: {command}"
            )
            actual_scope_name = result.resolved_destination.scope.get_name()
            assert actual_scope_name == expected_scope_name, (
                f"Failed scope for command: {command}. Expected '{expected_scope_name}', got '{actual_scope_name}'"
            )
            assert result.resolved_destination.path == expected_path, (
                f"Failed path for command: {command}. Expected '{expected_path}', got '{result.resolved_destination.path}'"
            )

    def test_source_path_scope_extraction(self):
        """Test pathC (source_path) scope extraction."""
        test_cases = [
            ("! @->task@{{value.var=prompt.data}}", "prompt", "data"),
            ("! @->task@{{value.var=value.content}}", "value", "content"),
            ("! @->task@{{value.var=outputs.results}}", "outputs", "results"),
            ("! @->task@{{value.var=task.analysis}}", "task", "analysis"),
            (
                "! @->task@{{value.var=unknown.source}}",
                "current_node",
                "unknown.source",
            ),
            ("! @->task@{{value.var=simple}}", "current_node", "simple"),
        ]

        for command, expected_scope_name, expected_path in test_cases:
            result = parse_command(command)
            mapping = result.variable_mappings[0]

            assert mapping.resolved_source is not None
            assert mapping.resolved_source.scope is not None, (
                f"Scope should never be None for command: {command}"
            )
            actual_scope_name = mapping.resolved_source.scope.get_name()
            assert actual_scope_name == expected_scope_name, (
                f"Failed scope for command: {command}. Expected '{expected_scope_name}', got '{actual_scope_name}'"
            )
            assert mapping.resolved_source.path == expected_path, (
                f"Failed path for command: {command}. Expected '{expected_path}', got '{mapping.resolved_source.path}'"
            )

    def test_multiple_scopes_in_same_command(self):
        """Test multiple different scopes in the same command.

        Note: 'value.task' means value scope with 'task' as field name, not two scopes.
        The 'task' here is a field name in the value scope, not the task scope.
        This test demonstrates parser's ability to distinguish scope.field patterns.
        """
        # Using 'value.result_field' instead of 'value.task' to avoid confusion
        command = (
            "! @each[prompt.sections]->value.result_field@{{outputs.title=task.data}}*"
        )
        result = parse_command(command)

        # Test inclusion path scope
        assert result.resolved_inclusion is not None
        assert result.resolved_inclusion.scope.get_name() == "prompt"
        assert result.resolved_inclusion.path == "sections"

        # Test destination path scope
        assert result.resolved_destination is not None
        assert result.resolved_destination.scope.get_name() == "value"
        assert result.resolved_destination.path == "result_field"

        # Test variable mapping scopes
        mapping = result.variable_mappings[0]
        assert mapping.resolved_target is not None
        assert mapping.resolved_target.scope.get_name() == "outputs"
        assert mapping.resolved_target.path == "title"

        assert mapping.resolved_source is not None
        assert mapping.resolved_source.scope.get_name() == "task"
        assert mapping.resolved_source.path == "data"

    def test_same_scope_in_multiple_places(self):
        """Test same scope modifier in multiple path components.

        This tests the parser's ability to handle the same scope (prompt) appearing
        in multiple positions. While this can be confusing, it's syntactically valid.
        """
        command = (
            "! @each[prompt.items]->prompt.destination@{{prompt.var=prompt.source}}*"
        )
        result = parse_command(command)

        # Should be unambiguous - each scope only affects its own path
        assert result.resolved_inclusion is not None
        assert result.resolved_inclusion.scope.get_name() == "prompt"
        assert result.resolved_inclusion.path == "items"

        assert result.resolved_destination is not None
        assert result.resolved_destination.scope.get_name() == "prompt"
        assert result.resolved_destination.path == "destination"

        mapping = result.variable_mappings[0]
        assert mapping.resolved_target is not None
        assert mapping.resolved_target.scope.get_name() == "prompt"
        assert mapping.resolved_target.path == "var"

        assert mapping.resolved_source is not None
        assert mapping.resolved_source.scope.get_name() == "prompt"
        assert mapping.resolved_source.path == "source"

    def test_confusing_edge_case_scope_as_field_name(self):
        """Test edge case where scope names are used as field names.

        This is a confusing but syntactically valid pattern that should be
        documented and tested separately. In real usage, this should be avoided.
        """
        # Test where 'prompt' could be misunderstood as a field name
        command = (
            "! @each[current.prompt]->task.processor@{{value.result=current.prompt}}*"
        )
        result = parse_command(command)

        # 'current' is the scope, 'prompt' is the field name (not prompt scope)
        assert result.resolved_inclusion is not None
        assert result.resolved_inclusion.scope.get_name() == "current_node"
        assert result.resolved_inclusion.path == "current.prompt"

        # Similarly for the mapping source
        mapping = result.variable_mappings[0]
        assert mapping.resolved_source.scope.get_name() == "current_node"
        assert mapping.resolved_source.path == "current.prompt"

    def test_inclusion_scope_extraction_expected_behavior(self):
        """Test expected behavior for inclusion path scope extraction.

        Note: This test demonstrates parser functionality with minimal paths.
        In real usage, paths would typically be more complete (e.g., sections.title).
        """
        command = "! @each[document.sections]->task.analyzer@{{value.section_title=document.sections.title}}*"
        result = parse_command(command)

        assert result.resolved_inclusion is not None
        assert result.resolved_inclusion.scope.get_name() == "current_node"
        assert result.resolved_inclusion.path == "document.sections"

    def test_destination_scope_extraction_expected_behavior(self):
        """Test expected behavior for destination path scope extraction."""
        command = "! @->value.task@{{outputs.var=source}}"
        result = parse_command(command)

        assert result.resolved_destination is not None
        assert result.resolved_destination.scope.get_name() == "value"
        assert result.resolved_destination.path == "task"

    def test_source_scope_extraction_expected_behavior(self):
        """Test expected behavior for source path scope extraction."""
        command = "! @->task@{{value.var=outputs.data}}"
        result = parse_command(command)

        mapping = result.variable_mappings[0]
        assert mapping.resolved_source is not None
        assert mapping.resolved_source.scope.get_name() == "outputs"
        assert mapping.resolved_source.path == "data"


def test_all_documented_examples():
    """Test all examples from the documentation work correctly."""
    parser = CommandParser()

    documented_examples = [
        "! @each[sections.subsections]->task.analyze_comparison@{{value.main_analysis.title=sections.title,value.main_analysis.subsections.title=sections.subsections}}*",
        "! @each[main_analysis]->summary@{{outputs.main_analysis=main_analysis}}*",
        "! @->task.output_aggregator@{{prompt.source_data=summary}}",
        "! @->task.summarize_analysis@{{prompt.task_strategic_recommendations=*}}",
        "! @all->task.output_aggregator@{{prompt.source_data}}*",
    ]

    for example in documented_examples:
        result = parser.parse(example)
        assert isinstance(result, ParsedCommand)


def test_complex_nested_paths_with_scopes():
    """Test complex nested paths with scope modifiers - standalone function.

    Note: Uses realistic task references that would exist in a typical document
    processing tree structure rather than undefined placeholder tasks.
    """
    test_cases = [
        "! @each[document.sections]->task.section_analyzer@{{outputs.analysis.title=document.sections.header.text}}*",
        "! @->outputs.summary.result@{{task.document_processor.data=value.source.content}}",
    ]

    for command in test_cases:
        # Should parse without errors
        result = parse_command(command)
        assert len(result.variable_mappings) > 0

        # Print current behavior for analysis
        print(f"Command: {command}")
        print(f"  Inclusion: {result.inclusion_path}")
        print(f"  Destination: {result.destination_path}")
        for i, mapping in enumerate(result.variable_mappings):
            print(f"  Mapping {i + 1}: {mapping.target_path} = {mapping.source_path}")


# New command features tests (V2.0 functionality)


class TestVariableAssignmentCommands:
    """Test suite for variable assignment command parsing (! var=value)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_integer_assignment(self):
        """Test parsing integer variable assignments."""
        command = "! count=5"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "count"
        assert result.value == 5
        assert isinstance(result.value, int)
        assert result.comment is None

    def test_parse_float_assignment(self):
        """Test parsing float variable assignments."""
        command = "! threshold=2.5"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "threshold"
        assert result.value == 2.5
        assert isinstance(result.value, float)
        assert result.comment is None

    def test_parse_string_assignment_double_quotes(self):
        """Test parsing string variable assignments with double quotes."""
        command = '! name="analysis_task"'
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "name"
        assert result.value == "analysis_task"
        assert isinstance(result.value, str)
        assert result.comment is None

    def test_parse_string_assignment_single_quotes(self):
        """Test parsing string variable assignments with single quotes."""
        command = "! title='Multi-word title'"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "title"
        assert result.value == "Multi-word title"
        assert isinstance(result.value, str)
        assert result.comment is None

    def test_parse_string_with_escaped_quotes(self):
        """Test parsing strings with escaped quote characters."""
        command = r'! path="file with \"quotes\""'
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "path"
        assert result.value == 'file with "quotes"'
        assert result.comment is None

    def test_parse_boolean_true_assignment(self):
        """Test parsing boolean true variable assignments."""
        command = "! debug=true"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "debug"
        assert result.value is True
        assert isinstance(result.value, bool)
        assert result.comment is None

    def test_parse_boolean_false_assignment(self):
        """Test parsing boolean false variable assignments."""
        command = "! verbose=false"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "verbose"
        assert result.value is False
        assert isinstance(result.value, bool)
        assert result.comment is None

    def test_parse_assignment_with_comment(self):
        """Test parsing variable assignment with comment."""
        command = "! iterations=10 # Number of resampling iterations"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == "iterations"
        assert result.value == 10
        assert result.comment == "Number of resampling iterations"

    def test_parse_assignment_with_whitespace(self):
        """Test parsing variable assignment with various whitespace patterns."""
        commands = [
            "! count = 5",  # Spaces around equals
            "!count=5",  # No spaces around exclamation
            "! count=5 ",  # Trailing space
        ]

        for command in commands:
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert result.variable_name == "count"
            assert result.value == 5

    def test_invalid_variable_name_format(self):
        """Test that invalid variable names raise parse errors."""
        invalid_commands = [
            "! 123invalid=5",  # Starts with number
            "! invalid-name=5",  # Contains hyphen
            "! invalid name=5",  # Contains space
            "! =5",  # Empty variable name
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError, match="Invalid variable name"):
                self.parser.parse(command)

    def test_invalid_unquoted_value(self):
        """Test that invalid unquoted values raise parse errors."""
        test_cases = [
            (
                "! var=invalid_string",
                "Unquoted value must be a number or boolean",
            ),  # Unquoted non-numeric string
            (
                "! var=mixed123string",
                "Unquoted value must be a number or boolean",
            ),  # Mixed alphanumeric
            ("! var=", "Empty value in assignment"),  # Empty value
        ]

        for command, expected_error in test_cases:
            with pytest.raises(CommandParseError, match=expected_error):
                self.parser.parse(command)

    def test_variable_name_validation(self):
        """Test variable name validation follows Python identifier rules."""
        valid_names = [
            "valid_name",
            "_private_var",
            "CamelCase",
            "variable123",
            "CONSTANT_VALUE",
        ]

        for name in valid_names:
            command = f"! {name}=42"
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert result.variable_name == name


class TestExecutionCommands:
    """Test suite for execution command parsing (! command(args))."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_command_no_arguments(self):
        """Test parsing execution commands with invalid arguments raises error."""
        # Both resample and llm require arguments
        with pytest.raises(
            CommandParseError, match="resample command requires exactly 1 argument"
        ):
            self.parser.parse("! resample()")

        with pytest.raises(
            CommandParseError, match="llm command requires 1-2 arguments"
        ):
            self.parser.parse("! llm()")

    def test_parse_command_single_argument(self):
        """Test parsing execution commands with single argument."""
        command = "! resample(5)"
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "resample"
        assert result.arguments == [5]
        assert result.comment is None

    def test_parse_command_multiple_arguments(self):
        """Test parsing execution commands with multiple arguments."""
        command = '! llm("gpt-4", true)'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4", True]
        assert result.comment is None

    def test_parse_command_mixed_argument_types(self):
        """Test parsing valid llm command with variable arguments."""
        command = "! llm(model_var, override_var)"
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["model_var", "override_var"]
        assert result.comment is None

    def test_parse_command_with_comment(self):
        """Test parsing execution commands with comments."""
        command = "! resample(3) # Resample 3 times for better results"
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "resample"
        assert result.arguments == [3]
        assert result.comment == "Resample 3 times for better results"

    def test_parse_command_string_arguments(self):
        """Test parsing llm commands with string arguments."""
        command = '! llm("claude-3")'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["claude-3"]

    def test_parse_command_boolean_arguments(self):
        """Test parsing llm commands with boolean arguments."""
        command = "! llm(model_name, true)"
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["model_name", True]

    def test_parse_command_with_whitespace(self):
        """Test parsing execution commands with various whitespace patterns."""
        commands = [
            "! resample( 5 )",  # Spaces inside parentheses
            "!resample(5)",  # No space after exclamation
            "! resample (5)",  # Space before parentheses
        ]

        for command in commands:
            result = self.parser.parse(command)
            assert isinstance(result, ExecutionCommand)
            assert result.command_name == "resample"
            assert result.arguments == [5]

    def test_invalid_command_name_format(self):
        """Test that invalid command names raise parse errors."""
        invalid_commands = [
            "! 123invalid()",  # Starts with number
            "! invalid-name()",  # Contains hyphen
            "! invalid name()",  # Contains space
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError, match="Invalid command name"):
                self.parser.parse(command)

    def test_empty_argument_error(self):
        """Test that empty arguments raise parse errors."""
        command = "! resample(5, , 3)"  # Empty middle argument
        with pytest.raises(CommandParseError, match="Empty argument in command"):
            self.parser.parse(command)

    def test_malformed_parentheses(self):
        """Test that malformed parentheses raise parse errors."""
        invalid_commands = [
            "! resample(5",  # Missing closing parenthesis
            "! resample 5)",  # Missing opening parenthesis
            "! resample((5))",  # Double parentheses
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError):
                self.parser.parse(command)

    def test_unknown_command_error(self):
        """Test that unknown commands raise parse errors."""
        invalid_commands = [
            "! initialize()",  # Unknown command
            "! configure()",  # Unknown command
            "! unknown()",  # Unknown command
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError,
                match="Unknown command.*Valid commands are: llm, resample",
            ):
                self.parser.parse(command)

    def test_resample_argument_validation(self):
        """Test resample command argument validation."""
        # Test wrong number of arguments
        with pytest.raises(
            CommandParseError, match="resample command requires exactly 1 argument"
        ):
            self.parser.parse("! resample()")

        with pytest.raises(
            CommandParseError, match="resample command requires exactly 1 argument"
        ):
            self.parser.parse("! resample(5, 10)")

        # Test negative values
        with pytest.raises(
            CommandParseError, match="resample n_times must be positive"
        ):
            self.parser.parse("! resample(-5)")

        with pytest.raises(
            CommandParseError, match="resample n_times must be positive"
        ):
            self.parser.parse("! resample(0)")

        # Test invalid argument types
        with pytest.raises(
            CommandParseError, match="resample argument must be int or variable name"
        ):
            self.parser.parse("! resample(3.14)")

        with pytest.raises(
            CommandParseError, match="resample argument must be int or variable name"
        ):
            self.parser.parse("! resample(true)")

    def test_llm_argument_validation(self):
        """Test llm command argument validation."""
        # Test wrong number of arguments
        with pytest.raises(
            CommandParseError, match="llm command requires 1-2 arguments"
        ):
            self.parser.parse("! llm()")

        with pytest.raises(
            CommandParseError, match="llm command requires 1-2 arguments"
        ):
            self.parser.parse('! llm("model", true, "extra")')

        # Test invalid model_key types
        with pytest.raises(CommandParseError, match="llm model_key must be string"):
            self.parser.parse("! llm(123)")

        with pytest.raises(CommandParseError, match="llm model_key must be string"):
            self.parser.parse("! llm(true)")

        # Test invalid override types
        with pytest.raises(
            CommandParseError, match="llm override must be bool or variable name"
        ):
            self.parser.parse('! llm("model", 123)')

        with pytest.raises(
            CommandParseError, match="llm override must be bool or variable name"
        ):
            self.parser.parse('! llm("model", 3.14)')

        # Test valid cases
        result = self.parser.parse('! llm("gpt-4")')
        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4"]

        result = self.parser.parse('! llm("gpt-4", true)')
        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4", True]

        result = self.parser.parse("! llm(model_var, override_var)")
        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["model_var", "override_var"]


class TestResamplingCommands:
    """Test suite for resampling aggregation command parsing (! @resampled[field]->function)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_numerical_aggregation_mean(self):
        """Test parsing resampling commands with numerical aggregation (mean)."""
        command = "! @resampled[rating]->mean"
        result = self.parser.parse(command)

        assert isinstance(result, ResamplingCommand)
        assert result.field_name == "rating"
        assert result.aggregation_function == "mean"
        assert result.comment is None
        assert result.is_numerical_function() is True

    def test_parse_numerical_aggregation_median(self):
        """Test parsing resampling commands with numerical aggregation (median)."""
        command = "! @resampled[score]->median"
        result = self.parser.parse(command)

        assert isinstance(result, ResamplingCommand)
        assert result.field_name == "score"
        assert result.aggregation_function == "median"
        assert result.is_numerical_function() is True

    def test_parse_numerical_aggregation_min_max(self):
        """Test parsing resampling commands with min/max aggregation."""
        commands = [
            ("! @resampled[value]->min", "min"),
            ("! @resampled[value]->max", "max"),
        ]

        for command_str, expected_function in commands:
            result = self.parser.parse(command_str)
            assert isinstance(result, ResamplingCommand)
            assert result.field_name == "value"
            assert result.aggregation_function == expected_function
            assert result.is_numerical_function() is True

    def test_parse_universal_aggregation_mode(self):
        """Test parsing resampling commands with universal aggregation (mode)."""
        command = "! @resampled[status]->mode"
        result = self.parser.parse(command)

        assert isinstance(result, ResamplingCommand)
        assert result.field_name == "status"
        assert result.aggregation_function == "mode"
        assert result.is_numerical_function() is False  # Mode works with any enum

    def test_parse_resampling_with_comment(self):
        """Test parsing resampling commands with comments."""
        command = "! @resampled[priority]->mean # Mean priority rating"
        result = self.parser.parse(command)

        assert isinstance(result, ResamplingCommand)
        assert result.field_name == "priority"
        assert result.aggregation_function == "mean"
        assert result.comment == "Mean priority rating"

    def test_parse_resampling_with_whitespace(self):
        """Test parsing resampling commands with valid whitespace patterns."""
        commands = [
            "! @resampled[ rating ]->mean",  # Spaces inside brackets (valid per spec)
            "!@resampled[rating]->mean",  # No space after exclamation (valid per spec)
            "!   @resampled[rating]->mean",  # Multiple spaces after exclamation (valid per spec)
        ]

        for command in commands:
            result = self.parser.parse(command)
            assert isinstance(result, ResamplingCommand)
            assert result.field_name == "rating"
            assert result.aggregation_function == "mean"

    def test_invalid_field_name_format(self):
        """Test that invalid field names raise parse errors."""
        test_cases = [
            (
                "! @resampled[123invalid]->mean",
                "Invalid field name",
            ),  # Starts with number
            (
                "! @resampled[invalid-name]->mean",
                "Invalid field name",
            ),  # Contains hyphen
            (
                "! @resampled[invalid name]->mean",
                "Invalid field name",
            ),  # Contains space
            (
                "! @resampled[]->mean",
                "Empty field name in resampling command",
            ),  # Empty field name
        ]

        for command, expected_error in test_cases:
            with pytest.raises(CommandParseError, match=expected_error):
                self.parser.parse(command)

    def test_invalid_aggregation_function(self):
        """Test that invalid aggregation functions raise parse errors."""
        test_cases = [
            (
                "! @resampled[field]->invalid",
                "Invalid aggregation function",
            ),  # Unknown function
            (
                "! @resampled[field]->count",
                "Invalid aggregation function",
            ),  # Not in valid functions list
            (
                "! @resampled[field]->aggregate",
                "Invalid aggregation function",
            ),  # Generic term not allowed
            (
                "! @resampled[field]->",
                "Empty aggregation function in resampling command",
            ),  # Empty function
        ]

        for command, expected_error in test_cases:
            with pytest.raises(CommandParseError, match=expected_error):
                self.parser.parse(command)

    def test_malformed_brackets(self):
        """Test that malformed brackets raise parse errors."""
        invalid_commands = [
            "! @resampled[field->mean",  # Missing closing bracket
            "! @resampled field]->mean",  # Missing opening bracket
            "! @resampled[[field]]->mean",  # Double brackets
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError):
                self.parser.parse(command)

    def test_valid_aggregation_functions_list(self):
        """Test that all valid aggregation functions are accepted."""
        valid_functions = ["mean", "median", "mode", "min", "max"]

        for function in valid_functions:
            command = f"! @resampled[test_field]->{function}"
            result = self.parser.parse(command)
            assert isinstance(result, ResamplingCommand)
            assert result.aggregation_function == function


class TestCommentParsing:
    """Test suite for comment parsing and stripping across all command types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_variable_assignment_comment_stripping(self):
        """Test that comments are properly stripped from variable assignments."""
        command = "! var=42 # This is a comment"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value == 42
        assert result.comment == "This is a comment"

    def test_execution_command_comment_stripping(self):
        """Test that comments are properly stripped from execution commands."""
        command = "! resample(5) # Resample for better accuracy"
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.arguments == [5]
        assert result.comment == "Resample for better accuracy"

    def test_resampling_command_comment_stripping(self):
        """Test that comments are properly stripped from resampling commands."""
        command = "! @resampled[rating]->mean # Mean of the ratings"
        result = self.parser.parse(command)

        assert isinstance(result, ResamplingCommand)
        assert result.aggregation_function == "mean"
        assert result.comment == "Mean of the ratings"

    def test_comment_with_special_characters(self):
        """Test comments containing special characters."""
        command = "! var=42 # Comment with #hashtag and @symbols!"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.comment == "Comment with #hashtag and @symbols!"

    def test_comment_whitespace_handling(self):
        """Test that comment whitespace is properly handled."""
        commands = [
            ("! var=42 #No space before comment", "No space before comment"),
            ("! var=42 # Single space", "Single space"),
            (
                "! var=42 #   Multiple spaces",
                "Multiple spaces",
            ),  # Leading spaces in comment trimmed
            (
                "! var=42 # Trailing spaces   ",
                "Trailing spaces",
            ),  # Trailing spaces in comment trimmed
        ]

        for command_str, expected_comment in commands:
            result = self.parser.parse(command_str)
            # Only check comment for new command types that support comments
            if isinstance(
                result, VariableAssignmentCommand | ExecutionCommand | ResamplingCommand
            ):
                assert result.comment == expected_comment

    def test_no_comment_returns_none(self):
        """Test that commands without comments return None for comment field."""
        commands = [
            "! var=42",
            "! resample(5)",
            "! @resampled[field]->mean",
        ]

        for command in commands:
            result = self.parser.parse(command)
            # Only check comment for new command types that support comments
            if isinstance(
                result, VariableAssignmentCommand | ExecutionCommand | ResamplingCommand
            ):
                assert result.comment is None


class TestBooleanValueSupport:
    """Test suite for boolean value support across command types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_boolean_true_in_variable_assignment(self):
        """Test boolean true values in variable assignments."""
        command = "! override=true"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value is True
        assert isinstance(result.value, bool)

    def test_boolean_false_in_variable_assignment(self):
        """Test boolean false values in variable assignments."""
        command = "! enabled=false"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value is False
        assert isinstance(result.value, bool)

    def test_boolean_in_execution_command_arguments(self):
        """Test boolean values in execution command arguments."""
        command = '! llm("gpt-4", true)'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.arguments == ["gpt-4", True]
        assert isinstance(result.arguments[1], bool)

    def test_boolean_case_sensitivity(self):
        """Test that boolean parsing is case-sensitive."""
        invalid_commands = [
            "! var=True",  # Capital T
            "! var=FALSE",  # All caps
            "! var=True",  # Mixed case
            "! var=tRuE",  # Mixed case
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError, match="Unquoted value must be a number"
            ):
                self.parser.parse(command)

    def test_boolean_vs_string_distinction(self):
        """Test that boolean values are distinct from string values."""
        commands = [
            ('! quoted="true"', str, "true"),  # Quoted string
            ("! unquoted=true", bool, True),  # Unquoted boolean
            ('! quoted="false"', str, "false"),  # Quoted string
            ("! unquoted=false", bool, False),  # Unquoted boolean
        ]

        for command_str, expected_type, expected_value in commands:
            result = self.parser.parse(command_str)
            assert isinstance(result, VariableAssignmentCommand)
            assert isinstance(result.value, expected_type)
            assert result.value == expected_value


class TestValueTypeCoercion:
    """Test suite for enhanced value type coercion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_integer_coercion(self):
        """Test that integer values are properly coerced."""
        test_cases = [
            ("42", int, 42),
            ("0", int, 0),
            ("-5", int, -5),
            ("1000", int, 1000),
        ]

        for value_str, expected_type, expected_value in test_cases:
            command = f"! var={value_str}"
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert isinstance(result.value, expected_type)
            assert result.value == expected_value

    def test_float_coercion(self):
        """Test that float values are properly coerced."""
        test_cases = [
            ("3.14", float, 3.14),
            ("0.0", float, 0.0),
            ("-2.5", float, -2.5),
            ("1000.001", float, 1000.001),
        ]

        for value_str, expected_type, expected_value in test_cases:
            command = f"! var={value_str}"
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert isinstance(result.value, expected_type)
            assert result.value == expected_value

    def test_string_preservation(self):
        """Test that quoted strings preserve their content without coercion."""
        test_cases = [
            ('"42"', str, "42"),  # Numeric string
            ('"true"', str, "true"),  # Boolean string
            ('"3.14"', str, "3.14"),  # Float string
            ('""', str, ""),  # Empty string
            ('"  spaces  "', str, "  spaces  "),  # String with spaces
        ]

        for quoted_value, expected_type, expected_value in test_cases:
            command = f"! var={quoted_value}"
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert isinstance(result.value, expected_type)
            assert result.value == expected_value

    def test_escape_sequence_handling(self):
        """Test that escape sequences in strings are properly handled."""
        test_cases = [
            (r'"\"quoted\""', str, '"quoted"'),  # Escaped quotes
            (r'"line1\nline2"', str, "line1\nline2"),  # Escaped newline (if supported)
            (r'"path\\file"', str, "path\\file"),  # Escaped backslash
        ]

        for escaped_string, expected_type, expected_value in test_cases:
            command = f"! var={escaped_string}"
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert result.value == expected_value


class TestErrorHandling:
    """Test suite for comprehensive error handling and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_command_must_start_with_exclamation(self):
        """Test that commands must start with exclamation mark."""
        invalid_commands = [
            "var=42",  # Missing exclamation
            " ! var=42",  # Space before exclamation
            "? var=42",  # Wrong prefix
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError, match="Command must start with '!'"):
                self.parser.parse(command)

    def test_empty_command_handling(self):
        """Test handling of empty or whitespace-only commands."""
        invalid_commands = [
            "",  # Empty string
            "   ",  # Whitespace only
            "!",  # Just exclamation
            "! ",  # Exclamation with space
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError):
                self.parser.parse(command)

    def test_invalid_command_syntax_detection(self):
        """Test detection of invalid command syntax patterns."""
        invalid_commands = [
            "! invalid syntax here",  # No recognized pattern
            "! @notacommand",  # Unknown @ command
            "! var=",  # Assignment without value
            "! command(",  # Incomplete parentheses
            "! @resampled[]",  # Incomplete resampling syntax
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError):
                self.parser.parse(command)

    def test_descriptive_error_messages(self):
        """Test that error messages are descriptive and helpful."""
        error_cases = [
            ("! 123var=42", "Invalid variable name"),
            ("! var=invalid_unquoted", "Unquoted value must be a number"),
            ("! resample(,)", "Empty argument in command"),
            ("! @resampled[field]->invalid_func", "Invalid aggregation function"),
        ]

        for command, expected_error_text in error_cases:
            with pytest.raises(CommandParseError, match=expected_error_text):
                self.parser.parse(command)


class TestNewCommandIntegrationWithExisting:
    """Test suite for integration between new and existing command types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_command_type_detection(self):
        """Test that parser correctly identifies different command types."""
        commands = [
            ("! var=42", VariableAssignmentCommand),
            ("! resample(5)", ExecutionCommand),
            ("! @resampled[field]->mean", ResamplingCommand),
            (
                "! @each[items]->task@{{value.item=items}}*",
                "ParsedCommand",
            ),  # Existing type
            ("! @->task@{{prompt.var=value}}", "ParsedCommand"),  # Existing type
        ]

        for command_str, expected_type in commands:
            result = self.parser.parse(command_str)
            if isinstance(expected_type, str):
                # For existing ParsedCommand type, check class name
                assert result.__class__.__name__ == expected_type
            else:
                assert isinstance(result, expected_type)

    def test_convenience_function_with_new_commands(self):
        """Test that parse_command convenience function works with new command types."""
        commands = [
            "! var=42",
            "! resample(5)",
            "! @resampled[field]->mode",
        ]

        for command in commands:
            result = parse_command(command)
            assert result is not None
            # Should not raise any exceptions

    def test_parser_maintains_existing_functionality(self):
        """Test that new command parsing doesn't break existing @each/@all parsing."""
        existing_commands = [
            "! @each[sections]->task.analyze@{{value.title=sections.title}}*",
            "! @->task.summarize@{{prompt.context=*}}",
            "! @all->task.process@{{value.data=input}}*",
        ]

        for command in existing_commands:
            result = self.parser.parse(command)
            # Should parse successfully without affecting new command types
            # Only check these attributes on ParsedCommand (existing type)
            if hasattr(result, "command_type"):
                assert hasattr(result, "destination_path")
                assert hasattr(result, "variable_mappings")


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_very_long_variable_names(self):
        """Test parsing with very long variable names."""
        long_name = "very_long_variable_name_" + "x" * 100
        command = f"! {long_name}=42"
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.variable_name == long_name

    def test_very_long_string_values(self):
        """Test parsing with very long string values."""
        long_value = "This is a very long string value " * 50
        command = f'! var="{long_value}"'
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value == long_value

    def test_special_characters_in_strings(self):
        """Test strings containing special characters."""
        special_chars = (
            "!@#$%^&*(){}[]|\\:;<>?,./"  # Removed quotes that would break parsing
        )
        command = f'! special="{special_chars}"'
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value == special_chars

    def test_unicode_characters_in_strings(self):
        """Test strings containing unicode characters."""
        unicode_string = "Hello 世界 🌍 café naïve résumé"
        command = f'! unicode="{unicode_string}"'
        result = self.parser.parse(command)

        assert isinstance(result, VariableAssignmentCommand)
        assert result.value == unicode_string

    def test_maximum_numeric_values(self):
        """Test parsing with very large numeric values."""
        large_int = str(2**63 - 1)  # Maximum 64-bit signed integer
        large_float = "1.7976931348623157e+308"  # Near maximum float

        commands = [
            (f"! big_int={large_int}", int),
            (f"! big_float={large_float}", float),
        ]

        for command, expected_type in commands:
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert isinstance(result.value, expected_type)

    def test_zero_and_negative_edge_cases(self):
        """Test edge cases with zero and negative values."""
        commands = [
            ("! zero=0", 0),
            ("! neg_zero=-0", 0),
            ("! float_zero=0.0", 0.0),
            ("! negative=-42", -42),
            ("! neg_float=-3.14", -3.14),
        ]

        for command, expected_value in commands:
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert result.value == expected_value


def test_language_specification_compliance():
    """Test that parser implementation complies with LANGUAGE_SPECIFICATION.md examples."""
    parser = CommandParser()

    # Test examples from LANGUAGE_SPECIFICATION.md
    specification_examples = [
        # Variable assignment examples
        ("! count=5", VariableAssignmentCommand),
        ("! threshold=2.5", VariableAssignmentCommand),
        ('! name="analysis_task"', VariableAssignmentCommand),
        ("! debug=true", VariableAssignmentCommand),
        ("! verbose=false", VariableAssignmentCommand),
        # Execution command examples
        ("! resample(5)", ExecutionCommand),
        ('! llm("gpt-4")', ExecutionCommand),
        ('! llm("claude-3", true)', ExecutionCommand),
        # Resampling command examples
        ("! @resampled[rating]->mean", ResamplingCommand),
        ("! @resampled[status]->mode", ResamplingCommand),
        ("! @resampled[priority]->median", ResamplingCommand),
    ]

    for command_str, expected_type in specification_examples:
        result = parser.parse(command_str)
        assert isinstance(result, expected_type), f"Failed for command: {command_str}"


def test_comment_examples_from_specification():
    """Test comment examples from LANGUAGE_SPECIFICATION.md."""
    parser = CommandParser()

    comment_examples = [
        ("! count=5 # Integer variable", "Integer variable"),
        (
            "! resample(3) # Resample for better accuracy",
            "Resample for better accuracy",
        ),
        ("! @resampled[rating]->mean # Mean of the ratings", "Mean of the ratings"),
    ]

    for command_str, expected_comment in comment_examples:
        result = parser.parse(command_str)
        # Only check comment for new command types that support comments
        if isinstance(
            result, VariableAssignmentCommand | ExecutionCommand | ResamplingCommand
        ):
            assert result.comment == expected_comment


# Assembly Variable Registry Tests
def test_assembly_variable_registry_basic_operations():
    """Test basic Assembly Variable Registry operations."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Test storing variables
    registry.store_variable("count", 5, "node1", 10)
    registry.store_variable("name", "test", "node1", 11)
    registry.store_variable("active", True, "node2", 12)

    # Test retrieval
    assert registry.get_variable_value("count") == 5
    assert registry.get_variable_value("name") == "test"
    assert registry.get_variable_value("active") is True
    assert registry.get_variable_value("nonexistent") is None

    # Test existence checks
    assert registry.has_variable("count")
    assert registry.has_variable("name")
    assert not registry.has_variable("nonexistent")

    # Test conflict checking
    assert registry.check_conflict("count")
    assert not registry.check_conflict("new_var")


def test_assembly_variable_registry_conflict_detection():
    """Test Assembly Variable conflict detection and error handling."""
    from langtree.structure.registry import (
        AssemblyVariableConflictError,
        AssemblyVariableRegistry,
    )

    registry = AssemblyVariableRegistry()

    # Store initial variable
    registry.store_variable("threshold", 2.5, "node1")

    # Attempt to store conflicting variable should raise error
    with pytest.raises(AssemblyVariableConflictError) as exc_info:
        registry.store_variable("threshold", 3.0, "node2")

    error = exc_info.value
    assert error.variable_name == "threshold"
    assert error.existing_value == 2.5
    assert error.new_value == 3.0
    assert "already exists" in str(error)


def test_assembly_variable_registry_variable_reference_resolution():
    """Test Assembly Variable reference pattern resolution."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Store test variables
    registry.store_variable("model_key", "gpt-4", "node1")
    registry.store_variable("temperature", 0.7, "node1")
    registry.store_variable("use_cache", False, "node2")

    # Test variable reference resolution with different patterns
    assert registry.resolve_variable_reference("model_key") == "gpt-4"
    assert registry.resolve_variable_reference("<model_key>") == "gpt-4"
    assert registry.resolve_variable_reference("temperature") == 0.7
    assert registry.resolve_variable_reference("<temperature>") == 0.7
    assert registry.resolve_variable_reference("use_cache") is False
    assert registry.resolve_variable_reference("<use_cache>") is False

    # Test nonexistent variables
    assert registry.resolve_variable_reference("nonexistent") is None
    assert registry.resolve_variable_reference("<nonexistent>") is None


def test_assembly_variable_registry_node_filtering():
    """Test filtering variables by source node."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Store variables from different nodes
    registry.store_variable("var1", "value1", "node1")
    registry.store_variable("var2", "value2", "node1")
    registry.store_variable("var3", "value3", "node2")
    registry.store_variable("var4", "value4", "node3")

    # Test filtering by node
    node1_vars = registry.get_variables_for_node("node1")
    assert len(node1_vars) == 2
    assert {var.name for var in node1_vars} == {"var1", "var2"}

    node2_vars = registry.get_variables_for_node("node2")
    assert len(node2_vars) == 1
    assert node2_vars[0].name == "var3"

    nonexistent_vars = registry.get_variables_for_node("nonexistent")
    assert len(nonexistent_vars) == 0


def test_assembly_variable_registry_list_all_variables():
    """Test listing all variables in registry."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Initially empty
    assert len(registry.list_variables()) == 0

    # Add variables
    registry.store_variable("count", 5, "node1")
    registry.store_variable("name", "test", "node2")
    registry.store_variable("threshold", 1.5, "node3")

    # Check all variables
    all_vars = registry.list_variables()
    assert len(all_vars) == 3

    var_names = {var.name for var in all_vars}
    assert var_names == {"count", "name", "threshold"}

    var_values = {var.value for var in all_vars}
    assert var_values == {5, "test", 1.5}


def test_assembly_variable_registry_variable_metadata():
    """Test Assembly Variable metadata storage and retrieval."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Store variable with metadata
    registry.store_variable("debug_mode", True, "config_node", 42)

    # Retrieve variable object
    var = registry.get_variable("debug_mode")
    assert var is not None
    assert var.name == "debug_mode"
    assert var.value is True
    assert var.source_node_tag == "config_node"
    assert var.defined_at_line == 42

    # Test string representation
    var_str = str(var)
    assert "debug_mode=True" in var_str


def test_assembly_variable_registry_type_support():
    """Test Assembly Variable Registry supports all required types."""
    from langtree.structure.registry import AssemblyVariableRegistry

    registry = AssemblyVariableRegistry()

    # Test all supported types
    registry.store_variable("string_var", "hello world", "node1")
    registry.store_variable("int_var", 42, "node1")
    registry.store_variable("float_var", 3.14159, "node1")
    registry.store_variable("bool_true", True, "node1")
    registry.store_variable("bool_false", False, "node1")

    # Verify types are preserved
    assert isinstance(registry.get_variable_value("string_var"), str)
    assert isinstance(registry.get_variable_value("int_var"), int)
    assert isinstance(registry.get_variable_value("float_var"), float)
    assert isinstance(registry.get_variable_value("bool_true"), bool)
    assert isinstance(registry.get_variable_value("bool_false"), bool)

    # Verify values
    assert registry.get_variable_value("string_var") == "hello world"
    assert registry.get_variable_value("int_var") == 42
    assert registry.get_variable_value("float_var") == 3.14159
    assert registry.get_variable_value("bool_true") is True
    assert registry.get_variable_value("bool_false") is False


class TestNamedParameters:
    """Test named parameter support in execution commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_llm_with_named_override_parameter(self):
        """Test llm command with named override parameter."""
        test_cases = [
            ('! llm("gpt-4", override=true)', "gpt-4", True),
            ('! llm("claude", override=false)', "claude", False),
            ('! llm("gpt-3.5", override=my_var)', "gpt-3.5", "my_var"),
        ]

        for command, expected_model, expected_override in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, ExecutionCommand)
            assert result.command_name == "llm"
            assert len(result.arguments) == 1
            assert result.arguments[0] == expected_model
            assert result.named_arguments is not None
            assert len(result.named_arguments) == 1
            assert "override" in result.named_arguments
            assert result.named_arguments["override"] == expected_override

    def test_llm_mixed_positional_and_named_arguments(self):
        """Test llm command with both positional and named arguments."""
        command = '! llm("gpt-4", override=true)'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4"]
        assert result.named_arguments == {"override": True}

    def test_llm_positional_override_still_works(self):
        """Test that positional override parameter still works."""
        command = '! llm("gpt-4", true)'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4", True]
        assert result.named_arguments == {}

    def test_invalid_named_parameter_for_llm(self):
        """Test that invalid named parameters are rejected for llm."""
        invalid_commands = [
            '! llm("gpt-4", invalid_param=true)',
            '! llm("gpt-4", model=true)',  # model should be positional
            '! llm("gpt-4", temperature=0.7)',  # not supported
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError,
                match="llm command only supports 'override' named argument",
            ):
                self.parser.parse(command)

    def test_resample_rejects_named_parameters(self):
        """Test that resample command rejects named parameters."""
        invalid_commands = [
            "! resample(n_times=5)",
            "! resample(5, retry=true)",
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError,
                match="resample command does not support named arguments",
            ):
                self.parser.parse(command)

    def test_named_parameter_validation(self):
        """Test validation of named parameter syntax."""
        invalid_commands = [
            '! llm("gpt-4", =true)',  # empty parameter name
            '! llm("gpt-4", 123=true)',  # invalid parameter name
            '! llm("gpt-4", override=)',  # empty value
            '! llm("gpt-4", over-ride=true)',  # invalid parameter name characters
        ]

        error_patterns = [
            "Empty parameter name in named argument",
            "Invalid parameter name: 123",
            "Empty value for parameter: override",
            "Invalid parameter name: over-ride",
        ]

        for command, expected_error in zip(invalid_commands, error_patterns):
            with pytest.raises(CommandParseError, match=expected_error):
                self.parser.parse(command)

    def test_named_parameter_type_coercion(self):
        """Test that named parameter values are properly type-coerced."""
        test_cases = [
            ('! llm("gpt-4", override=true)', bool, True),
            ('! llm("gpt-4", override=false)', bool, False),
            ('! llm("gpt-4", override="true")', str, "true"),  # quoted string
            ('! llm("gpt-4", override=my_var)', str, "my_var"),  # variable name
        ]

        for command, expected_type, expected_value in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, ExecutionCommand)
            assert result.named_arguments is not None
            override_value = result.named_arguments["override"]
            assert isinstance(override_value, expected_type)
            assert override_value == expected_value

    def test_named_parameters_with_comments(self):
        """Test named parameters work with comments."""
        command = '! llm("gpt-4", override=true) # Use override mode'
        result = self.parser.parse(command)

        assert isinstance(result, ExecutionCommand)
        assert result.command_name == "llm"
        assert result.arguments == ["gpt-4"]
        assert result.named_arguments == {"override": True}
        assert result.comment == "Use override mode"


# TODO: Following TDD principles from CODING_STANDARDS.md, comprehensive tests
# for new language features will be added here before implementation:
#
# 1. Variable assignment commands (! var=value) with string/number/boolean support
# 2. Resampling commands (! @resampled[field]->function) with enum field validation
# 3. Node modifier commands (! @sequential, ! @parallel, ! together)
# 4. Boolean literal parsing (true/false) across all command types
# 5. Comment parsing (# comment) for all command types
# 6. Enhanced validation and error handling for new syntax
#
# Each test will be linked to specific implementation TODOs in parser.py
class TestNodeModifierCommands:
    """
    Test suite for node modifier command parsing.

    Tests the parsing of node modifier commands like:
    ! @sequential
    ! @parallel
    ! together
    """

    def setup_method(self):
        """Create parser fixture for node modifier tests."""
        self.parser = CommandParser()

    def test_node_modifier_commands(self):
        """Test parsing of node modifier commands."""
        test_cases = [
            ("! @sequential", NodeModifierType.SEQUENTIAL),
            ("! @parallel", NodeModifierType.PARALLEL),
            ("! together", NodeModifierType.TOGETHER),
        ]

        for command, expected_modifier in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, NodeModifierCommand)
            assert result.modifier == expected_modifier

    def test_node_modifier_with_comments(self):
        """Test node modifier commands with comments."""
        test_cases = [
            (
                "! @sequential # Process fields in order",
                NodeModifierType.SEQUENTIAL,
                "Process fields in order",
            ),
            (
                "! @parallel # Process fields simultaneously",
                NodeModifierType.PARALLEL,
                "Process fields simultaneously",
            ),
            (
                "! together # Shorthand for parallel",
                NodeModifierType.TOGETHER,
                "Shorthand for parallel",
            ),
        ]

        for command, expected_modifier, expected_comment in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, NodeModifierCommand)
            assert result.modifier == expected_modifier
            assert result.comment == expected_comment


class TestBooleanLiteralParsing:
    """
    Test suite for boolean literal parsing in various contexts.

    Tests that boolean values (true/false) are properly parsed and type-coerced
    across different command types.
    """

    def setup_method(self):
        """Create parser fixture for boolean parsing tests."""
        self.parser = CommandParser()

    def test_boolean_in_variable_assignment(self):
        """Test boolean parsing in variable assignments."""
        test_cases = [
            ("! debug=true", True),
            ("! debug=false", False),
            ("! enabled=true", True),
            ("! disabled=false", False),
        ]

        for command, expected_value in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, VariableAssignmentCommand)
            assert result.value == expected_value
            assert isinstance(result.value, bool)

    def test_boolean_in_execution_arguments(self):
        """Test boolean parsing in execution command arguments."""
        test_cases = [
            ('! llm("gpt-4", true)', ["gpt-4", True]),
            ('! llm("gpt-4", false)', ["gpt-4", False]),
        ]

        for command, expected_args in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, ExecutionCommand)
            assert result.arguments == expected_args

    def test_boolean_in_named_arguments(self):
        """Test boolean parsing in named arguments."""
        test_cases = [
            ('! llm("gpt-4", override=true)', {"override": True}),
            ('! llm("gpt-4", override=false)', {"override": False}),
        ]

        for command, expected_named_args in test_cases:
            result = self.parser.parse(command)
            assert isinstance(result, ExecutionCommand)
            assert result.named_arguments == expected_named_args

    def test_invalid_boolean_literals(self):
        """Test that invalid boolean literals are rejected."""
        invalid_commands = [
            "! var=True",  # capitalized
            "! var=FALSE",  # all caps
            "! var=tru",  # misspelled
            "! var=fals",  # misspelled
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError, match="Unquoted value must be a number or boolean"
            ):
                self.parser.parse(command)


class TestStrictWhitespaceValidation:
    """Test suite for strict whitespace validation rules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_no_spaces_around_dots(self):
        """Test that spaces around dots are not allowed."""
        invalid_commands = [
            "! @each[document . sections]->task@{{value.item=items}}*",  # Space before dot
            "! @each[document. sections]->task@{{value.item=items}}*",  # Space after dot
            "! @each[document .sections]->task@{{value.item=items}}*",  # Space both sides
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError, match="Space.*'\\.'.*not allowed"):
                self.parser.parse(command)

    def test_no_spaces_around_at_symbol(self):
        """Test that spaces around @ symbol are not allowed."""
        invalid_commands = [
            "! @ each[items]->task@{{value.item=items}}*",  # Space after @
        ]

        # Test space after @
        with pytest.raises(CommandParseError, match="Space.*'@'.*not allowed"):
            self.parser.parse(invalid_commands[0])

        # Test space before @{{ (this is actually a braces rule, not @ rule)
        with pytest.raises(CommandParseError, match="Space.*'{{'.*not allowed"):
            self.parser.parse("! @each[items]->task @{{value.item=items}}*")

    def test_no_spaces_around_brackets(self):
        """Test that spaces around brackets are not allowed."""
        invalid_commands = [
            "! @each [items]->task@{{value.item=items}}*",  # Space before [
            "! @each[items] ->task@{{value.item=items}}*",  # Space after ]
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError,
                match="Space.*(before '\\['|after '\\]').*not allowed",
            ):
                self.parser.parse(command)

    def test_no_spaces_around_arrow(self):
        """Test that spaces around arrow are not allowed."""
        invalid_commands = [
            "! @each[items] ->task@{{value.item=items}}*",  # Space before ->
            "! @each[items]-> task@{{value.item=items}}*",  # Space after ->
            "! @each[items] -> task@{{value.item=items}}*",  # Spaces both sides
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError, match="Space.*(before|after).*'->'.*not allowed"
            ):
                self.parser.parse(command)

    def test_no_spaces_around_braces(self):
        """Test that spaces around braces are not allowed."""
        invalid_commands = [
            "! @each[items]->task @{{value.item=items}}*",  # Space before {{
            "! @each[items]->task@{{value.item=items}} *",  # Space after }}
        ]

        for command in invalid_commands:
            with pytest.raises(
                CommandParseError, match="Space.*(before|after).*('{{'|}}).*not allowed"
            ):
                self.parser.parse(command)

    def test_resampling_strict_whitespace(self):
        """Test strict whitespace rules for resampling commands."""
        invalid_commands = [
            "! @resampled [rating]->mean",  # Space before [
            "! @resampled[rating] ->mean",  # Space before ->
            "! @resampled[rating]-> mean",  # Space after ->
            "! @ resampled[rating]->mean",  # Space after @
            "! @resampled [rating] -> mean",  # Spaces around components
            "! @resampled[rating] -> mean",  # Space before arrow
        ]

        for command in invalid_commands:
            with pytest.raises(CommandParseError, match="Space.*not allowed"):
                self.parser.parse(command)

    def test_valid_strict_whitespace_commands(self):
        """Test that commands following strict whitespace rules are parsed correctly."""
        valid_commands = [
            "! @each[items]->task@{{value.item=items}}*",  # Perfect strict spacing
            "! @each[document.sections]->task.analyzer@{{value.title=document.sections.title}}*",  # Complex paths with correct full path
            "! @all->task.process@{{prompt.context=*}}*",  # @all with wildcard
            "! @each[items]->task@{{value.item=items}}*",  # Another valid command
            '! var="value"',  # Variable assignment with string
            "! count=42",  # Variable assignment with number
            "! enabled=true",  # Variable assignment with boolean
            "! llm('gpt-4',override=true)",  # Valid execution command with positional + named args
            "! @resampled[field]->mean",  # Resampling command
        ]

        for command in valid_commands:
            # Should not raise exceptions
            result = self.parser.parse(command)
            assert result is not None


class TestMultilineCommandSupport:
    """Test suite for multiline command support within brackets/braces."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_multiline_variable_mappings(self):
        """Test multiline support within {{}} for variable mappings."""
        multiline_command = """! @each[items]->task@{{
            value.item=items,
            prompt.context=items.data
        }}*"""

        result = self.parser.parse(multiline_command)
        assert isinstance(result, ParsedCommand)
        assert len(result.variable_mappings) == 2

    def test_multiline_inclusion_paths(self):
        """Test multiline support within [] for inclusion paths."""
        multiline_command = """! @each[
            document.sections.subsections
        ]->task@{{value.item=document.sections.subsections}}*"""

        result = self.parser.parse(multiline_command)
        assert isinstance(result, ParsedCommand)
        assert result.inclusion_path == "document.sections.subsections"

    def test_multiline_execution_arguments(self):
        """Test multiline support within () for execution arguments."""
        multiline_command = """! llm(
            'gpt-4',
            override=true
        )"""

        result = self.parser.parse(multiline_command)
        assert isinstance(result, ExecutionCommand)
        assert len(result.arguments) == 1
        assert result.named_arguments is not None
        assert len(result.named_arguments) == 1

    def test_multiline_with_comments(self):
        """Test multiline support with comments."""
        multiline_command = """! @each[items]->task@{{
            # Main content mapping
            value.item=items,
            # Context mapping
            prompt.context=items.data
        }}*"""

        result = self.parser.parse(multiline_command)
        assert isinstance(result, ParsedCommand)
        assert len(result.variable_mappings) == 2

    def test_multiline_normalization(self):
        """Test that multiline commands normalize to single-line equivalents."""
        multiline_command = """! @each[
            items
        ]->task@{{
            value.item=items
        }}*"""

        # The parser should normalize this to the equivalent single-line form
        result = self.parser.parse(multiline_command)
        assert isinstance(result, ParsedCommand)


class TestParserAggressiveEdgeCases:
    """Aggressive tests designed to break parser implementation with edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_malformed_paths_comprehensive(self):
        """Test all possible malformed path variations."""
        malformed_commands = [
            # Empty components
            "! @each[]->task@{{value.item=items}}*",  # Empty inclusion
            "! @each[.]->task@{{value.item=items}}*",  # Dot only inclusion
            "! @each[items]->@{{value.item=items}}*",  # Empty destination
            "! @each[items]->task@{{=items}}*",  # Empty variable target
            "! @each[items]->task@{{value.item=}}*",  # Empty variable source
            # Invalid dots
            "! @each[items.]->task@{{value.item=items}}*",  # Trailing dot in inclusion
            "! @each[.items]->task@{{value.item=items}}*",  # Leading dot in inclusion
            "! @each[items..nested]->task@{{value.item=items}}*",  # Double dots in inclusion
            "! @each[items]->task.@{{value.item=items}}*",  # Trailing dot in destination
            "! @each[items]->task@{{value..item=items}}*",  # Double dots in variable
            "! @each[items]->task@{{value.item=items.}}*",  # Trailing dot in source
            # Invalid characters
            "! @each[items-invalid]->task@{{value.item=items}}*",  # Hyphen in path
            "! @each[items@invalid]->task@{{value.item=items}}*",  # @ in path
            "! @each[items#invalid]->task@{{value.item=items}}*",  # # in path
            "! @each[items]->task-invalid@{{value.item=items}}*",  # Hyphen in destination
            "! @each[items]->task@{{value.item@invalid=items}}*",  # @ in variable path
            # Invalid brackets/braces
            "! @each[items]]->task@{{value.item=items}}*",  # Extra closing bracket
            "! @each[[items]->task@{{value.item=items}}*",  # Extra opening bracket
            "! @each[items]->task@{{{value.item=items}}*",  # Extra opening brace
            "! @each[items]->task@{{value.item=items}}}*",  # Extra closing brace
        ]

        for i, command in enumerate(malformed_commands):
            print(f"Testing command {i}: {command}")
            try:
                result = self.parser.parse(command)
                print(f"  UNEXPECTED SUCCESS: {result}")
                # This should have failed but didn't - this reveals a parser weakness
                assert False, (
                    f"Command should have failed but parsed successfully: {command}"
                )
            except CommandParseError as e:
                print(f"  Expected failure: {e}")
                # Expected behavior
                pass

    def test_boundary_condition_commands(self):
        """Test boundary conditions for command parsing."""
        boundary_commands = [
            # Minimal valid commands
            "! @->a@{{b=c}}",  # Shortest possible @all command
            "! a=b",  # Shortest variable assignment
            "! a()",  # Shortest execution command
            # Maximum reasonable length commands
            "! @each["
            + "a" * 100
            + "]->task@{{value.item=items}}*",  # Very long inclusion path
            "! @each[items]->"
            + "task." * 50
            + "target@{{value.item=items}}*",  # Very long destination
            # Edge case multiplicity
            "! @each[items]->task@{{value.item=items}}**",  # Double multiplicity (invalid)
            "! @each[items]->task@{{value.item=items}}*#comment",  # Multiplicity with comment
        ]

        for command in boundary_commands:
            try:
                result = self.parser.parse(command)
                # If it doesn't raise an exception, it should at least return a valid result
                assert result is not None
            except CommandParseError:
                # Expected for invalid boundary cases
                pass

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_commands = [
            # Unicode in paths (should be rejected)
            "! @each[items_ñ]->task@{{value.item=items}}*",  # Non-ASCII in inclusion
            "! @each[items]->tâsk@{{value.item=items}}*",  # Non-ASCII in destination
            "! @each[items]->task@{{valué.item=items}}*",  # Non-ASCII in variable
            # Special ASCII characters
            "! @each[items$]->task@{{value.item=items}}*",  # Dollar sign
            "! @each[items%]->task@{{value.item=items}}*",  # Percent sign
            "! @each[items&]->task@{{value.item=items}}*",  # Ampersand
            # Control characters (should be rejected)
            "! @each[items\t]->task@{{value.item=items}}*",  # Tab in path
            # Note: newlines within [] are allowed per multiline specification
        ]

        for i, command in enumerate(unicode_commands):
            print(f"Testing unicode command {i}: {repr(command)}")
            try:
                result = self.parser.parse(command)
                print(f"  UNEXPECTED SUCCESS: {result}")
                # This should have failed but didn't - another parser weakness!
                assert False, (
                    f"Command with unicode/special chars should have failed: {command}"
                )
            except CommandParseError as e:
                print(f"  Expected failure: {e}")
                # Expected behavior
                pass

    def test_valid_newlines_in_contexts(self):
        """Test that newlines are correctly allowed within brackets, braces, and parentheses."""
        valid_commands = [
            # Newline within brackets (inclusion paths)
            "! @each[items\n]->task@{{value.item=items}}*",
            "! @each[\nitems\n]->task@{{value.item=items}}*",
            # Newline within braces (variable mappings) - tested in multiline tests
            "! @each[items]->task@{{\nvalue.item=items\n}}*",
            # Newline within parentheses (execution arguments)
            "! llm(\n'gpt-4'\n)",
        ]

        for command in valid_commands:
            # These should all parse successfully
            result = self.parser.parse(command)
            assert result is not None

    def test_command_prefix_edge_cases(self):
        """Test edge cases with command prefix and structure."""
        prefix_commands = [
            # Missing command prefix
            "@each[items]->task@{{value.item=items}}*",  # No ! prefix
            "  @each[items]->task@{{value.item=items}}*",  # No ! prefix with spaces
            # Multiple command prefixes
            "!! @each[items]->task@{{value.item=items}}*",  # Double !
            "! ! @each[items]->task@{{value.item=items}}*",  # Spaced double !
            # Invalid command structure
            "!",  # Just prefix
            "! ",  # Prefix with space only
            "! invalid_command",  # Invalid command type
        ]

        for command in prefix_commands:
            with pytest.raises(CommandParseError, match=".*"):
                self.parser.parse(command)

    def test_nested_structure_edge_cases(self):
        """Test edge cases with nested structures and complex mappings."""
        nested_commands = [
            # Nested braces (invalid)
            "! @each[items]->task@{{value.item={{nested}}}}*",  # Nested braces in value
            "! @each[items]->task@{{{{nested}}.item=items}}*",  # Nested braces in key
            # Nested brackets (invalid)
            "! @each[items[nested]]->task@{{value.item=items}}*",  # Nested brackets
            "! @each[[nested]items]->task@{{value.item=items}}*",  # Leading nested brackets
            # Mixed nesting
            "! @each[items{mixed}]->task@{{value.item=items}}*",  # Braces in brackets
            "! @each[items]->task@{{value[mixed].item=items}}*",  # Brackets in braces
            # Complex invalid combinations
            "! @each[items]->task@{{value.item=items[invalid]}}*",  # Brackets in value
            "! @each[items]->task@{{value.item=items{invalid}}}*",  # Braces in value
        ]

        for command in nested_commands:
            with pytest.raises(CommandParseError, match=".*"):
                self.parser.parse(command)

    def test_comment_edge_cases(self):
        """Test edge cases with comments."""
        comment_commands = [
            # Valid comment variations
            "! @each[items]->task@{{value.item=items}}* # comment",  # Space before comment
            "! @each[items]->task@{{value.item=items}}*# comment",  # No space before comment
            "! @each[items]->task@{{value.item=items}}*#",  # Empty comment
            "! var=value # comment with spaces and symbols !@#$%",  # Complex comment
            # Comments in wrong places (should be ignored or cause errors)
            "! @each[items#comment]->task@{{value.item=items}}*",  # Comment in inclusion (invalid)
            "! @each[items]->task#comment@{{value.item=items}}*",  # Comment in destination (invalid)
            "! @each[items]->task@{{value.item#comment=items}}*",  # Comment in mapping (invalid)
        ]

        for command in comment_commands:
            try:
                result = self.parser.parse(command)
                # Valid comment cases should parse successfully
                if "#" in command and (
                    "[" not in command.split("#")[1]
                    or "{{" not in command.split("#")[1]
                ):
                    assert result is not None
            except CommandParseError:
                # Invalid comment placements should raise errors
                pass


class TestIterableDepthTracking:
    """Test iterable depth assignment during path parsing."""

    def setup_method(self):
        """Create parser instance for testing."""
        self.parser = CommandParser()

    def test_simple_iterable_path_depth(self):
        """Test depth assignment for simple iterable path: iterable1."""

        class SimpleNode(TreeNode):
            items: list[str] = []

        # Mock node for path resolution
        node = SimpleNode()

        # Test path: items (1 iterable)
        path = ["items"]
        depth = self.parser.calculate_iterable_depth(node, path)

        assert depth == 1, f"Expected depth 1 for single iterable, got {depth}"

    def test_nested_iterable_path_depth(self):
        """Test depth assignment for nested iterable path: iterable1.noniterable.iterable2."""

        class Level2Node(TreeNode):
            subitems: list[str] = []
            data: str = ""

        class Level1Node(TreeNode):
            items: list[Level2Node] = []

        # Mock node for path resolution
        node = Level1Node()

        # Test path: items.subitems (2 iterables)
        path = ["items", "subitems"]
        depth = self.parser.calculate_iterable_depth(node, path)

        assert depth == 2, f"Expected depth 2 for two iterables, got {depth}"

    def test_complex_iterable_path_depth(self):
        """Test depth for complex path: iterable1.noniterable.iterable2.noniterable.iterable3."""

        class Level3Node(TreeNode):
            tags: list[str] = []
            name: str = ""

        class Level2Node(TreeNode):
            elements: list[Level3Node] = []
            title: str = ""

        class Level1Node(TreeNode):
            items: list[Level2Node] = []

        # Mock node for path resolution
        node = Level1Node()

        # Test path: items.elements.tags (3 iterables)
        path = ["items", "elements", "tags"]
        depth = self.parser.calculate_iterable_depth(node, path)

        assert depth == 3, f"Expected depth 3 for three iterables, got {depth}"

    def test_noniterable_only_path_depth(self):
        """Test depth for path with only non-iterable fields."""

        class ConfigNode(TreeNode):
            data: str = ""

        class RootNode(TreeNode):
            config: ConfigNode = ConfigNode()

        # Mock node for path resolution
        node = RootNode()

        # Test path: config.data (0 iterables)
        path = ["config", "data"]
        depth = self.parser.calculate_iterable_depth(node, path)

        assert depth == 0, f"Expected depth 0 for no iterables, got {depth}"

    def test_mixed_spacing_same_depth(self):
        """Test that different non-iterable spacing gives same depth for same iterable count."""

        # Structure A: iterable1.iterable2.iterable3 (minimal spacing)
        class NodeA3(TreeNode):
            data: str = ""

        class NodeA2(TreeNode):
            level3: list[NodeA3] = []

        class NodeA1(TreeNode):
            level2: list[NodeA2] = []

        node_a = NodeA1()
        path_a = ["level2", "level3"]  # 2 iterables
        depth_a = self.parser.calculate_iterable_depth(node_a, path_a)

        # Structure B: iterable1.noniterable.noniterable.iterable2 (maximal spacing)
        class NodeB4(TreeNode):
            data: str = ""

        class NodeB3(TreeNode):
            level4: list[NodeB4] = []

        class NodeB2(TreeNode):
            level3: NodeB3 = NodeB3()  # non-iterable

        class NodeB1(TreeNode):
            level2: NodeB2 = NodeB2()  # non-iterable

        class RootB(TreeNode):
            level1: list[NodeB1] = []

        node_b = RootB()
        path_b = ["level1", "level2", "level3", "level4"]  # 2 iterables with spacing
        depth_b = self.parser.calculate_iterable_depth(node_b, path_b)

        assert depth_a == depth_b == 2, (
            f"Expected both depths to be 2, got A={depth_a}, B={depth_b}"
        )


class TestParserIterableDepthIntegration:
    """Test that parser automatically assigns iterable depths during command parsing."""

    def setup_method(self):
        """Create parser instance for testing."""
        self.parser = CommandParser()

    def test_parser_calculates_inclusion_path_depth(self):
        """Test that parser can calculate inclusion path depth using existing functionality."""

        # Create TreeNode structure: sections.items (depth 2)
        class ItemNode(TreeNode):
            text: str = ""

        class SectionNode(TreeNode):
            items: list[ItemNode] = []
            name: str = ""

        class DocumentNode(TreeNode):
            sections: list[SectionNode] = []

        node = DocumentNode()
        path_components = ["sections", "items"]

        # Test existing calculate_iterable_depth functionality
        depth = self.parser.calculate_iterable_depth(node, path_components)
        assert depth == 2, f"Expected depth 2, got {depth}"

    def test_parser_calculates_rhs_path_depths(self):
        """Test that parser can calculate RHS path depths using existing functionality."""

        # Create TreeNode structure matching the command
        class ItemNode(TreeNode):
            text: str = ""

        class SectionNode(TreeNode):
            items: list[ItemNode] = []
            name: str = ""  # depth 1 (sections.name)

        class DocumentNode(TreeNode):
            sections: list[SectionNode] = []

        node = DocumentNode()

        # Test different RHS paths
        rhs_path1 = ["sections", "items", "text"]  # depth 2
        rhs_path2 = ["sections", "name"]  # depth 1

        depth1 = self.parser.calculate_iterable_depth(node, rhs_path1)
        depth2 = self.parser.calculate_iterable_depth(node, rhs_path2)

        assert depth1 == 2, f"Expected depth 2 for sections.items.text, got {depth1}"
        assert depth2 == 1, f"Expected depth 1 for sections.name, got {depth2}"


class TestLastMatchingIterableValidation:
    """Test last-matching-iterable algorithm for enhanced subchain validation."""

    def setup_method(self):
        """Create parser instance for testing."""
        self.parser = CommandParser()

    def test_complete_coverage_valid(self):
        """Test case with complete coverage - at least one RHS reaches inclusion's last iterable."""

        class Sentence(TreeNode):
            text: str = ""

        class Paragraph(TreeNode):
            sentences: list[Sentence] = []
            title: str = ""

        class Section(TreeNode):
            paragraphs: list[Paragraph] = []
            title: str = ""

        node = Section()
        inclusion_path = ["paragraphs", "sentences"]
        rhs_paths = [
            ["paragraphs", "sentences", "text"],  # extends beyond last iterable ✅
            ["paragraphs", "title"],  # shorter subchain ✅
        ]

        result = self.parser.validate_last_matching_iterable(
            node, inclusion_path, rhs_paths
        )
        assert result["valid"], f"Expected valid, got {result}"

    def test_empty_paths_illegal(self):
        """Test that empty paths raise CommandParseError (violate LangTree DSL syntax)."""
        from langtree.parsing.parser import CommandParseError

        class SimpleNode(TreeNode):
            items: list[str] = []

        node = SimpleNode()

        # Empty inclusion path - should raise CommandParseError
        with pytest.raises(CommandParseError, match="Empty inclusion path"):
            self.parser.validate_last_matching_iterable(node, [], [["items"]])

        # Empty RHS paths - should raise CommandParseError
        with pytest.raises(CommandParseError, match="Empty RHS paths"):
            self.parser.validate_last_matching_iterable(node, ["items"], [])

    def test_identical_paths_valid(self):
        """Test case where RHS exactly matches inclusion_path."""

        class Item(TreeNode):
            text: str = ""

        class Section(TreeNode):
            items: list[Item] = []

        node = Section()
        inclusion_path = ["items"]
        rhs_paths = [["items"]]  # Exact match

        result = self.parser.validate_last_matching_iterable(
            node, inclusion_path, rhs_paths
        )
        assert result["valid"]

    def test_non_iterable_only_paths_invalid(self):
        """Test that paths with no iterables violate @each semantics."""
        from langtree.parsing.parser import CommandParseError

        class Config(TreeNode):
            name: str = ""

        class Root(TreeNode):
            config: Config = Config()

        node = Root()
        inclusion_path = ["config", "name"]  # No iterables - violates @each semantics
        rhs_paths = [["config", "name"]]

        # Should raise error because @each requires at least one iterable
        with pytest.raises(
            CommandParseError, match="@each requires at least one iterable"
        ):
            self.parser.validate_last_matching_iterable(node, inclusion_path, rhs_paths)

    def test_mixed_iterable_positions_complex(self):
        """Test complex case with iterables at different positions."""

        class Tag(TreeNode):
            name: str = ""

        class Item(TreeNode):
            tags: list[Tag] = []
            title: str = ""

        class Config(TreeNode):
            items: list[Item] = []

        class Root(TreeNode):
            configs: list[Config] = []
            title: str = ""

        node = Root()
        inclusion_path = ["configs", "items", "tags"]  # iterables at pos 0, 1, 2
        rhs_paths = [
            ["configs", "items", "tags", "name"],  # complete coverage ✅
            ["configs", "items", "title"],  # partial coverage ✅
            ["configs", "items"],  # valid subchain ✅
        ]

        result = self.parser.validate_last_matching_iterable(
            node, inclusion_path, rhs_paths
        )
        assert result["valid"]
