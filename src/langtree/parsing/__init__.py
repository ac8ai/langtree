"""
LangTree DSL parsing components.

This package provides command parsing, path resolution, and parsing validation
for the LangTree DSL framework.
"""

from langtree.core.path_utils import ScopeModifier
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
    VariableMapping,
    parse_command,
)
from langtree.parsing.path_resolver import analyze_scope_coverage
from langtree.parsing.validation import validate_field_types

__all__ = [
    "CommandParseError",
    "CommandParser",
    "CommandType",
    "ExecutionCommand",
    "NodeModifierCommand",
    "NodeModifierType",
    "ParsedCommand",
    "ResamplingCommand",
    "ScopeModifier",
    "VariableAssignmentCommand",
    "VariableMapping",
    "parse_command",
    "analyze_scope_coverage",
    "validate_field_types",
]
