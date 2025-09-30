"""
Parser for Action Chaining Language commands.

This module provides parsing functionality for commands that control data flow
between TreeNode instances in hierarchical prompt execution.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from langtree.core.path_utils import PathResolver, ResolvedPath

if TYPE_CHECKING:
    pass


class CommandType(Enum):
    """Type of command operation."""

    EACH = "each"
    ALL = "all"
    VARIABLE_ASSIGNMENT = "variable_assignment"
    EXECUTION = "execution"
    RESAMPLING = "resampling"
    NODE_MODIFIER = "node_modifier"
    COMMENT = "comment"


class NodeModifierType(Enum):
    """Type of node modifier."""

    SEQUENTIAL = "@sequential"
    PARALLEL = "@parallel"
    TOGETHER = "together"


@dataclass
class VariableMapping:
    """Represents a variable mapping in the command."""

    target_path: str  # Left side of the assignment (e.g., "value.main_analysis.title")
    source_path: str  # Right side of the assignment (e.g., "sections.title")

    # Scope resolution for all path components
    resolved_target: ResolvedPath | None = None
    resolved_source: ResolvedPath | None = None

    def __post_init__(self):
        """Resolve scope modifiers from all paths."""
        self.resolved_target = PathResolver.resolve_path_with_scope_instance(
            self.target_path
        )
        self.resolved_source = PathResolver.resolve_path_with_scope_instance(
            self.source_path
        )


@dataclass
class VariableAssignmentCommand:
    """Represents a variable assignment command (! var=value)."""

    variable_name: str
    value: str | int | float | bool  # Parsed value (string, number, or boolean)
    comment: str | None = None

    def __post_init__(self):
        """Validate variable name format."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.variable_name):
            raise CommandParseError(f"Invalid variable name: {self.variable_name}")

    def __str__(self) -> str:
        """Return a string representation of the command."""
        return f"{self.variable_name}={self.value}"


@dataclass
class ExecutionCommand:
    """Represents an execution command (! command(args))."""

    command_name: str
    arguments: list[
        str | int | float | bool
    ]  # Positional arguments (variables or literals)
    named_arguments: dict[str, str | int | float | bool] | None = (
        None  # Named arguments (key=value)
    )
    comment: str | None = None

    # Built-in commands only
    VALID_COMMANDS = {"resample", "llm"}

    def __post_init__(self):
        """Validate command name and arguments."""
        if self.named_arguments is None:
            self.named_arguments = {}

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.command_name):
            raise CommandParseError(f"Invalid command name: {self.command_name}")

        if self.command_name not in self.VALID_COMMANDS:
            raise CommandParseError(
                f"Unknown command: {self.command_name}. Valid commands are: {', '.join(sorted(self.VALID_COMMANDS))}"
            )

        # Validate arguments based on command
        if self.command_name == "resample":
            self._validate_resample_args()
        elif self.command_name == "llm":
            self._validate_llm_args()

    def _validate_resample_args(self):
        """Validate resample command arguments."""
        # Named arguments not supported for resample - check this first
        if self.named_arguments:
            raise CommandParseError("resample command does not support named arguments")

        named_args_count = len(self.named_arguments) if self.named_arguments else 0
        total_args = len(self.arguments) + named_args_count
        if total_args != 1:
            raise CommandParseError(
                f"resample command requires exactly 1 argument (n_times), got {total_args}"
            )

        # Argument can be int literal or variable name (string) - NOT boolean
        arg = self.arguments[0]
        if isinstance(arg, bool):
            # Boolean values are not valid for resample
            raise CommandParseError(
                f"resample argument must be int or variable name, got {type(arg).__name__}: {arg}"
            )
        elif isinstance(arg, str):
            # Variable name - validate format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", arg):
                raise CommandParseError(
                    f"Invalid variable name in resample argument: {arg}"
                )
        elif isinstance(arg, int):
            # Integer literal - validate positive
            if arg <= 0:
                raise CommandParseError(f"resample n_times must be positive, got {arg}")
        else:
            raise CommandParseError(
                f"resample argument must be int or variable name, got {type(arg).__name__}: {arg}"
            )

    def _validate_llm_args(self):
        """Validate llm command arguments."""
        named_args_count = len(self.named_arguments) if self.named_arguments else 0
        total_args = len(self.arguments) + named_args_count
        if not (1 <= total_args <= 2):
            raise CommandParseError(
                f"llm command requires 1-2 arguments (model_key, [override]), got {total_args}"
            )

        # First argument: model_key (string literal or variable name) - always positional
        if len(self.arguments) == 0:
            raise CommandParseError(
                "llm command requires model_key as first positional argument"
            )

        model_key = self.arguments[0]
        if not isinstance(model_key, str):
            raise CommandParseError(
                f"llm model_key must be string, got {type(model_key).__name__}: {model_key}"
            )

        # Second argument: override (bool literal or variable name) - can be positional or named
        if total_args == 2:
            if len(self.arguments) == 2:
                # Positional override argument
                override = self.arguments[1]
                self._validate_override_value(override)
            elif self.named_arguments and "override" in self.named_arguments:
                # Named override argument
                override = self.named_arguments["override"]
                self._validate_override_value(override)
            else:
                # Unknown named argument
                unknown_args = (
                    list(self.named_arguments.keys()) if self.named_arguments else []
                )
                raise CommandParseError(
                    f"llm command only supports 'override' named argument, got: {', '.join(unknown_args)}"
                )

        # Validate no unknown named arguments
        if self.named_arguments and "override" not in self.named_arguments:
            unknown_args = list(self.named_arguments.keys())
            raise CommandParseError(
                f"llm command only supports 'override' named argument, got: {', '.join(unknown_args)}"
            )

    def _validate_override_value(self, override):
        """Validate the override parameter value."""
        if isinstance(override, str):
            # Variable name - validate format
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", override):
                raise CommandParseError(
                    f"Invalid variable name in llm override argument: {override}"
                )
        elif isinstance(override, bool):
            # Boolean literal - always valid
            pass
        else:
            raise CommandParseError(
                f"llm override must be bool or variable name, got {type(override).__name__}: {override}"
            )


@dataclass
class NodeModifierCommand:
    """Represents a node modifier command (! @sequential, ! @parallel, ! together)."""

    modifier: NodeModifierType
    comment: str | None = None

    @classmethod
    def from_string(
        cls, modifier_str: str, comment: str | None = None
    ) -> "NodeModifierCommand":
        """Create NodeModifierCommand from string modifier."""
        modifier_map = {
            "@sequential": NodeModifierType.SEQUENTIAL,
            "@parallel": NodeModifierType.PARALLEL,
            "together": NodeModifierType.TOGETHER,
        }

        if modifier_str not in modifier_map:
            raise CommandParseError(f"Invalid node modifier: {modifier_str}")

        return cls(modifier=modifier_map[modifier_str], comment=comment)


@dataclass
class ResamplingCommand:
    """Represents a resampling aggregation command (! @resampled[field]->function)."""

    field_name: str
    aggregation_function: str
    comment: str | None = None

    NUMERICAL_FUNCTIONS = {"mean", "median", "min", "max"}
    VALID_FUNCTIONS = {"mode"} | NUMERICAL_FUNCTIONS

    def __post_init__(self):
        """Validate field name and aggregation function."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.field_name):
            raise CommandParseError(f"Invalid field name: {self.field_name}")

        if self.aggregation_function not in self.VALID_FUNCTIONS:
            raise CommandParseError(
                f"Invalid aggregation function: {self.aggregation_function}"
            )

    def is_numerical_function(self) -> bool:
        """Check if this function requires numerical enum values."""
        return self.aggregation_function in self.NUMERICAL_FUNCTIONS


@dataclass
class CommentCommand:
    """Represents a standalone comment command (! # comment)."""

    comment: str

    def __str__(self) -> str:
        """Return a string representation of the comment."""
        return f"# {self.comment}"


@dataclass
class ParsedCommand:
    """
    Represents a parsed LangTree DSL command with source location tracking.

    Params:
        command_type: Type of command (EACH, ALL, etc.)
        destination_path: Target path for the command
        variable_mappings: List of variable mappings
        inclusion_path: Optional inclusion path for @each commands
        has_multiplicity: True if command ends with *
        is_wildcard_assignment: True if uses * assignment
        comment: Optional comment text
        docstring_line: Line number within docstring or field description
        source_node_tag: TreeNode class name where command is defined
        source_node_file: Python file where TreeNode is defined
        source_node_line: Line number where TreeNode class starts
        field_name: Field name if command is in Field description
        field_line: Line number where Field is defined
        raw_command_text: Original command text as written
    """

    command_type: CommandType
    destination_path: str
    variable_mappings: list[VariableMapping]
    inclusion_path: str | None = None  # For @each commands
    has_multiplicity: bool = False
    is_wildcard_assignment: bool = False
    comment: str | None = None
    docstring_line: int | None = None
    source_node_tag: str | None = None
    source_node_file: str | None = None
    source_node_line: int | None = None
    field_name: str | None = None
    field_line: int | None = None
    raw_command_text: str | None = None
    resolved_destination: ResolvedPath | None = None
    resolved_inclusion: ResolvedPath | None = None

    def __post_init__(self):
        """Resolve scope modifiers from all paths."""
        self.resolved_destination = PathResolver.resolve_path_with_scope_instance(
            self.destination_path
        )
        if self.inclusion_path:
            self.resolved_inclusion = PathResolver.resolve_path_with_scope_instance(
                self.inclusion_path
            )

    def __str__(self) -> str:
        """Return a string representation of the command."""
        command_str = f"@{self.command_type.value}"
        if self.inclusion_path:
            command_str += f"[{self.inclusion_path}]"
        command_str += f"->{self.destination_path}"
        return command_str

    def is_one_to_many(self) -> bool:
        """Check if this represents a 1:n relationship."""
        return self.command_type == CommandType.ALL and self.has_multiplicity

    def is_many_to_many(self) -> bool:
        """Check if this represents an n:n relationship."""
        return self.command_type == CommandType.EACH and self.has_multiplicity


class CommandParseError(Exception):
    """Exception raised when command parsing fails."""

    pass


class CommandParser:
    """Parser for Action Chaining Language commands."""

    # Regex patterns for command parsing
    COMMAND_PATTERN = re.compile(
        r"^!\s*@(?P<command_type>each|all)?\s*"
        r"(?:\[(?P<inclusion>[^\]]*)\])?\s*"
        r"->\s*(?P<destination>\S+)"  # No space allowed after destination
        r"@\{\{\s*(?P<mappings>[^}]*)\s*\}\}\s*"
        r"(?P<multiplicity>\*)?"
        r"(?:\s*#(?P<comment>.*))?$"  # Optional comment support
    )

    # New patterns for extended syntax - handle quoted strings properly
    # Variable assignment: ! var=value (no @ or -> to avoid conflict with traditional commands)
    VARIABLE_ASSIGNMENT_PATTERN = re.compile(
        r'^!\s*(?P<variable_name>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?P<value>(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^#@\s][^#@]*?))(?:\s*#(?P<comment>.*))?$'
    )

    # Execution command: ! command(args) (no @ or -> to avoid conflict with traditional commands)
    EXECUTION_PATTERN = re.compile(
        r"^!\s*(?P<command_name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(?P<arguments>.*?)\s*\)(?:\s*#(?P<comment>.*))?$"
    )

    # Resampling command: ! @resampled[field]->function
    RESAMPLING_PATTERN = re.compile(
        r"^!\s*@resampled\s*\[\s*(?P<field_name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*->\s*(?P<function>[a-zA-Z_][a-zA-Z0-9_]*)(?:\s*#(?P<comment>.*))?$"
    )

    # Node modifier command: ! @sequential, ! @parallel, ! together
    NODE_MODIFIER_PATTERN = re.compile(
        r"^!\s*(?P<modifier>@sequential|@parallel|together)(?:\s*#(?P<comment>.*))?$"
    )

    # Standalone comment pattern: ! # comment or !# comment
    COMMENT_PATTERN = re.compile(r"^!\s*#(?P<comment>.*)$")

    VARIABLE_MAPPING_PATTERN = re.compile(
        r"(?P<target>[^=,:\s]+)\s*=\s*(?P<source>[^=,:\s]*)",  # Added colon to forbidden chars
        re.VERBOSE,
    )

    IMPLICIT_MAPPING_PATTERN = re.compile(
        r"^(?P<target>[^=,:\s]+)$",  # Added colon to forbidden chars to prevent matching colon syntax
        re.VERBOSE,
    )

    VALID_SCOPE_MODIFIERS = {"prompt", "value", "outputs", "task"}

    def _normalize_multiline_command(self, command: str) -> str:
        """
        Normalize multiline commands to single-line equivalents.

        Multiline continuation is only allowed within:
        - Brackets: [...]
        - Braces: {{...}}
        - Parentheses: (...)

        Params:
            command: The potentially multiline command string

        Returns:
            Normalized single-line command string
        """
        if "\n" not in command:
            return command  # Already single-line

        lines = command.split("\n")
        result = []
        context_stack = []
        current_segment = ""

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments within contexts
            if not stripped or stripped.startswith("#"):
                if not context_stack:
                    # Preserve structure outside contexts
                    if current_segment:
                        result.append(current_segment)
                        current_segment = ""
                    if stripped and not stripped.startswith("#"):
                        result.append(stripped)
                continue

            # Track context depth
            for char in stripped:
                if char in "[{(":
                    context_stack.append(char)
                elif char in "]})":
                    if context_stack:
                        context_stack.pop()

            if context_stack:
                # Within context - accumulate with spaces
                if current_segment:
                    current_segment += " " + stripped
                else:
                    current_segment = stripped
            else:
                # Outside context - end current segment
                if current_segment:
                    current_segment += " " + stripped
                    result.append(current_segment)
                    current_segment = ""
                else:
                    result.append(stripped)

        if current_segment:
            result.append(current_segment)

        # Join and clean up extra spaces
        normalized = " ".join(result)
        # Remove extra spaces around operators while preserving quotes
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _validate_forbidden_characters(self, command: str) -> None:
        """
        Validate that command doesn't contain forbidden characters in inappropriate contexts.

        This validates:
        - Non-ASCII unicode characters (outside quoted strings)
        - Control characters (tab, etc.) except newlines in multiline contexts

        Params:
            command: Raw command string before normalization

        Raises:
            CommandParseError: If forbidden characters are found
        """
        # Check for non-ASCII characters outside quoted strings
        in_string = False
        quote_char = None
        escaped = False

        for i, char in enumerate(command):
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    quote_char = char
                elif ord(char) > 127:  # Non-ASCII outside strings
                    raise CommandParseError(
                        f"Invalid non-ASCII character '{char}' at position {i}"
                    )
            else:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote_char:
                    in_string = False
                    quote_char = None
                # Unicode allowed inside strings, so no validation needed

        # Check for forbidden control characters (but allow newlines - they'll be validated separately)
        forbidden_control_chars = []
        for char in command:
            if char in "\t\r\f\v":  # Removed \n from this check
                if char == "\t":
                    forbidden_control_chars.append("tab")
                elif char == "\r":
                    forbidden_control_chars.append("carriage return")
                elif char == "\f":
                    forbidden_control_chars.append("form feed")
                elif char == "\v":
                    forbidden_control_chars.append("vertical tab")

        if forbidden_control_chars:
            raise CommandParseError(
                f"Invalid control characters in command: {', '.join(set(forbidden_control_chars))}"
            )

        # Special validation for newlines - only allowed in multiline contexts
        if "\n" in command:
            # Only validate multiline usage if command contains bracket/brace/parenthesis contexts
            if any(char in command for char in ["[", "]", "{", "}", "(", ")"]):
                self._validate_newline_usage(command)

    def _validate_newline_usage(self, command: str) -> None:
        """
        Validate that newlines only appear in legitimate multiline contexts.

        Newlines are only allowed between complete tokens within:
        - Brackets: [...]
        - Braces: {{...}}
        - Parentheses: (...)

        Params:
            command: Command string that contains newlines

        Raises:
            CommandParseError: If newlines appear outside valid contexts or within tokens
        """
        # For control characters like literal \n (not multiline), we need different validation
        # Check if this looks like a control character injection rather than actual multiline
        if "\\n" in command or "\\t" in command or any(c in command for c in "\r\f\v"):
            raise CommandParseError("Invalid control characters in command: newline")

        # This is actual multiline (contains \n characters)
        # Validate that newlines only appear where line breaks make sense
        lines = command.split("\n")

        # Single line commands should not reach here, but just in case
        if len(lines) <= 1:
            return

        # For multiline: validate context and structure
        context_stack = []
        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            # Skip empty lines and comment-only lines
            if not stripped_line or stripped_line.startswith("#"):
                continue

            # Track bracket/brace/paren depth through this line
            for char in line:
                if char in "[{(":
                    context_stack.append(char)
                elif char in "]})":
                    if context_stack:
                        context_stack.pop()

            # If this is not the last line and we're not in any context,
            # this multiline structure is invalid
            if line_num < len(lines) - 1 and not context_stack:
                # Allow empty or comment lines to terminate outside contexts
                remaining_lines = lines[line_num + 1 :]
                if any(
                    line.strip() and not line.strip().startswith("#")
                    for line in remaining_lines
                ):
                    raise CommandParseError(
                        "Invalid newline outside of brackets [], braces {{}}, or parentheses ()"
                    )

        # Ensure all contexts are properly closed
        if context_stack:
            raise CommandParseError(
                "Unclosed brackets, braces, or parentheses in multiline command"
            )

    def _validate_iteration_matching(
        self, inclusion_path: str, variable_mappings: list[VariableMapping]
    ) -> None:
        """
        Validate iteration matching rules for @each commands.

        For @each[sections.subsections], validates that:
        1. Right side paths start from iteration root (sections.subsections)
        2. Left side mirrors iteration depth implicitly

        Params:
            inclusion_path: The inclusion path like "sections.subsections"
            variable_mappings: List of variable mappings to validate

        Raises:
            CommandParseError: If iteration matching rules are violated
        """
        if not inclusion_path or not variable_mappings:
            return

        # Parse iteration depth from inclusion path
        inclusion_parts = inclusion_path.split(".")
        iteration_root = inclusion_parts[
            0
        ]  # e.g., "sections" from "sections.subsections"

        for mapping in variable_mappings:
            source_path = mapping.source_path

            # Skip wildcard and scope-modified paths
            if (
                source_path == "*"
                or "." in source_path
                and source_path.split(".")[0] in ("prompt", "value", "outputs", "task")
            ):
                continue

            # Validate right side: must start from iteration root
            source_parts = source_path.split(".")
            if len(source_parts) > 0 and source_parts[0] != iteration_root:
                # Allow access to parent fields (e.g., sections.title for [sections.subsections])
                # But the first part should still be from the iteration context
                if not self._is_valid_iteration_context_access(
                    source_path, inclusion_path
                ):
                    raise CommandParseError(
                        f"Source path '{source_path}' must start from iteration root '{iteration_root}' "
                        f"for @each[{inclusion_path}]"
                    )

    def _is_valid_iteration_context_access(
        self, source_path: str, inclusion_path: str
    ) -> bool:
        """
        Check if source path is valid access within iteration context.

        Examples for @each[document.sections]:
        - document.sections ✅ (iteration items)
        - document.sections.content ✅ (field of each item)
        - document.title ✅ (parent level field)
        - sections.title ✅ (accessing current item collection's fields)
        - title ✅ (shorthand for current item field)
        """
        inclusion_parts = inclusion_path.split(".")
        source_parts = source_path.split(".")

        if not source_parts:
            return False

        iteration_root = inclusion_parts[0]  # "document"
        iteration_collection = (
            inclusion_parts[-1] if len(inclusion_parts) > 1 else inclusion_parts[0]
        )  # "sections"

        # Case 1: Full path starting from iteration root (e.g., document.sections.title)
        if source_parts[0] == iteration_root:
            return True

        # Case 2: Path starting from the iteration variable (collection being iterated)
        # For @each[sections], allow "sections.title" ✅
        # For @each[document.sections], allow "sections.title" ✅ (sections is the iteration variable)
        if source_parts[0] == iteration_collection:
            return True

        # Case 3: Shorthand field access for current iteration item
        # This is a complex case - we need to be very strict here
        # For @each[items], a shorthand like "title" could mean "items.title"
        # But we should NOT allow arbitrary names like "elements" for @each[items]
        # The shorthand should make semantic sense as a field of the iteration item

        # For now, disable shorthand entirely to match test expectations
        # Shorthand support can be added later with proper field validation
        return False

    def _validate_path(self, path: str, path_type: str) -> None:
        """
        Validate that a path conforms to the language specification.

        Valid paths:
        - Must not be empty or just whitespace
        - Must not be just dots (., .., etc.)
        - Must not contain invalid characters (-, @, #, etc.)
        - Must not have leading/trailing dots
        - Must not have consecutive dots (..)
        - Components must be valid identifiers ([a-zA-Z_][a-zA-Z0-9_]*)

        Params:
            path: The path to validate
            path_type: Description of the path type for error messages

        Raises:
            CommandParseError: If the path is invalid
        """
        if not path or path.isspace():
            raise CommandParseError(f"Invalid {path_type}: path cannot be empty")

        # Check for just dots
        if re.match(r"^\.+$", path):
            raise CommandParseError(f"Invalid {path_type}: path cannot be just dots")

        # Check for leading/trailing dots
        if path.startswith(".") or path.endswith("."):
            raise CommandParseError(
                f"Invalid {path_type}: path cannot start or end with dots"
            )

        # Check for consecutive dots
        if ".." in path:
            raise CommandParseError(
                f"Invalid {path_type}: path cannot contain consecutive dots"
            )

        # Check for invalid characters
        if re.search(r"[^a-zA-Z0-9_.]", path):
            invalid_chars = re.findall(r"[^a-zA-Z0-9_.]", path)
            raise CommandParseError(
                f"Invalid {path_type}: contains invalid characters: {', '.join(set(invalid_chars))}"
            )

        # Validate each path component
        components = path.split(".")
        for component in components:
            if not component:  # This catches cases like "a..b"
                raise CommandParseError(f"Invalid {path_type}: empty path component")

            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", component):
                raise CommandParseError(
                    f"Invalid {path_type}: '{component}' is not a valid identifier"
                )

    def _validate_strict_whitespace(self, command: str) -> None:
        """
        Validate that command follows strict whitespace rules.

        Strict rules:
        - No spaces around dots (.)
        - No spaces around @ symbol
        - No spaces around brackets [, ]
        - No spaces around arrow (->)
        - No spaces around braces {{, }}

        Params:
            command: The command string to validate

        Raises:
            CommandParseError: If whitespace rules are violated
        """
        errors = []

        # Check for space around dots in paths (outside quoted strings)
        if re.search(r"[a-zA-Z_0-9]\s+\.", command):
            errors.append(
                "Space before '.' not allowed - use 'document.sections' not 'document . sections'"
            )
        if re.search(r"\.\s+[a-zA-Z_0-9]", command):
            errors.append(
                "Space after '.' not allowed - use 'document.sections' not 'document. sections'"
            )

        # Check for space around @ symbol (but not the legitimate space after !)
        # This regex matches: space followed by @ followed by (each|all|resampled)
        # But excludes commands that start with ! followed by spaces
        if re.search(r"\s@(?=each|all|resampled)", command) and not re.match(
            r"^!\s+@(?:each|all|resampled)", command
        ):
            errors.append("Space before '@' not allowed - use '@each' not ' @each'")
        if re.search(r"@\s+(?=each|all|resampled)", command):
            errors.append("Space after '@' not allowed - use '@each' not '@ each'")

        # Check for space before [
        if re.search(r"(?:@each|@all|@resampled)\s+\[", command):
            errors.append(
                "Space before '[' not allowed - use '@each[items]' not '@each [items]'"
            )

        # Check for space after ]
        if re.search(r"\]\s+(-|>)", command):
            errors.append(
                "Space after ']' not allowed - use '[items]->' not '[items] ->'"
            )

        # Check for space around ->
        if re.search(r"\s->", command):
            errors.append("Space before '->' not allowed - use ']->' not '] ->'")
        if re.search(r"->\s+", command):
            errors.append("Space after '->' not allowed - use '->task' not '-> task'")

        # Check for space before {{
        if re.search(r"\s@?\{\{", command):
            errors.append(
                "Space before '{{' not allowed - use 'task@{{' not 'task @{{'"
            )

        # Check for space after }} when followed by *
        if re.search(r"\}\}\s+\*", command):
            errors.append("Space after '}}' not allowed - use '}}*' not '}} *'")

        if errors:
            raise CommandParseError(
                f"Invalid command syntax - strict whitespace violations: {'; '.join(errors)}"
            )

    def parse(
        self, command: str
    ) -> "VariableAssignmentCommand | ExecutionCommand | ResamplingCommand | NodeModifierCommand | CommentCommand | ParsedCommand":
        """
        Parse a command string into the appropriate command object.

        Params:
            command: The command string to parse

        Returns:
            Appropriate command object based on command type

        Raises:
            CommandParseError: If the command is malformed or invalid
        """
        if not command.strip():
            raise CommandParseError("Empty command")

        # Must start with ! (no leading whitespace allowed)
        if not command.startswith("!"):
            raise CommandParseError("Command must start with '!'")

        # Validate forbidden character content BEFORE normalization
        self._validate_forbidden_characters(command)

        # Normalize multiline commands first
        command = self._normalize_multiline_command(command)

        # Validate strict whitespace rules
        self._validate_strict_whitespace(command)

        # Remove trailing whitespace only
        command = command.rstrip()

        # Check for specific patterns and provide detailed error messages

        # Try to parse as execution command first (to avoid conflicts with variable assignment validation)
        exec_match = self.EXECUTION_PATTERN.match(command)
        if exec_match:
            return self._parse_execution_command(exec_match)

        # Try to parse as standalone comment (early to avoid conflicts)
        comment_match = self.COMMENT_PATTERN.match(command)
        if comment_match:
            return self._parse_comment_command(comment_match)

        # Try to parse as variable assignment
        var_match = self.VARIABLE_ASSIGNMENT_PATTERN.match(command)
        if var_match:
            return self._parse_variable_assignment(var_match)

        # Check if it looks like a variable assignment but with invalid syntax
        # Variable assignments should not contain @ or -> which are used in traditional commands
        # Also exclude patterns that look like execution commands (contain parentheses)
        if (
            re.match(r"^!\s*[^=@()]*=", command)
            and "@" not in command
            and "->" not in command
            and "(" not in command
            and ")" not in command
        ):
            # Extract variable name for validation
            var_name_match = re.match(r"^!\s*([^=\s]*(?:\s+[^=\s]*)*)\s*=", command)
            if var_name_match:
                var_name = var_name_match.group(1)
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                    raise CommandParseError(f"Invalid variable name: {var_name}")
                # If name is valid but pattern didn't match, it's likely a value issue
                value_match = re.match(r"^!\s*[^=]*=\s*(.*?)(?:\s*#.*)?$", command)
                if value_match:
                    value = value_match.group(1).strip()
                    if not value:
                        raise CommandParseError("Empty value in assignment")
                    # Check if it looks like an invalid unquoted value
                    if not (
                        value.startswith('"')
                        or value.startswith("'")
                        or value in ["true", "false"]
                        or re.match(r"^-?\d+(?:\.\d+)?$", value)
                    ):
                        raise CommandParseError(
                            f"Unquoted value must be a number or boolean, got: {value}"
                        )

        # Check if it looks like an execution command but with invalid syntax
        if re.match(r"^!\s*[^()]*\(", command):
            # Extract command name for validation (everything before first parenthesis)
            cmd_name_match = re.match(r"^!\s*([^()]*?)\s*\(", command)
            if cmd_name_match:
                cmd_name = cmd_name_match.group(1).strip()
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", cmd_name):
                    raise CommandParseError(f"Invalid command name: {cmd_name}")

        # Try to parse as resampling command
        resample_match = self.RESAMPLING_PATTERN.match(command)
        if resample_match:
            return self._parse_resampling_command(resample_match)

        # Try to parse as node modifier command
        node_match = self.NODE_MODIFIER_PATTERN.match(command)
        if node_match:
            return self._parse_node_modifier_command(node_match)

        # Check if it looks like a resampling command but with invalid syntax
        if re.match(r"^!\s*@resampled\s*\[", command):
            # Extract field name for validation (everything between brackets)
            field_match = re.match(r"^!\s*@resampled\s*\[\s*([^\]]*?)\s*\]", command)
            if field_match:
                field_name = field_match.group(1).strip()
                if not field_name:
                    raise CommandParseError("Empty field name in resampling command")
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", field_name):
                    raise CommandParseError(f"Invalid field name: {field_name}")
            # Check for function validation
            func_match = re.match(
                r"^!\s*@resampled\s*\[\s*[^\]]*\s*\]\s*->\s*(\S*)(?:\s*#.*)?$", command
            )
            if func_match:
                func_name = func_match.group(1)
                if not func_name:
                    raise CommandParseError(
                        "Empty aggregation function in resampling command"
                    )
                valid_functions = ResamplingCommand.VALID_FUNCTIONS
                if func_name not in valid_functions:
                    raise CommandParseError(
                        f"Invalid aggregation function: {func_name}"
                    )

        # Try to parse as traditional @each/@all command
        traditional_match = self.COMMAND_PATTERN.match(command)
        if traditional_match:
            return self._parse_traditional_command(traditional_match)

        # If no pattern matches, provide helpful error
        raise CommandParseError(
            f"Invalid command syntax - malformed command: {command}"
        )

    def _parse_variable_assignment(self, match) -> VariableAssignmentCommand:
        """Parse variable assignment command."""
        variable_name = match.group("variable_name")
        value_str = match.group("value").strip()
        comment = match.group("comment")

        # Parse the value (string, number, or boolean)
        parsed_value = self._parse_value(value_str)

        return VariableAssignmentCommand(
            variable_name=variable_name,
            value=parsed_value,
            comment=comment.strip() if comment else None,
        )

    def _parse_execution_command(self, match) -> ExecutionCommand:
        """Parse execution command."""
        command_name = match.group("command_name")
        arguments_str = match.group("arguments").strip()
        comment = match.group("comment")

        # Parse arguments (both positional and named)
        positional_args, named_args = (
            self._parse_execution_arguments(arguments_str)
            if arguments_str
            else ([], {})
        )

        return ExecutionCommand(
            command_name=command_name,
            arguments=positional_args,
            named_arguments=named_args if named_args else None,
            comment=comment.strip() if comment else None,
        )

    def _parse_resampling_command(self, match) -> ResamplingCommand:
        """Parse resampling aggregation command."""
        field_name = match.group("field_name")
        function_name = match.group("function")
        comment = match.group("comment")

        return ResamplingCommand(
            field_name=field_name,
            aggregation_function=function_name,
            comment=comment.strip() if comment else None,
        )

    def _parse_node_modifier_command(self, match) -> NodeModifierCommand:
        """Parse node modifier command."""
        modifier = match.group("modifier")
        comment = match.group("comment")

        return NodeModifierCommand.from_string(
            modifier, comment.strip() if comment else None
        )

    def _parse_comment_command(self, match) -> CommentCommand:
        """Parse standalone comment command."""
        comment = match.group("comment")
        return CommentCommand(comment=comment.strip() if comment else "")

    def _parse_traditional_command(self, match) -> ParsedCommand:
        """Parse traditional @each/@all command."""
        groups = match.groupdict()

        # Parse command type
        command_type_str = groups.get("command_type")
        if command_type_str == "each":
            command_type = CommandType.EACH
        elif command_type_str == "all" or command_type_str is None:
            command_type = CommandType.ALL
        else:
            raise CommandParseError(f"Invalid command type: {command_type_str}")

        # Parse inclusion path
        inclusion_path = groups.get("inclusion")
        if inclusion_path is not None:
            inclusion_path = inclusion_path.strip()
            if command_type == CommandType.ALL and inclusion_path:
                raise CommandParseError("@all commands cannot have inclusion brackets")
            if command_type == CommandType.EACH and inclusion_path == "":
                raise CommandParseError("@each inclusion brackets cannot be empty")
            if inclusion_path:
                self._validate_path(inclusion_path, "inclusion path")

        # Parse destination
        destination_path = groups.get("destination", "").strip()
        if not destination_path:
            raise CommandParseError("Destination path is required")
        self._validate_path(destination_path, "destination path")

        # Parse variable mappings
        mappings_str = groups.get("mappings", "").strip()
        if not mappings_str:
            raise CommandParseError("Variable mappings cannot be empty")

        variable_mappings = self._parse_variable_mappings(mappings_str)

        # Validate variable mapping paths
        for mapping in variable_mappings:
            if mapping.target_path != "*":  # Skip wildcard validation
                self._validate_path(mapping.target_path, "variable target path")
            if mapping.source_path != "*":  # Skip wildcard validation
                self._validate_path(mapping.source_path, "variable source path")

        # Check for multiplicity indicator
        has_multiplicity = groups.get("multiplicity") == "*"

        # For @each commands, multiplicity is required
        if command_type == CommandType.EACH and not has_multiplicity:
            raise CommandParseError("@each commands require '*' multiplicity indicator")

        # Validate iteration matching for @each commands
        if command_type == CommandType.EACH and inclusion_path:
            self._validate_iteration_matching(inclusion_path, variable_mappings)

        # Check for wildcard assignment
        is_wildcard_assignment = any(
            mapping.source_path == "*" for mapping in variable_mappings
        )

        # Validate wildcard usage
        if is_wildcard_assignment:
            if command_type == CommandType.EACH:
                raise CommandParseError(
                    "@each commands cannot use wildcard (*) in variable mappings"
                )

            # Check for mixed wildcard/non-wildcard usage (not allowed)
            if len(variable_mappings) > 1:
                wildcard_count = sum(
                    1 for mapping in variable_mappings if mapping.source_path == "*"
                )
                non_wildcard_count = len(variable_mappings) - wildcard_count

                if wildcard_count > 0 and non_wildcard_count > 0:
                    raise CommandParseError(
                        "Cannot mix wildcard (*) with non-wildcard mappings in the same command"
                    )

                # Multiple wildcard mappings are allowed (e.g., value.item=*, value.result=*)

        # Parse comment
        comment = groups.get("comment")

        return ParsedCommand(
            command_type=command_type,
            destination_path=destination_path,
            variable_mappings=variable_mappings,
            inclusion_path=inclusion_path,
            has_multiplicity=has_multiplicity,
            is_wildcard_assignment=is_wildcard_assignment,
            comment=comment.strip() if comment else None,
        )

    def _parse_value(self, value_str: str) -> str | int | float | bool:
        """
        Parse a value string into appropriate type (string, number, or boolean).

        Params:
            value_str: The value string to parse

        Returns:
            Parsed value as string, int, float, or bool

        Raises:
            CommandParseError: If unquoted value is not a valid number or boolean
        """
        if not value_str:
            raise CommandParseError("Empty value in assignment")

        # Check if it's a quoted string
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            # It's a quoted string - remove quotes and handle escape sequences
            quote_char = value_str[0]
            unquoted = value_str[1:-1]

            # Handle escape sequences
            if quote_char == '"':
                # Replace escape sequences for double quotes
                unquoted = (
                    unquoted.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
            else:
                # Replace escape sequences for single quotes
                unquoted = (
                    unquoted.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\'", "'")
                    .replace("\\\\", "\\")
                )

            return unquoted

        # Not quoted - check for boolean literals first
        if value_str == "true":
            return True
        elif value_str == "false":
            return False

        # Not boolean - must be a number
        try:
            # Try int first
            if "." not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            raise CommandParseError(
                f"Unquoted value must be a number or boolean, got: {value_str}"
            )

    def _parse_execution_argument(self, value_str: str) -> str | int | float | bool:
        """
        Parse an execution command argument - allows variable names as strings.

        Params:
            value_str: The argument value string to parse

        Returns:
            Parsed value as string, int, float, or bool

        Raises:
            CommandParseError: If the argument is empty
        """
        if not value_str:
            raise CommandParseError("Empty argument in command")

        # Check if it's a quoted string
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            # It's a quoted string - remove quotes and handle escape sequences
            quote_char = value_str[0]
            unquoted = value_str[1:-1]

            # Handle escape sequences
            if quote_char == '"':
                # Replace escape sequences for double quotes
                unquoted = (
                    unquoted.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
            else:
                # Replace escape sequences for single quotes
                unquoted = (
                    unquoted.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\'", "'")
                    .replace("\\\\", "\\")
                )

            return unquoted

        # Not quoted - check for boolean literals first
        if value_str == "true":
            return True
        elif value_str == "false":
            return False

        # Try to parse as number
        try:
            # Try int first
            if "." not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            # Not a number - treat as variable name (string)
            return value_str

    def _parse_arguments(self, args_str: str) -> list[str | int | float | bool]:
        """
        Parse command arguments string into list of values.

        Params:
            args_str: Comma-separated arguments string

        Returns:
            List of parsed argument values

        Raises:
            CommandParseError: If argument parsing fails
        """
        if not args_str.strip():
            return []

        # Split by comma and parse each argument
        arg_parts = [part.strip() for part in args_str.split(",")]
        arguments = []

        for part in arg_parts:
            if not part:
                raise CommandParseError("Empty argument in command")

            # Parse execution arguments (allows variable names)
            arguments.append(self._parse_execution_argument(part))

        return arguments

    def _parse_execution_arguments(
        self, args_str: str
    ) -> tuple[list[str | int | float | bool], dict[str, str | int | float | bool]]:
        """
        Parse execution command arguments into positional and named arguments.

        Params:
            args_str: Comma-separated arguments string (may contain key=value pairs)

        Returns:
            Tuple of (positional_args, named_args)

        Raises:
            CommandParseError: If argument parsing fails
        """
        if not args_str.strip():
            return [], {}

        # Split by comma and parse each argument
        arg_parts = [part.strip() for part in args_str.split(",")]
        positional_args = []
        named_args = {}

        for part in arg_parts:
            if not part:
                raise CommandParseError("Empty argument in command")

            # Check if this is a named argument (contains =)
            if "=" in part and not (part.startswith('"') or part.startswith("'")):
                # Named argument: key=value
                key, value_str = part.split("=", 1)
                key = key.strip()
                value_str = value_str.strip()

                if not key:
                    raise CommandParseError("Empty parameter name in named argument")
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                    raise CommandParseError(f"Invalid parameter name: {key}")
                if not value_str:
                    raise CommandParseError(f"Empty value for parameter: {key}")

                # Parse the value
                value = self._parse_execution_argument(value_str)
                named_args[key] = value
            else:
                # Positional argument
                value = self._parse_execution_argument(part)
                positional_args.append(value)

        return positional_args, named_args

    def _parse_variable_mappings(self, mappings_str: str) -> list[VariableMapping]:
        """Parse variable mappings from the mappings string."""
        if not mappings_str:
            return []

        mappings = []

        # Handle multiline mappings with comments
        # First, process line by line to strip comments and filter comment-only lines
        lines = mappings_str.split("\n")
        cleaned_parts = []

        for line in lines:
            # Strip comments from the line
            if "#" in line:
                # Find the first # that's not inside quotes
                in_quote = False
                quote_char = None
                for i, char in enumerate(line):
                    if not in_quote and char in ('"', "'"):
                        in_quote = True
                        quote_char = char
                    elif in_quote and char == quote_char:
                        in_quote = False
                        quote_char = None
                    elif not in_quote and char == "#":
                        line = line[:i]
                        break

            line = line.strip()
            if line:  # Skip empty lines and comment-only lines
                cleaned_parts.append(line)

        # Join the cleaned lines and split by comma
        cleaned_mappings_str = " ".join(cleaned_parts)
        mapping_parts = [part.strip() for part in cleaned_mappings_str.split(",")]

        for part in mapping_parts:
            if not part:
                continue

            # Try explicit assignment pattern first
            explicit_match = self.VARIABLE_MAPPING_PATTERN.match(part)
            if explicit_match:
                target = explicit_match.group("target").strip()
                source = explicit_match.group("source").strip()

                if not target:
                    raise CommandParseError(
                        "Invalid command syntax - empty variable name in mapping"
                    )
                if source == "":  # Empty source should fail
                    raise CommandParseError(
                        "Invalid command syntax - empty source path in mapping"
                    )

                # Don't validate unknown scope modifiers - they are allowed

                mappings.append(VariableMapping(target_path=target, source_path=source))
                continue

            # Try implicit assignment pattern
            implicit_match = self.IMPLICIT_MAPPING_PATTERN.match(part)
            if implicit_match:
                target = implicit_match.group("target").strip()

                if not target:
                    raise CommandParseError(
                        "Invalid command syntax - empty variable name in mapping"
                    )

                # For implicit mapping, source equals the field name (last part of target)
                if "." in target:
                    source = target.split(".")[-1]
                else:
                    source = target

                # Don't validate unknown scope modifiers - they are allowed

                mappings.append(VariableMapping(target_path=target, source_path=source))
                continue

            raise CommandParseError(
                f"Invalid command syntax - malformed variable mapping: {part}"
            )

        return mappings

    def calculate_iterable_depth(self, node, path_components: list[str]) -> int:
        """
        Calculate iterable depth for a path by counting iterable traversals.

        Traverses the path through TreeNode hierarchy and counts how many
        list fields are encountered. Non-iterable fields don't affect depth count.

        Params:
            node: Starting TreeNode instance for path traversal
            path_components: List of field names forming the path

        Returns:
            Integer count of iterable fields traversed in the path

        Raises:
            CommandParseError: If path cannot be traversed due to missing fields
        """
        from typing import get_origin, get_type_hints

        current_node_type = type(node)
        iterable_count = 0

        for component in path_components:
            # Get type hints for current node
            try:
                hints = get_type_hints(current_node_type)
            except (NameError, AttributeError):
                raise CommandParseError(
                    f"Cannot analyze type hints for {current_node_type.__name__}"
                )

            if component not in hints:
                raise CommandParseError(
                    f"Field '{component}' does not exist in {current_node_type.__name__}"
                )

            field_type = hints[component]
            origin = get_origin(field_type)

            # Validate field type specifications
            from langtree.parsing.validation import validate_field_types

            validate_field_types(component, field_type)

            # Check if field is iterable (properly typed collections)
            if origin in (list, dict, set, tuple):
                iterable_count += 1

                # Get element type for next traversal
                from typing import get_args

                args = get_args(field_type)
                if args:
                    element_type = args[0]
                    # Only continue if element is TreeNode
                    from langtree.structure.builder import TreeNode

                    try:
                        if hasattr(element_type, "__mro__") and issubclass(
                            element_type, TreeNode
                        ):
                            current_node_type = element_type
                        else:
                            # List of primitives - can't traverse further
                            break
                    except TypeError:
                        # Element type is not a class - can't traverse further
                        break
                else:
                    # No type args - can't traverse further
                    break

            elif hasattr(field_type, "__mro__"):
                # Check if it's a TreeNode (non-iterable)
                from langtree.structure.builder import TreeNode

                try:
                    if issubclass(field_type, TreeNode):
                        current_node_type = field_type
                    else:
                        # Regular object - can't traverse further with structure
                        break
                except TypeError:
                    # Not a class - can't traverse further
                    break
            else:
                # Primitive type - can't traverse further
                break

        return iterable_count

    def _is_proper_subchain(
        self, inclusion_path: list[str], rhs_path: list[str]
    ) -> bool:
        """
        Check if rhs_path is a proper subchain of inclusion_path by tracking actual node chains.

        This validates that:
        1. Both paths describe valid traversal chains in the node structure
        2. RHS path either extends inclusion_path OR is a prefix of inclusion_path
        3. Each tag exists and is reachable from the previous tag

        The actual node chain validation happens in the calling method via field existence checks.
        This method focuses on logical subchain relationships.

        Params:
            inclusion_path: The inclusion path components
            rhs_path: The RHS path components to check

        Returns:
            True if rhs_path is a valid subchain relationship, False otherwise
        """
        # Case 1: RHS is longer or equal - must start with inclusion_path
        if len(rhs_path) >= len(inclusion_path):
            # Check if RHS starts with inclusion_path (extends the chain)
            for i in range(len(inclusion_path)):
                if rhs_path[i] != inclusion_path[i]:
                    return False
            return True

        # Case 2: RHS is shorter - must be a prefix of inclusion_path (parent access)
        else:
            # Check if inclusion_path starts with rhs_path (accesses parent in chain)
            for i in range(len(rhs_path)):
                if inclusion_path[i] != rhs_path[i]:
                    return False
            return True

    def _validate_node_chain_traversal(self, node, path: list[str]) -> dict:
        """
        Validate that a path describes a valid node chain traversal.

        This method ensures that each tag in the path exists and is reachable
        from the previous tag in the actual node structure, tracking the complete chain.

        Params:
            node: Starting TreeNode
            path: List of field names to traverse

        Returns:
            Dict with {'valid': bool, 'error': str} indicating chain validity
        """
        if not path:
            return {"valid": False, "error": "Empty path"}

        current_node = node
        traversed_chain = []

        for i, field_name in enumerate(path):
            traversed_chain.append(field_name)
            current_path = ".".join(traversed_chain)

            # Check if field exists on current node
            if not hasattr(current_node, field_name):
                return {
                    "valid": False,
                    "error": f"Field '{field_name}' does not exist on node at path '{'.'.join(traversed_chain[:-1]) or 'root'}'",
                }

            # Get field type information
            field_info = current_node.__annotations__.get(field_name)
            if field_info is None:
                # Try to get from class definition or field defaults
                try:
                    field_value = getattr(current_node, field_name)
                    field_info = type(field_value)
                except AttributeError:
                    return {
                        "valid": False,
                        "error": f"Cannot determine type for field '{field_name}' at path '{current_path}'",
                    }

            # If this is not the last element, verify we can traverse deeper
            if i < len(path) - 1:
                # Extract the element type if this is a list/iterable
                if hasattr(field_info, "__origin__") and field_info.__origin__ is list:
                    # Get list element type
                    element_type = field_info.__args__[0]
                    if hasattr(element_type, "__annotations__"):
                        # Create instance of element type for further traversal
                        current_node = element_type()
                    else:
                        return {
                            "valid": False,
                            "error": f"Field '{field_name}' at path '{current_path}' contains primitive type '{element_type}', cannot traverse further to '{path[i + 1]}'",
                        }
                elif hasattr(field_info, "__annotations__"):
                    # Direct node type, create instance
                    current_node = field_info()
                else:
                    return {
                        "valid": False,
                        "error": f"Field '{field_name}' at path '{current_path}' is primitive type '{field_info}', cannot traverse further to '{path[i + 1]}'",
                    }

        return {"valid": True, "error": ""}

    def validate_last_matching_iterable(
        self, node, inclusion_path: list[str], rhs_paths: list[list[str]]
    ) -> dict:
        """
        Validate subchain relationships using last-matching-iterable algorithm.

        Enhanced validation that checks:
        1. Field existence (paths refer to real structures)
        2. Last matching iterable identification
        3. Complete coverage requirement (at least one RHS reaches inclusion's last iterable)
        4. Depth constraint enforcement (no RHS exceeds inclusion iterable depth)

        Params:
            node: Starting TreeNode for path traversal
            inclusion_path: Components of the @each inclusion path
            rhs_paths: List of RHS path components from variable mappings

        Returns:
            Dict with validation result: {'valid': bool, 'error': str, 'has_complete_coverage': bool}

        Raises:
            CommandParseError: If path traversal fails due to structural issues
        """
        # Empty paths should never occur - they violate LangTree DSL syntax
        # These would be caught during command parsing, but add defensive check
        if not inclusion_path:
            raise CommandParseError(
                "Empty inclusion path - violates LangTree DSL syntax"
            )
        if not rhs_paths:
            raise CommandParseError("Empty RHS paths - violates LangTree DSL syntax")

        # Step 1: Calculate iterable depths and validate field existence
        inclusion_depth = self.calculate_iterable_depth(node, inclusion_path)

        # Validate @each semantics: inclusion path must have at least one iterable
        if inclusion_depth == 0:
            raise CommandParseError(
                "@each requires at least one iterable in inclusion path"
            )

        try:
            rhs_depths = []

            for rhs_path in rhs_paths:
                try:
                    rhs_depth = self.calculate_iterable_depth(node, rhs_path)
                    rhs_depths.append(rhs_depth)
                except CommandParseError as e:
                    return {
                        "valid": False,
                        "error": f"RHS path validation failed: {str(e)}",
                        "has_complete_coverage": False,
                    }

            # Step 2: Check for complete coverage - at least one RHS must match inclusion depth
            has_complete_coverage = any(
                rhs_depth >= inclusion_depth for rhs_depth in rhs_depths
            )

            if not has_complete_coverage:
                return {
                    "valid": False,
                    "error": f"Source path validation failed: no complete coverage. Source paths must be subchains of inclusion path (iteration root). Inclusion depth {inclusion_depth}, max RHS depth {max(rhs_depths) if rhs_depths else 0}",
                    "has_complete_coverage": False,
                }

            # Step 3: Check post-inclusion constraint - RHS cannot have iterables after inclusion path ends
            # This enforces that once inclusion defines the iteration context, RHS can only access
            # non-iterable fields beyond that point
            for i, rhs_path in enumerate(rhs_paths):
                rhs_depth = rhs_depths[i]

                # If RHS depth exceeds inclusion depth, verify no additional iterables exist
                if rhs_depth > inclusion_depth:
                    return {
                        "valid": False,
                        "error": f"Post-inclusion constraint violated: RHS path {i} has iterable depth {rhs_depth} beyond inclusion depth {inclusion_depth}. RHS cannot contain iterables after inclusion path ends.",
                        "has_complete_coverage": has_complete_coverage,
                    }

            # Step 4: Validate proper subchain relationships for same-tree paths
            # This checks that RHS paths don't branch improperly from inclusion path
            subchain_errors = self._validate_proper_subchains(
                node, inclusion_path, rhs_paths
            )
            if subchain_errors:
                return {
                    "valid": False,
                    "error": f"Subchain validation failed: {subchain_errors[0]}",
                    "has_complete_coverage": has_complete_coverage,
                }

            return {"valid": True, "error": "", "has_complete_coverage": True}

        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "has_complete_coverage": False,
            }

    def _validate_proper_subchains(
        self, node, inclusion_path: list[str], rhs_paths: list[list[str]]
    ) -> list[str]:
        """
        Validate that RHS paths meet iterable field sharing requirements and are proper subchains.

        Core requirements:
        1. Inclusion path must start and end with iterable fields
        2. RHS paths must share at least 1 iterable field with inclusion path
        3. After finding common prefix, RHS paths must be proper subchains (not siblings)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate inclusion path starts and ends with iterables
        try:
            inclusion_iterable_count = self.calculate_iterable_depth(
                node, inclusion_path
            )
            if inclusion_iterable_count == 0:
                errors.append(
                    f"Inclusion path {'.'.join(inclusion_path)} must contain at least one iterable field"
                )
                return errors  # Can't proceed with further validation
        except Exception as e:
            errors.append(
                f"Cannot validate inclusion path {'.'.join(inclusion_path)}: {str(e)}"
            )
            return errors

        # Check if inclusion path starts with iterable
        try:
            first_field_iterable = (
                self.calculate_iterable_depth(node, inclusion_path[:1]) > 0
            )
            if not first_field_iterable:
                errors.append(
                    f"Inclusion path {'.'.join(inclusion_path)} must start with an iterable field"
                )
        except Exception:
            pass  # Field validation error will be caught elsewhere

        # Check if inclusion path ends with iterable
        try:
            full_path_count = self.calculate_iterable_depth(node, inclusion_path)
            if len(inclusion_path) > 1:
                parent_count = self.calculate_iterable_depth(node, inclusion_path[:-1])
                last_field_adds_iterable = full_path_count > parent_count
                if not last_field_adds_iterable:
                    errors.append(
                        f"Inclusion path {'.'.join(inclusion_path)} must end with an iterable field"
                    )
        except Exception:
            pass  # Field validation error will be caught elsewhere

        # Validate each RHS path
        for i, rhs_path in enumerate(rhs_paths):
            try:
                # Find common prefix to determine shared iterable count
                common_prefix_len = 0
                for j in range(min(len(inclusion_path), len(rhs_path))):
                    if inclusion_path[j] == rhs_path[j]:
                        common_prefix_len += 1
                    else:
                        break

                # Calculate shared iterable fields
                if common_prefix_len > 0:
                    shared_iterable_count = self.calculate_iterable_depth(
                        node, inclusion_path[:common_prefix_len]
                    )
                else:
                    shared_iterable_count = 0

                # RHS must share at least 1 iterable field with inclusion path
                if shared_iterable_count == 0:
                    errors.append(
                        f"Source path {'.'.join(rhs_path)} must start from iteration root '{'.'.join(inclusion_path)}' - RHS paths must share at least 1 iterable field with inclusion path"
                    )
                    continue  # Skip subchain validation for this path

            except Exception:
                errors.append(f"Cannot validate RHS path {'.'.join(rhs_path)}")
                continue

            # Validate subchain relationship after iterable sharing is confirmed
            # Get remaining paths after common prefix
            inclusion_remaining = inclusion_path[common_prefix_len:]
            rhs_remaining = rhs_path[common_prefix_len:]

            # Validate true subchain relationship
            if len(rhs_remaining) == 0:
                # RHS stops at common prefix (accessing parent level) - VALID
                continue
            elif len(inclusion_remaining) == 0:
                # Inclusion exhausted, RHS extends deeper - VALID
                continue
            else:
                # Both have remaining components - RHS must start with inclusion_remaining
                is_valid_subchain = True

                # Check if RHS remaining starts with inclusion remaining
                for k in range(len(inclusion_remaining)):
                    if (
                        k >= len(rhs_remaining)
                        or rhs_remaining[k] != inclusion_remaining[k]
                    ):
                        is_valid_subchain = False
                        break

                if not is_valid_subchain:
                    # Check if this is a valid sibling pattern (accessing different field of same parent)
                    # vs invalid iterable mismatch (accessing completely different iterables)

                    if len(inclusion_remaining) == 1 and len(rhs_remaining) == 1:
                        # Single-level divergence - could be valid sibling access
                        # E.g., sentences vs title (both fields of same paragraphs)

                        # Check if the divergent RHS component is non-iterable (valid sibling field access)
                        try:
                            # Get the parent path for context
                            parent_path = rhs_path[:common_prefix_len]
                            divergent_path = parent_path + [rhs_remaining[0]]
                            divergent_depth = self.calculate_iterable_depth(
                                node, divergent_path
                            )
                            parent_depth = self.calculate_iterable_depth(
                                node, parent_path
                            )

                            # If RHS divergent field doesn't add iterables, it's valid sibling access
                            if divergent_depth == parent_depth:
                                continue  # Valid sibling access (non-iterable field)
                            else:
                                # RHS divergent field is iterable - this is invalid iterable mismatch
                                errors.append(
                                    f"Iterable path mismatch: source path {'.'.join(rhs_path)} accesses different iterable '{rhs_remaining[0]}' than inclusion path '{'.'.join(inclusion_path)}' which requires '{inclusion_remaining[0]}'"
                                )
                                continue
                        except Exception:
                            # If we can't determine field types, fall back to strict validation
                            pass

                    # Multi-level divergence or other cases - apply strict subchain validation
                    divergent_component = (
                        rhs_remaining[0] if rhs_remaining else "<empty>"
                    )
                    expected_component = (
                        inclusion_remaining[0] if inclusion_remaining else "<empty>"
                    )
                    errors.append(
                        f"Subchain mismatch: source path {'.'.join(rhs_path)} diverges from inclusion path {'.'.join(inclusion_path)} - expected '{expected_component}' but found '{divergent_component}' (paths must be proper subchains, not siblings)"
                    )

        return errors


def parse_command(
    command: str,
) -> "VariableAssignmentCommand | ExecutionCommand | ResamplingCommand | NodeModifierCommand | CommentCommand | ParsedCommand":
    """
    Convenience function to parse a command string.

    Params:
        command: The command string to parse

    Returns:
        Appropriate command object based on command type

    Raises:
        CommandParseError: If the command is malformed or invalid
    """
    parser = CommandParser()
    return parser.parse(command)
