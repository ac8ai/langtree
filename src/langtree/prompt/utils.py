"""
Utility functions for LangTree DSL prompt processing.

This module contains utility functions for extracting commands from text,
processing prompts, and handling naming conventions.
"""

import re
from textwrap import dedent
from typing import TYPE_CHECKING

from inflection import underscore

if TYPE_CHECKING:
    from langtree.prompt.structure import TreeNode

# Text processing utilities - matches commands that may span multiple lines within brackets/braces
COMMAND_PATTERN = re.compile(r"^\s*!\s*[^!\n]*(?:\n(?!\s*!)[^\n]*)*", re.MULTILINE)


def extract_commands(content: str | None) -> tuple[list[str], str]:
    """
    Extract commands from text content and return clean prompt content.

    Commands can only appear at the start of the content. As soon as regular text
    (non-command, non-whitespace) is encountered, command parsing stops permanently.

    Params:
        content: Raw text content that may contain command lines (starting with !)

    Returns:
        Tuple of (extracted_commands, clean_content) where commands are stripped
        of leading whitespace and clean_content has commands removed
    """
    if not content:
        return [], ""

    lines = content.split("\n")
    commands = []
    command_indices = set()
    current_command = None
    in_multiline_context = False
    bracket_depth = 0
    parsing_commands = True  # Start in command parsing mode

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not parsing_commands:
            # Once we stop parsing commands, everything goes to content
            continue

        # Check if this line starts a new command
        if stripped.startswith("!"):
            # Finish previous command if any
            if current_command is not None:
                commands.append(current_command.strip())

            # Start new command
            current_command = line
            command_indices.add(i)

            # Check if this command has multiline contexts
            in_multiline_context = any(char in line for char in ["[", "{", "("])
            bracket_depth = (
                line.count("[")
                + line.count("{")
                + line.count("(")
                - line.count("]")
                - line.count("}")
                - line.count(")")
            )

        elif current_command is not None and in_multiline_context and bracket_depth > 0:
            # Continue multiline command if we're in a bracket/brace context
            current_command += "\n" + line
            command_indices.add(i)
            bracket_depth += (
                line.count("[")
                + line.count("{")
                + line.count("(")
                - line.count("]")
                - line.count("}")
                - line.count(")")
            )

            if bracket_depth <= 0:
                in_multiline_context = False

        elif stripped == "":
            # Empty/whitespace line - continue parsing commands
            continue

        else:
            # Regular text found - stop parsing commands permanently
            if current_command is not None:
                commands.append(current_command.strip())
                current_command = None
                in_multiline_context = False
                bracket_depth = 0
            parsing_commands = False

    # Finish last command if any
    if current_command is not None:
        commands.append(current_command.strip())

    # Remove command lines from content
    clean_lines = [line for i, line in enumerate(lines) if i not in command_indices]
    clean_content = "\n".join(clean_lines)

    # Clean up the content - remove empty lines and dedent
    clean_content = dedent(clean_content).strip()

    return commands, clean_content


def get_root_tag(subtree: type["TreeNode"], kind: str = "task") -> str:
    """
    Generate a root tag for a TreeNode type based on naming conventions.

    Validates that the class name follows strict CamelCase and Task prefix conventions:
    - Must be CamelCase (no underscores, starts with capital letter)
    - Must start with 'Task' followed by a capital letter (e.g., TaskEarly, TaskA)
    - Invalid: task_early, taskEarly, Tasks, Taskx, task

    Params:
        subtree: The TreeNode type to generate a tag for
        kind: The expected designation/kind (default: 'task')

    Returns:
        A root tag string in the format 'designation.name'

    Raises:
        ValueError: If the class name doesn't follow naming conventions or expected kind
    """
    class_name = subtree.__name__.split(".")[-1]

    # Validate CamelCase format
    if not _is_valid_camelcase(class_name):
        raise ValueError(
            f"Class name '{class_name}' must be in CamelCase format (no underscores, starts with capital letter)"
        )

    # For task classes, validate and process Task prefix
    if kind == "task":
        is_valid, matched_prefix = _is_valid_prefix(class_name, ["Task"])
        if not is_valid:
            raise ValueError(
                f"Task class name '{class_name}' has invalid prefix - must start with 'Task' followed by a capital letter (e.g., TaskEarly, TaskA)"
            )

        # Extract the part after matched prefix and convert to underscore
        task_suffix = class_name[len(matched_prefix) :]  # Remove prefix
        if task_suffix:
            name = underscore(task_suffix)
        else:
            raise ValueError(
                f"Task class name '{class_name}' must have content after '{matched_prefix}' prefix"
            )

        root_tag = f"task.{name}"
        return root_tag

    # For non-task classes, use the original logic
    name_underscore = underscore(class_name)
    parts = name_underscore.split("_", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Class name '{class_name}' does not follow expected '{kind.title()}Name' pattern"
        )

    designation, name = parts
    if designation != kind:
        raise ValueError(
            f"Expected a {kind} class, got: {class_name} of a kind {designation}."
        )

    root_tag = f"{designation}.{name}"
    return root_tag


def _is_valid_camelcase(name: str) -> bool:
    """
    Check if a name follows CamelCase conventions.

    Valid: TaskEarly, TaskA, Task2Analysis
    Invalid: task_early, taskEarly, Task_early, task
    """
    if not name:
        return False

    # Must start with capital letter
    if not name[0].isupper():
        return False

    # No underscores allowed
    if "_" in name:
        return False

    return True


def _is_valid_prefix(name: str, allowed_prefixes: list[str] = None) -> tuple[bool, str]:
    """
    Check if a class name has valid prefix format from allowed prefixes.

    Valid: TaskEarly, TaskA, Task2Analysis
    Invalid: Tasks, Taskx, task, Task

    Params:
        name: Class name to validate
        allowed_prefixes: List of allowed prefixes (defaults to ['Task'])

    Returns:
        Tuple of (is_valid, matched_prefix)
    """
    if allowed_prefixes is None:
        allowed_prefixes = ["Task"]

    for prefix in allowed_prefixes:
        if name.startswith(prefix):
            # Must have at least one character after prefix
            if len(name) <= len(prefix):
                continue

            # Character after prefix must be uppercase letter or digit
            # This rejects "Tasks" (lowercase 's') and "Taskx" (lowercase 'x')
            next_char = name[len(prefix)]
            if next_char.isupper() or next_char.isdigit():
                return True, prefix

    return False, ""
