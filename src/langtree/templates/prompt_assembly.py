"""
Prompt assembly with template variables.

This module handles the assembly of prompts using COLLECTED_CONTEXT and
PROMPT_SUBTREE template variables, including automatic addition, field
title generation, and output formatting.
"""

import re
from typing import TYPE_CHECKING, Any

from langtree.exceptions.core import (
    ComprehensiveStructuralValidationError,
    TemplateVariableConflictError,
)
from langtree.templates.prompt_operations import has_template_variable
from langtree.templates.prompt_parser import parse_prompt_to_list
from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from langtree.core.tree_node import TreeNode
    from langtree.templates.variables import ProcessedDocstring


def ensure_template_variables(
    elements: list[PromptElement],
    config: dict[str, Any] | None = None,
) -> list[PromptElement]:
    """
    Ensure template variables exist in the prompt element list.

    Adds COLLECTED_CONTEXT and PROMPT_SUBTREE template variables if they
    don't already exist. Respects user placement if templates already exist.
    Adds templates at the end of existing content with appropriate titles.

    Params:
        elements: List of prompt elements to check
        config: Optional configuration for titles and behavior

    Returns:
        List with template variables ensured (may be unchanged if already present)
    """
    config = config or {}
    result = elements.copy()

    # Check what's missing
    has_context = has_template_variable(result, "COLLECTED_CONTEXT")
    has_subtree = has_template_variable(result, "PROMPT_SUBTREE")

    # If both exist, nothing to do
    if has_context and has_subtree:
        return result

    # Add COLLECTED_CONTEXT if missing
    if not has_context:
        # Add title if list is empty or last element isn't a title
        if not result or not isinstance(result[-1], PromptTitle):
            context_title = config.get("collected_context_title", "Context")
            result.append(PromptTitle(content=context_title, level=1))

        result.append(
            PromptTemplate(
                variable_name="COLLECTED_CONTEXT",
                level=2,
                optional=True,
            )
        )

    # Add PROMPT_SUBTREE if missing
    if not has_subtree:
        # Add title if last element isn't a title
        if not result or not isinstance(result[-1], PromptTitle):
            subtree_title = config.get("prompt_subtree_title", "Task")
            result.append(PromptTitle(content=subtree_title, level=1))

        result.append(
            PromptTemplate(
                variable_name="PROMPT_SUBTREE",
                level=2,
                optional=True,
            )
        )

    return result


def field_name_to_title_text(field_name: str) -> str:
    """
    Convert field name to title text with proper formatting.

    Handles underscore_case, camelCase, and numbers in field names.

    Params:
        field_name: Field name to convert (e.g., 'myField2Name')

    Returns:
        Formatted title text (e.g., 'My Field 2 Name')
    """
    # Insert space before uppercase letters (camelCase)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name)

    # Insert space before numbers
    spaced = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", spaced)

    # Insert space after numbers when followed by letters
    spaced = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", spaced)

    # Replace underscores with spaces
    spaced = spaced.replace("_", " ")

    # Title case each word
    return spaced.title()


def create_field_title(
    field_name: str,
    level: int = 1,
    is_leaf: bool = False,
    title_prefix: str | None = None,
) -> PromptTitle:
    """
    Create a prompt title for a field.

    Converts field name to proper title format and optionally adds
    a prefix for leaf nodes.

    Params:
        field_name: Name of the field
        level: Heading level (1-6+)
        is_leaf: Whether this is a leaf node (affects prefix)
        title_prefix: Optional prefix text (only applied to leaf nodes)

    Returns:
        PromptTitle element with formatted content
    """
    title_text = field_name_to_title_text(field_name)

    # Add prefix only for leaf nodes if provided
    if is_leaf and title_prefix:
        title_text = f"{title_prefix}: {title_text}"

    return PromptTitle(content=title_text, level=level)


def create_markdown_tag_instruction(field_name: str) -> str:
    """
    Create instruction for wrapping markdown output in tags.

    Params:
        field_name: Name of the field being generated

    Returns:
        Instruction text for using langtree-markdown-output tags
    """
    return f"""Wrap your response in the following tags:
<langtree-markdown-output field="{field_name}">
[Your markdown content here]
</langtree-markdown-output>

The content between tags can use full markdown formatting."""


def assemble_field_prompt(
    field_name: str,
    field_description: str | None = None,
    is_leaf: bool = True,
    config: dict[str, Any] | None = None,
) -> list[PromptElement]:
    """
    Assemble complete prompt for a field.

    Creates a structured prompt with field title, description,
    and optional output formatting instructions.

    Params:
        field_name: Name of the field to generate
        field_description: Description/prompt for the field
        is_leaf: Whether this is a leaf node (affects title and formatting)
        config: Configuration for output formatting and customization

    Returns:
        List of prompt elements for the field
    """
    config = config or {}
    elements = []

    # Create field title (with prefix for leaf nodes)
    title = create_field_title(
        field_name,
        level=1,
        is_leaf=is_leaf,
        title_prefix=config.get("title_prefix"),
    )
    elements.append(title)

    # Add field description if provided
    if field_description:
        # Parse description as it might contain markdown
        desc_elements = parse_prompt_to_list(field_description, normalize=False)
        if desc_elements:
            # If description starts with its own title, adjust levels
            if isinstance(desc_elements[0], PromptTitle):
                for elem in desc_elements:
                    if hasattr(elem, "level"):
                        elem.level = elem.level + 1
            elements.extend(desc_elements)
        else:
            # Plain text description
            elements.append(PromptText(content=field_description, level=2))

    # Add PROMPT_SUBTREE for non-leaf nodes
    if not is_leaf:
        # Non-leaf nodes need template for their children
        elements.append(
            PromptTemplate(
                variable_name="PROMPT_SUBTREE",
                level=2,
                optional=True,
            )
        )

    # Add output formatting for leaf nodes (unless disabled)
    if is_leaf and not config.get("skip_formatting"):
        output_format = config.get("output_format")
        use_tags = config.get("use_tags", False)

        # Add output format section
        elements.append(PromptTitle(content="Output Format", level=2))

        # Special handling for markdown with tags
        if output_format == "markdown" and use_tags:
            instruction = create_markdown_tag_instruction(field_name)
            elements.append(PromptText(content=instruction, level=2))
        elif output_format:
            # Standard format instruction
            format_text = get_format_instruction(output_format)
            elements.append(PromptText(content=format_text, level=2))

    return elements


def get_format_instruction(
    output_format: str,
    enum_values: list[str] | None = None,
    include_markdown_desc: bool = True,
) -> str:
    """
    Get standard format instruction for output type.

    Params:
        output_format: Type of output (int, str, bool, enum, etc.)
        enum_values: List of valid enum string values (only for enum type)
        include_markdown_desc: If True, include markdown formatting description (default: True)

    Returns:
        Format instruction text
    """
    format_map = {
        "int": "Return only the integer value.",
        "str": "Return only the text content.",
        "bool": "Return exactly 'true' or 'false' (lowercase).",
        "float": "Return only the numeric value.",
        "json": "Return valid JSON.",
    }

    # Handle enum type with values
    if output_format == "enum":
        if enum_values:
            values_str = ", ".join(f'"{v}"' for v in enum_values)
            return f"Return one of the following values: {values_str}"
        else:
            return "Return one of the defined enum values."

    # Handle markdown with optional detailed description
    if output_format == "markdown":
        if include_markdown_desc:
            return (
                "Return formatted markdown content. "
                "You may use headings, lists, bold, italic, code blocks, and other markdown syntax. "
                "Ensure proper markdown structure with blank lines between sections."
            )
        else:
            return "Return formatted markdown content."

    return format_map.get(output_format, f"Return output as {output_format}.")


def get_field_output_type(
    field_info: "FieldInfo",
    node: Any = None,
    field_name: str | None = None,
) -> tuple[str, list[str] | None]:
    """
    Determine output type and enum values for a field.

    Checks for @output_format command override first, then falls back to
    Pydantic field type inspection.

    Params:
        field_info: Pydantic FieldInfo for the field
        node: Optional StructureTreeNode containing command information
        field_name: Optional field name for command lookup

    Returns:
        Tuple of (output_type, enum_values):
        - output_type: "int", "float", "bool", "str", "enum", "markdown", "json", etc.
        - enum_values: List of enum value strings (only for enum type), None otherwise
    """
    from enum import Enum

    # Check for @output_format command override in node's commands
    if node and field_name and hasattr(node, "extracted_commands"):
        for cmd in node.extracted_commands:
            if hasattr(cmd, "text") and "@output_format" in cmd.text:
                # Extract format value from @output_format("value")
                match = re.match(r'.*@output_format\s*\(\s*"([^"]*)"', cmd.text)
                if match:
                    format_value = match.group(1)
                    # User-specified format, no enum values
                    return (format_value, None)

    # Fall back to Pydantic type inspection
    field_type = field_info.annotation

    # Handle basic types
    if field_type is int:
        return ("int", None)
    elif field_type is float:
        return ("float", None)
    elif field_type is bool:
        return ("bool", None)
    elif field_type is str:
        return ("str", None)
    elif field_type is dict or (
        hasattr(field_type, "__origin__") and field_type.__origin__ is dict
    ):
        return ("json", None)

    # Handle Enum types
    try:
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            # Extract enum values as strings
            enum_values = [member.value for member in field_type]
            return ("enum", enum_values)
    except TypeError:
        pass

    # Default to str for unknown types
    return ("str", None)


def get_output_format(processed: "ProcessedDocstring", validate: bool = False) -> str:
    """
    Extract output format from processed docstring.

    Params:
        processed: Processed docstring containing commands
        validate: Whether to validate the format value

    Returns:
        Output format ("markdown" or "plain")

    Raises:
        ValueError: If format is invalid or empty (when validate=True)
        TemplateVariableConflictError: If multiple output_format commands found
    """
    # Find all output_format commands
    output_formats = []
    for cmd in processed.commands:
        if "@output_format" in cmd.text:
            # Extract the format value from @output_format("value")
            import re

            match = re.match(r'.*@output_format\s*\(\s*"([^"]*)"', cmd.text)
            if match:
                output_formats.append(match.group(1))

    # Validate constraints
    if validate:
        if len(output_formats) > 1:
            raise TemplateVariableConflictError(
                f"Multiple output_format commands found in {processed.source_class}"
            )

        if output_formats:
            format_value = output_formats[0]
            if not format_value:
                raise ValueError("Empty output format value")
            if format_value not in ("markdown", "plain"):
                raise ValueError(f"Unsupported output format: {format_value}")

    # Return the format or default to "plain"
    if output_formats:
        return output_formats[0]
    return "plain"


def validate_output_format_usage(node_class: type["TreeNode"]) -> None:
    """
    Validate that @output_format is only used on str-typed leaf fields.

    Params:
        node_class: TreeNode class to validate

    Raises:
        ComprehensiveStructuralValidationError: If validation fails
    """
    from langtree.templates.variables import process_class_docstring

    # Check if class has output_format command
    docstring = node_class.__doc__ or ""
    processed = process_class_docstring(docstring, node_class.__name__)

    # Get output format if specified
    try:
        output_format = get_output_format(processed, validate=True)
    except (ValueError, TemplateVariableConflictError) as e:
        raise ComprehensiveStructuralValidationError(str(e))

    # If no markdown format, nothing to validate
    if output_format == "plain":
        return

    # Check all fields
    errors = []
    for field_name, field_info in node_class.model_fields.items():
        field_type = field_info.annotation

        # Check if field is a TreeNode subclass (non-leaf)
        try:
            from langtree.core.tree_node import TreeNode

            if isinstance(field_type, type) and issubclass(field_type, TreeNode):
                errors.append(
                    f"Field '{field_name}' is a nested TreeNode "
                    f"(output_format is only valid on leaf nodes)"
                )
                continue
        except TypeError:
            pass  # Not a class type, continue checking

        # Check if field is str type
        if field_type is not str:
            # Handle Optional[str] and similar
            import typing

            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)

            # Check for Optional[str] which is Union[str, None]
            if origin is typing.Union:
                non_none_types = [t for t in args if t is not type(None)]
                if len(non_none_types) == 1 and non_none_types[0] is str:
                    continue  # This is Optional[str], which is valid

            errors.append(
                f"Field '{field_name}' has type {field_type} "
                f"(output_format is only valid on str-typed fields)"
            )

    if errors:
        raise ComprehensiveStructuralValidationError(
            errors, f"{node_class.__name__} with @output_format('markdown')"
        )


def get_output_format_for_field(node_class: type["TreeNode"], field_name: str) -> str:
    """
    Get the output format for a specific field.

    This is used during structured generation to determine how to
    format the field's output.

    Params:
        node_class: The TreeNode class containing the field
        field_name: Name of the field

    Returns:
        Output format for the field ("markdown" or "plain")
    """
    from langtree.templates.variables import process_class_docstring

    # Get class-level output format
    docstring = node_class.__doc__ or ""
    processed = process_class_docstring(docstring, node_class.__name__)
    output_format = get_output_format(processed)

    # Verify field is valid for this format
    if output_format == "markdown":
        field_info = node_class.model_fields.get(field_name)
        if field_info and field_info.annotation is str:
            return "markdown"

    return "plain"


def assemble_field_prompt_with_format(
    field_name: str,
    field_info: "FieldInfo",
    node_class: type["TreeNode"],
    is_leaf: bool = True,
    heading_level: int = 1,
) -> str:
    """
    Assemble a field prompt with output formatting support.

    This is the main entry point for prompt assembly that considers
    the @output_format command.

    Params:
        field_name: Name of the field
        field_info: Pydantic FieldInfo for the field
        node_class: The TreeNode class containing this field
        is_leaf: Whether this field is a leaf node
        heading_level: Markdown heading level to use

    Returns:
        Formatted prompt string
    """
    from langtree.templates.prompt_parser import prompt_list_to_string

    # Get output format for this field
    output_format = get_output_format_for_field(node_class, field_name)

    # Create base elements
    elements = []

    # Add field title
    title = create_field_title(field_name, heading_level, is_leaf)
    elements.append(title)

    # Add field description if present
    if field_info.description:
        elements.append(
            PromptText(content=field_info.description, level=heading_level + 1)
        )

    # Add output format instructions for markdown fields
    if is_leaf and output_format == "markdown":
        elements.append(
            PromptText(content="\n<langtree-output>\n", level=heading_level + 1)
        )
        elements.append(
            PromptTemplate(
                variable_name="FIELD_CONTENT", level=heading_level + 1, optional=False
            )
        )
        elements.append(
            PromptText(content="\n</langtree-output>", level=heading_level + 1)
        )

    # Add PROMPT_SUBTREE for non-leaf fields
    if not is_leaf:
        elements.append(
            PromptTemplate(
                variable_name="PROMPT_SUBTREE", level=heading_level + 1, optional=True
            )
        )

    # Convert to string
    return prompt_list_to_string(elements)


# DEPRECATED STRING-BASED FUNCTIONS REMOVED
# These functions have been replaced with the element-based resolution API:
# - assemble_full_prompt() → use node.get_prompt(previous_values={...})
# - resolve_template_variables_in_content() → use resolve_template_elements()
# - resolve_template_variables_recursively() → use node.get_prompt(previous_values={...})
# See StructureTreeNode.get_prompt() in langtree.structure.builder for production API
