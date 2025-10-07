"""
Prompt element resolution system.

This module handles the resolution of template variables using the structured
PromptElement list approach. Each node maintains a list of PromptElements,
and template variables are resolved bottom-up through the tree.
"""

from typing import TYPE_CHECKING, Any

from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)
from langtree.templates.variables import field_name_to_title

if TYPE_CHECKING:
    from langtree.structure.builder import StructureTreeNode
    from langtree.structure.format_descriptions import OutputFormatDescriptions


def resolve_prompt_subtree_elements(
    node: "StructureTreeNode",
    base_heading_level: int = 1,
) -> list[PromptElement]:
    """
    Generate PromptElement list for {PROMPT_SUBTREE} template variable.

    Creates structured prompt elements for all fields in the node's type.
    For nested TreeNode fields, includes a PromptTemplate placeholder that
    will be replaced by the child's resolved content during bottom-up resolution.

    Args:
        node: Structure tree node to generate subtree for
        base_heading_level: Base heading level for field titles

    Returns:
        List of PromptElements representing the subtree structure
    """
    if not node or not hasattr(node, "field_type") or not node.field_type:
        return []

    elements = []

    # Process each field in the node's type
    if hasattr(node.field_type, "model_fields"):
        for field_name, field_def in node.field_type.model_fields.items():
            # Check if this field is a nested TreeNode
            is_nested = False
            try:
                from langtree.core.tree_node import TreeNode

                field_type = field_def.annotation
                if isinstance(field_type, type) and issubclass(field_type, TreeNode):
                    is_nested = True
            except (TypeError, AttributeError):
                pass

            # Add field title
            title_text = field_name_to_title(field_name, base_heading_level)
            # Remove the markdown heading prefix to get clean title
            clean_title = title_text.lstrip("#").strip()
            elements.append(PromptTitle(content=clean_title, level=base_heading_level))

            # Add field description if available
            if (
                hasattr(node, "clean_field_descriptions")
                and field_name in node.clean_field_descriptions
            ):
                description = node.clean_field_descriptions[field_name]
                if description:
                    elements.append(
                        PromptText(content=description, level=base_heading_level)
                    )
            elif field_def.description:
                elements.append(
                    PromptText(content=field_def.description, level=base_heading_level)
                )

            # For nested TreeNode fields, add a PROMPT_SUBTREE placeholder
            # This will be replaced with the child's resolved content
            if is_nested:
                elements.append(
                    PromptTemplate(
                        variable_name="PROMPT_SUBTREE",
                        level=base_heading_level + 1,  # Child content is subordinate
                        resolved_content=None,  # Will be filled during resolution
                    )
                )

    return elements


def _add_field_tag_to_elements(
    elements: list[PromptElement],
    field_name: str,
    field_path: str,
    base_heading_level: int,
    field_type_class: type | None = None,
) -> None:
    """
    Helper to add field tag references to the elements list.

    For TreeNode fields, recursively generates tags for all leaf fields.
    For leaf fields (str, int, float, bool, etc.), generates a single FULL tag.

    IMPORTANT: Tags are FULL paths like {task.parent.child.leaf}, not partial.

    Args:
        elements: List to append formatted elements to
        field_name: Name of the field being added
        field_path: Full path for the tag (e.g., "task.parent.child")
        base_heading_level: Heading level for the field title
        field_type_class: The type/class of this field for recursive TreeNode detection
    """
    import typing

    from langtree.core.tree_node import TreeNode

    # Build field title
    title_text = field_name_to_title(field_name, base_heading_level)
    clean_title = title_text.lstrip("#").strip()

    elements.append(PromptTitle(content=clean_title, level=base_heading_level))

    # Check if this is a TreeNode field that needs recursive tag generation
    is_treenode_field = False
    nested_type = None

    if field_type_class:
        try:
            # Handle forward references and resolve them
            actual_type = field_type_class

            # If it's a ForwardRef or string, try to get the actual type
            if isinstance(field_type_class, str | typing.ForwardRef):
                # Can't resolve forward refs here without more context
                # But Pydantic should have already resolved them in model_fields
                actual_type = None
            elif hasattr(typing, "get_origin") and typing.get_origin(field_type_class):
                # Handle Optional, Union, etc.
                actual_type = field_type_class

            if (
                actual_type
                and isinstance(actual_type, type)
                and issubclass(actual_type, TreeNode)
            ):
                is_treenode_field = True
                nested_type = actual_type
        except (TypeError, AttributeError):
            pass

    if is_treenode_field and nested_type and hasattr(nested_type, "model_fields"):
        # For TreeNode fields, recursively add tags for ALL nested fields
        for nested_field_name, nested_field_def in nested_type.model_fields.items():
            nested_path = f"{field_path}.{nested_field_name}"
            nested_field_type = nested_field_def.annotation

            # Recursively process (handles TreeNode within TreeNode)
            _add_field_tag_to_elements(
                elements,
                nested_field_name,
                nested_path,
                base_heading_level + 1,
                nested_field_type,
            )
    else:
        # For leaf fields, add the FULL tag reference
        tag = f"{{{field_path}}}"
        elements.append(PromptText(content=tag, level=base_heading_level))


def resolve_collected_context_elements(
    node: "StructureTreeNode",
    previous_values: dict[str, Any] | None = None,
    base_heading_level: int = 1,
) -> list[PromptElement]:
    """
    Generate PromptElement list for {COLLECTED_CONTEXT} template variable.

    Creates structured prompt elements from previously generated field values.
    Handles two cases:
    1. For child nodes: Shows sibling field values that were completed before this field
    2. For root nodes: Shows previous values of the node's own fields from prior execution

    These are already completed values, so they do NOT get any output field prefix.

    Args:
        node: Structure tree node to resolve context for
        previous_values: Dictionary of already generated field values
        base_heading_level: Base heading level for context sections

    Returns:
        List of PromptElements representing the collected context
    """
    if not previous_values:
        return []

    if not node:
        return []

    elements = []

    # Determine if this is a task-level node or a field within a task
    # Task-level nodes: direct children of "task" namespace, no field_name
    # Field nodes: have a field_name, represent individual fields within a task
    is_task_level = (
        not hasattr(node, "field_name") or getattr(node, "field_name", None) is None
    )

    # Get the node's name/tag for building full paths
    node_tag = (
        getattr(node, "name", None) or getattr(node, "tag", None) or "task.unknown"
    )

    # Case 1: Field node - show sibling field tag references
    if not is_task_level and hasattr(node, "parent") and node.parent:
        parent = node.parent
        if hasattr(parent, "children") and parent.children:
            current_field = getattr(node, "field_name", None)

            # Use get_type_hints to resolve forward references for parent type
            parent_type_hints = {}
            if hasattr(parent, "field_type"):
                try:
                    from typing import get_type_hints

                    parent_type_hints = get_type_hints(parent.field_type)
                except Exception:
                    pass

            # Iterate through siblings in definition order
            for field_name, field_node in parent.children.items():
                # Skip current field
                if field_name == current_field:
                    continue

                # Only include fields that are in previous_values (already generated)
                if field_name in previous_values:
                    # Build full tag path
                    full_path = f"{node_tag}.{field_name}"

                    # Get resolved field type (handles forward references)
                    field_type = parent_type_hints.get(field_name)
                    if (
                        not field_type
                        and hasattr(parent, "field_type")
                        and hasattr(parent.field_type, "model_fields")
                    ):
                        if field_name in parent.field_type.model_fields:
                            field_type = parent.field_type.model_fields[
                                field_name
                            ].annotation

                    _add_field_tag_to_elements(
                        elements, field_name, full_path, base_heading_level, field_type
                    )

    # Case 2: Task-level node - show tag references for the task's own fields
    else:
        # Get the node's field definitions to iterate in order
        if hasattr(node, "field_type") and hasattr(node.field_type, "model_fields"):
            # Use get_type_hints to resolve forward references
            try:
                from typing import get_type_hints

                type_hints = get_type_hints(node.field_type)
            except Exception:
                type_hints = {}

            for field_name in node.field_type.model_fields.keys():
                if field_name in previous_values:
                    # Build full tag path
                    full_path = f"{node_tag}.{field_name}"

                    # Get resolved field type (handles forward references)
                    field_type = type_hints.get(field_name)
                    if not field_type:
                        field_def = node.field_type.model_fields.get(field_name)
                        field_type = field_def.annotation if field_def else None

                    _add_field_tag_to_elements(
                        elements, field_name, full_path, base_heading_level, field_type
                    )
        else:
            # Fallback: iterate previous_values in insertion order
            for field_name in previous_values.keys():
                full_path = f"{node_tag}.{field_name}"
                _add_field_tag_to_elements(
                    elements, field_name, full_path, base_heading_level, None
                )

    return elements


def parse_docstring_to_elements(
    content: str, base_level: int = 1
) -> list[PromptElement]:
    """
    Parse a docstring into a list of PromptElements.

    Converts raw docstring text into structured elements, identifying
    template variables and converting them to PromptTemplate placeholders.

    Template variable levels are detected based on the most recent heading
    in the docstring before the template variable.

    Args:
        content: Raw docstring content
        base_level: Base heading level for the content

    Returns:
        List of PromptElements parsed from the docstring
    """
    import re

    from langtree.templates.variables import (
        COLLECTED_CONTEXT_PATTERN,
        PROMPT_SUBTREE_PATTERN,
    )

    if not content:
        return []

    elements = []
    lines = content.split("\n")
    current_text = []
    current_content_level = (
        base_level  # Track current level for content and template variables
    )

    for line in lines:
        # Check if this line is a heading (allow leading whitespace)
        heading_match = re.match(r"^\s*(#+)\s+(.+)$", line)
        if heading_match:
            heading_level = len(heading_match.group(1))
            # Everything after this heading should be one level below it
            current_content_level = heading_level + 1

        # Check for template variables
        if PROMPT_SUBTREE_PATTERN.search(line):
            # Add any accumulated text first
            if current_text:
                text_content = "\n".join(current_text).strip()
                if text_content:
                    elements.append(
                        PromptText(content=text_content, level=current_content_level)
                    )
                current_text = []

            # Add PROMPT_SUBTREE template at current content level
            elements.append(
                PromptTemplate(
                    variable_name="PROMPT_SUBTREE",
                    level=current_content_level,
                    resolved_content=None,
                )
            )

        elif COLLECTED_CONTEXT_PATTERN.search(line):
            # Add any accumulated text first
            if current_text:
                text_content = "\n".join(current_text).strip()
                if text_content:
                    elements.append(
                        PromptText(content=text_content, level=current_content_level)
                    )
                current_text = []

            # Add COLLECTED_CONTEXT template at current content level
            elements.append(
                PromptTemplate(
                    variable_name="COLLECTED_CONTEXT",
                    level=current_content_level,
                    resolved_content=None,
                )
            )

        else:
            # Check if line is a heading (allow leading whitespace)
            heading_match = re.match(r"^\s*(#+)\s+(.+)$", line)
            if heading_match:
                # Add any accumulated text first
                if current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:
                        elements.append(
                            PromptText(content=text_content, level=base_level)
                        )
                    current_text = []

                # Add heading
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                elements.append(PromptTitle(content=title, level=level))
            else:
                # Accumulate regular text
                current_text.append(line)

    # Add any remaining text
    if current_text:
        text_content = "\n".join(current_text).strip()
        if text_content:
            elements.append(PromptText(content=text_content, level=base_level))

    return elements


def is_output_field(node: "StructureTreeNode", field_name: str) -> bool:
    """
    Check if a field is marked as an output field via outputs.* mapping.

    Args:
        node: Structure tree node containing command information
        field_name: Name of the field to check

    Returns:
        True if field is mapped with outputs.* scope
    """
    if not hasattr(node, "extracted_commands") or not node.extracted_commands:
        return False

    # Check all commands for outputs.* mappings to this field
    for command in node.extracted_commands:
        if hasattr(command, "variable_mapping") and command.variable_mapping:
            for mapping in command.variable_mapping.mappings:
                # Check if target path starts with "outputs." and matches field
                if hasattr(mapping, "target_path") and mapping.target_path:
                    target = mapping.target_path
                    if (
                        target.startswith("outputs.")
                        and target.split(".")[-1] == field_name
                    ):
                        return True

    return False


def generate_prompt_subtree_with_children(
    node: "StructureTreeNode",
    base_heading_level: int,
    child_resolutions: dict[str, list[PromptElement]] | None = None,
    output_field_prefix: str | None = "To generate: ",
    format_descriptions: "OutputFormatDescriptions | None" = None,
    previous_values: dict[str, Any] | None = None,
) -> list[PromptElement]:
    """
    Generate PROMPT_SUBTREE elements including resolved child content.

    Creates the complete subtree structure with field titles, descriptions,
    and for nested TreeNode fields, inserts the already-resolved child content.
    Fields marked as outputs via outputs.* mappings get optional prefix.

    IMPORTANT: Only shows the current leaf field being generated, not sibling fields.
    The current field is the first field NOT in previous_values.

    Args:
        node: Structure tree node
        base_heading_level: Base heading level for field titles
        child_resolutions: Resolved elements from child nodes
        output_field_prefix: Optional prefix for output field titles (e.g., "To generate: ").
                           Set to None to disable. Defaults to "To generate: "
        format_descriptions: Optional configuration for output format descriptions
        previous_values: Dictionary of already generated field values (used to determine current field)

    Returns:
        List of PromptElements for the complete subtree
    """
    if not node or not hasattr(node, "field_type") or not node.field_type:
        return []

    elements = []
    child_resolutions = child_resolutions or {}
    previous_values = previous_values or {}

    # Determine the current field being generated (first field NOT in previous_values)
    current_field = None
    if hasattr(node.field_type, "model_fields"):
        for field_name in node.field_type.model_fields.keys():
            if field_name not in previous_values:
                current_field = field_name
                break

    # If all fields are in previous_values, show nothing (all fields already generated)
    if current_field is None:
        return []

    # Process ONLY the current field
    if hasattr(node.field_type, "model_fields"):
        field_name = current_field
        field_def = node.field_type.model_fields[field_name]

        # Check if this field is a nested TreeNode
        is_nested = False
        try:
            from langtree.core.tree_node import TreeNode

            field_type = field_def.annotation
            if isinstance(field_type, type) and issubclass(field_type, TreeNode):
                is_nested = True
        except (TypeError, AttributeError):
            pass

        # Build field title with optional output prefix
        title_text = field_name_to_title(field_name, base_heading_level)
        clean_title = title_text.lstrip("#").strip()

        # Add prefix if this is an output field and prefix is enabled
        if output_field_prefix and is_output_field(node, field_name):
            clean_title = f"{output_field_prefix}{clean_title}"

        elements.append(PromptTitle(content=clean_title, level=base_heading_level))

        # Add field description if available
        if (
            hasattr(node, "clean_field_descriptions")
            and field_name in node.clean_field_descriptions
        ):
            description = node.clean_field_descriptions[field_name]
            if description:
                elements.append(
                    PromptText(content=description, level=base_heading_level)
                )
        elif field_def.description:
            elements.append(
                PromptText(content=field_def.description, level=base_heading_level)
            )

        # Add output format instruction for leaf fields (non-nested)
        if (
            not is_nested
            and format_descriptions
            and format_descriptions.include_format_sections
        ):
            from langtree.templates.prompt_assembly import get_field_output_type

            output_type, enum_values = get_field_output_type(
                field_def, node, field_name
            )

            # Add "Output Format" subheading
            elements.append(
                PromptTitle(content="Output Format", level=base_heading_level + 1)
            )

            # Get format instruction from configuration
            format_instruction = format_descriptions.get_description(
                output_type, enum_values
            )
            elements.append(
                PromptText(content=format_instruction, level=base_heading_level + 1)
            )

        # For nested TreeNode fields, insert the resolved child content
        if is_nested and field_name in child_resolutions:
            # Insert the already-resolved child elements
            child_elements = child_resolutions[field_name]
            # Adjust their level to be subordinate
            adjusted = adjust_element_levels(child_elements, base_heading_level + 1)
            elements.extend(adjusted)

    return elements


def resolve_template_elements(
    elements: list[PromptElement],
    node: "StructureTreeNode",
    previous_values: dict[str, Any] | None = None,
    child_resolutions: dict[str, list[PromptElement]] | None = None,
    output_field_prefix: str | None = "To generate: ",
    format_descriptions: "OutputFormatDescriptions | None" = None,
) -> list[PromptElement]:
    """
    Resolve template placeholders in a list of PromptElements.

    Replaces PromptTemplate placeholders with their resolved content:
    - PROMPT_SUBTREE: Replaced with child node's resolved elements
    - COLLECTED_CONTEXT: Replaced with sibling field values

    Args:
        elements: List of PromptElements potentially containing templates
        node: Current structure tree node
        previous_values: Dictionary of sibling field values for COLLECTED_CONTEXT
        child_resolutions: Dictionary mapping child field names to their resolved elements
        output_field_prefix: Optional prefix for output field titles
        format_descriptions: Optional configuration for output format descriptions

    Returns:
        New list with template placeholders replaced by resolved content
    """
    if not elements:
        return []

    resolved = []
    child_resolutions = child_resolutions or {}

    for element in elements:
        if isinstance(element, PromptTemplate):
            if element.variable_name == "PROMPT_SUBTREE":
                # Generate the subtree structure for this node's fields
                # This includes field titles, descriptions, and nested content
                # Only shows the current leaf field (first NOT in previous_values)
                subtree_elements = generate_prompt_subtree_with_children(
                    node,
                    element.level,
                    child_resolutions,
                    output_field_prefix,
                    format_descriptions,
                    previous_values,
                )

                # If no subtree was generated but we have child resolutions,
                # just insert them with adjusted levels (e.g., in test scenarios)
                if not subtree_elements and child_resolutions:
                    for child_elements in child_resolutions.values():
                        adjusted = adjust_element_levels(child_elements, element.level)
                        resolved.extend(adjusted)
                else:
                    resolved.extend(subtree_elements)

            elif element.variable_name == "COLLECTED_CONTEXT":
                # Replace with collected context elements (already generated, no prefix)
                context_elements = resolve_collected_context_elements(
                    node, previous_values, element.level
                )
                resolved.extend(context_elements)
        else:
            # Keep non-template elements as-is
            resolved.append(element)

    return resolved


def adjust_element_levels(
    elements: list[PromptElement], base_level: int
) -> list[PromptElement]:
    """
    Adjust heading levels of all elements to be subordinate to a base level.

    Used when inserting child content under a parent section to maintain
    proper hierarchy. Preserves relative level differences within the content.

    Args:
        elements: List of elements to adjust
        base_level: New base level for the top-level elements

    Returns:
        New list with adjusted heading levels, preserving hierarchy
    """
    from dataclasses import replace

    if not elements:
        return []

    # Find the minimum level in the elements (the "top" level)
    min_level = min(
        (
            elem.level
            for elem in elements
            if isinstance(elem, PromptTitle | PromptTemplate)
        ),
        default=1,
    )

    # Calculate the shift needed to align min_level to base_level
    level_shift = base_level - min_level

    adjusted = []
    for element in elements:
        # Shift all levels by the same amount to preserve hierarchy
        new_level = element.level + level_shift
        adjusted_element = replace(element, level=new_level)
        adjusted.append(adjusted_element)

    return adjusted


def elements_to_markdown(elements: list[PromptElement]) -> str:
    """
    Convert a list of PromptElements to markdown string.

    This is the final step that converts the fully resolved element list
    into the actual prompt string.

    Args:
        elements: List of resolved PromptElements

    Returns:
        Markdown formatted string
    """
    if not elements:
        return ""

    parts = []

    for element in elements:
        if isinstance(element, PromptTitle):
            # Generate heading with proper level
            heading_prefix = "#" * element.level
            parts.append(f"{heading_prefix} {element.content}")

        elif isinstance(element, PromptText):
            # Add text content
            if element.content:
                parts.append(element.content)

        elif isinstance(element, PromptTemplate):
            # This shouldn't happen in fully resolved content
            # But if it does, add as placeholder
            parts.append(f"{{{element.variable_name}}}")

    # Join with double newlines for proper markdown spacing
    return "\n\n".join(parts)


def resolve_node_prompt_elements(
    node: "StructureTreeNode",
    previous_values: dict[str, Any] | None = None,
    child_resolutions: dict[str, list[PromptElement]] | None = None,
    format_descriptions: "OutputFormatDescriptions | None" = None,
) -> list[PromptElement]:
    """
    Resolve all prompt elements for a node.

    This is the main entry point for node resolution. It:
    1. Parses the node's docstring to elements (or uses cached elements)
    2. Resolves any template placeholders
    3. Returns the fully resolved element list

    Args:
        node: Structure tree node to resolve
        previous_values: Sibling field values for COLLECTED_CONTEXT
        child_resolutions: Already resolved elements from child nodes
        format_descriptions: Optional configuration for output format descriptions

    Returns:
        Fully resolved list of PromptElements for this node
    """
    # Get elements - use cached if available, otherwise parse docstring
    if hasattr(node, "clean_docstring_elements") and node.clean_docstring_elements:
        # Use cached parsed elements (copy to avoid mutation)
        elements = list(node.clean_docstring_elements)
    else:
        # Parse docstring on-the-fly
        docstring = getattr(node, "clean_docstring", "")
        elements = parse_docstring_to_elements(docstring)

    # Get output_field_prefix from node if available
    output_field_prefix = getattr(node, "output_field_prefix", "To generate: ")

    # Resolve template placeholders
    resolved = resolve_template_elements(
        elements,
        node,
        previous_values,
        child_resolutions,
        output_field_prefix=output_field_prefix,
        format_descriptions=format_descriptions,
    )

    return resolved
