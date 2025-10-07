"""
Prompt parsing and serialization functions.

This module provides functions to convert between markdown text and structured
prompt element lists, enabling manipulation of prompts as structured data.
"""

from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)


def parse_prompt_to_list(
    content: str, *, normalize: bool = True
) -> list[PromptElement]:
    """
    Parse markdown content into structured prompt element list.

    Converts markdown text with headings, paragraphs, and template variables
    into a list of PromptElement objects. Text elements inherit their level
    from the most recent heading.

    By default, normalizes heading levels so the minimum level becomes 1,
    preserving relative structure. This makes prompts easier to compose.

    Params:
        content: Markdown content to parse
        normalize: If True, normalize minimum heading level to 1 (default True)

    Returns:
        List of PromptElement objects with optionally normalized levels
    """
    if not content or not content.strip():
        return []

    elements = []
    current_level = 1  # Default level for text without preceding heading

    # Split by lines first to handle headings properly
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Track line number for this element
        element_line_number = i

        # Check for heading (starts with #)
        if line.startswith("#"):
            # Count number of # to determine level
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break

            # Extract heading text (after # and space)
            heading_text = line[level:].strip()
            elements.append(
                PromptTitle(
                    content=heading_text, level=level, line_number=element_line_number
                )
            )
            current_level = level  # Update current level for subsequent text
            i += 1

        # Check for template variable
        elif line.startswith("{") and line.endswith("}"):
            var_content = line[1:-1]  # Remove { and }
            if var_content in ["PROMPT_SUBTREE", "COLLECTED_CONTEXT"]:
                elements.append(
                    PromptTemplate(
                        variable_name=var_content,
                        level=current_level,
                        line_number=element_line_number,
                    )
                )
            else:
                # Not a template variable, treat as text
                elements.append(
                    PromptText(
                        content=line,
                        level=current_level,
                        line_number=element_line_number,
                    )
                )
            i += 1

        # Regular text - collect until next empty line or heading
        else:
            text_lines = [line]
            i += 1
            # Collect continuation lines until empty line or next element
            while i < len(lines):
                next_line = lines[i].strip()
                if (
                    not next_line
                    or next_line.startswith("#")
                    or (next_line.startswith("{") and next_line.endswith("}"))
                ):
                    break
                text_lines.append(next_line)
                i += 1

            # Join text lines and add as single text element
            text_content = " ".join(text_lines)
            elements.append(
                PromptText(
                    content=text_content,
                    level=current_level,
                    line_number=element_line_number,
                )
            )

    # Normalize levels if requested
    if normalize and elements:
        from langtree.templates.prompt_operations import adjust_element_levels

        elements = adjust_element_levels(elements, align_to=1)

    return elements


def prompt_list_to_string(
    elements: list[PromptElement], include_optional: bool = True
) -> str:
    """
    Serialize prompt element list to markdown string.

    Converts structured prompt elements back to markdown text format.
    Optionally filters out optional elements.

    Params:
        elements: List of prompt elements to serialize
        include_optional: If True, includes optional elements; if False, excludes them

    Returns:
        Markdown formatted string
    """
    if not elements:
        return ""

    # Filter optional elements if needed
    if not include_optional:
        elements = [e for e in elements if not e.optional]

    parts = []

    for elem in elements:
        if isinstance(elem, PromptTitle):
            # Render heading with appropriate number of #
            heading = "#" * elem.level + " " + elem.content
            parts.append(heading)

        elif isinstance(elem, PromptText):
            parts.append(elem.content)

        elif isinstance(elem, PromptTemplate):
            # Render unresolved template variable
            parts.append(f"{{{elem.variable_name}}}")

    # Join with double newlines between blocks
    return "\n\n".join(parts)
