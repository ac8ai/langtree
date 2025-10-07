"""
Operations on prompt element lists.

This module provides utility functions for manipulating structured prompt
element lists, including level adjustments, template checks, and insertions.
"""

from typing import Literal

from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)


def adjust_element_levels(
    elements: list[PromptElement],
    *,
    align_by: int | None = None,
    align_to: int | None = None,
) -> list[PromptElement]:
    """
    Adjust levels of all elements by offset or to a target minimum level.

    Creates new element instances with adjusted levels. Levels are clamped
    to the valid range [1, 6] for markdown headings.

    Params:
        elements: List of prompt elements to adjust
        align_by: Amount to add to each element's level (can be negative)
        align_to: Target level for the minimum element (adjusts all relatively)

    Returns:
        New list with adjusted element levels

    Raises:
        ValueError: If both or neither align_by and align_to are specified
    """
    if (align_by is None and align_to is None) or (
        align_by is not None and align_to is not None
    ):
        raise ValueError("Exactly one of 'align_by' or 'align_to' must be specified")

    if not elements:
        return []

    # Calculate offset
    if align_by is not None:
        offset = align_by
    else:
        # align_to: adjust so minimum level becomes align_to
        min_level = min(e.level for e in elements)
        offset = align_to - min_level

    result = []
    for elem in elements:
        # Create new instance with adjusted level
        # Only enforce minimum of 1, allow levels > 6 for deep nesting
        new_level = max(1, elem.level + offset)

        if isinstance(elem, PromptTitle):
            result.append(
                PromptTitle(
                    content=elem.content,
                    level=new_level,
                    optional=elem.optional,
                    line_number=elem.line_number,
                )
            )
        elif isinstance(elem, PromptText):
            result.append(
                PromptText(
                    content=elem.content,
                    level=new_level,
                    optional=elem.optional,
                    line_number=elem.line_number,
                )
            )
        elif isinstance(elem, PromptTemplate):
            result.append(
                PromptTemplate(
                    variable_name=elem.variable_name,
                    level=new_level,
                    optional=elem.optional,
                    resolved_content=elem.resolved_content,
                    line_number=elem.line_number,
                )
            )

    return result


def has_template_variable(
    elements: list[PromptElement],
    variable_name: Literal["PROMPT_SUBTREE", "COLLECTED_CONTEXT"],
    *,
    include_optional: bool = True,
) -> bool:
    """
    Check if a template variable exists in the element list.

    Params:
        elements: List of prompt elements to search
        variable_name: Template variable name to look for
        include_optional: If False, only search non-optional elements

    Returns:
        True if template variable is found, False otherwise
    """
    for elem in elements:
        if isinstance(elem, PromptTemplate) and elem.variable_name == variable_name:
            # If we're excluding optional elements, check the flag
            if not include_optional and elem.optional:
                continue
            return True
    return False


def insert_elements_at_template(
    elements: list[PromptElement],
    variable_name: Literal["PROMPT_SUBTREE", "COLLECTED_CONTEXT"],
    to_insert: list[PromptElement],
) -> list[PromptElement]:
    """
    Insert elements at the location of a template variable.

    Replaces the first occurrence of the specified template variable with
    the provided elements. Inserted elements are level-adjusted to match
    the template's level.

    Params:
        elements: List of prompt elements containing template
        variable_name: Template variable to replace
        to_insert: Elements to insert at template location

    Returns:
        New list with template replaced by inserted elements
    """
    # Find template index
    template_idx = next(
        (
            i
            for i, elem in enumerate(elements)
            if isinstance(elem, PromptTemplate) and elem.variable_name == variable_name
        ),
        None,
    )

    # If template not found, return original list
    if template_idx is None:
        return elements

    # Get template level for adjustment
    template_level = elements[template_idx].level

    # Adjust inserted elements to template level
    if to_insert:
        base_level = min(e.level for e in to_insert)
        offset = template_level - base_level
        adjusted = adjust_element_levels(to_insert, align_by=offset)
    else:
        adjusted = []

    # Build result: before + inserted + after
    return elements[:template_idx] + adjusted + elements[template_idx + 1 :]
