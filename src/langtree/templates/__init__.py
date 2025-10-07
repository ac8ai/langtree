"""
LangTree DSL template processing components.

This package provides template variable processing and template utilities
for the LangTree DSL framework.
"""

from langtree.templates.element_resolution import (
    elements_to_markdown,
    parse_docstring_to_elements,
    resolve_collected_context_elements,
    resolve_node_prompt_elements,
    resolve_prompt_subtree_elements,
    resolve_template_elements,
)
from langtree.templates.prompt_assembly import (
    assemble_field_prompt,
    create_field_title,
    ensure_template_variables,
    field_name_to_title_text,
)
from langtree.templates.prompt_operations import (
    adjust_element_levels,
    has_template_variable,
    insert_elements_at_template,
)
from langtree.templates.prompt_parser import (
    parse_prompt_to_list,
    prompt_list_to_string,
)
from langtree.templates.prompt_structure import (
    PromptElement,
    PromptTemplate,
    PromptText,
    PromptTitle,
)
from langtree.templates.utils import extract_commands, get_root_tag
from langtree.templates.variables import (
    add_automatic_prompt_subtree,
    add_automatic_template_variables,
    collect_inherited_docstrings,
    detect_template_variables,
    process_template_variables,
    validate_no_duplicate_template_variables,
    validate_template_variable_conflicts,
    validate_template_variable_names,
    validate_template_variable_spacing,
)

__all__ = [
    # Prompt structure types
    "PromptElement",
    "PromptTitle",
    "PromptText",
    "PromptTemplate",
    # Element resolution
    "parse_docstring_to_elements",
    "resolve_prompt_subtree_elements",
    "resolve_collected_context_elements",
    "resolve_template_elements",
    "resolve_node_prompt_elements",
    "elements_to_markdown",
    # Parsing and serialization
    "parse_prompt_to_list",
    "prompt_list_to_string",
    # Operations
    "adjust_element_levels",
    "has_template_variable",
    "insert_elements_at_template",
    # Prompt assembly
    "ensure_template_variables",
    "field_name_to_title_text",
    "create_field_title",
    "assemble_field_prompt",
    # Template variable processing
    "extract_commands",
    "get_root_tag",
    "add_automatic_prompt_subtree",
    "add_automatic_template_variables",
    "collect_inherited_docstrings",
    "detect_template_variables",
    "process_template_variables",
    "validate_no_duplicate_template_variables",
    "validate_template_variable_conflicts",
    "validate_template_variable_names",
    "validate_template_variable_spacing",
]
