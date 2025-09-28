"""
LangTree DSL template processing components.

This package provides template variable processing and template utilities
for the LangTree DSL framework.
"""

from langtree.templates.utils import extract_commands, get_root_tag
from langtree.templates.variables import (
    add_automatic_prompt_subtree,
    detect_template_variables,
    process_template_variables,
    validate_template_variable_conflicts,
    validate_template_variable_names,
    validate_template_variable_spacing,
)

__all__ = [
    "extract_commands",
    "get_root_tag",
    "add_automatic_prompt_subtree",
    "detect_template_variables",
    "process_template_variables",
    "validate_template_variable_conflicts",
    "validate_template_variable_names",
    "validate_template_variable_spacing",
]
