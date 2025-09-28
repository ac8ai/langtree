"""
Core type definitions for LangTree DSL framework.

This module contains fundamental type aliases and type definitions used
throughout the LangTree DSL framework for type safety and consistency.
"""

from typing import Any

PromptValue = str | int | float | bool | list | dict | None

ConfigDict = dict[str, Any]

ProcessingResult = tuple[bool, str | None, dict[str, Any]]

ResolutionResult = dict[str, Any]

# NOTE: ParsedCommandUnion will be defined in builder.py to avoid circular imports
