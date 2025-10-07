"""
Structured prompt representation data classes.

This module defines the core data structures for representing prompts as
structured lists of elements (titles, text, templates) that can be parsed
from markdown and serialized back to markdown with proper level management
and optional content handling.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptElement(ABC):
    """
    Base class for all prompt elements.

    All prompt elements share common attributes for hierarchical structure
    (level), optional content management (optional flag), and line tracking
    for error reporting.

    Params:
        level: Heading level (1 for #, 2 for ##, etc.)
        optional: Whether this element is optional (excluded when parent renders children)
        line_number: Line number in original markdown (0-indexed, None if not from parsing)
    """

    level: int = 1
    optional: bool = False
    line_number: int | None = None


@dataclass
class PromptTitle(PromptElement):
    """
    Markdown heading element.

    Represents a heading in the prompt structure. Headings organize content
    into hierarchical sections and their level determines the markdown heading
    syntax (# for level 1, ## for level 2, etc.).

    Params:
        content: The heading text
        level: Heading level (inherited from base)
        optional: Optional flag (inherited from base)
    """

    content: str = ""


@dataclass
class PromptText(PromptElement):
    """
    Text content element.

    Represents paragraph text or content in the prompt. Text elements inherit
    their level from the most recent title element, maintaining proper
    hierarchical structure.

    Params:
        content: The text content
        level: Inherited level from last title (from base)
        optional: Optional flag (inherited from base)
    """

    content: str = ""


@dataclass
class PromptTemplate(PromptElement):
    """
    Template variable placeholder element.

    Represents template variables (PROMPT_SUBTREE, COLLECTED_CONTEXT) that
    are replaced during prompt assembly. Can optionally store resolved content
    for recursive template replacement.

    Params:
        variable_name: Template variable name (PROMPT_SUBTREE or COLLECTED_CONTEXT)
        level: Level for alignment of inserted content (from base)
        optional: Optional flag (inherited from base)
        resolved_content: Optional list of elements that replace this template
    """

    variable_name: Literal["PROMPT_SUBTREE", "COLLECTED_CONTEXT"] = "PROMPT_SUBTREE"
    resolved_content: list[PromptElement] | None = None
