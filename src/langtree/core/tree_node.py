"""
Core TreeNode base class for LangTree DSL framework.

This module contains the fundamental TreeNode class that serves as the base
for all prompt tree node types in the LangTree DSL framework.
"""

from pydantic import BaseModel


class TreeNode(BaseModel):
    """
    Base class for all prompt tree node types.

    This class provides the foundation for creating structured prompt components
    that can be organized in a tree hierarchy for complex LLM interactions.
    """

    pass
