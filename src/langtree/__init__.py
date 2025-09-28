"""
LangTree - A tool to orchestrate prompts for structured generation with LangChain

LangTree provides a DSL for building hierarchical prompt structures with action chaining.
"""

from importlib.metadata import version

from langtree.core import TreeNode
from langtree.models import LLMProvider
from langtree.structure import RunStructure

__version__ = version("langtree")

__all__ = [
    "__version__",
    "TreeNode",
    "RunStructure",
    "LLMProvider",
]
