"""
Utility functions for structure registry operations.

This module contains helper functions used by the structure registry
that don't have circular import dependencies.
"""

from typing import TYPE_CHECKING

from langtree.exceptions import (
    FieldTypeError,
    NodeInstantiationError,
    NodeTagValidationError,
    PathValidationError,
)

if TYPE_CHECKING:
    from langtree.core.tree_node import TreeNode
    from langtree.structure.builder import StructureTreeNode


def validate_path_and_node_tag(path: str, node_tag: str) -> None:
    """
    Validate path and node_tag parameters for resolution methods.

    Common validation logic shared across registry methods to ensure
    consistent error handling and parameter validation.

    Params:
        path: The path string to validate for basic format requirements
        node_tag: The node tag string to validate for basic format requirements

    Raises:
        PathValidationError: If path format is invalid
        NodeTagValidationError: If node_tag format is invalid
    """
    if not path or not isinstance(path, str):
        raise PathValidationError(path, "must be a non-empty string")

    if not node_tag or not isinstance(node_tag, str):
        raise NodeTagValidationError(node_tag, "must be a non-empty string")


def get_node_instance(node: "StructureTreeNode", node_tag: str) -> "TreeNode":
    """
    Get an instance of a node's field type for data access.

    Creates an instance of the TreeNode class associated with
    a structure tree node, enabling access to node data and attributes
    for resolution and validation purposes.

    Params:
        node: The structure tree node containing the field_type reference
        node_tag: The node tag used for error reporting and debugging

    Returns:
        An instance of the node's field_type (TreeNode subclass)

    Raises:
        FieldTypeError: When node has no field_type defined
        NodeInstantiationError: When node cannot be instantiated due to initialization errors
    """
    if not node.field_type:
        raise FieldTypeError(node_tag)

    try:
        return node.field_type()
    except Exception as e:
        raise NodeInstantiationError(node_tag, str(e))
