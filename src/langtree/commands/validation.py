"""
Field type validation for DPCL command parsing.

This module contains validation functions for field types, ensuring proper type
specifications and compliance with DPCL framework requirements.
"""

# Group 4: Internal from imports (alphabetical by source module)
from langtree.commands.parser import CommandParseError


def validate_field_types(component: str, field_type) -> None:
    """
    Validate field type specifications for DPCL compatibility.

    Ensures that field types meet DPCL framework requirements including
    proper collection type specifications. Bare collection types (list, dict,
    set, tuple) are rejected as underspecified.

    Params:
        component: Field component name being validated
        field_type: The field type annotation to validate

    Raises:
        CommandParseError: When field type is invalid or underspecified
    """
    # Reject bare collection types - they are underspecified
    if field_type in (list, dict, set, tuple):
        type_name = field_type.__name__
        raise CommandParseError(
            f"Bare collection type '{type_name}' is not allowed in field '{component}'. "
            f"Use properly typed collections like '{type_name}[ElementType]' for type specification."
        )