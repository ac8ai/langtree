"""
LangTree DSL exception classes.

This package provides all exception types used throughout the LangTree DSL
framework for consistent error handling and reporting.
"""

from langtree.exceptions.core import (
    ComprehensiveStructuralValidationError,
    DuplicateTargetError,
    FieldTypeError,
    FieldValidationError,
    LangTreeDSLError,
    NodeInstantiationError,
    NodeTagValidationError,
    PathValidationError,
    RuntimeVariableError,
    TemplateVariableConflictError,
    TemplateVariableError,
    TemplateVariableNameError,
    TemplateVariableSpacingError,
    VariableSourceValidationError,
    VariableTargetValidationError,
)

__all__ = [
    "LangTreeDSLError",
    "FieldTypeError",
    "FieldValidationError",
    "NodeInstantiationError",
    "NodeTagValidationError",
    "PathValidationError",
    "DuplicateTargetError",
    "RuntimeVariableError",
    "VariableSourceValidationError",
    "VariableTargetValidationError",
    "TemplateVariableError",
    "TemplateVariableConflictError",
    "TemplateVariableNameError",
    "TemplateVariableSpacingError",
    "ComprehensiveStructuralValidationError",
]
