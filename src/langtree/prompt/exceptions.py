"""
Exception classes for LangTree DSL prompt processing.

This module defines specific exception types for different error conditions
that can occur during LangTree DSL parsing, validation, and execution planning.
"""


class LangTreeDSLError(Exception):
    """Base exception for all LangTree DSL-related errors."""

    pass


class NodeInstantiationError(LangTreeDSLError):
    """Raised when a node cannot be instantiated from its field type."""

    def __init__(self, node_tag: str, reason: str):
        """
        Initialize the exception.

        Params:
            node_tag: The tag of the node that failed to instantiate
            reason: The underlying reason for the failure
        """
        self.node_tag = node_tag
        self.reason = reason
        super().__init__(f"Cannot instantiate node '{node_tag}': {reason}")


class FieldTypeError(LangTreeDSLError):
    """Raised when a node has no field type or invalid field type."""

    def __init__(self, node_tag: str, message: str = "has no field type"):
        """
        Initialize the exception.

        Params:
            node_tag: The tag of the node with the field type issue
            message: Specific error message
        """
        self.node_tag = node_tag
        super().__init__(f"Node '{node_tag}' {message}")


class PathValidationError(LangTreeDSLError):
    """Raised when path validation fails."""

    def __init__(self, path: str, reason: str):
        """
        Initialize the exception.

        Params:
            path: The invalid path
            reason: Why the path is invalid
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid path '{path}': {reason}")


class NodeTagValidationError(LangTreeDSLError):
    """Raised when node tag validation fails."""

    def __init__(self, node_tag: str, reason: str):
        """
        Initialize the exception.

        Params:
            node_tag: The invalid node tag
            reason: Why the node tag is invalid
        """
        self.node_tag = node_tag
        self.reason = reason
        super().__init__(f"Invalid node tag '{node_tag}': {reason}")


class RuntimeVariableError(LangTreeDSLError):
    """Raised when runtime variable resolution fails."""

    def __init__(self, message: str):
        """
        Initialize the exception.

        Params:
            message: Error message describing the variable resolution failure
        """
        super().__init__(message)


class FieldValidationError(LangTreeDSLError):
    """Raised when command references non-existent fields in source structure."""

    def __init__(
        self,
        field_path: str,
        container: str,
        message: str = "does not exist",
        command_context: str = None,
    ):
        """
        Initialize the exception.

        Params:
            field_path: The field path that was not found
            container: The structure/class where the field was expected
            message: Specific error message
            command_context: Optional higher-level command context for error chaining
        """
        self.field_path = field_path
        self.container = container
        self.command_context = command_context

        # Build error message with optional context
        primary_error = f"Field '{field_path}' {message} in {container}"
        if command_context:
            full_message = f"{primary_error}\n  Context: {command_context}"
        else:
            full_message = primary_error

        super().__init__(full_message)


class VariableTargetValidationError(LangTreeDSLError):
    """Raised when variable target structure cannot be satisfied."""

    def __init__(self, target_path: str, source_node: str, reason: str):
        """
        Initialize the exception.

        Params:
            target_path: The variable target path that cannot be satisfied
            source_node: The node where this target was declared
            reason: Why the target structure is invalid
        """
        self.target_path = target_path
        self.source_node = source_node
        super().__init__(f"Variable target '{target_path}' in {source_node}: {reason}")


class VariableSourceValidationError(LangTreeDSLError):
    """Raised when variable source field does not exist in referenced structure."""

    def __init__(self, source_path: str, structure_type: str, command_context: str):
        """
        Initialize the exception.

        Params:
            source_path: The source field path that does not exist
            structure_type: The structure type where field was expected
            command_context: Context of the command that references this source
        """
        self.source_path = source_path
        self.structure_type = structure_type
        super().__init__(
            f"Source field '{source_path}' does not exist in {structure_type} (referenced by {command_context})"
        )


class TemplateVariableError(LangTreeDSLError):
    """Base exception for template variable processing errors."""

    def __init__(self, message: str):
        """
        Initialize the exception.

        Params:
            message: Error message describing the template variable failure
        """
        super().__init__(message)


class TemplateVariableSpacingError(TemplateVariableError):
    """Exception for template variable spacing violations."""

    pass


class TemplateVariableConflictError(TemplateVariableError):
    """Exception for template variable conflicts with Assembly Variables."""

    pass


class TemplateVariableNameError(TemplateVariableError):
    """Exception for unknown or invalid template variable names."""

    pass


class ComprehensiveStructuralValidationError(LangTreeDSLError):
    """Raised when multiple structural validation issues are detected."""

    def __init__(self, issues: list[str], node_context: str):
        """
        Initialize the exception.

        Params:
            issues: List of structural validation issues found
            node_context: The node where these issues were detected
        """
        self.issues = issues
        self.node_context = node_context
        issue_summary = "; ".join(issues)
        super().__init__(
            f"Multiple structural issues in {node_context}: {issue_summary}"
        )
