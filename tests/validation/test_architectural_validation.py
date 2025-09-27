"""
TDD Tests for Architectural Validation - Define Expected Behavior

These tests define the correct validation behavior BEFORE implementation.
Tests should FAIL initially, then we implement code to make them PASS.

Focus Areas:
1. Bare collection type rejection
2. DPCL structural validation at assembly time
3. Runtime variable enumeration
"""

import pytest
from pydantic import Field

from langtree.commands.parser import CommandParseError
from langtree.prompt import PromptTreeNode, RunStructure
from langtree.prompt.exceptions import FieldValidationError


class TestBareCollectionTypeRejection:
    """Test that bare collection types are rejected during validation."""

    def test_bare_list_should_be_rejected(self):
        """Bare list type should be rejected as underspecified."""

        class TaskWithBareList(PromptTreeNode):
            """! @each[items]->task.processor@{{value.item=items}}*"""

            items: list  # ❌ Bare list - should be rejected

        class TaskProcessor(PromptTreeNode):
            item: str

        structure = RunStructure()

        # Should raise error for bare list type (may be wrapped in FieldValidationError)
        with pytest.raises((CommandParseError, FieldValidationError)) as exc_info:
            structure.add(TaskWithBareList)

        error_msg = str(exc_info.value).lower()
        assert "bare" in error_msg or "underspecified" in error_msg
        assert "list" in error_msg

    def test_bare_dict_should_be_rejected(self):
        """Bare dict type should be rejected as underspecified."""

        class TaskWithBareDict(PromptTreeNode):
            """! @each[metadata]->task.processor@{{value.key=metadata}}*"""

            metadata: dict  # ❌ Bare dict - should be rejected

        class TaskProcessor(PromptTreeNode):
            key: str

        structure = RunStructure()

        # Should raise error for bare dict type
        with pytest.raises((CommandParseError, FieldValidationError)) as exc_info:
            structure.add(TaskWithBareDict)

        error_msg = str(exc_info.value).lower()
        assert "bare" in error_msg or "underspecified" in error_msg
        assert "dict" in error_msg

    def test_bare_set_should_be_rejected(self):
        """Bare set type should be rejected as underspecified."""

        class TaskWithBareSet(PromptTreeNode):
            """! @each[tags]->task.processor@{{value.tag=tags}}*"""

            tags: set  # ❌ Bare set - should be rejected

        class TaskProcessor(PromptTreeNode):
            tag: str

        structure = RunStructure()

        # Should raise error for bare set type
        with pytest.raises((CommandParseError, FieldValidationError)) as exc_info:
            structure.add(TaskWithBareSet)

        error_msg = str(exc_info.value).lower()
        assert "bare" in error_msg or "underspecified" in error_msg
        assert "set" in error_msg

    def test_properly_typed_collections_should_be_accepted(self):
        """Properly typed collections should be accepted."""

        class TaskWithTypedCollections(PromptTreeNode):
            """
            Task with properly typed collections.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.processor@{{value.processed_items=items}}*",
            )  # ✅ Properly typed
            metadata: dict[str, int] = {}  # ✅ Properly typed
            tags: set[str] = set()  # ✅ Properly typed

        class TaskProcessor(PromptTreeNode):
            processed_items: list[str]  # ✅ Matching iteration level with source

        structure = RunStructure()

        # Should NOT raise error for properly typed collections
        structure.add(TaskWithTypedCollections)
        structure.add(TaskProcessor)

        # Verify nodes were added successfully
        assert structure.get_node("task.with_typed_collections") is not None
        assert structure.get_node("task.processor") is not None


class TestDPCLStructuralValidation:
    """Test DPCL structural validation at assembly time."""

    def test_nonexistent_field_in_inclusion_path_should_fail(self):
        """DPCL commands referencing nonexistent fields should fail."""

        class TaskWithBadInclusionPath(PromptTreeNode):
            """! @each[nonexistent_field]->task.processor@{{value.item=items}}*"""

            items: list[str]  # Field exists, but inclusion path wrong

        class TaskProcessor(PromptTreeNode):
            item: str

        structure = RunStructure()

        # Should fail because 'nonexistent_field' doesn't exist
        with pytest.raises((CommandParseError, FieldValidationError)) as exc_info:
            structure.add(TaskWithBadInclusionPath)

        error_msg = str(exc_info.value).lower()
        assert "nonexistent_field" in error_msg
        # Structural validation catches this before field existence validation
        assert (
            "must start from iteration root" in error_msg
            or "does not exist" in error_msg
            or "not found" in error_msg
        )

    def test_nonexistent_rhs_field_should_fail(self):
        """DPCL commands with nonexistent RHS fields should fail."""

        class TaskWithBadRHS(PromptTreeNode):
            """! @each[items]->task.processor@{{value.processed_items=nonexistent_rhs}}*"""

            items: list[str]  # Inclusion path is correct
            # Missing: nonexistent_rhs field

        class TaskProcessor(PromptTreeNode):
            processed_items: list[str]  # ✅ Matching iteration level

        structure = RunStructure()

        # Should fail because 'nonexistent_rhs' doesn't exist
        # Structural validation catches this before field existence validation
        from langtree.prompt.exceptions import VariableSourceValidationError

        with pytest.raises(
            (CommandParseError, VariableSourceValidationError)
        ) as exc_info:
            structure.add(TaskWithBadRHS)

        error_msg = str(exc_info.value).lower()
        assert "nonexistent_rhs" in error_msg
        # Accept either structural validation error or field existence error
        assert (
            "must start from iteration root" in error_msg
            or "does not exist" in error_msg
        )

    def test_forward_references_should_be_allowed(self):
        """Forward references to future nodes should be allowed."""

        class TaskEarly(PromptTreeNode):
            """
            Task with forward reference to later node.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.late@{{value.results=items}}*",
            )

        class TaskLate(PromptTreeNode):  # Defined after being referenced
            results: list[str]  # ✅ Matching iteration level with source

        structure = RunStructure()

        # Should NOT fail - forward references are OK
        structure.add(TaskEarly)  # References task.late before it exists
        structure.add(TaskLate)  # Now task.late exists

        # Both nodes should be added successfully
        assert structure.get_node("task.early") is not None
        assert structure.get_node("task.late") is not None

    def test_valid_structural_references_should_pass(self):
        """Valid DPCL structural references should pass validation."""

        class TaskValid(PromptTreeNode):
            """
            Task with valid DPCL structural references.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.processor@{{value.processed_items=items}}*",
            )  # ✅ Inclusion path exists, RHS path is valid subchain

        class TaskProcessor(PromptTreeNode):
            processed_items: list[str]  # ✅ Matching iteration level with source

        structure = RunStructure()

        # Should NOT raise any errors
        structure.add(TaskValid)
        structure.add(TaskProcessor)

        # Verify successful validation
        assert structure.get_node("task.valid") is not None
        assert structure.get_node("task.processor") is not None


class TestRuntimeVariableEnumeration:
    """Test simple runtime variable enumeration function."""

    def test_list_all_runtime_variables_needed(self):
        """Should provide simple list of all runtime variables needing values."""

        class TaskWithVariables(PromptTreeNode):
            """Template with variables: {field_var} and {another_var}"""

            field_var: str = "default"
            another_var: int = 42

        structure = RunStructure()
        structure.add(TaskWithVariables)

        # Get all runtime variables that need values at invoke time
        runtime_vars = structure.list_runtime_variables()

        # Should return simple list of expanded variable names
        expected_vars = {
            "prompt__with_variables__field_var",
            "prompt__with_variables__another_var",
        }
        assert set(runtime_vars) == expected_vars

    def test_empty_list_when_no_runtime_variables(self):
        """Should return empty list when no runtime variables exist."""

        class TaskNoVariables(PromptTreeNode):
            """Template with no variables"""

            field: str = "static"

        structure = RunStructure()
        structure.add(TaskNoVariables)

        # Should return empty list
        runtime_vars = structure.list_runtime_variables()
        assert runtime_vars == []
