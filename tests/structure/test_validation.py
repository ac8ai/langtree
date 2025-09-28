"""
TDD Tests for Architectural Validation - Define Expected Behavior

These tests define the correct validation behavior BEFORE implementation.
Tests should FAIL initially, then we implement code to make them PASS.

Focus Areas:
1. Bare collection type rejection
2. LangTree DSL structural validation at assembly time
3. Runtime variable enumeration
"""

import pytest
from pydantic import Field

from langtree import TreeNode
from langtree.exceptions import (
    FieldValidationError,
    VariableSourceValidationError,
)
from langtree.parsing.parser import CommandParseError
from langtree.structure import RunStructure


class TestBareCollectionTypeRejection:
    """Test that bare collection types are rejected during validation."""

    def test_bare_list_should_be_rejected(self):
        """Bare list type should be rejected as underspecified."""

        class TaskWithBareList(TreeNode):
            """! @each[items]->task.processor@{{value.item=items}}*"""

            items: list  # ❌ Bare list - should be rejected

        class TaskProcessor(TreeNode):
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

        class TaskWithBareDict(TreeNode):
            """! @each[metadata]->task.processor@{{value.key=metadata}}*"""

            metadata: dict  # ❌ Bare dict - should be rejected

        class TaskProcessor(TreeNode):
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

        class TaskWithBareSet(TreeNode):
            """! @each[tags]->task.processor@{{value.tag=tags}}*"""

            tags: set  # ❌ Bare set - should be rejected

        class TaskProcessor(TreeNode):
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

        class TaskWithTypedCollections(TreeNode):
            """
            Task with properly typed collections.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.processor@{{value.processed_items=items}}*",
            )  # ✅ Properly typed
            metadata: dict[str, int] = {}  # ✅ Properly typed
            tags: set[str] = set()  # ✅ Properly typed

        class TaskProcessor(TreeNode):
            processed_items: list[str]  # ✅ Matching iteration level with source

        structure = RunStructure()

        # Should NOT raise error for properly typed collections
        structure.add(TaskWithTypedCollections)
        structure.add(TaskProcessor)

        # Verify nodes were added successfully
        assert structure.get_node("task.with_typed_collections") is not None
        assert structure.get_node("task.processor") is not None


class TestLangTreeDSLStructuralValidation:
    """Test LangTree DSL structural validation at assembly time."""

    def test_nonexistent_field_in_inclusion_path_should_fail(self):
        """LangTree DSL commands referencing nonexistent fields should fail."""

        class TaskWithBadInclusionPath(TreeNode):
            """! @each[nonexistent_field]->task.processor@{{value.item=items}}*"""

            items: list[str]  # Field exists, but inclusion path wrong

        class TaskProcessor(TreeNode):
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
        """LangTree DSL commands with nonexistent RHS fields should fail."""

        class TaskWithBadRHS(TreeNode):
            """! @each[items]->task.processor@{{value.processed_items=nonexistent_rhs}}*"""

            items: list[str]  # Inclusion path is correct
            # Missing: nonexistent_rhs field

        class TaskProcessor(TreeNode):
            processed_items: list[str]  # ✅ Matching iteration level

        structure = RunStructure()

        # Should fail because 'nonexistent_rhs' doesn't exist
        # Structural validation catches this before field existence validation

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

        class TaskEarly(TreeNode):
            """
            Task with forward reference to later node.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.late@{{value.results=items}}*",
            )

        class TaskLate(TreeNode):  # Defined after being referenced
            results: list[str]  # ✅ Matching iteration level with source

        structure = RunStructure()

        # Should NOT fail - forward references are OK
        structure.add(TaskEarly)  # References task.late before it exists
        structure.add(TaskLate)  # Now task.late exists

        # Both nodes should be added successfully
        assert structure.get_node("task.early") is not None
        assert structure.get_node("task.late") is not None

    def test_valid_structural_references_should_pass(self):
        """Valid LangTree DSL structural references should pass validation."""

        class TaskValid(TreeNode):
            """
            Task with valid LangTree DSL structural references.
            """

            items: list[str] = Field(
                default=[],
                description="! @each[items]->task.processor@{{value.processed_items=items}}*",
            )  # ✅ Inclusion path exists, RHS path is valid subchain

        class TaskProcessor(TreeNode):
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

        class TaskWithVariables(TreeNode):
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

        class TaskNoVariables(TreeNode):
            """Template with no variables"""

            field: str = "static"

        structure = RunStructure()
        structure.add(TaskNoVariables)

        # Should return empty list
        runtime_vars = structure.list_runtime_variables()
        assert runtime_vars == []


# Key semantic validation tests added from test_semantic_validation_specification.py
class TestFieldExistenceValidationRHS:
    """Test field existence validation for variable mapping source fields (RHS).

    Per LANGUAGE_SPECIFICATION.md: RHS fields must exist in current node scope.
    """

    def setup_method(self):
        """Create fixtures for field existence tests."""
        self.structure = RunStructure()

    def test_nonexistent_field_in_variable_mapping_fails(self):
        """Test that nonexistent fields in variable mappings cause immediate validation failure."""

        class Section(TreeNode):
            title: str
            content: str

        class TaskWithNonexistentField(TreeNode):
            """Task referencing nonexistent field in variable mapping."""

            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should fail: sections.nonexistent_field doesn't exist on Section objects
            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.nonexistent_field}}*"
            )
            valid_list: list[
                Section
            ] = []  # Needed to satisfy "at least one must match" rule

        # Should fail immediately during tree building - field validation
        with pytest.raises(VariableSourceValidationError) as exc_info:
            self.structure.add(TaskWithNonexistentField)

        assert "nonexistent_field" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

    def test_existing_field_in_variable_mapping_passes(self):
        """Test that existing fields in variable mappings pass validation."""

        class Section(TreeNode):
            title: str
            content: str

        class TaskWithExistingField(TreeNode):
            """Task referencing existing field in variable mapping."""

            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should pass: sections.title exists on Section objects and we have a matching level
            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.title}}*"
            )
            # Need a list field to satisfy "at least one must match" rule (1 level for @each[sections])
            valid_list: list[Section] = []

        # Should pass validation
        self.structure.add(TaskWithExistingField)
        assert self.structure.get_node("task.with_existing_field") is not None

    def test_invalid_nested_field_access_fails(self):
        """Test that invalid nested field access fails validation."""

        class SubSection(TreeNode):
            title: str
            content: str

        class TaskWithInvalidNested(TreeNode):
            """Task with invalid nested field access."""

            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should fail: sections.nonexistent doesn't exist
            sections: list[SubSection] = Field(
                description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.nonexistent}}*"
            )
            # Need a list field to satisfy "at least one must match" rule (1 level for @each[sections])
            valid_list: list[SubSection] = []

        # Should fail - nested field doesn't exist
        with pytest.raises(VariableSourceValidationError) as exc_info:
            self.structure.add(TaskWithInvalidNested)

        assert "nonexistent" in str(exc_info.value)


# ===== RESTORED FROM BACKUP: validation tests =====


# From test_semantic_validation_specification.py:


# Common task classes referenced by tests
class TaskDocumentProcessor(TreeNode):
    """Generic analyzer task referenced by test commands."""

    pass


class TaskProcessor(TreeNode):
    """Generic processor task referenced by test commands."""

    pass


class TaskProcessorFour(TreeNode):
    """Four-level processor task."""

    pass


class TaskStructureAThreeLevels(TreeNode):
    """Three-level structure task."""

    pass


class TaskProcessorFive(TreeNode):
    """Five-level processor task."""

    pass


class TaskDocumentProcessorSeven(TreeNode):
    """Seven-level analyzer task."""

    pass


class TaskStructureAMinimalSpacing(TreeNode):
    """Minimal spacing structure task."""

    pass


class TaskTarget(TreeNode):
    """Target task for general processing."""

    pass


class TaskStructureA(TreeNode):
    """Structure A task."""

    pass


class TaskStructureBZeroLayers(TreeNode):
    """Structure B zero layers task."""

    pass


class TaskStructureATwoLayers(TreeNode):
    """Structure A two layers task."""

    pass
