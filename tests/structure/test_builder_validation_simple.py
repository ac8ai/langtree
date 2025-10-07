"""
Focused validation tests for builder.py defensive code.

Tests the untested validation methods by triggering them through
Field-level @each commands which ARE allowed to reference fields.
"""

import pytest
from pydantic import Field

from langtree import TreeNode
from langtree.exceptions import (
    FieldValidationError,
    VariableSourceValidationError,
)
from langtree.structure import RunStructure


class TestBareCollectionValidation:
    """Test _validate_against_bare_collection_types() - lines 2101-2141."""

    def test_reject_bare_list(self):
        """Bare list without type parameters should raise."""

        class TaskBad(TreeNode):
            """Task with bare list."""

            items: list = Field(description="Items")  # Should be list[str]

        structure = RunStructure()

        with pytest.raises(FieldValidationError) as exc:
            structure.add(TaskBad)

        assert "list" in str(exc.value).lower()
        assert "underspecified" in str(exc.value).lower()

    def test_reject_bare_dict(self):
        """Bare dict should raise."""

        class TaskBad(TreeNode):
            """Task with bare dict."""

            data: dict = Field(description="Data")  # Should be dict[str, int]

        structure = RunStructure()

        with pytest.raises(FieldValidationError) as exc:
            structure.add(TaskBad)

        assert "dict" in str(exc.value).lower()

    def test_reject_bare_set(self):
        """Bare set should raise."""

        class TaskBad(TreeNode):
            """Task with bare set."""

            tags: set = Field(description="Tags")  # Should be set[str]

        structure = RunStructure()

        with pytest.raises(FieldValidationError) as exc:
            structure.add(TaskBad)

        assert "set" in str(exc.value).lower()

    def test_accept_typed_list(self):
        """list[str] should be accepted."""

        class TaskGood(TreeNode):
            """Task with properly typed list."""

            items: list[str] = Field(description="Items")

        structure = RunStructure()
        structure.add(TaskGood)

        assert structure.get_node("task.good") is not None

    def test_accept_typed_dict(self):
        """dict[str, int] should be accepted."""

        class TaskGood(TreeNode):
            """Task with dict."""

            data: dict[str, int] = Field(description="Data")

        structure = RunStructure()
        structure.add(TaskGood)

        assert structure.get_node("task.good") is not None

    def test_accept_typed_set(self):
        """set[str] should be accepted."""

        class TaskGood(TreeNode):
            """Task with set."""

            tags: set[str] = Field(description="Tags")

        structure = RunStructure()
        structure.add(TaskGood)

        assert structure.get_node("task.good") is not None


class TestInclusionFieldValidation:
    """Test _validate_inclusion_field() - lines 1592-1651."""

    def test_missing_inclusion_field(self):
        """@each with non-existent field should raise."""

        class TaskDoc(TreeNode):
            """Document processor."""

            content: str = Field(description="Content")
            sections: list[str] = Field(
                description="! @each[nonexistent]->task.analyzer@{{value.text=nonexistent}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()

        with pytest.raises(FieldValidationError) as exc:
            structure.add(TaskDoc)
            structure.add(TaskAnalyzer)

        assert "nonexistent" in str(exc.value)

    def test_valid_inclusion_field(self):
        """@each with valid field should succeed."""

        class Section(TreeNode):
            """Section."""

            title: str = Field(description="Title")

        class TaskDoc(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.title}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()
        structure.add(TaskDoc)
        structure.add(TaskAnalyzer)

        assert structure.get_node("task.doc") is not None


class TestVariableSourceFieldValidation:
    """Test _validate_variable_source_field() - lines 1446-1536."""

    def test_iteration_variable_missing_field(self):
        """@each accessing non-existent field on iteration type should raise."""

        class Section(TreeNode):
            """Section with content only."""

            content: str = Field(description="Content")
            # NO 'title' field!

        class TaskDoc(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.title}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()

        with pytest.raises(VariableSourceValidationError) as exc:
            structure.add(TaskDoc)
            structure.add(TaskAnalyzer)

        assert "title" in str(exc.value)

    def test_iteration_variable_valid_field(self):
        """@each accessing valid field on iteration type should succeed."""

        class Section(TreeNode):
            """Section with title."""

            title: str = Field(description="Title")
            content: str = Field(description="Content")

        class TaskDoc(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.title}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()
        structure.add(TaskDoc)
        structure.add(TaskAnalyzer)

        assert structure.get_node("task.doc") is not None

    def test_nested_iteration_variable_path(self):
        """@each with nested path like sections.metadata.title should validate."""

        class Metadata(TreeNode):
            """Metadata."""

            author: str = Field(description="Author")
            # NO 'title' field!

        class Section(TreeNode):
            """Section."""

            metadata: Metadata = Field(description="Metadata")
            content: str = Field(description="Content")

        class TaskDoc(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.metadata.title}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()

        with pytest.raises(VariableSourceValidationError) as exc:
            structure.add(TaskDoc)
            structure.add(TaskAnalyzer)

        # Should fail - 'title' doesn't exist in Metadata
        assert "title" in str(exc.value)


class TestFieldPathValidation:
    """Test _validate_field_path_exists() helper - lines 1757-1815."""

    def test_path_through_list_unwrapping(self):
        """Path through list[T] should unwrap to T and continue validation."""

        class Section(TreeNode):
            """Section."""

            content: str = Field(description="Content")
            # NO 'title' field!

        class TaskDocument(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.title}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()

        with pytest.raises(VariableSourceValidationError) as exc:
            structure.add(TaskDocument)
            structure.add(TaskAnalyzer)

        # Should unwrap list[Section] and try to find 'title' on Section
        assert "title" in str(exc.value)

    def test_deeply_nested_valid_path(self):
        """Deep nesting should work when all fields exist."""

        class Author(TreeNode):
            """Author."""

            name: str = Field(description="Name")

        class Metadata(TreeNode):
            """Metadata."""

            author: Author = Field(description="Author")

        class Section(TreeNode):
            """Section."""

            metadata: Metadata = Field(description="Metadata")

        class TaskDocument(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.analyzer@{{value.text=sections.metadata.author.name}}*\n\nSections"
            )

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            text: str = Field(description="Text")

        structure = RunStructure()
        structure.add(TaskDocument)
        structure.add(TaskAnalyzer)

        # Should succeed - full path exists
        assert structure.get_node("task.document") is not None


class TestNestedInclusionPath:
    """Test nested inclusion paths in @each commands."""

    def test_nested_inclusion_path_validation(self):
        """Nested path like document.sections should validate each level."""

        class Section(TreeNode):
            """Section."""

            title: str = Field(description="Title")

        class TaskDocument(TreeNode):
            """Document."""

            content: str = Field(description="Content")
            # NO 'sections' field!

        class TaskAnalyzer(TreeNode):
            """Analyzer."""

            container: TaskDocument = Field(description="Container")
            results: list[str] = Field(
                description="! @each[container.sections]->task.processor@{{value.text=container.sections}}*\n\nResults"
            )

        class TaskProcessor(TreeNode):
            """Processor."""

            text: str = Field(description="Text")

        structure = RunStructure()

        with pytest.raises(FieldValidationError) as exc:
            structure.add(TaskAnalyzer)
            structure.add(TaskProcessor)

        # Should fail - 'sections' doesn't exist in Document
        assert "sections" in str(exc.value)

    def test_deeply_nested_inclusion_valid(self):
        """Deep nested inclusion path should work when valid and follows scoping rules."""

        class Section(TreeNode):
            """Section."""

            title: str = Field(description="Title")
            content: str = Field(description="Content")

        class TaskDocument(TreeNode):
            """Document."""

            sections: list[Section] = Field(
                description="! @each[sections]->task.processor@{{value.text=sections.title}}*\n\nSections"
            )

        class TaskProcessor(TreeNode):
            """Processor."""

            text: str = Field(description="Text")

        structure = RunStructure()
        structure.add(TaskDocument)
        structure.add(TaskProcessor)

        # Should succeed - sections field is properly scoped
        assert structure.get_node("task.document") is not None
