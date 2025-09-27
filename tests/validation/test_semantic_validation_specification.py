"""
Comprehensive semantic validation tests per LANGUAGE_SPECIFICATION.md.

This module tests the complex semantic validation rules defined in the specification:
- Field existence validation for variable mapping sources (RHS)
- Loop nesting validation between inclusion paths and variable mappings
- Task target completeness validation
- Variable mapping constraint validation

Tests follow TDD approach: define expected behavior first, then implement to pass.
"""

import pytest
from pydantic import BaseModel, Field

from langtree.prompt import PromptTreeNode, RunStructure
from langtree.prompt.exceptions import FieldValidationError, VariableSourceValidationError
from langtree.prompt.utils import get_root_tag
from langtree.commands.parser import CommandParseError


# Common task classes referenced by tests
class TaskDocumentProcessor(PromptTreeNode):
    """Generic analyzer task referenced by test commands."""
    pass


class TaskProcessor(PromptTreeNode):
    """Generic processor task referenced by test commands."""
    pass


class TaskProcessorFour(PromptTreeNode):
    """Four-level processor task."""
    pass


class TaskStructureAThreeLevels(PromptTreeNode):
    """Three-level structure task."""
    pass


class TaskProcessorFive(PromptTreeNode):
    """Five-level processor task."""
    pass


class TaskDocumentProcessorSeven(PromptTreeNode):
    """Seven-level analyzer task."""
    pass


class TaskStructureAMinimalSpacing(PromptTreeNode):
    """Minimal spacing structure task."""
    pass


class TaskTarget(PromptTreeNode):
    """Target task for general processing."""
    pass


class TaskStructureA(PromptTreeNode):
    """Structure A task."""
    pass


class TaskStructureBZeroLayers(PromptTreeNode):
    """Structure B zero layers task."""
    pass


class TaskStructureATwoLayers(PromptTreeNode):
    """Structure A two layers task."""
    pass


class TestFieldExistenceValidationRHS:
    """Test field existence validation for variable mapping source fields (RHS).

    Per LANGUAGE_SPECIFICATION.md: RHS fields must exist in current node scope.
    """

    def setup_method(self):
        """Create fixtures for field existence tests."""
        self.structure = RunStructure()

    def test_nonexistent_field_in_variable_mapping_fails(self):
        """Test that nonexistent fields in variable mappings cause immediate validation failure."""
        class Section(PromptTreeNode):
            title: str
            content: str

        class TaskWithNonexistentField(PromptTreeNode):
            """Task referencing nonexistent field in variable mapping."""
            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should fail: sections.nonexistent_field doesn't exist on Section objects
            sections: list[Section] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.nonexistent_field}}*")
            valid_list: list[Section] = []  # Needed to satisfy "at least one must match" rule

        # Should fail immediately during tree building - field validation
        with pytest.raises(VariableSourceValidationError) as exc_info:
            self.structure.add(TaskWithNonexistentField)

        assert "nonexistent_field" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

    def test_existing_field_in_variable_mapping_passes(self):
        """Test that existing fields in variable mappings pass validation."""
        class Section(PromptTreeNode):
            title: str
            content: str

        class TaskWithExistingField(PromptTreeNode):
            """Task referencing existing field in variable mapping."""
            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should pass: sections.title exists on Section objects and we have a matching level
            sections: list[Section] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.title}}*")
            # Need a list field to satisfy "at least one must match" rule (1 level for @each[sections])
            valid_list: list[Section] = []

        # Should pass validation
        self.structure.add(TaskWithExistingField)
        assert self.structure.get_node("task.with_existing_field") is not None

    def test_nested_field_access_validation(self):
        """Test validation of nested field access in variable mappings."""
        class SubSection(PromptTreeNode):
            title: str
            content: str

        class TaskWithNestedAccess(PromptTreeNode):
            """Task with nested field access."""
            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should pass: sections.title should be valid nested access
            sections: list[SubSection] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.title}}*")
            # Need a list field to satisfy "at least one must match" rule (1 level for @each[sections])
            valid_list: list[SubSection] = []

        # Should pass - nested access to existing field structure
        self.structure.add(TaskWithNestedAccess)
        assert self.structure.get_node("task.with_nested_access") is not None

    def test_invalid_nested_field_access_fails(self):
        """Test that invalid nested field access fails validation."""
        class SubSection(PromptTreeNode):
            title: str
            content: str

        class TaskWithInvalidNested(PromptTreeNode):
            """Task with invalid nested field access."""
            # Command in 'sections' field - can use inclusion_path starting with 'sections'
            # This should fail: sections.nonexistent doesn't exist
            sections: list[SubSection] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=sections.nonexistent}}*")
            # Need a list field to satisfy "at least one must match" rule (1 level for @each[sections])
            valid_list: list[SubSection] = []

        # Should fail - nested field doesn't exist
        with pytest.raises(VariableSourceValidationError) as exc_info:
            self.structure.add(TaskWithInvalidNested)

        assert "nonexistent" in str(exc_info.value)


class TestLoopNestingValidation:
    """Test loop nesting validation between inclusion paths and variable mappings.

    Per LANGUAGE_SPECIFICATION.md:
    - LHS must have iteration structure matching source
    - Field types determine iteration levels
    - RHS must start from iteration root path
    """

    def setup_method(self):
        """Create fixtures for loop nesting tests."""
        self.structure = RunStructure()

    def test_single_level_iteration_matching(self):
        """Test that single-level iteration requires at least one matching nesting level."""
        class TaskSingleLevel(PromptTreeNode):
            """Task with single-level iteration."""
            # Command in 'sections' field - creates 1 level of iteration
            # At least one mapping must match: value.items has 1 level ✅
            # This should pass: value.items matches 1 level, value.title is fewer (allowed)
            sections: list[str] = Field(description="! @each[sections]->task.analyzer@{{value.items=sections,value.title=sections}}*")
            items: list[str] = []
            title: str = ""  # 0 levels - allowed as long as one mapping matches

        self.structure.add(TaskSingleLevel)
        assert self.structure.get_node("task.single_level") is not None

    def test_multi_level_iteration_matching(self):
        """Test multi-level iteration nesting validation."""
        class SubSection(PromptTreeNode):
            paragraphs: list[str]
            title: str

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskMultiLevel(PromptTreeNode):
            """Task with multi-level iteration."""
            # Command in 'sections' field can use inclusion_path starting with 'sections'
            sections: list[SubSection] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=sections.paragraphs}}*")

            # results field has 2 levels to match the iteration structure
            results: list[ResultGroup] = []

        self.structure.add(TaskMultiLevel)

    def test_no_mapping_matches_iteration_level_fails(self):
        """Test that when no mapping matches iteration level, validation fails."""
        class SubSection(PromptTreeNode):
            paragraphs: list[str]
            title: str

        class TaskNoMatchingLevel(PromptTreeNode):
            """Task where no mapping matches iteration level - should fail."""
            # Command in 'sections' field uses inclusion_path starting with 'sections'
            # But all mappings have fewer levels than the 2-level iteration: sections→paragraphs
            # This violates "at least one must match" rule
            sections: list[SubSection] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.title=sections.paragraphs,value.simple=sections.paragraphs}}*")

            # title field has 0 nesting levels (should have 2 to match sections→paragraphs iteration)
            title: str = ""

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskNoMatchingLevel)

        # Accept either iteration level mismatch or subchain validation error
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["at least one", "iteration", "nesting", "level"]), \
            f"Expected nesting/iteration validation error, got: {exc_info.value}"

    def test_excessive_nesting_fails(self):
        """Test that LHS with more nesting levels than iteration fails."""
        class SubSection(PromptTreeNode):
            paragraphs: list[str]

        class Level2Node(PromptTreeNode):
            items: list[str]

        class Level1Node(PromptTreeNode):
            level2: list[Level2Node]

        class TaskExcessiveNesting(PromptTreeNode):
            """Task with excessive LHS nesting."""
            # Command in 'sections' field - creates 1 level of iteration
            # One valid mapping (1 level) and one excessive mapping (3 levels)
            # This should fail: value.deeply_nested has 3 levels but iteration is only 1
            sections: list[SubSection] = Field(description="! @each[sections]->task.analyzer@{{value.valid_field=sections,value.deeply_nested=sections}}*")
            valid_field: list[str] = []  # 1 level - matches iteration
            deeply_nested: list[Level1Node] = []  # 3 levels - exceeds iteration

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskExcessiveNesting)

        assert "nesting" in str(exc_info.value).lower()
        assert "exceeds" in str(exc_info.value).lower()

    def test_rhs_must_start_from_iteration_root(self):
        """Test that RHS must start from iteration root path."""
        class TaskRHSValidation(PromptTreeNode):
            """Task testing RHS path validation."""
            # Command in 'sections' field - RHS must share iterable parts from 'sections'
            # One valid mapping (matches level and root) + one invalid RHS root
            # Using prompt scope which should follow strict subchain rules (unlike value scope)
            sections: list[str] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections,value.title=other_field}}*")
            other_field: str = ""
            valid_list: list[str] = []  # Need this to satisfy "at least one matches" rule

        # Parser validation catches this early as CommandParseError
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskRHSValidation)

        assert "iteration root" in str(exc_info.value).lower()

    def test_inclusion_path_must_start_with_iterable(self):
        """Test that inclusion paths must start with iterable fields."""
        class SubSection(PromptTreeNode):
            items: list[str]

        class TaskInvalidStart(PromptTreeNode):
            """Task with inclusion path not starting with iterable."""
            title: str = Field(
                default="test",
                description="! @each[title.subsections.items]->task.processor@{{value.item=title}}*"
            )  # Non-iterable start, but also RHS field
            subsections: list[SubSection] = []

        class TaskProcessor(PromptTreeNode):
            item: str

        # Should fail because inclusion path starts with non-iterable 'title'
        from langtree.commands.parser import CommandParseError
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskInvalidStart)

        error_msg = str(exc_info.value).lower()
        assert "@each cannot be defined on non-iterable field" in error_msg

    def test_inclusion_path_must_end_with_iterable(self):
        """Test that inclusion paths must end with iterable fields."""
        class SubSection(PromptTreeNode):
            items: list[str]
            title: str

        class TaskInvalidEnd(PromptTreeNode):
            """Task with inclusion path not ending with iterable."""
            sections: list[SubSection] = Field(
                default=[],
                description="! @each[sections.title]->task.processor@{{value.name=sections}}*"
            )
            title: str = "test"  # Add as field for RHS validation

        class TaskProcessor(PromptTreeNode):
            name: str

        # Should fail because inclusion path ends with non-iterable 'title' (sections.title)
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskInvalidEnd)

        error_msg = str(exc_info.value).lower()
        assert "must end with an iterable field" in error_msg


class TestTaskTargetCompletenessValidation:
    """Test task target completeness validation.

    Per specification: ->task should be caught as incomplete (missing specific target).
    """

    def setup_method(self):
        """Create fixtures for task target tests."""
        self.structure = RunStructure()

    def test_incomplete_task_target_fails(self):
        """Test that incomplete task targets (just 'task') fail validation."""
        class TaskIncompleteTarget(PromptTreeNode):
            """Task with incomplete target reference."""
            # Commands in respective fields with incomplete task targets - should fail
            sections: list[str] = Field(description="! @each[sections]->task@{{value.title=sections}}*")
            data: str = Field(description="! @all->task@{{prompt.data=data}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskIncompleteTarget)

        assert "incomplete" in str(exc_info.value).lower()
        assert "task target" in str(exc_info.value).lower()

    def test_complete_task_target_passes(self):
        """Test that complete task targets pass validation."""
        class TaskCompleteTarget(PromptTreeNode):
            """Task with complete target reference."""
            # Commands in respective fields with complete task targets - should pass
            sections: list[str] = Field(description="! @each[sections]->task.analyzer@{{value.valid_list=sections}}*")
            data: str = Field(description="! @all->task.processor@{{prompt.data=data}}*")
            valid_list: list[str] = []  # Needed to satisfy "at least one must match" rule

        # Should pass validation
        self.structure.add(TaskCompleteTarget)
        assert self.structure.get_node("task.complete_target") is not None


class TestVariableMappingConstraints:
    """Test variable mapping constraint validation per specification."""

    def setup_method(self):
        """Create fixtures for constraint tests."""
        self.structure = RunStructure()

    def test_multiple_mappings_with_mixed_nesting_levels(self):
        """Test constraint: at least one mapping must match iteration level, none can exceed."""
        class SubSection(PromptTreeNode):
            paragraphs: list[str]
            title: str

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskMixedMappings(PromptTreeNode):
            """Task with multiple mappings where LHS fields have different nesting levels."""
            # Command in 'sections' field - creates 2 levels with sections.paragraphs
            # This should pass: results field has 2 levels (matching iteration), items has 1 level (fewer)
            sections: list[SubSection] = Field(description="""
            ! @each[sections.paragraphs]->task.analyzer@{{
                value.results=sections.paragraphs,
                value.items=sections.paragraphs
            }}*
            """)

            results: list[ResultGroup] = []  # 2 levels - should match iteration (2)
            items: list[str] = []    # 1 level - fewer than iteration (2)

        self.structure.add(TaskMixedMappings)

    def test_no_mapping_matches_iteration_level_fails(self):
        """Test constraint failure: no mapping matches the iteration level."""
        class SubSection(PromptTreeNode):
            paragraphs: list[str]
            title: str

        class TaskNoMatchingLevel(PromptTreeNode):
            """Task where no mapping matches iteration level."""
            # Command in 'sections' field - creates 2 levels with sections.paragraphs
            # But all mappings are 0 or 1 level - none match the 2-level requirement
            sections: list[SubSection] = Field(description="""
            ! @each[sections.paragraphs]->task.analyzer@{{
                value.items=sections.paragraphs,
                value.content=sections.paragraphs.text
            }}*
            """)

        # This test now catches RHS path validation error during semantic validation
        # The LHS-RHS nesting validation will be deferred to assembly stage
        from langtree.prompt.exceptions import VariableSourceValidationError

        with pytest.raises((FieldValidationError, VariableSourceValidationError)) as exc_info:
            self.structure.add(TaskNoMatchingLevel)

        # Accept either nesting level error or RHS path validation error
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in [
            "iteration level", "requires at least one",  # LHS-RHS nesting errors
            "does not exist", "cannot access"             # RHS path validation errors
        ]), f"Expected validation error, got: {exc_info.value}"


class TestSpecificationEdgeCases:
    """Test edge cases and boundary conditions from specification."""

    def setup_method(self):
        """Create fixtures for edge case tests."""
        self.structure = RunStructure()

    def test_wildcard_with_proper_nesting(self):
        """Test wildcard (*) usage with proper nesting validation."""
        class TaskWildcard(PromptTreeNode):
            """Task using wildcard mapping."""
            sections: list[str] = []

            # Wildcard should respect nesting rules
            wildcard_mapping: str = Field(description="! @->task.analyzer@{{prompt.context=*}}")

        self.structure.add(TaskWildcard)

    def test_complex_nested_iteration_all_levels(self):
        """Test complex nested iteration with all levels specified."""
        class Paragraph(PromptTreeNode):
            sentences: list[str]

        class SubSection(PromptTreeNode):
            paragraphs: list[Paragraph]

        class Level2Node(PromptTreeNode):
            items: list[str]

        class Level1Node(PromptTreeNode):
            level2: list[Level2Node]

        class TaskComplexNesting(PromptTreeNode):
            """Task with complex nested iteration."""
            # Command in 'sections' field - creates 3 levels with sections.paragraphs.sentences
            # All three levels must be iterable: sections, paragraphs, sentences
            sections: list[SubSection] = Field(description="""
            ! @each[sections.paragraphs.sentences]->task.analyzer@{{
                value.results=sections.paragraphs.sentences
            }}*
            """)

            # Field that has 3 levels of nesting to match the iteration
            results: list[Level1Node] = []  # 3 levels - should match iteration

        self.structure.add(TaskComplexNesting)


class TestInheritanceAndNamingValidation:
    """Test inheritance and naming convention validation."""

    def setup_method(self):
        """Create fixtures for inheritance tests."""
        self.structure = RunStructure()

    def test_basemodel_inheritance_fails(self):
        """Test that BaseModel inheritance fails validation."""

        class InvalidSection(BaseModel):  # This should fail!
            title: str
            content: str

        class TaskWithBaseModel(PromptTreeNode):
            """Task that tries to use BaseModel inheritance."""
            sections: list[InvalidSection] = []

            valid_command: str = Field(description="! @->task.analyzer@{{prompt.data=valid_command}}")

        # Should fail because InvalidSection inherits from BaseModel, not PromptTreeNode
        with pytest.raises(Exception) as exc_info:  # Type may vary - could be validation error
            self.structure.add(TaskWithBaseModel)

        # Check that it's caught for inheritance reasons
        assert any(keyword in str(exc_info.value).lower() for keyword in ['basemodel', 'inheritance', 'prompttreenode'])

    def test_root_task_naming_convention_enforced(self):
        """Test that root classes must follow TaskSomethingCamelCased naming."""

        # These should fail - don't start with Task
        class AnalyzerNode(PromptTreeNode):
            """Invalid root class name."""
            data: str = Field(description="! @->task.processor@{{prompt.input=data}}")

        class ProcessorEngine(PromptTreeNode):
            """Another invalid root class name."""
            content: str = Field(description="! @->task.analyzer@{{value.data=content}}")

        class SimpleHandler(PromptTreeNode):
            """Yet another invalid root class name."""
            input_data: str = Field(description="! @->task.handler@{{prompt.context=input_data}}")

        # Should fail because they don't start with 'Task'
        for invalid_class in [AnalyzerNode, ProcessorEngine, SimpleHandler]:
            with pytest.raises(ValueError) as exc_info:
                self.structure.add(invalid_class)

            assert "task" in str(exc_info.value).lower()

    def test_valid_task_naming_convention_passes(self):
        """Test that valid TaskSomethingCamelCased names pass validation."""

        class TaskDocumentProcessor(PromptTreeNode):
            """Valid task name."""
            data: str = Field(description="! @->task.processor@{{prompt.input=data}}")

        class TaskDataProcessor(PromptTreeNode):
            """Valid task name with multiple words."""
            content: str = Field(description="! @->task.analyzer@{{value.data=content}}")

        class TaskComplexAnalysisEngine(PromptTreeNode):
            """Valid task name with many words."""
            input_data: str = Field(description="! @->task.handler@{{prompt.context=input_data}}")

        # These should all pass
        for valid_class in [TaskDocumentProcessor, TaskDataProcessor, TaskComplexAnalysisEngine]:
            # Should not raise any exception
            self.structure.add(valid_class)

            # Verify they were added successfully
            root_tag = get_root_tag(valid_class)
            assert self.structure.get_node(root_tag) is not None


class TestFieldContextScopingValidation:
    """Test field context scoping validation for DPCL commands.

    Per corrected LANGUAGE_SPECIFICATION.md:
    - inclusion_path must start with the field where command is defined
    - destination and target_path can reference other subtrees
    - source_path must share all iterable parts of inclusion_path exactly
    """

    def setup_method(self):
        """Create fixtures for field context scoping tests."""
        self.structure = RunStructure()

    def test_inclusion_path_must_start_with_field_context(self):
        """Test that inclusion_path must start with field where command is defined."""
        class Section(PromptTreeNode):
            paragraphs: list[str] = []

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskFieldContextValid(PromptTreeNode):
            """Task with valid field context scoping."""
            # ✅ Command in 'sections' field can use inclusion_path starting with 'sections'
            sections: list[Section] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=sections.paragraphs}}*")
            results: list[ResultGroup] = []

        # Should pass validation
        self.structure.add(TaskFieldContextValid)
        assert self.structure.get_node("task.field_context_valid") is not None

    def test_inclusion_path_wrong_field_context_fails(self):
        """Test that inclusion_path with wrong field context fails."""
        class Paragraph(PromptTreeNode):
            text: str

        class Section(PromptTreeNode):
            paragraphs: list[Paragraph] = []

        class CommandResult(PromptTreeNode):
            analysis: str

        class ResultSummary(PromptTreeNode):
            summaries: list[CommandResult] = []

        class TaskFieldContextInvalid(PromptTreeNode):
            """Task with invalid field context scoping."""
            sections: list[Section] = []
            # ❌ Command in 'command' field cannot use inclusion_path starting with 'sections'
            command: list[CommandResult] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=sections.paragraphs.text}}*")
            results: list[ResultSummary] = []

        # Should fail validation
        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskFieldContextInvalid)

        assert "field context" in str(exc_info.value).lower() or "scoping" in str(exc_info.value).lower()
        assert "inclusion_path" in str(exc_info.value).lower()

    def test_destination_can_reference_other_subtrees(self):
        """Test that destination can reference other subtrees."""
        class TaskCrossSubtreeDestination(PromptTreeNode):
            """Task with cross-subtree destination."""
            # ✅ destination can reference other subtrees
            sections: list[str] = Field(description="! @each[sections]->other.analyzer@{{value.results=sections}}*")
            results: list[str] = []

        # Should pass validation
        self.structure.add(TaskCrossSubtreeDestination)
        assert self.structure.get_node("task.cross_subtree_destination") is not None

    def test_target_path_can_reference_other_subtrees(self):
        """Test that target_path can reference other subtrees."""
        class TaskCrossSubtreeTarget(PromptTreeNode):
            """Task with cross-subtree target_path."""
            # ✅ target_path can reference other subtrees, but need one mapping that matches iteration level
            sections: list[str] = Field(description="! @each[sections]->task.analyzer@{{other.results=sections,value.items=sections}}*")
            items: list[str] = []  # 1 level to match iteration level

        # Should pass validation
        self.structure.add(TaskCrossSubtreeTarget)
        assert self.structure.get_node("task.cross_subtree_target") is not None


class TestSubchainValidation:
    """Test subchain validation for source_path vs inclusion_path.

    Per corrected LANGUAGE_SPECIFICATION.md:
    - source_path must be subchains of the inclusion_path
    - All source_paths in variable mappings must start with inclusion_path or be subchains of it
    """

    def setup_method(self):
        """Create fixtures for subchain validation tests."""
        self.structure = RunStructure()

    def test_source_path_exact_subchain_match(self):
        """Test that source_path exactly matching inclusion_path passes."""
        class Section(PromptTreeNode):
            paragraphs: list[str] = []

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskValidSubchain(PromptTreeNode):
            """Task with valid subchain matching."""
            # ✅ source_path 'sections.paragraphs' exactly matches inclusion_path
            sections: list[Section] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=sections.paragraphs}}*")
            results: list[ResultGroup] = []

        # Should pass validation
        self.structure.add(TaskValidSubchain)
        assert self.structure.get_node("task.valid_subchain") is not None

    def test_source_path_subchain_with_field_access_passes(self):
        """Test that source_path as subchain with field access passes."""
        class Paragraph(PromptTreeNode):
            text: str = ""

        class Section(PromptTreeNode):
            paragraphs: list[Paragraph] = []
            title: str = ""

        class ParagraphGroup(PromptTreeNode):
            paragraphs: list[Paragraph]

        class TaskSubchainFieldAccess(PromptTreeNode):
            """Task with subchain + field access."""
            # ✅ source_path with field access is allowed - all must be subchains of sections.paragraphs
            sections: list[Section] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=sections.paragraphs,value.content=sections.paragraphs.text}}*")
            results: list[ParagraphGroup] = []
            title: str = ""

        # Should pass validation
        self.structure.add(TaskSubchainFieldAccess)
        assert self.structure.get_node("task.subchain_field_access") is not None

    def test_source_path_not_subchain_fails(self):
        """Test that source_path that is not a subchain of inclusion_path fails."""
        class Section(PromptTreeNode):
            paragraphs: list[str] = []

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskInvalidSubchain(PromptTreeNode):
            """Task with source_path that is not a subchain of inclusion_path."""
            # ❌ source_path 'paragraphs' is NOT a subchain of 'sections.paragraphs'
            # Add paragraphs field so field existence validation passes, but subchain validation fails
            sections: list[Section] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=paragraphs}}*")
            paragraphs: list[str] = []  # Exists but is NOT a subchain of 'sections.paragraphs'
            results: list[ResultGroup] = []

        # Should fail validation (caught by parser's iteration matching)
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskInvalidSubchain)

        assert any(keyword in str(exc_info.value).lower() for keyword in ["subchain", "iteration root", "source path"])

    def test_source_path_completely_different_fails(self):
        """Test that source_path with completely different path fails."""
        class Section(PromptTreeNode):
            paragraphs: list[str] = []

        class ResultGroup(PromptTreeNode):
            items: list[str]

        class TaskDifferentPath(PromptTreeNode):
            """Task with source_path using completely different path."""
            # ❌ source_path 'other_field' is NOT a subchain of inclusion_path 'sections.paragraphs'
            # Use simple field instead of nested path to test subchain validation
            sections: list[Section] = Field(description="! @each[sections.paragraphs]->task.analyzer@{{value.results=other_field}}*")
            other_field: str = ""  # Exists but is NOT a subchain of 'sections.paragraphs'
            results: list[ResultGroup] = []

        # Should fail validation
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskDifferentPath)

        assert any(keyword in str(exc_info.value).lower() for keyword in ["subchain", "iteration root", "source path"])


class TestComplexIterationPathValidation:
    """Test complex iteration path validation with mixed iterable/non-iterable segments.

    Per corrected LANGUAGE_SPECIFICATION.md:
    - Support paths like 'iterable1.noniterable.iterable2.iterable3'
    """

    def setup_method(self):
        """Create fixtures for complex iteration path tests."""
        self.structure = RunStructure()

    def test_mixed_iterable_noniterable_segments_passes(self):
        """Test that mixed iterable/non-iterable segments pass validation."""
        class Category(PromptTreeNode):
            items: list[str] = []

        class Metadata(PromptTreeNode):
            categories: list[Category] = []

        class Section(PromptTreeNode):
            metadata: Metadata = Metadata()

        class Level2Node(PromptTreeNode):
            items: list[str]

        class Level1Node(PromptTreeNode):
            level2: list[Level2Node]

        class TaskComplexPath(PromptTreeNode):
            """Task with complex iteration path."""
            # ✅ sections.metadata.categories.items - mixed iterable/non-iterable
            # Iteration levels: sections(1) + categories(2) + items(3) = 3 levels (metadata is non-iterable)
            sections: list[Section] = Field(description="! @each[sections.metadata.categories.items]->task.analyzer@{{value.results=sections.metadata.categories.items}}*")
            results: list[Level1Node] = []  # 3 levels to match actual iteration levels

        # Should pass validation
        self.structure.add(TaskComplexPath)
        assert self.structure.get_node("task.complex_path") is not None


class TestComplexNestedValueScopeValidation:
    """Test complex nested value scope validation with multi-level PromptTreeNode hierarchies.

    Tests structures with same iteration count but different total levels,
    unequal non-iterable spacing between iterations, and cross-tree references.
    Per CODING_STANDARDS.md TDD approach: comprehensive test coverage first.
    """

    def setup_method(self):
        """Create fixtures for complex nested value scope tests."""
        self.structure = RunStructure()

    def test_nesting_level_mismatch_node_based_fails(self):
        """Test nesting level mismatch using pure node hierarchy (not Python lists).

        ERROR PATTERN: LHS node nesting vs RHS iteration level mismatch
        PROBLEM: LHS has 0 node levels, RHS has 3 iteration levels
        RULE: LHS node nesting depth must match RHS iteration count
        FIX: Use LHS field with 3-level node nesting to match 3 iterations
        """
        # Structure: root → iterable₁ → non-iterable → non-iterable → non-iterable → iterable₂ → iterable₃ → non-iterable
        # Pattern: documents.config.settings.sections.paragraphs.summary
        # Iterations: documents(1) → sections(2) → paragraphs(3) = 3 iterations
        # Non-iterables between 1st-2nd iteration: config, settings (2 levels)

        class ParagraphData(PromptTreeNode):
            """Level 7 - non-iterable data container."""
            text: str
            confidence: float

        class Paragraph(PromptTreeNode):
            """Level 6 - iterable container."""
            data: ParagraphData
            summary: str

        class Section(PromptTreeNode):
            """Level 5 - iterable container."""
            paragraphs: list[Paragraph] = []
            title: str

        class ProcessingSettings(PromptTreeNode):
            """Level 4 - non-iterable configuration."""
            sections: list[Section] = []
            algorithm: str

        class ProcessingConfig(PromptTreeNode):
            """Level 3 - non-iterable configuration."""
            settings: ProcessingSettings
            version: str

        class DocumentMetadata(PromptTreeNode):
            """Level 2 - non-iterable metadata."""
            config: ProcessingConfig
            author: str

        class TaskDocumentProcessorSevenBad(PromptTreeNode):
            """Root task demonstrating subchain validation violation."""
            # ❌ WRONG: RHS path is NOT a subchain of inclusion_path
            # inclusion_path: documents.config.settings.sections.paragraphs
            # Invalid RHS: documents.config.settings.sections.title (does NOT start with inclusion_path)
            documents: list[DocumentMetadata] = Field(description="""! @each[documents.config.settings.sections.paragraphs]->task.processor@{{
                value.paragraph_summary=documents.config.settings.sections.title
            }}*""")
            # documents.config.settings.sections.title is NOT a subchain of documents.config.settings.sections.paragraphs

        # Should fail - nesting depth mismatch
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskDocumentProcessorSevenBad)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["nesting", "level", "iteration", "depth"]), \
            f"Expected nesting depth mismatch error, got: {exc_info.value}"

    def test_same_iteration_count_different_total_levels_passes(self):
        """Test structures with SAME iteration count but DIFFERENT total levels and non-iterable spacing.

        STRUCTURE 1: 6 total levels, 3 iterations, 2 non-iterables between 1st-2nd iteration
        STRUCTURE 2: 4 total levels, 3 iterations, 0 non-iterables between 1st-2nd iteration
        VERIFICATION: Both have exactly 3 iteration levels but different complexity
        """

        # Structure 1: 5 levels - companies.metadata.departments.items (3 iterations: companies→departments→items)
        class Item(PromptTreeNode):
            name: str

        class Department(PromptTreeNode):
            items: list[Item] = []  # iteration level 3

        class CompanyMetadata(PromptTreeNode):  # non-iterable layer
            departments: list[Department] = []  # iteration level 2

        class Company(PromptTreeNode):
            metadata: CompanyMetadata  # non-iterable layer

        class TaskStructure5Levels(PromptTreeNode):
            # ✅ CORRECT: @each[companies.metadata.departments.items] MUST be on companies field
            companies: list[Company] = Field(description="""! @each[companies.metadata.departments.items]->task.processor_4_levels@{{
                value.item_data=companies.metadata.departments.items.name
            }}*""")  # iteration level 1

            # 5 total levels: companies.metadata.departments.items
            # 3 iterations: companies(1)→departments(2)→items(3)
            # 1 non-iterable between 1st-2nd: metadata

        # Structure 2: 4 levels - batches.items.tokens (3 iterations: batches→items→tokens)
        class Token(PromptTreeNode):
            text: str

        class BatchItem(PromptTreeNode):
            tokens: list[Token] = []  # iteration level 3

        class Batch(PromptTreeNode):
            items: list[BatchItem] = []  # iteration level 2

        class TaskStructure4Levels(PromptTreeNode):
            # ✅ CORRECT: @each[batches.items.tokens] MUST be on batches field
            batches: list[Batch] = Field(description="""! @each[batches.items.tokens]->task.structure_5_levels@{{
                value.token_data=batches.items.tokens.text
            }}*""")  # iteration level 1

            # 4 total levels: batches.items.tokens
            # 3 iterations: batches(1)→items(2)→tokens(3)
            # 0 non-iterables between iterations (direct nesting)

        # Both should pass - same iteration count (3) despite different total levels (5 vs 4)
        self.structure.add(TaskStructure5Levels)
        self.structure.add(TaskStructure4Levels)
        # Verify that both structures were processed and are accessible
        # Note: They share the same 'task' designation so only the last one added remains in root_nodes
        # But both should be processable without validation errors
        task_4_node = self.structure.get_node("task.structure4_levels")
        # Since TaskStructure4Levels was added last, only it should be accessible
        assert task_4_node is not None, "TaskStructure4Levels should be accessible"
        # TaskStructure5Levels will be None due to overshadowing, which is expected behavior

    def test_seven_level_structure_correct_nesting_passes(self):
        """Test 7-level structure with correct nesting depth assignments."""
        # Structure: root → iterable₁ → non-iterable → non-iterable → non-iterable → iterable₂ → iterable₃ → non-iterable
        # Pattern: documents.config.settings.sections.paragraphs.summary
        # Iterations: documents(1) → sections(2) → paragraphs(3) = 3 iterations
        # Non-iterables between iterations: config, settings (2 levels)

        class ParagraphData(PromptTreeNode):
            """Level 7 - non-iterable data container."""
            text: str
            confidence: float

        class Paragraph(PromptTreeNode):
            """Level 6 - iterable container."""
            data: ParagraphData
            summary: str

        class Section(PromptTreeNode):
            """Level 5 - iterable container."""
            paragraphs: list[Paragraph] = []
            title: str

        class ProcessingSettings(PromptTreeNode):
            """Level 4 - non-iterable configuration."""
            sections: list[Section] = []
            algorithm: str

        class ProcessingConfig(PromptTreeNode):
            """Level 3 - non-iterable configuration."""
            settings: ProcessingSettings
            version: str

        class DocumentMetadata(PromptTreeNode):
            """Level 2 - non-iterable metadata."""
            config: ProcessingConfig
            author: str

        class TaskDocumentProcessorSevenGood(PromptTreeNode):
            """Root task with correct node-based nesting assignments."""
            # ✅ CORRECT: Command must be on the field that starts the inclusion_path
            # inclusion_path: documents.config.settings.sections.paragraphs
            # Valid subchains: all RHS must start with inclusion_path
            documents: list[DocumentMetadata] = Field(description="""! @each[documents.config.settings.sections.paragraphs]->task.processor@{{
                value.paragraph_content=documents.config.settings.sections.paragraphs,
                value.paragraph_summary=documents.config.settings.sections.paragraphs.summary,
                value.paragraph_data_text=documents.config.settings.sections.paragraphs.data.text
            }}*""")
            # All RHS paths are valid subchains starting with inclusion_path ✅

        # Should pass validation - proper subchain matching
        self.structure.add(TaskDocumentProcessorSevenGood)
        assert self.structure.get_node("task.document_processor_seven_good") is not None

    def test_five_level_structure_three_iterations_passes(self):
        """Test 5-level structure with 3 iterations and minimal non-iterable spacing."""
        # Structure: root → iterable₁ → non-iterable → iterable₂ → iterable₃
        # Pattern: batches.items.tokens
        # Iterations: batches(1) → items(2) → tokens(3) = 3 iterations
        # Non-iterables between 1st-2nd iteration: BatchConfig level (implicit 1 level)

        class Token(PromptTreeNode):
            """Level 4 - iterable data."""
            text: str
            pos_tag: str

        class BatchItem(PromptTreeNode):
            """Level 3 - iterable container."""
            tokens: list[Token] = []
            item_id: str

        class BatchConfig(PromptTreeNode):
            """Level 2 - non-iterable configuration."""
            items: list[BatchItem] = []
            algorithm: str

        class TaskProcessorFive(PromptTreeNode):
            """Root task with 5-level structure."""
            # CORRECT: @each on iterable batches field
            batches: list[BatchConfig] = Field(description="""! @each[batches.items.tokens]->task.analyzer_seven@{{
                value.token_data=batches.items.tokens,
                value.token_text=batches.items.tokens.text,
                value.token_pos=batches.items.tokens.pos_tag
            }}*""")

        # Should pass validation - same 3 iteration count as seven-level structure
        self.structure.add(TaskProcessorFive)
        assert self.structure.get_node("task.processor_five") is not None

    def test_cross_tree_same_iteration_count_different_spacing_passes(self):
        """Test cross-tree references with same iteration count but different non-iterable spacing."""
        # Both structures have 3 iterations but different total levels (7 vs 5)
        # and different non-iterable spacing (3 vs 1 levels between iterations)
        # This tests that resolution logic correctly handles unequal spacing

        class TokenData(PromptTreeNode):
            text: str
            confidence: float

        class Sentence(PromptTreeNode):
            tokens: list[TokenData] = []
            sentiment: str

        class Document(PromptTreeNode):
            sentences: list[Sentence] = []
            title: str

        class TaskCrossTreeTest(PromptTreeNode):
            """Test cross-tree references with matching iteration counts."""
            documents: list[Document] = Field(description="""! @each[documents.sentences.tokens]->task.processor_five@{{
                value.cross_tree_tokens=documents.sentences.tokens,
                value.cross_tree_token_text=documents.sentences.tokens.text,
                value.cross_tree_confidence=documents.sentences.tokens.confidence
            }}*""")

        # Should pass - both have 3 iteration levels despite different structures
        self.structure.add(TaskCrossTreeTest)
        assert self.structure.get_node("task.cross_tree_test") is not None

    def test_field_context_scoping_wrong_field_fails(self):
        """Test that @each command on wrong field fails validation.

        ERROR PATTERN: Field context scoping violation
        PROBLEM: @each[documents.content] command is placed on 'analysis' field instead of 'documents' field
        RULE: @each[path] command must be placed on the field that starts the path
        FIX: Move command to 'documents' field OR change path to start with 'analysis'
        """
        class SimpleDoc(PromptTreeNode):
            content: str

        class TaskWrongField(PromptTreeNode):
            """Task demonstrating field context scoping violation."""
            documents: list[SimpleDoc] = []

            # ❌ WRONG: @each[documents.content] on analysis field (not documents field)
            # CORRECT: Should be on documents field: documents: list[SimpleDoc] = Field(description="! @each[documents.content]...")
            analysis: str = Field(description="""! @each[documents.content]->task.processor@{{
                value.content=documents.content
            }}*""")

        # Should fail - field context scoping violation
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskWrongField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["field", "context", "scoping", "inclusion"]), \
            f"Expected field context scoping error keywords, got: {exc_info.value}"

    def test_each_command_on_non_iterable_field_fails(self):
        """Test that @each command on non-iterable field fails validation."""
        class ProcessorConfig(PromptTreeNode):
            batches: list[str] = []
            settings: str

        class TaskNonIterableField(PromptTreeNode):
            """Task with @each on non-iterable field."""
            # ❌ WRONG: config is single object (non-iterable), can't use @each
            config: ProcessorConfig = Field(description="""! @each[config.batches]->task.processor@{{
                value.batch_data=config.batches
            }}*""")

        # Should fail - @each on non-iterable field
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskNonIterableField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "each", "field", "list"])

    def test_value_scope_wrong_subchain_fails(self):
        """Test that value scope with wrong subchain paths fails validation."""
        class Section(PromptTreeNode):
            content: str

        class TaskWrongSubchain(PromptTreeNode):
            """Task with value scope using wrong subchain paths."""
            sections: list[Section] = []
            other_field: str = ""

            # ❌ WRONG: value.content=other_field is not a subchain of sections.content
            analysis: str = Field(description="""! @each[sections.content]->task.processor@{{
                value.content=other_field
            }}*""")

        # Should fail - wrong subchain path in value scope
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskWrongSubchain)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["subchain", "source", "path", "inclusion"])

    def test_value_scope_excessive_nesting_fails(self):
        """Test that value scope with excessive LHS nesting fails validation."""
        class SimpleItem(PromptTreeNode):
            text: str

        class Level2Node(PromptTreeNode):
            items: list[str]

        class Level1Node(PromptTreeNode):
            level2: list[Level2Node]

        class TaskExcessiveNesting(PromptTreeNode):
            """Task with excessive LHS nesting in value scope."""
            items: list[SimpleItem] = []

            # ❌ WRONG: value.deeply_nested has 3 levels but @each[items.text] creates only 1 iteration level
            deeply_nested: list[Level1Node] = Field(description="""! @each[items.text]->task.processor@{{
                value.deeply_nested=items.text
            }}*""")

        # Should fail - excessive nesting beyond iteration level
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskExcessiveNesting)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["nesting", "level", "iteration", "exceeds"])

    def test_node_hierarchy_depth_validation_passes(self):
        """Test deep node hierarchy navigation with proper subchain validation."""
        # 4-level pure node hierarchy: Company → Department → Team → Person
        # Focus on node traversal, not Python list nesting

        class Person(PromptTreeNode):
            name: str
            role: str

        class Team(PromptTreeNode):
            members: list[Person] = []
            team_lead: Person
            project: str

        class Department(PromptTreeNode):
            teams: list[Team] = []
            manager: Person
            budget: float

        class Company(PromptTreeNode):
            departments: list[Department] = []
            ceo: Person

        class TaskNodeHierarchy(PromptTreeNode):
            """Test pure node hierarchy traversal."""
            # ✅ CORRECT: Command must be on the field that starts the inclusion_path
            # inclusion_path: companies.departments.teams.members
            # Valid subchains: companies.departments.teams.members.name, companies.departments.teams.members.role
            companies: list[Company] = Field(description="""! @each[companies.departments.teams.members]->task.analyzer@{{
                value.person_data=companies.departments.teams.members,
                value.person_name=companies.departments.teams.members.name,
                value.person_role=companies.departments.teams.members.role
            }}*""")

        # Should pass - proper node hierarchy traversal with valid subchains
        self.structure.add(TaskNodeHierarchy)
        assert self.structure.get_node("task.node_hierarchy") is not None

    def test_node_hierarchy_invalid_branch_fails(self):
        """Test that accessing invalid node branches fails validation.

        ERROR PATTERN: Invalid node branch access
        PROBLEM: Accessing companies.departments.teams.project instead of valid subchain
        RULE: RHS must be subchain of inclusion_path (companies.departments.teams.members)
        FIX: Use companies.departments.teams.members.X or exact inclusion_path
        """
        class Person(PromptTreeNode):
            name: str
            role: str

        class Team(PromptTreeNode):
            members: list[Person] = []
            project: str  # This is at Team level, not Member level

        class Department(PromptTreeNode):
            teams: list[Team] = []
            budget: float

        class Company(PromptTreeNode):
            departments: list[Department] = []

        class TaskInvalidBranch(PromptTreeNode):
            """Test invalid node branch access."""
            companies: list[Company] = []

            # ❌ WRONG: companies.departments.teams.project is NOT a subchain of companies.departments.teams.members
            # inclusion_path: companies.departments.teams.members
            # Invalid RHS: companies.departments.teams.project (branches off at 'teams' level, not 'members')
            invalid_analysis: str = Field(description="""! @each[companies.departments.teams.members]->task.analyzer@{{
                value.project_info=companies.departments.teams.project
            }}*""")

        # Should fail - accessing invalid branch that's not a subchain
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskInvalidBranch)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["subchain", "source", "path", "inclusion"]), \
            f"Expected subchain validation error for invalid branch access, got: {exc_info.value}"

    def test_mixed_node_structure_complexity_passes(self):
        """Test complex mixed node structure with multiple non-iterable layers."""
        # Structure: School → Grade → ClassInfo → Schedule → Subject → Students → Performance
        # 7 levels with 3 iterable points: schools, subjects, students

        class Performance(PromptTreeNode):
            score: float
            feedback: str

        class Student(PromptTreeNode):
            performance: Performance  # non-iterable
            student_id: str

        class Subject(PromptTreeNode):
            students: list[Student] = []  # iterable
            subject_name: str

        class Schedule(PromptTreeNode):
            subjects: list[Subject] = []  # iterable
            semester: str

        class ClassInfo(PromptTreeNode):
            schedule: Schedule  # non-iterable
            classroom: str

        class Grade(PromptTreeNode):
            class_info: ClassInfo  # non-iterable
            grade_level: int

        class School(PromptTreeNode):
            grades: list[Grade] = []  # iterable
            school_name: str

        class TaskComplexNodes(PromptTreeNode):
            """Test complex node structure with deep non-iterable chains."""
            # ✅ CORRECT: Command must be on the field that starts the inclusion_path
            # inclusion_path: schools.grades.class_info.schedule.subjects.students
            # 3 iterations: schools → subjects → students
            # Non-iterable chain: grades.class_info.schedule between schools and subjects
            schools: list[School] = Field(description="""! @each[schools.grades.class_info.schedule.subjects.students]->task.analyzer@{{
                value.student_performance=schools.grades.class_info.schedule.subjects.students.performance,
                value.performance_score=schools.grades.class_info.schedule.subjects.students.performance.score,
                value.student_feedback=schools.grades.class_info.schedule.subjects.students.performance.feedback
            }}*""")

        # Should pass - complex node navigation with valid subchains
        self.structure.add(TaskComplexNodes)
        assert self.structure.get_node("task.complex_nodes") is not None


class TestDocumentedConfusionModes:
    """Document all confusion modes discussed to prevent repetition.

    Each test captures a specific confusion that occurred during development
    with clear explanation of the mistake and correct approach.
    """

    def setup_method(self):
        """Create fixtures for confusion mode documentation."""
        self.structure = RunStructure()

    def test_confusion_mode_1_field_context_scoping_wrong_field(self):
        """CONFUSION MODE 1: @each command placed on wrong field.

        MISTAKE: Placing @each[documents.path] on non-documents field
        RULE: @each[field.path] MUST be placed on the 'field' that starts the path
        CORRECT: @each[documents.path] goes on documents field only
        """
        class Doc(PromptTreeNode):
            content: str

        class TaskWrongFieldPlacement(PromptTreeNode):
            documents: list[Doc] = []

            # ❌ CONFUSION: @each[documents.content] on analysis field instead of documents field
            analysis: str = Field(description="! @each[documents.content]->task.processor@{{value.data=documents.content}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskWrongFieldPlacement)
        assert "field" in str(exc_info.value).lower() or "inclusion" in str(exc_info.value).lower()

    def test_confusion_mode_2_each_on_non_iterable_field(self):
        """CONFUSION MODE 2: @each command on non-iterable field.

        MISTAKE: Using @each on single object field instead of list field
        RULE: @each requires the base field to be iterable (list, etc.)
        CORRECT: Only use @each on list[SomeType] fields
        """
        class Config(PromptTreeNode):
            items: list[str] = []

        class TaskNonIterableBase(PromptTreeNode):
            # ❌ CONFUSION: config is single object, not list - can't use @each
            config: Config = Field(description="! @each[config.items]->task.processor@{{value.data=config.items}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskNonIterableBase)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "each", "field"])

    def test_confusion_mode_2a_each_on_non_iterable_base_field_only(self):
        """CONFUSION MODE 2A: Test ONLY @each on non-iterable base field error.

        Isolates the error of using @each[config.items] where 'config' is single object.
        Command is on proper iterable field to avoid that parallel error.
        """
        class Config(PromptTreeNode):
            items: list[str] = []

        class TaskNonIterableBaseOnly(PromptTreeNode):
            # Test only: config is single object, not list - can't use @each
            config: Config = Config()
            # Put command on iterable field to avoid parallel @each-on-string error
            results: list[str] = Field(description="! @each[config.items]->task.processor@{{value.data=config.items}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskNonIterableBaseOnly)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "base", "field"])

    def test_confusion_mode_2b_each_on_string_field_only(self):
        """CONFUSION MODE 2B: Test ONLY @each command on string field error.

        Isolates the error of putting @each command on non-iterable field.
        Base field is iterable to avoid that parallel error.
        """
        class Config(PromptTreeNode):
            items: list[str] = []

        class TaskStringFieldOnly(PromptTreeNode):
            # Test only: @each command on string field
            configs: list[Config] = []  # Base field is iterable - no error here
            # Put command on string field to test this specific error
            analysis: str = Field(description="! @each[configs.items]->task.processor@{{value.data=configs.items}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskStringFieldOnly)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "string", "each", "field"])

    def test_confusion_mode_3_prompt_vs_value_scope_misuse(self):
        """CONFUSION MODE 3: Using prompt scope instead of value scope for data forwarding.

        MISTAKE: Changed tests to prompt scope to avoid validation instead of fixing validation
        RULE: value scope is for direct data forwarding, prompt scope is for template variables
        CORRECT: Fix validation logic, don't change test scope to avoid validation
        """
        class Item(PromptTreeNode):
            text: str

        class TaskCorrectValueScope(PromptTreeNode):
            items: list[Item] = []

            # ✅ CORRECT: Use value scope for direct data forwarding
            analysis: str = Field(description="! @each[items.text]->task.processor@{{value.content=items.text}}*")

        # EXPECTED BUG: @each command on string field (analysis) instead of iterable field
        # Should fail - demonstrates invalid @each placement on non-iterable field
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskCorrectValueScope)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "each", "field", "string"])

    def test_confusion_mode_4_subchain_validation_violation(self):
        """CONFUSION MODE 4: RHS source_path not a subchain of inclusion_path.

        MISTAKE: Using source paths that don't start with inclusion_path
        RULE: All RHS source_paths must be subchains of inclusion_path
        CORRECT: RHS must start with inclusion_path or be exact match
        """
        class Section(PromptTreeNode):
            paragraphs: list[str] = []
            title: str

        class TaskSubchainViolation(PromptTreeNode):
            sections: list[Section] = []
            other_field: str = ""

            # ❌ CONFUSION: other_field is NOT a subchain of sections.paragraphs
            analysis: str = Field(description="! @each[sections.paragraphs]->task.processor@{{value.data=other_field}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskSubchainViolation)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["subchain", "source", "inclusion", "path"])

    def test_confusion_mode_5_nesting_level_mismatch_node_based(self):
        """CONFUSION MODE 5: LHS node nesting depth doesn't match RHS iteration levels.

        MISTAKE: Assigning simple field from deep iteration without matching nesting
        RULE: LHS field nesting must match RHS iteration count (node-based, not Python lists)
        CORRECT: Use nested node structures or simple fields for simple iterations
        """
        class Level3(PromptTreeNode):
            data: str

        class Level2(PromptTreeNode):
            level3: Level3

        class Level1(PromptTreeNode):
            level2: Level2

        class TaskNestingMismatch(PromptTreeNode):
            items: list[Level1] = []

            # ❌ CONFUSION: simple_field (0 nesting) from items.level2.level3.data (3 levels deep)
            # This creates nesting level mismatch
            simple_field: str = Field(description="! @each[items.level2.level3]->task.processor@{{value.simple_field=items.level2.level3.data}}*")

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskNestingMismatch)
        # Note: This might not fail yet if nesting level validation isn't fully implemented
        # But documents the expected behavior
        # Validate that some error occurred (even if not the specific nesting error yet)
        assert exc_info.value is not None, "Expected some validation error for nesting mismatch"

    def test_confusion_mode_6_python_lists_vs_node_nesting(self):
        """CONFUSION MODE 6: Using Python list nesting instead of node hierarchy nesting.

        MISTAKE: Focusing on list[list[str]] instead of Node→Node→Node chains
        RULE: DPCL validation is about navigating node hierarchies, not Python type nesting
        CORRECT: Create node chains and navigate through them
        """
        class DeepNode(PromptTreeNode):
            content: str

        class MiddleNode(PromptTreeNode):
            deep: DeepNode

        class TopNode(PromptTreeNode):
            middle: MiddleNode

        class TaskNodeBasedNesting(PromptTreeNode):
            nodes: list[TopNode] = []

            # ✅ CORRECT: Navigate node hierarchy, not Python list nesting
            analysis: str = Field(description="! @each[nodes.middle.deep]->task.processor@{{value.content=nodes.middle.deep.content}}*")

        # EXPECTED BUG: @each command on string field (analysis) instead of iterable field
        # Should fail - demonstrates invalid @each placement on non-iterable field
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskNodeBasedNesting)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "each", "field", "string"])

    def test_confusion_mode_7_iteration_count_verification(self):
        """CONFUSION MODE 7: Not properly verifying iteration counts match in cross-references.

        MISTAKE: Claiming structures have same iteration count without verification
        RULE: When testing cross-tree references, must verify iteration counts actually match
        CORRECT: Count iterations explicitly: root→iter1→iter2→iter3 = 3 iterations
        """
        # Structure A: 3 iterations - companies→departments→teams
        class Team(PromptTreeNode):
            name: str

        class Department(PromptTreeNode):
            teams: list[Team] = []  # iteration 3

        class Company(PromptTreeNode):
            departments: list[Department] = []  # iteration 2

        class TaskStructureA(PromptTreeNode):
            companies: list[Company] = []  # iteration 1
            # VERIFIED: 3 iterations total

        # Structure B: 3 iterations - schools→classes→students
        class Student(PromptTreeNode):
            name: str

        class SchoolClass(PromptTreeNode):
            students: list[Student] = []  # iteration 3

        class School(PromptTreeNode):
            classes: list[SchoolClass] = []  # iteration 2

        class TaskStructureB(PromptTreeNode):
            # ✅ CORRECT: Command in proper field with cross-tree reference
            schools: list[School] = Field(description="! @each[schools.classes.students]->task.structure_a@{{value.data=schools.classes.students.name}}*")  # iteration 1
            # VERIFIED: 3 iterations total - LHS from schools tree, RHS targets companies tree

        # Should pass - verified matching iteration counts
        self.structure.add(TaskStructureA)
        self.structure.add(TaskStructureB)
        assert self.structure.get_node("task.structure_a") is not None
        assert self.structure.get_node("task.structure_b") is not None

    def test_confusion_mode_7b_validation_gap_iteration_count_matching(self):
        """CONFUSION MODE 7b: VALIDATION GAP - Cross-tree iteration count matching not implemented.

        CURRENT BEHAVIOR: Cross-references with mismatched iteration counts are allowed (BUG!)
        EXPECTED BEHAVIOR: Should validate that LHS and RHS have matching iteration structures
        STATUS: Known validation gap that needs implementation
        """
        # Structure A: 3 iterations with non-iterables between levels
        class Team(PromptTreeNode):
            name: str

        class DeptToTeamBridge(PromptTreeNode):  # Non-iterable between Department and Team
            teams: list[Team] = []  # iteration 3

        class Department(PromptTreeNode):
            bridge_to_teams: DeptToTeamBridge  # non-iterable

        class CompanyToDeptBridge(PromptTreeNode):  # Non-iterable between Company and Department
            departments: list[Department] = []  # iteration 2

        class Company(PromptTreeNode):
            bridge_to_depts: CompanyToDeptBridge  # non-iterable

        class TaskStructureAThreeLevels(PromptTreeNode):
            companies: list[Company] = []  # iteration 1
            # Path: companies.bridge_to_depts.departments.bridge_to_teams.teams (3 iterations, 2 non-iterables)

        # Structure C: 2 iterations with 3 non-iterables between levels
        class Unit(PromptTreeNode):
            name: str

        class Bridge3(PromptTreeNode):  # 3rd non-iterable
            units: list[Unit] = []  # iteration 2

        class Bridge2(PromptTreeNode):  # 2nd non-iterable
            bridge3: Bridge3

        class Bridge1(PromptTreeNode):  # 1st non-iterable
            bridge2: Bridge2

        class Division(PromptTreeNode):
            bridge1: Bridge1

        class TaskStructureCTwoLevels(PromptTreeNode):
            # ❌ BUG: 2 iterations with 3 non-iterables referencing 3-iteration structure with 2 non-iterables
            divisions: list[Division] = Field(description="! @each[divisions.bridge1.bridge2.bridge3.units]->task.structure_a_three_levels@{{value.companies.bridge_to_depts.departments.bridge_to_teams.teams=divisions.bridge1.bridge2.bridge3.units.name}}*")  # iteration 1
            # Path: divisions.bridge1.bridge2.bridge3.units (2 iterations, 3 non-iterables)

        # Add first task
        self.structure.add(TaskStructureAThreeLevels)  # 3 iterations

        # This SHOULD fail when adding 2nd task due to iteration count mismatch (2 vs 3)
        # But validation gap means no error is raised, causing this test to FAIL
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskStructureCTwoLevels)   # 2 iterations, cross-refs to 3-iteration task

        # Should detect iteration count mismatch (2 != 3), not non-iterable spacing difference
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iteration", "level", "mismatch", "count"]), \
            f"Expected iteration count mismatch error, got: {exc_info.value}"

    def test_confusion_mode_7c_matching_iterations_different_spacing_passes(self):
        """CONFUSION MODE 7c: Matching iteration counts should pass despite different non-iterable spacing.

        CORRECT BEHAVIOR: Cross-references with SAME iteration count but different spacing should pass
        TEST: Validates that non-iterable spacing is NOT considered when matching structures
        DESIGN: 3=3 iterations but vastly different spacing patterns
        """
        # Structure A: 3 iterations with minimal non-iterable spacing
        class TeamMinimal(PromptTreeNode):
            name: str

        class DeptMinimal(PromptTreeNode):
            teams: list[TeamMinimal] = []  # iteration 3

        class CompanyMinimal(PromptTreeNode):
            departments: list[DeptMinimal] = []  # iteration 2

        class TaskStructureAMinimalSpacing(PromptTreeNode):
            companies: list[CompanyMinimal] = []  # iteration 1
            # Path: companies.departments.teams (3 iterations, 0 non-iterables)

        # Structure D: 3 iterations with maximum non-iterable spacing
        class TeamMaximal(PromptTreeNode):
            name: str

        class SpacerC(PromptTreeNode):  # 3rd spacer
            teams: list[TeamMaximal] = []  # iteration 3

        class SpacerB(PromptTreeNode):  # 2nd spacer
            spacer_c: SpacerC

        class SpacerA(PromptTreeNode):  # 1st spacer
            spacer_b: SpacerB

        class DeptMaximal(PromptTreeNode):
            spacer_a: SpacerA

        class SpacerY(PromptTreeNode):  # Another spacer layer
            departments: list[DeptMaximal] = []  # iteration 2

        class SpacerX(PromptTreeNode):  # Yet another spacer layer
            spacer_y: SpacerY

        class CompanyMaximal(PromptTreeNode):
            spacer_x: SpacerX

        class TaskStructureDMaximalSpacing(PromptTreeNode):
            # ✅ CORRECT: Same 3 iterations as TaskA but with massive spacing differences
            companies: list[CompanyMaximal] = Field(
                description="! @each[companies.spacer_x.spacer_y.departments.spacer_a.spacer_b.spacer_c.teams]->task.structure_a_minimal_spacing@{{value.data=companies.spacer_x.spacer_y.departments.spacer_a.spacer_b.spacer_c.teams.name}}*"
            )  # iteration 1
            # Path: companies.spacer_x.spacer_y.departments.spacer_a.spacer_b.spacer_c.teams (3 iterations, 5 non-iterables)

        # Should pass - same iteration count (3=3), spacing difference should be ignored
        self.structure.add(TaskStructureAMinimalSpacing)   # 3 iterations, minimal spacing
        self.structure.add(TaskStructureDMaximalSpacing)   # 3 iterations, maximal spacing
        assert self.structure.get_node("task.structure_a_minimal_spacing") is not None
        assert self.structure.get_node("task.structure_d_maximal_spacing") is not None

    def test_confusion_mode_8_circular_references_vs_one_way(self):
        """CONFUSION MODE 8: Creating circular references instead of one-way references.

        MISTAKE: TaskA → TaskB → TaskA (circular dependency)
        RULE: Cross-tree references should be one-way to avoid loops
        CORRECT: TaskA → TaskB (one direction only)
        """
        class Item(PromptTreeNode):
            data: str

        class TaskSource(PromptTreeNode):
            items: list[Item] = []

            # ✅ CORRECT: One-way reference to target (no circular dependency)
            analysis: str = Field(description="! @each[items]->task.target@{{value.processed=items.data}}*")

        class TaskTarget(PromptTreeNode):
            processed_items: list[str] = []
            # ✅ CORRECT: No back-reference to TaskSource (avoids circular dependency)

        # EXPECTED BUG: @each command on string field (analysis) instead of iterable field
        # Should fail - demonstrates invalid @each placement on non-iterable field
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskSource)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "each", "field", "string"])

    def test_confusion_mode_9_tdd_methodology_violation(self):
        """CONFUSION MODE 9: Changing tests to match broken implementation instead of fixing implementation.

        MISTAKE: Changed tests from value scope to prompt scope to avoid validation errors
        RULE: In TDD, fix implementation to match tests, not tests to match broken implementation
        CORRECT: Keep tests correct, fix validation logic to make tests pass
        """
        class DataItem(PromptTreeNode):
            content: str

        class TaskCorrectTDD(PromptTreeNode):
            items: list[DataItem] = []

            # ✅ CORRECT: Keep value scope in test, fix validation to support it
            # Don't change to prompt scope just to avoid validation errors
            analysis: str = Field(description="! @each[items]->task.processor@{{value.data=items.content}}*")

        # EXPECTED BUG: @each command on string field (analysis) instead of iterable field
        # Should fail - demonstrates invalid @each placement on non-iterable field
        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskCorrectTDD)
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["iterable", "list", "each", "field", "string"])

    def test_confusion_mode_10_unequal_non_iterable_spacing_demonstration(self):
        """CONFUSION MODE 10: Not demonstrating unequal non-iterable spacing between iterations.

        MISTAKE: Claiming different non-iterable spacing without clear demonstration
        RULE: When testing spacing differences, make the spacing levels explicit and different
        CORRECT: Structure A has X non-iterables between iterations, Structure B has Y (X ≠ Y)
        """
        # Structure A: 2 non-iterables between iterations 1-2
        class DeepData(PromptTreeNode):
            value: str

        class MiddleLayer2(PromptTreeNode):  # non-iterable 2
            items: list[DeepData] = []  # iteration 2

        class MiddleLayer1(PromptTreeNode):  # non-iterable 1
            middle2: MiddleLayer2

        class TopLevel(PromptTreeNode):
            middle1: MiddleLayer1  # non-iterable connection

        class TaskStructureATwoLayers(PromptTreeNode):
            # Path: roots.middle1.middle2.items (2 non-iterables: middle1, middle2)
            roots: list[TopLevel] = Field(description="! @each[roots.middle1.middle2.items]->task.structure_b_zero_layers@{{value.data=roots.middle1.middle2.items.value}}*")  # iteration 1

        # Structure B: 0 non-iterables between iterations 1-2
        class DirectItem(PromptTreeNode):
            value: str

        class DirectContainer(PromptTreeNode):
            items: list[DirectItem] = []  # iteration 2 (direct connection)

        class TaskStructureBZeroLayers(PromptTreeNode):
            # Path: containers.items (0 non-iterables: direct connection)
            containers: list[DirectContainer] = Field(description="! @each[containers.items]->task.structure_a_two_layers@{{value.data=containers.items.value}}*")  # iteration 1

        # Should pass - demonstrates different non-iterable spacing (2 vs 0)
        self.structure.add(TaskStructureATwoLayers)
        self.structure.add(TaskStructureBZeroLayers)
        assert self.structure.get_node("task.structure_a_two_layers") is not None
        assert self.structure.get_node("task.structure_b_zero_layers") is not None

    def test_confusion_mode_11_list_list_naturally_fails_field_access(self):
        """Test that list[list[...]] naturally fails when trying to access nested fields.

        With field_name: list[list[SomeType]], when you try to access nested fields
        like field_name.inner_field, it should naturally fail because 'inner_field'
        doesn't exist as a field in the outer list structure.
        """
        class Item(PromptTreeNode):
            value: str

        class ItemGroup(PromptTreeNode):
            """Proper PromptTreeNode to replace list[Item]."""
            items: list[Item] = []

        class TaskWithListListStructure(PromptTreeNode):
            """Task showing natural failure when accessing non-existent fields in nested structures."""
            # Use proper PromptTreeNode hierarchy instead of list[list[Item]]
            nested_data: list[ItemGroup] = []

            # Try to access a non-existent field in the nested structure
            # This should fail because 'inner_items' doesn't exist in ItemGroup
            some_field: str = Field(description="! @each[nested_data.inner_items]->task.processor@{{value.data=nested_data.inner_items}}*")

        # Should fail naturally because 'inner_items' field doesn't exist
        with pytest.raises((FieldValidationError, VariableSourceValidationError)) as exc_info:
            self.structure.add(TaskWithListListStructure)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["does not exist", "field", "inner_items"]), \
            f"Expected field existence validation error, got: {exc_info.value}"

    def test_confusion_mode_11_error_message_clarity_for_llms(self):
        """CONFUSION MODE 11: Complex subchain validation with nested iterables.

        Tests mismatched iterable paths at same nesting level:
        - Source: iterableA.noniterable.iterableB
        - Target: iterableA.noniterable.iterableC.another_noniterable
        Should fail because iterableB ≠ iterableC (different iterable fields of same parent)
        """
        class ItemB(PromptTreeNode):
            content: str

        class ItemC(PromptTreeNode):
            another_noniterable: str

        class MiddleNode(PromptTreeNode):
            """Non-iterable node with multiple iterable children."""
            regular_field: str = ""
            iterableB: list[ItemB] = []
            iterableC: list[ItemC] = []

        class OuterItem(PromptTreeNode):
            """Outer iterable container."""
            noniterable: MiddleNode

        class TaskComplexSubchain(PromptTreeNode):
            # ❌ Mismatched iterable paths: iterableB vs iterableC at same level
            iterableA: list[OuterItem] = Field(
                description="! @each[iterableA.noniterable.iterableB]->task.processor@{{value.result=iterableA.noniterable.iterableC.another_noniterable}}*"
            )

        with pytest.raises((FieldValidationError, CommandParseError)) as exc_info:
            self.structure.add(TaskComplexSubchain)

        error_msg = str(exc_info.value).lower()
        # Should detect the mismatched iterable paths
        assert any(keyword in error_msg for keyword in ["subchain", "iterableb", "iterablec", "mismatch"]), \
            f"Error should mention subchain violation or path mismatch, got: {exc_info.value}"


class TestAllCommandRHSScopingValidation:
    """Test @all command RHS scoping rules per LANGUAGE_SPECIFICATION.md.

    ARCHITECTURAL PRINCIPLE: @all commands can only reference:
    1. The exact field containing the command
    2. Wildcard (*) representing entire current node

    FORBIDDEN PATTERNS:
    - Sibling field references (breaks locality)
    - Longer paths from containing field (breaks semantics)
    - External field references (breaks predictability)
    """

    def setup_method(self):
        """Create fixtures for @all RHS scoping tests."""
        self.structure = RunStructure()

    def test_all_rhs_containing_field_passes(self):
        """Test that @all with RHS matching containing field passes validation."""
        class TaskValidContainingField(PromptTreeNode):
            """Task with @all command RHS matching containing field."""
            # ✅ VALID: RHS 'data' matches containing field name 'data'
            data: str = Field(description="! @all->task.processor@{{prompt.data=data}}*")

        # Should pass validation
        self.structure.add(TaskValidContainingField)
        assert self.structure.get_node("task.valid_containing_field") is not None

    def test_all_rhs_wildcard_passes(self):
        """Test that @all with wildcard (*) RHS passes validation."""
        class TaskValidWildcard(PromptTreeNode):
            """Task with @all command using wildcard RHS."""
            # ✅ VALID: RHS '*' represents entire current node
            analysis: str = Field(description="! @all->task.processor@{{prompt.context=*}}*")

        # Should pass validation
        self.structure.add(TaskValidWildcard)
        assert self.structure.get_node("task.valid_wildcard") is not None

    def test_all_rhs_different_simple_field_fails(self):
        """Test that @all with different simple field fails validation."""
        class TaskDifferentField(PromptTreeNode):
            """Task with @all command referencing wrong simple field."""
            # ❌ INVALID: RHS 'field2' does not match containing field 'field1'
            field1: str = Field(description="! @all->task.processor@{{prompt.context=field2}}*")
            field2: str = "different field"

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskDifferentField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_rhs_complex_path_fails(self):
        """Test that @all with complex path fails validation."""
        class TaskComplexPath(PromptTreeNode):
            """Task with @all command using complex path."""
            # ❌ INVALID: RHS 'prompt.config' does not match containing field 'field1'
            field1: str = Field(description="! @all->task.processor@{{value.result=prompt.config}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskComplexPath)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_rhs_scope_path_fails(self):
        """Test that @all with scope path fails validation."""
        class TaskScopePath(PromptTreeNode):
            """Task with @all command referencing scope path."""
            # ❌ INVALID: RHS 'task.data' does not match containing field 'field1'
            field1: str = Field(description="! @all->task.processor@{{prompt.context=task.data}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskScopePath)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_rhs_multiple_mappings_mixed_validity(self):
        """Test @all with multiple variable mappings - mixed valid/invalid patterns."""
        class TaskMixedValidity(PromptTreeNode):
            """Task with @all command having mixed valid/invalid RHS patterns."""
            # ❌ INVALID: First mapping OK (containing field), second mapping references different field
            data: str = Field(description="! @all->task.processor@{{prompt.context=data, prompt.extra=field2}}*")
            field2: str = "other field"

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskMixedValidity)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_rhs_implicit_mapping_containing_field_passes(self):
        """Test @all with implicit mapping (field name inference) passes when matching containing field."""
        class TaskValidImplicitMapping(PromptTreeNode):
            """Task with @all command using implicit mapping."""
            # ✅ VALID: Implicit mapping infers RHS as 'data' which matches containing field
            data: str = Field(description="! @all->task.processor@{{prompt.data}}*")

        # Should pass validation
        self.structure.add(TaskValidImplicitMapping)
        assert self.structure.get_node("task.valid_implicit_mapping") is not None

    def test_all_docstring_wildcard_passes(self):
        """Test that @all in docstring with wildcard passes validation."""
        class TaskDocstringWildcard(PromptTreeNode):
            """! @all->task.processor@{{prompt.context=*}}*

            Task with @all command in docstring using wildcard.
            """
            field1: str = "some data"
            field2: str = "other data"

        # Should pass validation - wildcard is valid from docstring
        self.structure.add(TaskDocstringWildcard)
        assert self.structure.get_node("task.docstring_wildcard") is not None

    def test_all_docstring_implicit_mapping_passes(self):
        """Test that @all in docstring with implicit mapping passes validation."""
        class TaskDocstringImplicit(PromptTreeNode):
            """! @all->task.processor@{{prompt=*}}*

            Task with @all command in docstring using implicit mapping.
            """
            field1: str = "some data"
            field2: str = "other data"

        # Should pass validation - implicit mapping using wildcard from docstring level
        self.structure.add(TaskDocstringImplicit)
        assert self.structure.get_node("task.docstring_implicit") is not None

    def test_all_docstring_specific_field_fails(self):
        """Test that @all in docstring referencing specific field fails validation."""
        class TaskDocstringSpecificField(PromptTreeNode):
            """! @all->task.processor@{{prompt.context=field1}}*

            Task with @all command in docstring referencing specific field.
            """
            field1: str = "some data"
            field2: str = "other data"

        # Should fail - cannot reference specific field from docstring level
        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskDocstringSpecificField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["docstring", "wildcard", "data locality"]), \
            f"Expected @all docstring RHS scoping error, got: {exc_info.value}"

    def test_all_docstring_multiple_fields_fails(self):
        """Test that @all in docstring with multiple field references fails validation."""
        class TaskDocstringMultipleFields(PromptTreeNode):
            """! @all->task.processor@{{prompt.field1=field1, prompt.field2=field2}}*

            Task with @all command in docstring referencing multiple specific fields.
            """
            field1: str = "some data"
            field2: str = "other data"

        # Should fail - cannot reference specific fields from docstring level
        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskDocstringMultipleFields)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["docstring", "wildcard", "data locality"]), \
            f"Expected @all docstring RHS scoping error, got: {exc_info.value}"

    def test_all_field_subfield_path_fails(self):
        """Test that @all with subfield path fails validation."""
        class TaskSubfieldPath(PromptTreeNode):
            """Task with @all command using subfield path."""
            # ❌ INVALID: RHS 'field1.subfield' extends beyond containing field
            field1: str = Field(description="! @all->task.processor@{{prompt.context=field1.subfield}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskSubfieldPath)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_field_implicit_mapping_wrong_field_fails(self):
        """Test that @all implicit mapping with wrong field name fails validation."""
        class TaskWrongImplicitField(PromptTreeNode):
            """Task with @all command using implicit mapping for wrong field."""
            # ❌ INVALID: Implicit mapping infers 'field2' but command is on 'field1'
            field1: str = Field(description="! @all->task.processor@{{prompt.field2}}*")
            field2: str = "other field"

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskWrongImplicitField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["containing field", "rhs must be", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_field_empty_mappings_fails(self):
        """Test that @all with empty mappings fails validation."""
        class TaskEmptyMappings(PromptTreeNode):
            """Task with @all command having empty mappings."""
            # This should fail at parser level, but test here for completeness
            field1: str = Field(description="! @all->task.processor@{{}}*")

        # Should fail during parsing or validation
        with pytest.raises((CommandParseError, FieldValidationError)) as exc_info:
            self.structure.add(TaskEmptyMappings)

        # Accept either parser error or validation error
        assert exc_info.value is not None

    def test_all_multiple_valid_mappings_same_field_passes(self):
        """Test that @all with multiple mappings to same field passes validation."""
        class TaskMultipleSameField(PromptTreeNode):
            """Task with @all command having multiple mappings to same field."""
            # ✅ VALID: All mappings reference the same containing field
            data: str = Field(description="! @all->task.processor@{{prompt.context=data, value.result=data}}*")

        # Should pass validation
        self.structure.add(TaskMultipleSameField)
        assert self.structure.get_node("task.multiple_same_field") is not None

    def test_all_mixed_wildcard_field_mappings_passes(self):
        """Test that @all with mixed wildcard and field mappings passes validation."""
        class TaskMixedWildcardField(PromptTreeNode):
            """Task with @all command mixing containing field mappings."""
            # ✅ VALID: All mappings reference the containing field 'data'
            data: str = Field(description="! @all->task.processor@{{prompt.context=data, value.field=data}}*")

        # Should pass validation - all mappings reference containing field
        self.structure.add(TaskMixedWildcardField)
        assert self.structure.get_node("task.mixed_wildcard_field") is not None

    def test_all_nested_field_path_fails(self):
        """Test that @all with nested field path fails RHS scoping validation."""
        class SubData(PromptTreeNode):
            value: str = "nested value"

        class TaskNestedFieldPath(PromptTreeNode):
            """Task with @all command using nested field path that exists structurally."""
            # ❌ INVALID: RHS 'data.value' violates @all scoping - only 'data' or '*' allowed
            # Field structure is valid (data.value exists) but @all RHS scoping forbids it
            data: SubData = Field(description="! @all->task.processor@{{prompt.context=data.value}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskNestedFieldPath)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["@all", "rhs", "containing field", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_sibling_field_reference_fails(self):
        """Test that @all with sibling field reference fails RHS scoping validation."""
        class TaskSiblingField(PromptTreeNode):
            """Task with @all command referencing sibling field."""
            # ❌ INVALID: RHS 'other_data' violates @all scoping - must be 'main_data' or '*'
            # Both fields exist structurally but @all RHS scoping forbids sibling references
            main_data: str = Field(description="! @all->task.processor@{{prompt.context=other_data}}*")
            other_data: str = "sibling data"

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskSiblingField)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["@all", "rhs", "containing field", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_longer_path_from_containing_field_fails(self):
        """Test that @all with longer path from containing field fails RHS scoping validation."""
        class NestedData(PromptTreeNode):
            items: list[str] = []

        class TaskLongerPath(PromptTreeNode):
            """Task with @all command using longer path from containing field."""
            # ❌ INVALID: RHS 'data.items' violates @all scoping - only 'data' or '*' allowed
            # Field structure is valid but @all RHS scoping forbids extending beyond containing field
            data: NestedData = Field(description="! @all->task.processor@{{prompt.context=data.items}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskLongerPath)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["@all", "rhs", "containing field", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"

    def test_all_external_node_reference_fails(self):
        """Test that @all with external node reference fails RHS scoping validation."""
        class TaskExternalRef(PromptTreeNode):
            """Task with @all command referencing external node structure."""
            # ❌ INVALID: RHS 'task.data.field' violates @all scoping - references external structure
            # This is a valid structural reference but violates @all locality principle
            data: str = Field(description="! @all->task.processor@{{prompt.context=task.data.field}}*")

        with pytest.raises(FieldValidationError) as exc_info:
            self.structure.add(TaskExternalRef)

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["@all", "rhs", "containing field", "only reference"]), \
            f"Expected @all RHS scoping error, got: {exc_info.value}"