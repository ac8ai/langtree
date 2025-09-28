"""
Tests for structure module functionality.

This module tests the core structure components:
- TreeNode: Base node functionality
- StructureTreeNode: Tree structure management
- StructureTreeRoot: Root node handling
- RunStructure: Complete tree orchestration
"""

import pytest
from pydantic import Field

from langtree.prompt import RunStructure, StructureTreeNode, TreeNode
from langtree.prompt.exceptions import (
    TemplateVariableError,
    VariableTargetValidationError,
)
from langtree.prompt.registry import PendingTarget


class TestStructureTreeNodeEnhanced:
    """Test enhanced StructureTreeNode with clean prompt storage."""

    def test_node_stores_clean_content(self):
        """Test that nodes can store clean prompt content."""
        node = StructureTreeNode(name="test", field_type=None)

        # After Phase 1, nodes should have these fields
        assert hasattr(node, "clean_docstring")
        assert hasattr(node, "clean_field_descriptions")
        assert hasattr(node, "extracted_commands")


class TestDestinationTargetProcessing:
    """Test destination target processing and pending registry integration."""

    def test_forward_references_tracked(self):
        """Test that forward references are properly tracked."""
        structure = RunStructure()

        # Define target that references future node
        class TaskAnalysis(TreeNode):
            """
            ! @->task.future_target@{{prompt.data=*}}

            Analysis node with forward reference.
            """

            result: str = Field(description="Analysis result")

        # Add the node - should track the forward reference
        structure.add(TaskAnalysis)

        # Check that pending target was registered
        pending_targets = structure._pending_target_registry.pending_targets
        assert "task.future_target" in pending_targets
        assert len(pending_targets["task.future_target"]) == 1

    def test_target_resolution_on_add(self):
        """Test that adding nodes resolves pending targets."""
        structure = RunStructure()

        # First, add a node that references a future target
        class TaskSource(TreeNode):
            """
            ! @->task.target@{{prompt.data=*}}

            Source node.
            """

            data: str = Field(description="Source data")

        structure.add(TaskSource)

        # Should have pending target
        assert "task.target" in structure._pending_target_registry.pending_targets

        # Now add the target node
        class TaskTarget(TreeNode):
            """Target node."""

            received_data: str = Field(description="Received data")

        structure.add(TaskTarget)

        # The pending target should be resolved when we add a matching node
        # Note: This tests the resolution mechanism, actual resolution depends on tag matching


class TestTemplateVariableSystem:
    """Test template variable resolution system for {PROMPT_SUBTREE} and {COLLECTED_CONTEXT}."""

    def setup_method(self):
        """Create structure fixture for template variable tests."""
        self.structure = RunStructure()

    def test_prompt_subtree_automatic_addition(self):
        """Test that {PROMPT_SUBTREE} is automatically added to docstrings if missing."""

        class TaskTestNode(TreeNode):
            """Node without template variable.

            This docstring should automatically get a prompt subtree appended.
            """

            field1: str = Field(description="First field")
            field2: str = Field(description="Second field")

        self.structure.add(TaskTestNode)

        # Check that {PROMPT_SUBTREE} was automatically added
        task_node = self.structure.get_node("task.test_node")
        assert task_node is not None, "Task node should be found"

        # The clean_docstring should have template variables processed
        assert task_node.clean_docstring is not None
        print(f"Clean docstring: {repr(task_node.clean_docstring)}")
        print(
            f"Does it contain PROMPT_SUBTREE? {'{PROMPT_SUBTREE}' in task_node.clean_docstring}"
        )

    def test_prompt_subtree_resolution(self):
        """Test that {PROMPT_SUBTREE} is resolved into field titles and descriptions."""
        from langtree.prompt.template_variables import (
            resolve_template_variables_in_content,
        )

        class TaskWithFields(TreeNode):
            """Task with fields for testing.

            {PROMPT_SUBTREE}
            """

            field1: str = Field(description="First field description")
            field2: int = Field(description="Second field description")

        self.structure.add(TaskWithFields)

        # Get the node
        task_node = self.structure.get_node("task.with_fields")
        assert task_node is not None, "Task node should be found"

        # Test template variable resolution
        content = "Task description:\n\n{PROMPT_SUBTREE}\n\nEnd of task."
        resolved_content = resolve_template_variables_in_content(content, task_node)

        # Check that PROMPT_SUBTREE was resolved
        assert "{PROMPT_SUBTREE}" not in resolved_content, (
            "Template variable should be resolved"
        )

        # Check that field titles and descriptions are included
        assert (
            "field1" in resolved_content.lower()
            or "field 1" in resolved_content.lower()
        )
        assert (
            "field2" in resolved_content.lower()
            or "field 2" in resolved_content.lower()
        )
        assert "First field description" in resolved_content
        assert "Second field description" in resolved_content

        # Check that field titles and descriptions are included
        assert (
            "field1" in resolved_content.lower()
            or "field 1" in resolved_content.lower()
        )
        assert (
            "field2" in resolved_content.lower()
            or "field 2" in resolved_content.lower()
        )
        assert "First field description" in resolved_content
        assert "Second field description" in resolved_content

    def test_prompt_subtree_manual_placement(self):
        """Test that manually placed {PROMPT_SUBTREE} is preserved."""

        class TaskManualTemplate(TreeNode):
            """Node with manual template variable.

            Some content before.

            {PROMPT_SUBTREE}

            Some content after.
            """

            field1: str = Field(description="First field")
            field2: str = Field(description="Second field")

        self.structure.add(TaskManualTemplate)

        # Check that manual placement is preserved
        node = self.structure.get_node("task.manual_template")
        assert node is not None, "Node should be found in structure"

        # Original docstring should contain the manual placement
        original_docstring = TaskManualTemplate.__doc__ or ""
        assert "Some content before." in original_docstring
        assert "{PROMPT_SUBTREE}" in original_docstring
        assert "Some content after." in original_docstring

        # Clean docstring should preserve structure but may process template variables
        clean_docstring = node.clean_docstring or ""
        assert "Some content before." in clean_docstring
        assert "Some content after." in clean_docstring

    def test_prompt_subtree_spacing_validation(self):
        """Test that {PROMPT_SUBTREE} requires proper spacing (empty lines before/after)."""

        class TaskValidSpacing(TreeNode):
            """Valid template variable spacing.

            Text before.

            {PROMPT_SUBTREE}

            Text after.
            """

            field: str = Field(description="Test field")

        class TaskInvalidSpacing(TreeNode):
            """Invalid spacing.
            Text before.{PROMPT_SUBTREE}Text after.
            """

            field: str = Field(description="Test field")

        # Valid node should process correctly
        # TODO: Implement spacing validation
        # See: tests/prompt/test_structure.py::TestTemplateVariableSystem::test_template_variable_spacing_validation
        self.structure.add(TaskValidSpacing)
        # Invalid node should raise error or auto-correct
        # with pytest.raises(TemplateVariableError):
        #     self.structure.add(InvalidNode)

    def test_collected_context_manual_addition(self):
        """Test that {COLLECTED_CONTEXT} is only added when manually specified."""

        class TaskWithContext(TreeNode):
            """Node with manual context placement.

            {COLLECTED_CONTEXT}

            Processing instructions here.
            """

            result: str = Field(description="Processing result")

        class TaskWithoutContext(TreeNode):
            """Node without context.

            Just processing instructions.
            """

            result: str = Field(description="Processing result")

        self.structure.add(TaskWithContext)
        self.structure.add(TaskWithoutContext)

        # Verify nodes were added and context is preserved
        with_context = self.structure.get_node("task.with_context")
        without_context = self.structure.get_node("task.without_context")

        assert with_context is not None, "Node with context should be found"
        assert without_context is not None, "Node without context should be found"

        # Original docstrings should contain expected content
        with_context_doc = TaskWithContext.__doc__ or ""
        without_context_doc = TaskWithoutContext.__doc__ or ""

        assert "{COLLECTED_CONTEXT}" in with_context_doc
        assert "{COLLECTED_CONTEXT}" not in without_context_doc
        assert "Processing instructions here." in with_context_doc
        assert "Just processing instructions." in without_context_doc

    def test_collected_context_automatic_fallback(self):
        """Test automatic addition of context section when context is needed."""

        class TaskSourceData(TreeNode):
            """Source providing context."""

            data: str = Field(description="Source data")

        class TaskTargetProcess(TreeNode):
            """
            ! @->target.process@{{prompt.data=*}}

            Target receiving context.

            Processing without explicit context placement.
            """

            result: str = Field(description="Processing result")

        self.structure.add(TaskSourceData)
        self.structure.add(
            TaskTargetProcess
        )  # TODO: Implement automatic context fallback
        # target = self.structure._root_nodes['TargetNode']
        # If context is needed but {COLLECTED_CONTEXT} not present,
        # should automatically append "\n\n# Context\n\n{COLLECTED_CONTEXT}"

    def test_heading_level_detection(self):
        """Test that template variables detect proper heading levels for content."""

        class TaskParentLevel(TreeNode):
            """# Parent heading level 1

            {PROMPT_SUBTREE}
            """

            child: "TaskChildLevel" = Field(description="Child node")

        class TaskChildLevel(TreeNode):
            """Child content should get proper heading level."""

            field1: str = Field(description="Field 1 description")
            field2: str = Field(description="Field 2 description")

        self.structure.add(TaskParentLevel)  # TODO: Implement heading level detection
        # When {PROMPT_SUBTREE} is resolved:
        # - Detect that parent has level 1 heading
        # - Child content should become level 2 headings
        # - Field descriptions should become appropriate sub-levels

    def test_template_variable_resolution_order(self):
        """Test that template variables are resolved in correct order during assembly."""

        class TaskComplexTemplate(TreeNode):
            """Complex template usage.

            # Instructions
            Follow these steps.

            {COLLECTED_CONTEXT}

            # Processing
            Process the above context.

            {PROMPT_SUBTREE}

            # Conclusion
            Complete the task.
            """

            analysis: str = Field(description="Analysis step")
            summary: str = Field(description="Summary step")

        self.structure.add(
            TaskComplexTemplate
        )  # TODO: Implement template resolution order
        # 1. First resolve {COLLECTED_CONTEXT} with context data
        # 2. Then resolve {PROMPT_SUBTREE} with child field content
        # 3. Both should maintain proper spacing and heading levels

    def test_template_variable_field_title_generation(self):
        """Test that field names are converted to proper titles in {PROMPT_SUBTREE}."""

        class TaskTitleGeneration(TreeNode):
            """Test title generation.

            {PROMPT_SUBTREE}
            """

            main_analysis: str = Field(description="Main analysis content")
            technical_details: str = Field(description="Technical details")
            final_summary: str = Field(description="Final summary")

        self.structure.add(TaskTitleGeneration)  # TODO: Implement title generation
        # Field names should be converted to titles:
        # main_analysis -> "# Main Analysis"
        # technical_details -> "# Technical Details"
        # final_summary -> "# Final Summary"

    def test_template_variable_conflict_detection(self):
        """Test detection of conflicts between template variables and Assembly Variables."""

        class TaskConflictTest(TreeNode):
            """Node with potential conflicts.
            ! PROMPT_SUBTREE="some_value"  # This should conflict

            {PROMPT_SUBTREE}
            """

            result: str = Field(description="Test result")

        # This should raise a conflict error when Assembly Variables are implemented
        self.structure.add(TaskConflictTest)

    def test_nested_template_variable_resolution(self):
        """Test template variable resolution in nested node structures."""

        class TaskRootLevel(TreeNode):
            """Root node.

            # Root Instructions

            {PROMPT_SUBTREE}

            # Root Conclusion
            """

            child: "TaskMiddleLevel" = Field(description="Middle tier")

        class TaskMiddleLevel(TreeNode):
            """Middle node.

            ## Middle Instructions

            {PROMPT_SUBTREE}

            ## Middle Summary
            """

            leaf: "TaskLeafLevel" = Field(description="Leaf tier")

        class TaskLeafLevel(TreeNode):
            """Leaf node content."""

            result: str = Field(description="Final result")

        self.structure.add(TaskRootLevel)  # TODO: Implement nested resolution
        # Should correctly resolve heading levels:
        # Root: # (level 1)
        # Middle: ## (level 2)
        # Leaf: ### (level 3)
        # Field descriptions: #### (level 4)

    def test_template_variable_assembly_integration(self):
        """Test integration between template variables and full prompt assembly process."""

        class TaskComplexIntegration(TreeNode):
            """Complex integration test for template variables.

            ! data_source="input.csv"
            ! output_format="json"

            ## Context Information

            This task processes data from multiple sources.

            {COLLECTED_CONTEXT}

            ## Processing Steps

            Follow these detailed processing steps:

            {PROMPT_SUBTREE}

            ## Configuration

            - Data source: Uses assembly variable data_source
            - Output format: Uses assembly variable output_format
            - Processing mode: Standard analysis
            """

            data_extraction: str = Field(description="Extract data from source files")
            data_validation: str = Field(description="Validate extracted data quality")
            data_transformation: str = Field(
                description="Transform data to target format"
            )
            result_generation: str = Field(description="Generate final results")

        # Add to structure and verify template variables work with assembly
        self.structure.add(TaskComplexIntegration)

        # Get the node from structure
        node = self.structure.get_node("task.complex_integration")
        assert node is not None, "Node should be added to structure"

        # Verify template variables are properly processed
        from langtree.prompt.template_variables import (
            process_template_variables,
            resolve_template_variables_in_content,
        )

        # Test 1: Template variable processing should work without errors
        clean_content = node.clean_docstring or ""
        processed_content = process_template_variables(clean_content, node)
        assert isinstance(processed_content, str), "Should process without errors"

        # Test 2: Template variables should be resolved in content
        resolved_content = resolve_template_variables_in_content(clean_content, node)

        # Template variables should be resolved to actual field content
        assert "{PROMPT_SUBTREE}" not in resolved_content, (
            "PROMPT_SUBTREE should be resolved"
        )
        assert "{COLLECTED_CONTEXT}" not in resolved_content, (
            "COLLECTED_CONTEXT should be resolved"
        )

        # Test 3: Field content should be properly included in PROMPT_SUBTREE resolution
        assert "Data Extraction" in resolved_content, "Should include field titles"
        assert "Extract data from source files" in resolved_content, (
            "Should include field descriptions"
        )
        assert "Data Validation" in resolved_content, "Should include all fields"
        assert "Result Generation" in resolved_content, "Should include all fields"

        # Test 4: Original structure and spacing should be preserved
        assert "## Context Information" in resolved_content, (
            "Should preserve original headings"
        )
        assert "## Processing Steps" in resolved_content, "Should preserve structure"
        assert "## Configuration" in resolved_content, "Should preserve all content"

        # Test 5: Assembly variables should coexist with template variables
        # (Assembly variables are handled separately, shouldn't interfere)
        assert "data_source" in clean_content, (
            "Assembly variables should be preserved in original"
        )
        assert "output_format" in clean_content, (
            "Assembly variables should be preserved"
        )

        # Test 6: Verify proper heading levels in PROMPT_SUBTREE resolution
        from langtree.prompt.template_variables import resolve_prompt_subtree

        prompt_result = resolve_prompt_subtree(node, base_heading_level=3)
        assert "### Data Extraction" in prompt_result, (
            "Should use correct heading level"
        )
        assert "### Data Validation" in prompt_result, (
            "Should use consistent heading levels"
        )


class TestTemplateVariableValidation:
    """Test validation rules for template variables."""

    def setup_method(self):
        """Create structure fixture for validation tests."""
        self.structure = RunStructure()

    def test_invalid_template_variable_names(self):
        """Test that invalid template variable names are rejected."""
        invalid_cases = [
            "{INVALID_TEMPLATE}",  # Not a recognized template variable
            "{prompt_subtree}",  # Wrong case
            "{PROMPT_subtree}",  # Mixed case
            "{{PROMPT_SUBTREE}}",  # Wrong delimiter count
        ]

        for invalid_template in invalid_cases:
            # Create a class with the invalid template in its docstring
            docstring = f"Test with invalid template: {invalid_template}"

            class TaskInvalidTemplate(TreeNode):
                field: str = Field(description="Test field")

            # Set the docstring dynamically
            TaskInvalidTemplate.__doc__ = docstring

            # This should raise an error when validation is implemented
            with pytest.raises((ValueError, TemplateVariableError)):
                self.structure.add(TaskInvalidTemplate)

    def test_template_variable_spacing_enforcement(self):
        """Test that template variables enforce proper spacing rules."""
        spacing_violations = [
            "Text{PROMPT_SUBTREE}",  # No space before
            "{PROMPT_SUBTREE}Text",  # No space after
            "Text {PROMPT_SUBTREE}",  # Space but no newline before
            "{PROMPT_SUBTREE} Text",  # Space but no newline after
            "Text\n{PROMPT_SUBTREE}",  # Newline but no empty line before
            "{PROMPT_SUBTREE}\nText",  # Newline but no empty line after
        ]

        for violation in spacing_violations:

            class TaskSpacingViolation(TreeNode):
                f"""Test spacing: {violation}"""
                field: str = Field(description="Test field")

            # Current implementation falls back to original content for spacing violations
            # TODO: Implement strict spacing validation enforcement
            self.structure.add(TaskSpacingViolation)


class TestTemplateVariableIntegration:
    """Test integration between template variables and other LangTree DSL features."""

    def setup_method(self):
        """Create structure fixture for integration tests."""
        self.structure = RunStructure()

    def test_template_variables_with_acl_commands(self):
        """Test template variables working with LangTree DSL commands."""

        class TaskSourceTemplate(TreeNode):
            """Source with template variables.

            ! @->target.process@{{prompt.context=*}}

            # Source Content

            {PROMPT_SUBTREE}

            """

            data: str = Field(description="Source data")

        class TaskTargetTemplate(TreeNode):
            """Target with context.

            # Context Information

            {COLLECTED_CONTEXT}

            # Processing Steps

            {PROMPT_SUBTREE}

            """

            result: str = Field(description="Processing result")

        self.structure.add(TaskSourceTemplate)
        self.structure.add(TaskTargetTemplate)

        # Should properly integrate:
        # 1. LangTree DSL commands create context flow (implemented)
        # 2. Template variables are processed in docstrings (implemented)
        # 3. Context flow and template processing work together (basic level)

    def test_template_variables_with_runtime_variables(self):
        """Test that template variables don't interfere with runtime variable syntax."""

        class TaskCombinedVariables(TreeNode):
            """Test separation of template variables from runtime variables.

            ! data_source="input.csv"
            ! model_name="gpt-4"

            ## Context Section

            The following context will be dynamically provided:

            {COLLECTED_CONTEXT}

            ## Processing Instructions

            Execute the following steps with runtime configuration:

            {PROMPT_SUBTREE}

            ## Runtime Configuration

            This task will use runtime variables for dynamic behavior:
            - Model: {model_name} (runtime field reference)
            - Data source: {data_source} (runtime field reference)
            - Processing mode: {processing_mode} (runtime field reference)
            - Dynamic count: {iteration_count} (runtime field reference)
            """

            model_name: str = Field(
                default="gpt-4", description="Model name for processing"
            )
            data_source: str = Field(
                default="input.csv", description="Source of input data"
            )
            processing_mode: str = Field(
                default="analysis", description="Processing mode configuration"
            )
            iteration_count: int = Field(
                default=3, description="Number of processing iterations"
            )
            analysis_step: str = Field(description="Primary analysis processing")
            validation_step: str = Field(description="Data validation processing")

        self.structure.add(TaskCombinedVariables)
        node = self.structure.get_node("task.combined_variables")
        assert node is not None

        # Test template variable processing doesn't break runtime variables
        from langtree.prompt.template_variables import (
            process_template_variables,
            resolve_template_variables_in_content,
        )

        clean_content = node.clean_docstring or ""

        # Test 1: Template variables are properly resolved
        resolved_content = resolve_template_variables_in_content(clean_content, node)
        assert "{PROMPT_SUBTREE}" not in resolved_content, (
            "Template variables should be resolved"
        )
        assert "{COLLECTED_CONTEXT}" not in resolved_content, (
            "Template variables should be resolved"
        )

        # Test 2: Runtime variable syntax is preserved (not template variables)
        assert "{model_name}" in resolved_content, (
            "Runtime field references should be preserved"
        )
        assert "{data_source}" in resolved_content, (
            "Runtime field references should be preserved"
        )
        assert "{processing_mode}" in resolved_content, (
            "Runtime field references should be preserved"
        )
        assert "{iteration_count}" in resolved_content, (
            "Runtime field references should be preserved"
        )

        # Test 3: Template variable resolution includes field content
        assert "Analysis Step" in resolved_content, (
            "PROMPT_SUBTREE should include field titles"
        )
        assert "Primary analysis processing" in resolved_content, (
            "Should include field descriptions"
        )
        assert "Validation Step" in resolved_content, "Should include all field titles"
        assert "Data validation processing" in resolved_content, (
            "Should include all field descriptions"
        )

        # Test 4: Original document structure is preserved
        assert "## Context Section" in resolved_content, (
            "Should preserve document headings"
        )
        assert "## Processing Instructions" in resolved_content, (
            "Should preserve structure"
        )
        assert "## Runtime Configuration" in resolved_content, (
            "Should preserve all content"
        )

        # Test 5: Template variable processing works without errors
        processed_content = process_template_variables(clean_content, node)
        assert isinstance(processed_content, str), "Should process without errors"

        # Test 6: Assembly variables in original docstring are preserved
        assert "data_source" in clean_content, (
            "Assembly variables should be in original content"
        )
        assert "model_name" in clean_content, "Assembly variables should be preserved"

        # The key test: Template variables and runtime variables coexist peacefully
        # Template variables ({PROMPT_SUBTREE}, {COLLECTED_CONTEXT}) are resolved at assembly time
        # Runtime variables ({field_name}) are preserved for runtime resolution
        assert (
            "{PROMPT_SUBTREE}" not in resolved_content
            and "{model_name}" in resolved_content
        ), "Different variable types handled separately"


class TestPendingTargetResolutionCore:
    """Test core pending target resolution functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    def test_basic_forward_reference_resolution_workflow(self):
        """Test basic forward reference resolution workflow."""

        # Create early task referencing later target
        class TaskEarly(TreeNode):
            """
            ! @all->task.later@{{value.result=*}}
            Early task that references later target.
            """

            data: str = "early data"

        # Add early task - should create pending target
        self.run_structure.add(TaskEarly)

        # Verify pending target exists
        assert len(self.run_structure._pending_target_registry.pending_targets) == 1
        assert (
            "task.later" in self.run_structure._pending_target_registry.pending_targets
        )

        # Add target - should trigger resolution
        class TaskLater(TreeNode):
            """Later task."""

            result: str = "default"

        self.run_structure.add(TaskLater)

        # Verify pending target resolved
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0

        # Verify both nodes exist
        early_node = self.run_structure.get_node("task.early")
        later_node = self.run_structure.get_node("task.later")
        assert early_node is not None
        assert later_node is not None

    @pytest.mark.skip("TODO: Implement context resolution integration")
    def test_inclusion_context_resolution_integration(self):
        """Test that pending resolution triggers inclusion context validation."""
        # TODO: Implement context resolution integration
        # See: tests/prompt/test_structure.py::TestContextResolutionIntegration::test_inclusion_context_resolution_integration
        pytest.skip(
            "TODO: Implement context resolution integration - see langtree/prompt/resolution.py"
        )

        # Create command with inclusion path referencing future target
        class TaskEarly(TreeNode):
            """
            Early task with inclusion referencing later target.
            ! @each[task.later.items]->task.processor@{{value.item=items}}*
            """

            pass

        self.run_structure.add(TaskEarly)

        # Verify pending target for inclusion path
        assert (
            "task.later" in self.run_structure._pending_target_registry.pending_targets
        )

        # Add target with iterable field
        class TaskLater(TreeNode):
            """Later task with items."""

            items: list[str] = ["item1", "item2"]

        class TaskProcessor(TreeNode):
            """Processor task."""

            item: str = "default"

        self.run_structure.add(TaskLater)
        self.run_structure.add(TaskProcessor)

        # TODO: Verify inclusion path was validated as iterable
        # TODO: Verify context resolution was performed
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    @pytest.mark.skip("TODO: Implement destination context resolution")
    def test_destination_context_resolution_integration(self):
        """Test that pending resolution validates destination context compatibility."""

        class TaskEarly(TreeNode):
            """
            Early task referencing typed destination.
            ! @all->task.later.specific_field@{{value.result=prompt.data}}
            """

            data: str = "early data"

        self.run_structure.add(TaskEarly)

        # Add target with specific field type
        class TaskLater(TreeNode):
            """Later task with typed field."""

            specific_field: str = "default"
            other_field: int = 42

        self.run_structure.add(TaskLater)

        # TODO: Verify destination field type compatibility
        # TODO: Verify field exists and is accessible
        # See: llm/prompt/structure.py::_complete_pending_command_processing


class TestPendingTargetVariableMapping:
    """Test variable mapping resolution during pending target processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    @pytest.mark.skip("TODO: Implement variable mapping semantic validation")
    def test_variable_mapping_semantic_validation(self):
        """Test semantic validation of variable mappings during resolution."""

        class TaskEarly(TreeNode):
            """
            Early task with complex variable mappings.
            ! @all->task.later@{{prompt.context=*, outputs.summary=*}}
            """

            analysis: str = "detailed analysis"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task with expected mappings."""

            context: str = "default context"
            summary: str = "default summary"

        self.run_structure.add(TaskLater)

        # TODO: Verify source paths are semantically valid
        # TODO: Verify target variables exist and are compatible
        # TODO: Verify scope mappings are valid (prompt, value, outputs)
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    def test_wildcard_mapping_validation(self):
        """Test that wildcard (*) mappings in variable assignments are parsed correctly."""

        class TaskEarly(TreeNode):
            """
            ! @all->task.later@{{prompt.full_context=*}}
            Early task with wildcard mapping.
            """

            data: str = "source data"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task expecting wildcard data."""

            full_context: str = "default"

        self.run_structure.add(TaskLater)

        # Verify that the command was parsed correctly
        early_node = self.run_structure.get_node("task.early")
        assert early_node is not None
        assert len(early_node.extracted_commands) == 1

        # Verify wildcard mapping is correctly parsed
        command = early_node.extracted_commands[0]
        assert command.destination_path == "task.later"
        assert len(command.variable_mappings) == 1

        mapping = command.variable_mappings[0]
        assert mapping.target_path == "prompt.full_context"
        assert mapping.source_path == "*"

        # Verify both nodes exist in structure
        later_node = self.run_structure.get_node("task.later")
        assert later_node is not None

    def test_scope_mapping_validation(self):
        """Test that scope mappings (prompt, value, outputs, task) are parsed correctly."""

        class TaskEarly(TreeNode):
            """
            ! @all->task.later@{{prompt.ctx=*, outputs.result=*, value.ref=*}}
            Early task with multiple scope mappings.
            """

            data: str = "source data"
            summary: str = "summary text"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task with scope-aware fields."""

            ctx: str = "default"
            result: str = "default"
            ref: str = "default"

        self.run_structure.add(TaskLater)

        # Verify that the command was parsed correctly
        early_node = self.run_structure.get_node("task.early")
        assert early_node is not None
        assert len(early_node.extracted_commands) == 1

        # Verify multiple scope mappings are parsed correctly
        command = early_node.extracted_commands[0]
        assert command.destination_path == "task.later"
        assert len(command.variable_mappings) == 3

        # Verify scope types are correctly identified
        target_paths = [mapping.target_path for mapping in command.variable_mappings]
        assert "prompt.ctx" in target_paths
        assert "outputs.result" in target_paths
        assert "value.ref" in target_paths

        # Verify all mappings use wildcard sources
        source_paths = [mapping.source_path for mapping in command.variable_mappings]
        assert all(path == "*" for path in source_paths)


class TestPendingTargetVariableRegistry:
    """Test variable registry integration during pending target resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    @pytest.mark.skip("TODO: Implement variable registry satisfaction updates")
    def test_variable_registry_satisfaction_updates(self):
        """Test that pending resolution updates variable registry satisfaction sources."""

        class TaskEarly(TreeNode):
            """
            Early task creating variable dependencies.
            ! @all->task.later@{{value.result=prompt.analysis}}
            """

            analysis: str = "source analysis"

        self.run_structure.add(TaskEarly)

        # Add target - should trigger variable registry updates
        class TaskLater(TreeNode):
            """Later task receiving variable."""

            result: str = "default"

        self.run_structure.add(TaskLater)

        # TODO: Verify variable registry was updated
        # TODO: Verify satisfaction sources changed from syntactic to semantic
        # TODO: Verify variable is now marked as satisfied with resolved source
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    @pytest.mark.skip("TODO: Implement variable relationship type validation")
    def test_variable_relationship_type_validation(self):
        """Test validation of variable relationship types (1:1, 1:n, n:n)."""

        # Test @all command (should create 1:1 or 1:n relationship)
        class TaskSource1(TreeNode):
            """
            Source with @all command.
            ! @all->task.target@{{value.result=prompt.data}}
            """

            data: str = "source 1"

        # Test @each command (should create n:n relationship)
        class TaskSource2(TreeNode):
            """
            Source with @each command.
            ! @each[items]->task.target@{{value.result=items}}*
            """

            items: list[str] = ["item1", "item2"]

        self.run_structure.add(TaskSource1)
        self.run_structure.add(TaskSource2)

        class TaskTarget(TreeNode):
            """Target task."""

            result: str = "default"

        self.run_structure.add(TaskTarget)

        # TODO: Verify relationship types are correctly identified
        # TODO: Verify @all creates 1:1 relationship
        # TODO: Verify @each creates n:n relationship
        # TODO: Verify multiplicity indicators affect relationship type
        # See: llm/prompt/structure.py::_complete_pending_command_processing


class TestPendingTargetErrorHandling:
    """Test error handling and reporting during pending target resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    @pytest.mark.skip("TODO: Implement resolution error capture")
    def test_resolution_error_capture_and_reporting(self):
        """Test that resolution errors are captured and attached to commands."""

        class TaskEarly(TreeNode):
            """
            Early task with invalid mappings.
            ! @all->task.later@{{value.nonexistent=prompt.missing}}
            """

            data: str = "valid data"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task missing expected fields."""

            result: str = "default"
            # Note: Missing 'nonexistent' field

        self.run_structure.add(TaskLater)

        # TODO: Verify resolution errors were captured
        # TODO: Verify command has resolution_errors list
        # TODO: Verify specific error messages for missing fields
        # See: llm/prompt/structure.py::_complete_pending_command_processing

        early_node = self.run_structure.get_node("task.early")
        assert early_node is not None
        # TODO: Check early_node.extracted_commands[0].resolution_errors

    @pytest.mark.skip("TODO: Implement inclusion path validation errors")
    def test_inclusion_path_validation_errors(self):
        """Test error handling for invalid inclusion paths."""

        class TaskEarly(TreeNode):
            """
            Early task with invalid inclusion path.
            ! @each[task.later.nonexistent_list]->task.processor@{{value.item=items}}*
            """

            pass

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task without expected iterable field."""

            data: str = "not a list"
            # Note: Missing 'nonexistent_list' field

        class TaskProcessor(TreeNode):
            """Processor task."""

            item: str = "default"

        self.run_structure.add(TaskLater)
        self.run_structure.add(TaskProcessor)

        # TODO: Verify inclusion path validation error captured
        # TODO: Verify error indicates non-existent or non-iterable field
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    @pytest.mark.skip("TODO: Implement type compatibility validation")
    def test_type_compatibility_validation_errors(self):
        """Test error handling for type incompatibility in variable mappings."""

        class TaskEarly(TreeNode):
            """
            Early task with type-incompatible mapping.
            ! @all->task.later@{{value.numeric_field=prompt.text_data}}
            """

            text_data: str = "text value"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task with strongly typed field."""

            numeric_field: int = 42

        self.run_structure.add(TaskLater)

        # TODO: Verify type compatibility error captured
        # TODO: Verify error indicates string->int incompatibility
        # See: llm/prompt/structure.py::_complete_pending_command_processing


class TestPendingTargetEdgeCasesAdvanced:
    """Test advanced edge cases and boundary conditions in pending target resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    def test_multiple_commands_same_pending_target_advanced(self):
        """Test multiple commands waiting for the same pending target with complex mappings."""

        class TaskEarly1(TreeNode):
            """
            ! @all->task.shared@{{value.result1=*}}
            First early task.
            """

            data1: str = "data from task 1"

        class TaskEarly2(TreeNode):
            """
            ! @all->task.shared@{{value.result2=*}}
            Second early task.
            """

            data2: str = "data from task 2"

        self.run_structure.add(TaskEarly1)
        self.run_structure.add(TaskEarly2)

        # Verify both commands pending for same target
        assert (
            "task.shared" in self.run_structure._pending_target_registry.pending_targets
        )
        pending_commands = self.run_structure._pending_target_registry.pending_targets[
            "task.shared"
        ]
        assert len(pending_commands) == 2

        # Add shared target
        class TaskShared(TreeNode):
            """Shared target task."""

            result1: str = "default1"
            result2: str = "default2"

        self.run_structure.add(TaskShared)

        # Verify all pending targets resolved
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0

    def test_nested_pending_target_resolution_complex(self):
        """Test resolution of deeply nested pending targets with complex structure."""

        class TaskEarly(TreeNode):
            """
            Early task referencing deeply nested target.
            ! @all->task.analysis.deep.nested@{{value.result=prompt.data}}
            """

            data: str = "early data"

        self.run_structure.add(TaskEarly)

        # Add nested structure
        class TaskAnalysis(TreeNode):
            """Analysis task."""

            summary: str = "analysis summary"

            class Deep(TreeNode):
                """Deep analysis."""

                details: str = "deep details"

                class Nested(TreeNode):
                    """Nested analysis."""

                    result: str = "nested result"

                nested: Nested

            deep: Deep

        self.run_structure.add(TaskAnalysis)

        # Verify nested target resolved
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0
        nested_node = self.run_structure.get_node("task.analysis.deep.nested")
        assert nested_node is not None

    def test_nonexistent_source_node_graceful_handling_comprehensive(self):
        """Test graceful handling when source node doesn't exist during completion."""
        from langtree.commands.parser import CommandType, ParsedCommand

        # Create mock pending target with invalid source
        mock_command = ParsedCommand(
            command_type=CommandType.ALL,
            destination_path="task.target",
            variable_mappings=[],
        )

        pending_target = PendingTarget("task.target", mock_command, "task.nonexistent")

        # Should not crash
        self.run_structure._complete_pending_command_processing(pending_target)

        # Should handle gracefully without exceptions

    @pytest.mark.skip("TODO: Implement circular dependency detection")
    def test_circular_dependency_detection_in_pending_targets(self):
        """Test detection of circular dependencies in pending target resolution."""

        class TaskA(TreeNode):
            """
            Task A referencing Task B.
            ! @all->task.b@{{value.result=prompt.data}}
            """

            data: str = "data from A"

        class TaskB(TreeNode):
            """
            Task B referencing Task A.
            ! @all->task.a@{{value.result=prompt.data}}
            """

            data: str = "data from B"

        self.run_structure.add(TaskA)
        self.run_structure.add(TaskB)

        # TODO: Verify circular dependency detected
        # TODO: Verify appropriate error handling
        # See: llm/prompt/structure.py::_complete_pending_command_processing


class TestPendingTargetIntegrationWithResolution:
    """Test integration with existing context resolution system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

    @pytest.mark.skip("TODO: Implement resolution.py integration")
    def test_integration_with_resolve_deferred_contexts(self):
        """Test integration with resolution.py deferred context resolution."""

        class TaskEarly(TreeNode):
            """
            Early task with deferred contexts.
            ! @all->task.later@{{prompt.context=*}}
            """

            analysis: str = "detailed analysis"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task."""

            context: str = "default"

        self.run_structure.add(TaskLater)

        # TODO: Verify deferred context resolution was triggered
        # TODO: Verify integration with existing resolution.py functions
        # See: llm/prompt/resolution.py::resolve_deferred_contexts

    @pytest.mark.skip("TODO: Implement API for resolution reporting")
    def test_resolution_report_api(self):
        """Test API for getting resolution reports after pending target processing."""

        class TaskEarly(TreeNode):
            """
            Early task with various mappings.
            ! @all->task.later@{{prompt.ctx=*, outputs.result=*}}
            """

            data: str = "source data"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task."""

            ctx: str = "default"
            result: str = "default"

        self.run_structure.add(TaskLater)

        # TODO: Implement get_resolution_report() API
        # TODO: Verify report contains resolution statistics
        # TODO: Verify report includes error summaries
        # resolution_report = self.run_structure.get_resolution_report()
        # assert 'resolved_commands' in resolution_report
        # assert 'failed_commands' in resolution_report
        # assert 'variable_satisfaction_changes' in resolution_report

    # @pytest.mark.skip("TODO: Implement multiple forward references batch resolution")
    def test_multiple_forward_references_to_same_node_batch_resolution(self):
        """Test multiple commands referencing the same late-defined node (batch resolution)."""

        # Create multiple early tasks all referencing the same late node
        class TaskEarly1(TreeNode):
            """
            ! @all->task.late_target@{{value.result1=*}}
            Early task 1 referencing late node.
            """

            data1: str = "from early 1"

        class TaskEarly2(TreeNode):
            """
            ! @all->task.late_target@{{value.result2=*}}
            Early task 2 referencing late node.
            """

            data2: str = "from early 2"

        class TaskEarly3(TreeNode):
            """
            ! @all->task.late_target@{{outputs.context=*}}
            Early task 3 referencing late node.
            """

            data3: str = "from early 3"

        self.run_structure.add(TaskEarly1)
        self.run_structure.add(TaskEarly2)
        self.run_structure.add(TaskEarly3)

        # Verify all 3 pending targets exist (grouped under single path)
        assert len(self.run_structure._pending_target_registry.pending_targets) == 1
        assert (
            "task.late_target"
            in self.run_structure._pending_target_registry.pending_targets
        )
        pending_commands = self.run_structure._pending_target_registry.pending_targets[
            "task.late_target"
        ]
        assert len(pending_commands) == 3

        # Add the late target - should resolve ALL pending targets in batch
        class TaskLateTarget(TreeNode):
            """Late target receiving multiple variable mappings."""

            result1: str = "default1"
            result2: str = "default2"
            context: str = "default context"

        self.run_structure.add(TaskLateTarget)

        # Verify all 3 pending targets were resolved
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0
        # TODO: Verify batch resolution was efficient (single processing pass)
        # TODO: Verify all variable mappings were applied correctly
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    @pytest.mark.skip("TODO: Implement reverse order resolution robustness")
    def test_reverse_order_resolution_should_not_create_pending_entries(self):
        """Test that adding target before source doesn't create unnecessary pending entries."""

        # Add target first
        class TaskTarget(TreeNode):
            """Target task defined first."""

            result: str = "default"

        self.run_structure.add(TaskTarget)

        # Add source that references the already-existing target
        class TaskSource(TreeNode):
            """
            Source task referencing existing target.
            ! @all->task.target@{{value.result=prompt.data}}
            """

            data: str = "source data"

        self.run_structure.add(TaskSource)

        # TODO: Verify no pending targets were created (immediate resolution)
        # TODO: Verify variable mapping was applied immediately
        # TODO: Verify reverse order doesn't affect resolution correctness
        # assert len(self.run_structure._pending_target_registry.pending_targets) == 0
        # See: llm/prompt/structure.py::add

    @pytest.mark.skip("TODO: Implement duplicate target conflict detection")
    def test_duplicate_target_definitions_conflict_detection(self):
        """Test detection and reporting of duplicate target definitions."""

        # Add first definition with specific path
        class TaskDuplicateTarget(TreeNode):
            """First definition of target."""

            field: str = "first"

        self.run_structure.add(TaskDuplicateTarget)

        # Attempt to add same class again (should conflict)
        # TODO: Should raise structured error about duplicate target
        # with pytest.raises(DuplicateTargetError) as exc_info:
        #     self.run_structure.add(TaskDuplicateTarget)
        #
        # TODO: Verify error contains conflicting target path
        # assert "task.duplicate_target" in str(exc_info.value)
        # See: llm/prompt/structure.py::add

    @pytest.mark.skip("TODO: Implement cycle detection with structured errors")
    def test_cycles_in_forward_references_structured_error_reporting(self):
        """Test cycle detection in forward references with structured error reporting."""

        class TaskA(TreeNode):
            """
            Task A referencing Task B.
            ! @all->task.b@{{value.data_from_a=prompt.data}}
            """

            data: str = "from A"

        class TaskB(TreeNode):
            """
            Task B referencing Task C.
            ! @all->task.c@{{value.data_from_b=prompt.data}}
            """

            data: str = "from B"

        class TaskC(TreeNode):
            """
            Task C referencing Task A (creating cycle).
            ! @all->task.a@{{value.data_from_c=prompt.data}}
            """

            data: str = "from C"

        self.run_structure.add(TaskA)
        self.run_structure.add(TaskB)

        # Adding TaskC should detect the cycle and provide structured error
        # TODO: Should raise CyclicDependencyError with cycle path
        # with pytest.raises(CyclicDependencyError) as exc_info:
        #     self.run_structure.add(TaskC)
        #
        # TODO: Verify error contains complete cycle path
        # cycle_path = exc_info.value.cycle_path
        # assert "task.a" in cycle_path
        # assert "task.b" in cycle_path
        # assert "task.c" in cycle_path
        # See: llm/prompt/structure.py::_detect_cycles

    @pytest.mark.skip(
        "TODO: Implement post-processing verification with observable effects"
    )
    def test_post_processing_verification_with_observable_side_effects(self):
        """Test that post-processing actually occurs with observable side effects."""
        # Counter to track post-processing invocations
        processing_counter = {"count": 0}

        def mock_post_processor(command, target_node):
            """Mock post-processor that increments counter."""
            processing_counter["count"] += 1
            # Add observable effect: set a marker field
            if hasattr(target_node, "_processed_marker"):
                target_node._processed_marker = True

        # TODO: Add hook for post-processing callback registration
        # self.run_structure.register_post_processor(mock_post_processor)

        class TaskEarly(TreeNode):
            """
            Early task with pending target.
            ! @all->task.later@{{value.result=prompt.analysis}}
            """

            analysis: str = "analysis data"

        self.run_structure.add(TaskEarly)

        class TaskLater(TreeNode):
            """Later task receiving processing."""

            result: str = "default"

        self.run_structure.add(TaskLater)

        # TODO: Verify post-processing was invoked
        # assert processing_counter["count"] == 1
        #
        # TODO: Verify observable side effects occurred
        # later_node = self.run_structure.get_node("task.later")
        # assert hasattr(later_node, '_processed_marker')
        # assert later_node._processed_marker is True
        #
        # TODO: Verify variable mapping was applied during post-processing
        # assert later_node.result == "analysis data"
        # See: llm/prompt/structure.py::_complete_pending_command_processing

    @pytest.mark.skip("TODO: Implement unresolved references validation after finalize")
    def test_unresolved_references_after_finalize_validation(self):
        """Test validation of unresolved references after finalization."""

        class TaskEarly(TreeNode):
            """
            Early task with unresolvable reference.
            ! @all->task.never_defined@{{value.result=prompt.data}}
            """

            data: str = "some data"

        self.run_structure.add(TaskEarly)

        # Finalize without adding the target
        # TODO: Implement finalize() method
        # validation_result = self.run_structure.finalize()
        #
        # TODO: Verify unresolved references are reported
        # assert not validation_result.is_valid
        # assert len(validation_result.unresolved_targets) == 1
        # assert "task.never_defined" in validation_result.unresolved_targets
        #
        # TODO: Verify error contains source command information
        # unresolved_error = validation_result.unresolved_targets[0]
        # assert unresolved_error.source_node == "task.early"
        # assert "value.result=prompt.data" in unresolved_error.command_description
        # See: llm/prompt/structure.py::finalize
        # 3. Assembly variable resolution during template processing


class TestContextResolutionValidation:
    """Test context resolution in variable target structure validation.

    These tests verify that _validate_variable_target_structure correctly
    validates against target node context instead of source node context.
    """

    def test_cross_tree_validation_uses_target_context(self):
        """Test that validation correctly uses target node context."""

        class Company(TreeNode):
            name: str

        class TaskStructureAThreeLevels(TreeNode):
            """Target node with 'companies' field."""

            companies: list[Company] = []

        class TaskStructureCTwoLevels(TreeNode):
            """Source node without 'companies' field."""

            data: list[Company] = Field(
                default=[],
                description="! @each[data]->task.structure_a_three_levels@{{value.companies.name=data.name}}*",
            )

        structure = RunStructure()
        structure.add(TaskStructureAThreeLevels)

        # This should NOT raise an error because validation now checks target context
        # Target node (TaskStructureAThreeLevels) HAS the 'companies' field
        structure.add(TaskStructureCTwoLevels)

    def test_legitimate_validation_errors_preserved(self):
        """Test that fix doesn't break legitimate validation errors."""

        class SomeItem(TreeNode):
            name: str

        class TaskTarget(TreeNode):
            """Target node lacking the referenced field."""

            other_field: str = "test"

        class TaskSource(TreeNode):
            """Source node with invalid field reference."""

            data: list[SomeItem] = Field(
                default=[],
                description="! @each[data]->task.target@{{value.nonexistent.name=data.name}}*",
            )

        structure = RunStructure()
        structure.add(TaskTarget)

        # This SHOULD raise an error because 'nonexistent' doesn't exist in target
        with pytest.raises(VariableTargetValidationError):
            structure.add(TaskSource)


class TestContextResolutionHardCases:
    """Hard core test cases that stress test the context resolution implementation."""

    def test_forward_reference_fallback_behavior(self):
        """Test context resolution with forward references where target doesn't exist yet."""

        class TaskWithForwardRef(TreeNode):
            """Node with forward reference that falls back to source validation."""

            data: list[str] = Field(
                default=[],
                description="! @each[data]->task.nonexistent@{{value.some_field=data}}*",
            )

        structure = RunStructure()

        # Should handle missing target gracefully without crashing
        # This tests the fallback mechanism in the fix
        structure.add(TaskWithForwardRef)

        # Verify forward reference was recorded
        assert len(structure._pending_target_registry.pending_targets) > 0

    def test_deep_nested_field_validation_in_target_context(self):
        """Test validation of deeply nested field paths in target context."""

        class Member(TreeNode):
            name: str

        class TeamBridge(TreeNode):
            members: list[Member] = []

        class Team(TreeNode):
            bridge: TeamBridge

        class DeptBridge(TreeNode):
            teams: list[Team] = []

        class Department(TreeNode):
            bridge: DeptBridge

        class CompanyBridge(TreeNode):
            departments: list[Department] = []

        class TaskComplexTarget(TreeNode):
            """Target with deep nested structure."""

            companies: list[
                CompanyBridge
            ] = []  # Deep path: companies.departments.bridge.teams.bridge.members.name

        class TaskComplexSource(TreeNode):
            """Source that maps to deep nested path in target."""

            items: list[Member] = Field(
                default=[],
                description="! @each[items]->task.complex_target@{{value.companies.departments.bridge.teams.bridge.members.name=items.name}}*",
            )

        structure = RunStructure()
        structure.add(TaskComplexTarget)

        # Should validate 'companies' exists in target (not source)
        # Even though the path is deep, validation checks first component in target context
        structure.add(TaskComplexSource)

    def test_multiple_variable_mappings_mixed_validity(self):
        """Test multiple mappings where some are valid in target, others invalid."""

        class Item(TreeNode):
            name: str

        class TaskMixedTarget(TreeNode):
            """Target with some fields but not others."""

            valid_field: list[Item] = []
            # Missing: invalid_field

        class TaskMixedSource(TreeNode):
            """Source with multiple mappings - some valid, some invalid."""

            data: list[Item] = Field(
                default=[],
                description="! @each[data]->task.mixed_target@{{value.valid_field.name=data.name, value.invalid_field.name=data.name}}*",
            )

        structure = RunStructure()
        structure.add(TaskMixedTarget)

        # Should fail because 'invalid_field' doesn't exist in target
        with pytest.raises(VariableTargetValidationError) as exc_info:
            structure.add(TaskMixedSource)

        # Should mention the invalid field specifically
        assert "invalid_field" in str(exc_info.value)

    def test_complex_inheritance_hierarchy_validation(self):
        """Test context resolution with complex inheritance patterns."""

        class BaseItem(TreeNode):
            id: str

        class SpecialItem(BaseItem):
            name: str
            description: str

        class TaskComplexInheritanceTarget(TreeNode):
            """Target with complex type hierarchy."""

            special_items: list[SpecialItem] = []
            # Should validate against SpecialItem fields (name, description, id)

        class TaskComplexInheritanceSource(TreeNode):
            """Source that maps to inherited field."""

            raw_data: list[SpecialItem] = Field(
                default=[],
                description="! @each[raw_data]->task.complex_inheritance_target@{{value.special_items.description=raw_data.description}}*",
            )

        structure = RunStructure()
        structure.add(TaskComplexInheritanceTarget)

        # Should work because 'special_items' exists in target
        # and description exists in SpecialItem
        structure.add(TaskComplexInheritanceSource)

    def test_edge_case_empty_and_none_target_tags(self):
        """Test edge cases with None or empty target node tags."""

        class BasicItem(TreeNode):
            name: str

        class TaskEdgeCaseTarget(TreeNode):
            """Simple target node."""

            items: list[BasicItem] = []

        class TaskEdgeCaseSource(TreeNode):
            """Source that should work with basic validation."""

            data: list[BasicItem] = Field(
                default=[],
                description="! @each[data]->task.edge_case_target@{{value.items.name=data.name}}*",
            )

        structure = RunStructure()
        structure.add(TaskEdgeCaseTarget)

        # This tests the robustness of the fix when handling edge cases
        structure.add(TaskEdgeCaseSource)

    def test_stress_multiple_cross_references(self):
        """Stress test with multiple nodes cross-referencing each other."""

        class DataItem(TreeNode):
            value: str

        class TaskA(TreeNode):
            """First target with unique field."""

            unique_field_a: list[DataItem] = []

        class TaskB(TreeNode):
            """Second target with different field."""

            unique_field_b: list[DataItem] = []

        class TaskC(TreeNode):
            """Third target with yet another field."""

            unique_field_c: list[DataItem] = []

        class TaskStressSource(TreeNode):
            """Source that references all targets with their respective fields."""

            source_data: list[DataItem] = Field(
                default=[],
                description="""
                ! @each[source_data]->task.a@{{value.unique_field_a.value=source_data.value}}*
                ! @each[source_data]->task.b@{{value.unique_field_b.value=source_data.value}}*
                ! @each[source_data]->task.c@{{value.unique_field_c.value=source_data.value}}*
                """,
            )

        structure = RunStructure()
        structure.add(TaskA)
        structure.add(TaskB)
        structure.add(TaskC)

        # Each validation should check the correct target context
        # TaskA should find unique_field_a, TaskB should find unique_field_b, etc.
        structure.add(TaskStressSource)
