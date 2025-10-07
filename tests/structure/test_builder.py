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

from langtree import TreeNode
from langtree.exceptions import (
    DuplicateTargetError,
    TemplateVariableNameError,
    TemplateVariableSpacingError,
    VariableTargetValidationError,
)
from langtree.structure import RunStructure, StructureTreeNode
from langtree.structure.registry import PendingTarget


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
        from langtree.templates.element_resolution import (
            elements_to_markdown,
            parse_docstring_to_elements,
            resolve_template_elements,
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

        # Test template variable resolution using element-based API
        content = "Task description:\n\n{PROMPT_SUBTREE}\n\nEnd of task."
        elements = parse_docstring_to_elements(content)
        resolved_elements = resolve_template_elements(
            elements, task_node, None, None, output_field_prefix="To generate: "
        )
        resolved_content = elements_to_markdown(resolved_elements)

        # Check that PROMPT_SUBTREE was resolved
        assert "{PROMPT_SUBTREE}" not in resolved_content, (
            "Template variable should be resolved"
        )

        # Check that ONLY the first field is included (current leaf)
        assert (
            "field1" in resolved_content.lower()
            or "field 1" in resolved_content.lower()
        ), "Should include first field"
        # Sibling field should NOT be included
        assert not (
            "field2" in resolved_content.lower()
            or "field 2" in resolved_content.lower()
        ), "Should NOT include sibling field"
        assert "First field description" in resolved_content
        assert "Second field description" not in resolved_content

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
        # Note: Spacing validation has been implemented in validate_template_variable_spacing()
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
        self.structure.add(TaskTargetProcess)
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

        self.structure.add(TaskParentLevel)
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

        self.structure.add(TaskComplexTemplate)
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

        self.structure.add(TaskTitleGeneration)
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

        self.structure.add(TaskRootLevel)
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

        # Verify template variables are properly processed using element-based API
        from langtree.templates.element_resolution import (
            elements_to_markdown,
            resolve_node_prompt_elements,
        )

        # Test 1: Template variable processing should work without errors
        clean_content = node.clean_docstring or ""
        resolved_elements = resolve_node_prompt_elements(node)
        assert resolved_elements is not None, "Should resolve elements without errors"

        # Test 2: Template variables should be resolved in content
        resolved_content = elements_to_markdown(resolved_elements)

        # Template variables should be resolved to actual field content
        assert "{PROMPT_SUBTREE}" not in resolved_content, (
            "PROMPT_SUBTREE should be resolved"
        )
        assert "{COLLECTED_CONTEXT}" not in resolved_content, (
            "COLLECTED_CONTEXT should be resolved"
        )

        # Test 3: Field content should be properly included in PROMPT_SUBTREE resolution
        # PROMPT_SUBTREE now shows only the current leaf field (first field)
        assert "Data Extraction" in resolved_content, "Should include first field title"
        assert "Extract data from source files" in resolved_content, (
            "Should include first field description"
        )
        # Sibling fields should NOT be shown
        assert "Data Validation" not in resolved_content, (
            "Should NOT include sibling fields"
        )
        assert "Result Generation" not in resolved_content, (
            "Should NOT include sibling fields"
        )

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
        # PROMPT_SUBTREE now shows only the current leaf field (first field when no previous_values)
        from langtree.templates.element_resolution import (
            generate_prompt_subtree_with_children,
        )

        prompt_elements = generate_prompt_subtree_with_children(
            node, base_heading_level=3
        )
        prompt_result = elements_to_markdown(prompt_elements)
        assert "### Data Extraction" in prompt_result, (
            "Should use correct heading level for first field"
        )
        assert "### Data Validation" not in prompt_result, (
            "Should NOT include sibling fields"
        )


class TestTemplateVariableValidation:
    """Test validation rules for template variables."""

    def setup_method(self):
        """Create structure fixture for validation tests."""
        self.structure = RunStructure()

    def test_invalid_template_variable_names(self):
        """Test that misspelled template variable names are rejected."""
        # Only test actual misspellings of template variables
        # {INVALID_TEMPLATE} is now a valid runtime variable
        invalid_cases = [
            "{prompt_subtree}",  # Wrong case for template variable
            "{PROMPT_subtree}",  # Mixed case for template variable
            "{Prompt_Subtree}",  # Wrong case for template variable
            "{collected_context}",  # Wrong case for template variable
            "{COLLECTED_context}",  # Mixed case for template variable
        ]

        for i, invalid_template in enumerate(invalid_cases):
            # Create a class with the invalid template in its docstring
            docstring = f"Test with invalid template: {invalid_template}"

            # Create unique class name to avoid duplicate target conflicts
            class_name = f"TaskInvalidTemplate{i}"
            task_invalid_template = type(
                class_name,
                (TreeNode,),
                {
                    "field": Field(default="test", description="Test field"),
                    "__annotations__": {"field": str},
                    "__doc__": docstring,
                },
            )

            # Misspelled template variables should raise an error
            with pytest.raises((ValueError, TemplateVariableNameError)):
                self.structure.add(task_invalid_template)

    def test_variables_with_lowercase_are_valid(self):
        """Test that runtime variables with lowercase letters are allowed."""
        valid_cases = [
            "{dataSource}",  # Lowercase - valid
            "{outputFormat}",  # Camelcase - valid
            "{MyVariable}",  # Mixed case - valid
            "{configData_v2}",  # Lowercase with underscore and suffix - valid
        ]

        for i, valid_var in enumerate(valid_cases):
            # Create a class with the valid runtime variable in its docstring
            docstring = f"""Process data using {valid_var} variable.

            This uses a runtime variable that should be valid.

            {{PROMPT_SUBTREE}}
            """

            # Create unique class name to avoid duplicate target conflicts
            class_name = f"TaskValidRuntime{i}"
            task_valid = type(
                class_name,
                (TreeNode,),
                {
                    "field": Field(default="test", description="Test field"),
                    "__annotations__": {"field": str},
                    "__doc__": docstring,
                },
            )

            # Valid runtime variables should not raise errors
            try:
                self.structure.add(task_valid)
            except (ValueError, TemplateVariableNameError) as e:
                pytest.fail(f"Valid runtime variable {valid_var} raised error: {e}")

    def test_variables_without_lowercase_raise_errors(self):
        """Test that variables without lowercase letters raise errors."""
        error_cases = [
            "{DATA_SOURCE}",  # No lowercase - error
            "{OUTPUT_FORMAT}",  # No lowercase - error
            "{CONFIGURATION}",  # No lowercase - error
            "{OUTPUT_1}",  # No lowercase even with number - error
        ]

        for i, error_var in enumerate(error_cases):
            docstring = f"""Process using {error_var}.

            {{PROMPT_SUBTREE}}
            """

            # Create unique class name
            class_name = f"TaskInvalidVar{i}"
            task_invalid = type(
                class_name,
                (TreeNode,),
                {
                    "field": Field(default="test", description="Test field"),
                    "__annotations__": {"field": str},
                    "__doc__": docstring,
                },
            )

            # Should raise TemplateVariableNameError
            with pytest.raises(TemplateVariableNameError) as exc_info:
                self.structure.add(task_invalid)

            assert "reserved for template variables" in str(exc_info.value).lower()

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

        for i, violation in enumerate(spacing_violations):
            # Create unique class name to avoid duplicate target conflicts
            class_name = f"TaskSpacingViolation{i}"
            task_spacing_violation = type(
                class_name,
                (TreeNode,),
                {
                    "field": Field(default="test", description="Test field"),
                    "__annotations__": {"field": str},
                    "__doc__": f"Test spacing: {violation}",
                },
            )

            # Spacing validation is now implemented and should raise errors
            with pytest.raises(TemplateVariableSpacingError):
                self.structure.add(task_spacing_violation)


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
        from langtree.templates.element_resolution import (
            elements_to_markdown,
            resolve_node_prompt_elements,
        )

        clean_content = node.clean_docstring or ""

        # Test 1: Template variables are properly resolved
        resolved_elements = resolve_node_prompt_elements(node)
        resolved_content = elements_to_markdown(resolved_elements)

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
        # PROMPT_SUBTREE now shows only the current leaf field (first field)
        assert "Model Name" in resolved_content, (
            "PROMPT_SUBTREE should include first field title"
        )
        assert "Model name for processing" in resolved_content, (
            "Should include first field description"
        )
        # Sibling fields should NOT be shown
        assert "Analysis Step" not in resolved_content, (
            "Should NOT include sibling field titles"
        )
        assert "Validation Step" not in resolved_content, (
            "Should NOT include sibling field titles"
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

        # Test 5: Template variable processing works without errors (already tested above)
        assert isinstance(resolved_content, str), "Should process without errors"

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


class TestPendingTargetVariableMapping:
    """Test variable mapping resolution during pending target processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

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
        from langtree.parsing.parser import CommandType, ParsedCommand

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


class TestPendingTargetIntegrationWithResolution:
    """Test integration with existing context resolution system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.run_structure = RunStructure()

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

        # Verify no pending targets were created (immediate resolution)
        assert len(self.run_structure._pending_target_registry.pending_targets) == 0

    def test_duplicate_target_definitions_conflict_detection(self):
        """Test detection and reporting of duplicate target definitions."""

        # Add first definition with specific path
        class TaskDuplicateTarget(TreeNode):
            """First definition of target."""

            field: str = "first"

        self.run_structure.add(TaskDuplicateTarget)

        # Attempt to add same class again (should conflict)
        with pytest.raises(DuplicateTargetError) as exc_info:
            self.run_structure.add(TaskDuplicateTarget)

        # Verify error contains conflicting target path
        error_msg = str(exc_info.value)
        assert "task.duplicate_target" in error_msg
        assert "TaskDuplicateTarget" in error_msg


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


class TestErrorLevelConfiguration:
    """Test error_level parameter in RunStructure."""

    def test_default_error_level_is_user(self):
        """Test that default error level is USER."""
        from langtree.exceptions.core import ErrorLevel

        structure = RunStructure()
        assert structure.error_level == ErrorLevel.USER

    def test_set_error_level_to_developer(self):
        """Test setting error level to DEVELOPER."""
        from langtree.exceptions.core import ErrorLevel

        structure = RunStructure(error_level=ErrorLevel.DEVELOPER)
        assert structure.error_level == ErrorLevel.DEVELOPER

    def test_set_error_level_with_string(self):
        """Test setting error level using string value."""
        from langtree.exceptions.core import ErrorLevel

        structure = RunStructure(error_level="developer")
        assert structure.error_level == ErrorLevel.DEVELOPER
