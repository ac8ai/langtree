"""
Integration tests for get_prompt() chain resolution.

Tests that get_prompt() correctly:
1. Determines the current field (first ungenerated field based on previous_values)
2. Walks up the parent chain from current node to root
3. Embeds child prompts into parent PROMPT_SUBTREE
4. Includes COLLECTED_CONTEXT with previously generated fields

This replaces the old resolve_tree_bottom_up() tests with the actual production functionality.
"""

from pydantic import Field as PydanticField

from langtree.core.tree_node import TreeNode
from langtree.structure.builder import RunStructure


class TestGetPromptChain:
    """Test get_prompt() with full parent chain resolution."""

    def test_simple_root_node(self):
        """Test get_prompt() on a root Task node."""

        class TaskAnalysis(TreeNode):
            """Analyze the data:

            {PROMPT_SUBTREE}
            """

            result: str = PydanticField(description="Analysis result")

        structure = RunStructure()
        structure.add(TaskAnalysis)

        node = structure.get_node("task.analysis")
        prompt = node.get_prompt(previous_values={})

        assert "Analyze the data:" in prompt
        assert "Result" in prompt
        assert "Analysis result" in prompt

    def test_shows_only_first_ungenerated_field(self):
        """Test that PROMPT_SUBTREE shows only the first ungenerated field."""

        class TaskMultiStep(TreeNode):
            """Multi-step task:

            {COLLECTED_CONTEXT}

            {PROMPT_SUBTREE}
            """

            step1: str = PydanticField(description="First step")
            step2: str = PydanticField(description="Second step")
            step3: str = PydanticField(description="Third step")

        structure = RunStructure()
        structure.add(TaskMultiStep)

        node = structure.get_node("task.multi_step")

        # No previous_values: should show step1
        prompt = node.get_prompt(previous_values={})
        assert "Step 1" in prompt  # Field name gets converted to title case with spaces
        assert "First step" in prompt
        assert "Step 2" not in prompt
        assert "Second step" not in prompt

        # step1 completed: should show step2
        prompt = node.get_prompt(previous_values={"step1": "done"})
        # COLLECTED_CONTEXT should show step1 tag (within same node)
        assert "{task.multi_step.step1}" in prompt
        # PROMPT_SUBTREE shows step2 (next ungenerated field)
        assert "Step 2" in prompt
        assert "Second step" in prompt
        assert "Step 3" not in prompt

    def test_parent_child_full_chain(self):
        """Test get_prompt() walks up from child to parent."""

        class DataNode(TreeNode):
            """Process the data:

            {PROMPT_SUBTREE}
            """

            output: str = PydanticField(description="Processed output")

        class TaskPipeline(TreeNode):
            """Pipeline task:

            {COLLECTED_CONTEXT}

            {PROMPT_SUBTREE}
            """

            config: str = PydanticField(description="Configuration")
            processing: DataNode = PydanticField(description="Data processing step")

        structure = RunStructure()
        structure.add(TaskPipeline)

        # Get prompt for the child node
        child_node = structure.get_node("task.pipeline.processing")
        prompt = child_node.get_prompt(previous_values={})

        # Should show BOTH parent and child levels
        assert "Pipeline task:" in prompt
        assert "Process the data:" in prompt

        # Parent's COLLECTED_CONTEXT shows config tag
        assert "{task.pipeline.config}" in prompt

        # Parent's PROMPT_SUBTREE shows processing field
        assert "Data processing step" in prompt

        # Child's PROMPT_SUBTREE shows output
        assert "Output" in prompt
        assert "Processed output" in prompt

    def test_deep_three_level_chain(self):
        """Test 3-level nesting: Task → Middle → Leaf."""

        class LeafData(TreeNode):
            """Leaf computation:

            {PROMPT_SUBTREE}
            """

            value: str = PydanticField(description="Computed value")

        class MiddleProcess(TreeNode):
            """Middle processing:

            {COLLECTED_CONTEXT}

            {PROMPT_SUBTREE}
            """

            interim: str = PydanticField(description="Interim result")
            leaf: LeafData = PydanticField(description="Final computation")

        class TaskWorkflow(TreeNode):
            """Workflow:

            {COLLECTED_CONTEXT}

            {PROMPT_SUBTREE}
            """

            setup: str = PydanticField(description="Initial setup")
            middle: MiddleProcess = PydanticField(description="Processing phase")

        structure = RunStructure()
        structure.add(TaskWorkflow)

        # Get prompt for leaf node
        leaf = structure.get_node("task.workflow.middle.leaf")
        prompt = leaf.get_prompt(previous_values={})

        # All three levels present
        assert "Workflow:" in prompt
        assert "Middle processing:" in prompt
        assert "Leaf computation:" in prompt

        # Root COLLECTED_CONTEXT
        assert "{task.workflow.setup}" in prompt

        # Middle COLLECTED_CONTEXT
        assert "{task.workflow.middle.interim}" in prompt

        # Verify current field at each level
        assert "Processing phase" in prompt  # middle is current in root
        assert "Final computation" in prompt  # leaf is current in middle
        assert "Computed value" in prompt  # value is current in leaf

    def test_collected_context_expands_treenode_to_tags(self):
        """Test COLLECTED_CONTEXT expands nested TreeNode to leaf field tags."""

        class NestedStruct(TreeNode):
            """Nested structure."""

            field_a: str = PydanticField(description="Field A")
            field_b: str = PydanticField(description="Field B")

        class TaskWithNested(TreeNode):
            """Task:

            {COLLECTED_CONTEXT}

            {PROMPT_SUBTREE}
            """

            nested: NestedStruct = PydanticField(description="Nested data")
            final: str = PydanticField(description="Final output")

        structure = RunStructure()
        structure.add(TaskWithNested)

        node = structure.get_node("task.with_nested")

        # Generate final (nested already done)
        prompt = node.get_prompt(previous_values={"nested": "done"})

        # COLLECTED_CONTEXT should have FULL tags for nested fields
        assert "{task.with_nested.nested.field_a}" in prompt
        assert "{task.with_nested.nested.field_b}" in prompt

        # PROMPT_SUBTREE shows final
        assert "Final" in prompt
        assert "Final output" in prompt


class TestLegacyBehaviorReplacement:
    """Verify get_prompt() replaces resolve_tree_bottom_up() functionality."""

    def test_replaces_bottom_up_resolution(self):
        """Confirm get_prompt() is used instead of resolve_tree_bottom_up()."""
        # This is a meta-test to document the change
        # resolve_tree_bottom_up() is deprecated and should not be used
        # Production code uses node.get_prompt(previous_values={...})

        class TaskExample(TreeNode):
            """Example:

            {PROMPT_SUBTREE}
            """

            field: str = PydanticField(description="A field")

        structure = RunStructure()
        structure.add(TaskExample)

        node = structure.get_node("task.example")

        # This is the production API
        prompt = node.get_prompt(previous_values={})

        assert "Example:" in prompt
        assert "Field" in prompt
