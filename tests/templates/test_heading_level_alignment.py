"""
Tests for heading level alignment in prompt generation.

These tests document the expected behavior for heading levels when:
1. Template variables ({PROMPT_SUBTREE}, {COLLECTED_CONTEXT}) are resolved
2. Child node prompts are embedded into parent prompts
3. Nested TreeNode fields appear in COLLECTED_CONTEXT

The rules:
- Template variables should be at the level one below the preceding heading
- When embedding child content, preserve relative hierarchy within the child
- All child content should be subordinate to the parent's level where it's embedded
"""

from pydantic import Field as PydanticField

from langtree.core.tree_node import TreeNode
from langtree.structure.builder import RunStructure


class TestTemplateVariableLevelDetection:
    """Test that template variables get assigned the correct level based on context."""

    def test_template_variable_after_heading(self):
        """Template variable should be one level below preceding heading."""

        class TaskSimple(TreeNode):
            """Task description.

            ## Section Title
            Some text here.

            {PROMPT_SUBTREE}
            """

            field: str = PydanticField(description="A field")

        structure = RunStructure()
        structure.add(TaskSimple)

        node = structure.get_node("task.simple")
        prompt = node.get_prompt(previous_values={})

        # Section Title is level 2 (##)
        # So PROMPT_SUBTREE content should start at level 3 (###)
        assert "## Section Title" in prompt
        assert "### Field" in prompt  # Field title should be level 3

    def test_template_variable_without_preceding_heading(self):
        """Template variable without preceding heading should use base level."""

        class TaskNoHeading(TreeNode):
            """Task description without headings.

            {PROMPT_SUBTREE}
            """

            field: str = PydanticField(description="A field")

        structure = RunStructure()
        structure.add(TaskNoHeading)

        node = structure.get_node("task.no_heading")
        prompt = node.get_prompt(previous_values={})

        # No preceding heading, so base_level = 1
        # Field title should be level 1
        assert "# Field" in prompt

    def test_template_variable_level_inheritance(self):
        """Content after a heading inherits that heading's level + 1."""

        class TaskMultiSection(TreeNode):
            """Task with multiple sections.

            ## First Section
            Text in first section.

            {COLLECTED_CONTEXT}

            ## Second Section
            Text in second section.

            {PROMPT_SUBTREE}
            """

            field1: str = PydanticField(description="First field")
            field2: str = PydanticField(description="Second field")

        structure = RunStructure()
        structure.add(TaskMultiSection)

        node = structure.get_node("task.multi_section")

        # Generate second field (first is in previous_values)
        prompt = node.get_prompt(previous_values={"field1": "done"})

        # COLLECTED_CONTEXT is after "## First Section" so level 3
        assert "## First Section" in prompt
        assert "{task.multi_section.field1}" in prompt  # Tag in COLLECTED_CONTEXT

        # PROMPT_SUBTREE is after "## Second Section" so level 3
        assert "## Second Section" in prompt
        assert "### Field 2" in prompt  # Field title should be level 3


class TestChildContentEmbedding:
    """Test heading level alignment when child node content is embedded in parent."""

    def test_child_embedded_preserves_hierarchy(self):
        """When child content is embedded, its internal hierarchy should be preserved."""

        class ChildNode(TreeNode):
            """Child prompt.

            ## Child Section
            Child section text.

            {PROMPT_SUBTREE}
            """

            data: str = PydanticField(description="Child data")

        class TaskParent(TreeNode):
            """Parent prompt.

            ## Parent Section
            Parent section text.

            {PROMPT_SUBTREE}
            """

            child: ChildNode = PydanticField(description="Nested child")

        structure = RunStructure()
        structure.add(TaskParent)

        # Get prompt for child (which includes parent chain)
        child_node = structure.get_node("task.parent.child")
        prompt = child_node.get_prompt(previous_values={})

        # Expected structure:
        # Parent prompt.                    (text)
        # ## Parent Section                 (level 2)
        # Parent section text.              (text)
        # ### Child                         (level 3 - field title in parent's PROMPT_SUBTREE)
        # Nested child                      (text - field description)
        # Child prompt.                     (text - child's docstring)
        # #### Child Section                (level 4 - child's heading, shifted from level 2)
        # Child section text.               (text)
        # ##### Data                        (level 5 - child's field in child's PROMPT_SUBTREE)

        assert "## Parent Section" in prompt
        assert "### Child" in prompt
        assert "#### Child Section" in prompt  # Child's level 2 heading becomes level 4
        assert "##### Data" in prompt  # Child's level 3 content becomes level 5

    def test_deep_nesting_three_levels(self):
        """Test 3-level nesting maintains proper hierarchy."""

        class LeafNode(TreeNode):
            """Leaf prompt.

            {PROMPT_SUBTREE}
            """

            value: str = PydanticField(description="Leaf value")

        class MiddleNode(TreeNode):
            """Middle prompt.

            ## Middle Section
            Middle text.

            {PROMPT_SUBTREE}
            """

            leaf: LeafNode = PydanticField(description="Leaf node")

        class TaskRoot(TreeNode):
            """Root prompt.

            ## Root Section
            Root text.

            {PROMPT_SUBTREE}
            """

            middle: MiddleNode = PydanticField(description="Middle node")

        structure = RunStructure()
        structure.add(TaskRoot)

        # Get prompt for leaf (includes full chain)
        leaf_node = structure.get_node("task.root.middle.leaf")
        prompt = leaf_node.get_prompt(previous_values={})

        # Expected structure:
        # Root prompt.                      (text)
        # ## Root Section                   (level 2)
        # Root text.                        (text)
        # ### Middle                        (level 3 - middle field title)
        # Middle node                       (text - middle field description)
        # Middle prompt.                    (text - middle's docstring)
        # #### Middle Section               (level 4 - middle's heading shifted from 2)
        # Middle text.                      (text)
        # ##### Leaf                        (level 5 - leaf field title in middle's PROMPT_SUBTREE)
        # Leaf node                         (text - leaf field description)
        # Leaf prompt.                      (text - leaf's docstring)
        # ###### Value                      (level 6 - leaf's field in leaf's PROMPT_SUBTREE)

        assert "## Root Section" in prompt
        assert "### Middle" in prompt
        assert "#### Middle Section" in prompt
        assert "##### Leaf" in prompt
        assert "###### Value" in prompt


class TestCollectedContextNesting:
    """Test heading level alignment for nested TreeNode fields in COLLECTED_CONTEXT."""

    def test_nested_treenode_in_collected_context(self):
        """Nested TreeNode fields in COLLECTED_CONTEXT should show proper hierarchy."""

        class NestedData(TreeNode):
            """Nested data structure."""

            field_a: str = PydanticField(description="Field A")
            field_b: str = PydanticField(description="Field B")

        class TaskWithNested(TreeNode):
            """Task prompt.

            ## Context Section

            {COLLECTED_CONTEXT}

            ## Output Section

            {PROMPT_SUBTREE}
            """

            nested: NestedData = PydanticField(description="Nested data")
            result: str = PydanticField(description="Final result")

        structure = RunStructure()
        structure.add(TaskWithNested)

        node = structure.get_node("task.with_nested")

        # Generate result field (nested already completed)
        prompt = node.get_prompt(previous_values={"nested": "done"})

        # COLLECTED_CONTEXT is after "## Context Section" (level 2), so content is level 3
        # Nested TreeNode should show:
        # ### Nested                        (level 3 - field title)
        # #### Field A                      (level 4 - nested field, one below Nested)
        # {task.with_nested.nested.field_a} (tag)
        # #### Field B                      (level 4)
        # {task.with_nested.nested.field_b} (tag)

        assert "## Context Section" in prompt
        assert "### Nested" in prompt
        assert "#### Field A" in prompt
        assert "{task.with_nested.nested.field_a}" in prompt
        assert "#### Field B" in prompt
        assert "{task.with_nested.nested.field_b}" in prompt

        # PROMPT_SUBTREE content
        assert "## Output Section" in prompt
        assert "### Result" in prompt


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_headings_same_level(self):
        """Multiple headings at same level should each set content level correctly."""

        class TaskMultiHeadings(TreeNode):
            """Task with multiple same-level headings.

            ## Section A
            Text A.

            ## Section B
            Text B.

            {PROMPT_SUBTREE}
            """

            field: str = PydanticField(description="A field")

        structure = RunStructure()
        structure.add(TaskMultiHeadings)

        node = structure.get_node("task.multi_headings")
        prompt = node.get_prompt(previous_values={})

        assert "## Section A" in prompt
        assert "## Section B" in prompt
        assert "### Field" in prompt  # Should be level 3 (one below last ## heading)

    def test_varying_heading_depths(self):
        """Template variable level should follow the most recent heading."""

        class TaskVaryingDepth(TreeNode):
            """Task with varying heading depths.

            ## Level 2
            Text.

            ### Level 3
            More text.

            {PROMPT_SUBTREE}
            """

            field: str = PydanticField(description="A field")

        structure = RunStructure()
        structure.add(TaskVaryingDepth)

        node = structure.get_node("task.varying_depth")
        prompt = node.get_prompt(previous_values={})

        assert "## Level 2" in prompt
        assert "### Level 3" in prompt
        assert "#### Field" in prompt  # Level 4 (one below ### Level 3)


class TestCurrentBehaviorDocumentation:
    """Document current behavior before fixes (will update after fixes)."""

    def test_current_issue_child_heading_not_shifted(self):
        """
        CURRENT ISSUE: Child node's headings from docstring are not being shifted.

        When a child node has headings in its docstring, those headings should be
        shifted to be subordinate to where the child is embedded in the parent.
        """

        class ChildWithHeading(TreeNode):
            """Child docstring.

            ## Child Heading
            Child text.

            {PROMPT_SUBTREE}
            """

            data: str = PydanticField(description="Child data")

        class TaskParentSimple(TreeNode):
            """Parent docstring.

            ## Parent Heading

            {PROMPT_SUBTREE}
            """

            child: ChildWithHeading = PydanticField(description="Child field")

        structure = RunStructure()
        structure.add(TaskParentSimple)

        child_node = structure.get_node("task.parent_simple.child")
        prompt = child_node.get_prompt(previous_values={})

        # Expected structure (NOW WORKING!):
        # ## Parent Heading          (level 2)
        # ### Child                  (level 3 - field title)
        # Child field                (text)
        # Child docstring.           (text)
        # #### Child Heading         (level 4 - shifted from 2)
        # Child text.                (text)
        # ##### Data                 (level 5 - shifted from 3)

        assert "## Parent Heading" in prompt
        assert "### Child" in prompt
        assert "#### Child Heading" in prompt  # ✅ Fixed! Shifted from level 2 to 4
        assert "##### Data" in prompt  # ✅ Shifted from level 3 to 5
