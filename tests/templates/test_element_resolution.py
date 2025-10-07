"""
Tests for element-based prompt resolution.

Tests the proper handling of PromptElement lists and bottom-up resolution.
"""

from pydantic import Field

from langtree.core.tree_node import TreeNode
from langtree.templates.element_resolution import (
    adjust_element_levels,
    elements_to_markdown,
    parse_docstring_to_elements,
    resolve_collected_context_elements,
    resolve_node_prompt_elements,
    resolve_prompt_subtree_elements,
    resolve_template_elements,
)
from langtree.templates.prompt_structure import (
    PromptTemplate,
    PromptText,
    PromptTitle,
)


class MockNode:
    """Mock node for testing."""

    def __init__(
        self, field_name=None, field_type=None, parent=None, clean_docstring=""
    ):
        self.field_name = field_name
        self.field_type = field_type
        self.parent = parent
        self.children = {}
        self.clean_field_descriptions = {}
        self.clean_docstring = clean_docstring


class TestParseDocstringToElements:
    """Test parsing docstrings into PromptElement lists."""

    def test_parse_simple_text(self):
        """Test parsing plain text."""
        content = "This is a simple docstring."
        elements = parse_docstring_to_elements(content)

        assert len(elements) == 1
        assert isinstance(elements[0], PromptText)
        assert elements[0].content == "This is a simple docstring."
        assert elements[0].level == 1

    def test_parse_with_heading(self):
        """Test parsing with markdown headings."""
        content = [
            "# Main Title",
            "Some text under the title.",
            "",
            "## Subtitle",
            "More text here.",
        ]
        elements = parse_docstring_to_elements("\n".join(content))

        assert len(elements) == 4
        assert isinstance(elements[0], PromptTitle)
        assert elements[0].content == "Main Title"
        assert elements[0].level == 1

        assert isinstance(elements[1], PromptText)
        assert elements[1].content == "Some text under the title."

        assert isinstance(elements[2], PromptTitle)
        assert elements[2].content == "Subtitle"
        assert elements[2].level == 2

        assert isinstance(elements[3], PromptText)
        assert elements[3].content == "More text here."

    def test_parse_prompt_subtree_template(self):
        """Test parsing PROMPT_SUBTREE template variable."""
        content = [
            "Generate the following:",
            "",
            "{PROMPT_SUBTREE}",
            "",
            "End of prompt.",
        ]
        elements = parse_docstring_to_elements("\n".join(content))

        assert len(elements) == 3
        assert isinstance(elements[0], PromptText)
        assert elements[0].content == "Generate the following:"

        assert isinstance(elements[1], PromptTemplate)
        assert elements[1].variable_name == "PROMPT_SUBTREE"
        assert elements[1].resolved_content is None

        assert isinstance(elements[2], PromptText)
        assert elements[2].content == "End of prompt."

    def test_parse_collected_context_template(self):
        """Test parsing COLLECTED_CONTEXT template variable."""
        content = ["Previous context:", "{COLLECTED_CONTEXT}", "", "Now continue."]
        elements = parse_docstring_to_elements("\n".join(content))

        assert len(elements) == 3
        assert isinstance(elements[0], PromptText)
        assert isinstance(elements[1], PromptTemplate)
        assert elements[1].variable_name == "COLLECTED_CONTEXT"
        assert isinstance(elements[2], PromptText)

    def test_parse_both_templates(self):
        """Test parsing both template variables."""
        content = ["Context:", "{COLLECTED_CONTEXT}", "", "Tasks:", "{PROMPT_SUBTREE}"]
        elements = parse_docstring_to_elements("\n".join(content))

        assert len(elements) == 4
        assert elements[1].variable_name == "COLLECTED_CONTEXT"
        assert elements[3].variable_name == "PROMPT_SUBTREE"


class TestResolvePromptSubtreeElements:
    """Test PROMPT_SUBTREE element generation."""

    def test_empty_node(self):
        """Test with empty node."""
        node = MockNode()
        elements = resolve_prompt_subtree_elements(node)
        assert elements == []

    def test_simple_fields(self):
        """Test with simple string fields."""

        class TestNode(TreeNode):
            name: str = Field(description="The name")
            value: str = Field(description="The value")

        node = MockNode(field_type=TestNode)
        node.clean_field_descriptions = {
            "name": "Clean name description",
            "value": "Clean value description",
        }

        elements = resolve_prompt_subtree_elements(node)

        # Should have title and description for each field
        assert len(elements) == 4

        assert isinstance(elements[0], PromptTitle)
        assert elements[0].content == "Name"

        assert isinstance(elements[1], PromptText)
        assert elements[1].content == "Clean name description"

        assert isinstance(elements[2], PromptTitle)
        assert elements[2].content == "Value"

        assert isinstance(elements[3], PromptText)
        assert elements[3].content == "Clean value description"

    def test_nested_treenode_field(self):
        """Test with nested TreeNode field."""

        class ChildNode(TreeNode):
            data: str = Field(description="Child data")

        class ParentNode(TreeNode):
            title: str = Field(description="Title field")
            child: ChildNode = Field(description="Nested child node")

        node = MockNode(field_type=ParentNode)
        node.clean_field_descriptions = {
            "title": "Title description",
            "child": "Child node description",
        }

        elements = resolve_prompt_subtree_elements(node)

        # Should include PROMPT_SUBTREE template for nested field
        assert len(elements) == 5

        # Title field
        assert elements[0].content == "Title"
        assert elements[1].content == "Title description"

        # Child field
        assert elements[2].content == "Child"
        assert elements[3].content == "Child node description"

        # PROMPT_SUBTREE placeholder for nested field
        assert isinstance(elements[4], PromptTemplate)
        assert elements[4].variable_name == "PROMPT_SUBTREE"
        assert elements[4].level == 2  # Subordinate to parent


class TestResolveCollectedContextElements:
    """Test COLLECTED_CONTEXT element generation."""

    def test_empty_previous_values(self):
        """Test with no previous values."""
        node = MockNode(field_name="current")
        elements = resolve_collected_context_elements(node)
        assert elements == []

    def test_no_parent(self):
        """Test node without parent - should show field history for task-level nodes."""
        # Task-level node (no field_name) should show previous values
        node = MockNode()  # No field_name means task-level node
        node.field_type = type("MockType", (), {"model_fields": {"other": None}})()
        elements = resolve_collected_context_elements(node, {"other": "value"})
        # Should now show the 'other' field from previous_values
        assert len(elements) == 2  # Title + text
        assert any("Other" in str(e.content) for e in elements if hasattr(e, "content"))

    def test_simple_sibling_values(self):
        """Test with simple sibling values."""
        parent = MockNode()
        current = MockNode(field_name="current", parent=parent)
        sibling1 = MockNode(field_name="sibling1", parent=parent)
        sibling2 = MockNode(field_name="sibling2", parent=parent)

        parent.children = {
            "sibling1": sibling1,
            "sibling2": sibling2,
            "current": current,
        }

        previous_values = {
            "sibling1": "First value",
            "sibling2": "Second value",
        }

        elements = resolve_collected_context_elements(current, previous_values)

        assert len(elements) == 4

        assert isinstance(elements[0], PromptTitle)
        assert elements[0].content == "Sibling 1"

        assert isinstance(elements[1], PromptText)
        assert elements[1].content == "{task.unknown.sibling1}"

        assert isinstance(elements[2], PromptTitle)
        assert elements[2].content == "Sibling 2"

        assert isinstance(elements[3], PromptText)
        assert elements[3].content == "{task.unknown.sibling2}"

    def test_dict_value_formatting(self):
        """Test formatting of dict values."""
        parent = MockNode()
        current = MockNode(field_name="current", parent=parent)
        sibling = MockNode(field_name="data", parent=parent)

        parent.children = {"data": sibling, "current": current}

        previous_values = {
            "data": {
                "key1": "value1",
                "key2": "value2",
            }
        }

        elements = resolve_collected_context_elements(current, previous_values)

        # Should have title and FULL tag reference (not actual dict values)
        assert len(elements) == 2
        assert elements[0].content == "Data"
        assert elements[1].content == "{task.unknown.data}"

    def test_list_value_formatting(self):
        """Test formatting of list values."""
        parent = MockNode()
        current = MockNode(field_name="current", parent=parent)
        sibling = MockNode(field_name="items", parent=parent)

        parent.children = {"items": sibling, "current": current}

        previous_values = {"items": ["first", "second", "third"]}

        elements = resolve_collected_context_elements(current, previous_values)

        # Should have title and FULL tag reference (not actual list items)
        assert len(elements) == 2
        assert elements[0].content == "Items"
        assert elements[1].content == "{task.unknown.items}"


class TestElementsToMarkdown:
    """Test converting elements to markdown."""

    def test_simple_conversion(self):
        """Test simple element to markdown conversion."""
        elements = [
            PromptTitle(content="Main Title", level=1),
            PromptText(content="Some text here.", level=1),
            PromptTitle(content="Subtitle", level=2),
            PromptText(content="More text.", level=2),
        ]

        markdown = elements_to_markdown(elements)

        expected = "# Main Title\n\nSome text here.\n\n## Subtitle\n\nMore text."
        assert markdown == expected

    def test_deep_nesting(self):
        """Test deeply nested headings."""
        elements = [
            PromptTitle(content="Level 1", level=1),
            PromptTitle(content="Level 2", level=2),
            PromptTitle(content="Level 3", level=3),
            PromptTitle(content="Level 4", level=4),
        ]

        markdown = elements_to_markdown(elements)

        assert "# Level 1" in markdown
        assert "## Level 2" in markdown
        assert "### Level 3" in markdown
        assert "#### Level 4" in markdown

    def test_unresolved_template(self):
        """Test handling of unresolved template (shouldn't happen normally)."""
        elements = [
            PromptText(content="Before template", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=1),
            PromptText(content="After template", level=1),
        ]

        markdown = elements_to_markdown(elements)

        assert "Before template" in markdown
        assert "{PROMPT_SUBTREE}" in markdown
        assert "After template" in markdown


class TestAdjustElementLevels:
    """Test heading level adjustment."""

    def test_adjust_levels(self):
        """Test adjusting element levels while preserving hierarchy."""
        elements = [
            PromptTitle(content="Title", level=1),
            PromptText(content="Text", level=1),
            PromptTitle(content="Subtitle", level=2),
        ]

        adjusted = adjust_element_levels(elements, 3)

        # Min level is 1, base_level is 3, so shift is +2
        # Level 1 becomes 3, level 2 becomes 4 (hierarchy preserved)
        assert adjusted[0].level == 3
        assert adjusted[1].level == 3
        assert adjusted[2].level == 4  # Preserves hierarchy
        assert adjusted[0].content == "Title"
        assert adjusted[1].content == "Text"
        assert adjusted[2].content == "Subtitle"


class TestResolveTemplateElements:
    """Test template placeholder resolution."""

    def test_resolve_collected_context(self):
        """Test resolving COLLECTED_CONTEXT template."""
        elements = [
            PromptText(content="Previous work:", level=1),
            PromptTemplate(variable_name="COLLECTED_CONTEXT", level=1),
            PromptText(content="Continue:", level=1),
        ]

        parent = MockNode()
        current = MockNode(field_name="current", parent=parent)
        sibling = MockNode(field_name="done", parent=parent)
        parent.children = {"done": sibling, "current": current}

        previous_values = {"done": "Completed task"}

        resolved = resolve_template_elements(elements, current, previous_values)

        # Template should be replaced with context elements (FULL tag references now)
        assert len(resolved) > 3  # More elements due to context expansion
        assert any(isinstance(e, PromptTitle) and "Done" in e.content for e in resolved)
        assert any(
            isinstance(e, PromptText) and "{task.unknown.done}" in e.content
            for e in resolved
        )

    def test_resolve_prompt_subtree(self):
        """Test resolving PROMPT_SUBTREE template."""
        elements = [
            PromptText(content="Generate:", level=1),
            PromptTemplate(variable_name="PROMPT_SUBTREE", level=2),
        ]

        # Child resolutions to insert
        child_resolutions = {
            "child1": [
                PromptTitle(content="Child Title", level=1),
                PromptText(content="Child content", level=1),
            ]
        }

        node = MockNode()
        resolved = resolve_template_elements(
            elements, node, child_resolutions=child_resolutions
        )

        # Template should be replaced with child elements at adjusted level
        assert len(resolved) == 3
        assert resolved[0].content == "Generate:"
        assert resolved[1].content == "Child Title"
        assert resolved[1].level == 2  # Adjusted to template's level
        assert resolved[2].content == "Child content"
        assert resolved[2].level == 2


class TestResolveNodePromptElements:
    """Test full node resolution."""

    def test_resolve_simple_node(self):
        """Test resolving a simple node."""
        node = MockNode(clean_docstring="This is the main prompt.\n\n{PROMPT_SUBTREE}")

        class TestNode(TreeNode):
            field1: str = Field(description="First field")

        node.field_type = TestNode
        node.clean_field_descriptions = {"field1": "Clean description"}

        elements = resolve_node_prompt_elements(node)

        # Should have parsed docstring and resolved PROMPT_SUBTREE
        assert any(
            isinstance(e, PromptText) and "main prompt" in e.content for e in elements
        )
        assert any(
            isinstance(e, PromptTitle) and "Field 1" in e.content for e in elements
        )
        assert any(
            isinstance(e, PromptText) and "Clean description" in e.content
            for e in elements
        )

    def test_resolve_with_context(self):
        """Test resolving with COLLECTED_CONTEXT."""
        parent = MockNode()
        current = MockNode(
            field_name="current",
            parent=parent,
            clean_docstring="Context:\n{COLLECTED_CONTEXT}\n\nTask:",
        )
        sibling = MockNode(field_name="prev", parent=parent)
        parent.children = {"prev": sibling, "current": current}

        previous_values = {"prev": "Previous result"}

        elements = resolve_node_prompt_elements(
            current, previous_values=previous_values
        )

        # Should have resolved COLLECTED_CONTEXT (with FULL tag references now)
        assert any(
            isinstance(e, PromptText) and "Context:" in e.content for e in elements
        )
        assert any(isinstance(e, PromptTitle) and "Prev" in e.content for e in elements)
        assert any(
            isinstance(e, PromptText) and "{task.unknown.prev}" in e.content
            for e in elements
        )
        assert any(isinstance(e, PromptText) and "Task:" in e.content for e in elements)
