"""
Integration tests for type mapping configuration with structure building.

Tests that type mapping configurations are properly integrated into the
structure building phase. Actual type conversions happen during runtime
resolution which is tested separately.
"""

from typing import Any

from langtree import TreeNode
from langtree.structure import RunStructure
from langtree.structure.type_mapping import TypeMappingConfig


class TestTypeMappingStructureIntegration:
    """Test type mapping configuration in RunStructure."""

    def test_structure_accepts_type_mismatches_with_conversion(self):
        """Test that structure accepts type mismatches when conversions are possible."""

        class TaskSource(TreeNode):
            """
            ! @all->target@{{count=*}}
            """

            count: int = 42

        class TaskTarget(TreeNode):
            """Task target."""

            # Receives int but expects str (should be convertible)
            count: str

        # Should not raise error - conversion will happen at runtime
        structure = RunStructure()
        structure.add(TaskSource)
        structure.add(TaskTarget)

        # Verify nodes were added
        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.target") is not None

    def test_structure_with_custom_type_mapping_config(self):
        """Test RunStructure accepts custom TypeMappingConfig."""

        class TaskSource(TreeNode):
            """
            ! @all->target@{{value=*}}
            """

            value: str = "true"

        class TaskTarget(TreeNode):
            """Task target."""

            value: bool

        # Create structure with custom config enabling string parsing
        config = TypeMappingConfig(allow_string_parsing=True, strict_bool_parsing=True)
        structure = RunStructure(type_mapping_config=config)
        structure.add(TaskSource)
        structure.add(TaskTarget)

        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.target") is not None

    def test_multiple_type_conversions_in_structure(self):
        """Test structure with multiple type conversions."""

        class TaskSource(TreeNode):
            """
            ! @all->processor@{{int_val=*, float_val=*, str_val=*}}
            """

            int_val: int = 42
            float_val: float = 3.14
            str_val: str = "hello"

        class TaskProcessor(TreeNode):
            """Process values."""

            # All converted to strings
            int_val: str
            float_val: str
            str_val: str

        structure = RunStructure()
        structure.add(TaskSource)
        structure.add(TaskProcessor)

        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.processor") is not None

    def test_union_type_in_structure(self):
        """Test structure with union types."""

        class TaskSource(TreeNode):
            """
            ! @all->target@{{flexible=*}}
            """

            flexible: int = 42

        class TaskTarget(TreeNode):
            """Task target."""

            # Union type should accept int
            flexible: str | int | None

        structure = RunStructure()
        structure.add(TaskSource)
        structure.add(TaskTarget)

        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.target") is not None

    def test_dict_to_treenode_structure(self):
        """Test dict to TreeNode mapping in structure."""

        class ConfigNode(TreeNode):
            """Config node."""

            setting: str = "default"
            enabled: bool = True

        class TaskSource(TreeNode):
            """
            ! @all->target@{{config=*}}
            """

            config: dict[str, Any] = {"setting": "custom", "enabled": False}

        class TaskTarget(TreeNode):
            """Task target."""

            # Dict to TreeNode conversion
            config: ConfigNode

        # Enable TreeNode/dict conversion
        config = TypeMappingConfig(allow_treenode_dict_conversion=True)
        structure = RunStructure(type_mapping_config=config)
        structure.add(TaskSource)
        structure.add(TaskTarget)

        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.target") is not None

    def test_numeric_to_string_structure(self):
        """Test numeric to string mapping in structure."""

        class TaskSource(TreeNode):
            """
            ! @all->target@{{number=*, decimal=*}}
            """

            number: int = 42
            decimal: float = 3.14

        class TaskTarget(TreeNode):
            """Task target."""

            # Numeric to string conversion
            number: str
            decimal: str

        structure = RunStructure()
        structure.add(TaskSource)
        structure.add(TaskTarget)

        assert structure.get_node("task.source") is not None
        assert structure.get_node("task.target") is not None
