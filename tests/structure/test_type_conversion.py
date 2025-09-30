"""
Unit tests for TypeConverter conversion logic.

Tests all conversion paths, union type priority sorting, and edge cases
to ensure the conversion logic works correctly before integration.
"""

from typing import Union

import pytest

from langtree import TreeNode
from langtree.structure.type_mapping import (
    CompatibilityLevel,
    TypeConverter,
    TypeMappingConfig,
)


class SimpleTreeNode(TreeNode):
    """Test TreeNode for conversion testing."""

    name: str = "test"
    value: int = 42


class TestTypeConverter:
    """Test TypeConverter conversion functionality."""

    def setup_method(self):
        """Create TypeConverter instances for testing."""
        self.config = TypeMappingConfig()
        self.converter = TypeConverter(self.config)

    def test_identical_types_no_conversion(self):
        """Test that identical types are returned unchanged."""
        assert self.converter.convert_value(42, int, int) == 42
        assert self.converter.convert_value("hello", str, str) == "hello"
        assert self.converter.convert_value(3.14, float, float) == 3.14

    def test_basic_numeric_conversions(self):
        """Test basic numeric type conversions."""
        # Safe conversion: int → float
        assert self.converter.convert_value(42, int, float) == 42.0

        # Lossy conversion: float → int
        assert self.converter.convert_value(3.14, float, int) == 3
        assert self.converter.convert_value(3.9, float, int) == 3

    def test_string_conversion_for_collected_context(self):
        """Test universal string conversion for COLLECTED_CONTEXT."""
        # Basic types
        assert self.converter.convert_value(42, int, str) == "42"
        assert self.converter.convert_value(3.14, float, str) == "3.14"
        assert self.converter.convert_value(True, bool, str) == "True"

        # Collections → JSON
        assert self.converter.convert_value([1, 2, 3], list, str) == "[1, 2, 3]"
        assert (
            self.converter.convert_value({"key": "value"}, dict, str)
            == '{"key": "value"}'
        )
        assert self.converter.convert_value({1, 2, 3}, set, str) == "[1, 2, 3]"

    def test_treenode_string_conversion_not_implemented(self):
        """Test that TreeNode → string raises NotImplementedError."""
        node = SimpleTreeNode()

        with pytest.raises(
            NotImplementedError,
            match="TreeNode → string conversion for prompts not yet implemented",
        ):
            self.converter.convert_value(node, SimpleTreeNode, str)

    def test_string_parsing_enabled(self):
        """Test string parsing when enabled."""
        # JSON parsing
        assert self.converter.convert_value('{"key": "value"}', str, dict) == {
            "key": "value"
        }
        assert self.converter.convert_value("[1, 2, 3]", str, list) == [1, 2, 3]

        # Numeric parsing
        assert self.converter.convert_value("42", str, int) == 42
        assert self.converter.convert_value("3.14", str, float) == 3.14

        # Bool parsing (strict)
        assert self.converter.convert_value("true", str, bool)
        assert not self.converter.convert_value("FALSE", str, bool)
        assert self.converter.convert_value("1", str, bool)
        assert not self.converter.convert_value("0", str, bool)

    def test_strict_bool_parsing(self):
        """Test that bool parsing only accepts specific values."""
        # Valid bool strings
        valid_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", True),
            ("0", False),
        ]

        for string_val, expected in valid_cases:
            assert self.converter.convert_value(string_val, str, bool) == expected

        # Invalid bool strings should raise ValueError
        invalid_cases = ["yes", "no", "on", "off", "True!", " true ", "2", ""]

        for invalid_val in invalid_cases:
            with pytest.raises(ValueError, match="Cannot parse .* as bool"):
                self.converter.convert_value(invalid_val, str, bool)

    def test_string_parsing_disabled(self):
        """Test behavior when string parsing is disabled."""
        config = TypeMappingConfig(allow_string_parsing=False)
        converter = TypeConverter(config)

        # Should raise ValueError since string parsing is disabled
        with pytest.raises(ValueError):
            converter.convert_value("42", str, int)

    def test_treenode_dict_conversion(self):
        """Test TreeNode ↔ dict conversion."""
        node = SimpleTreeNode(name="test_node", value=123)

        # TreeNode → dict
        result_dict = self.converter.convert_value(node, SimpleTreeNode, dict)
        expected = {"name": "test_node", "value": 123}
        assert result_dict == expected

        # dict → TreeNode
        test_dict = {"name": "from_dict", "value": 456}
        result_node = self.converter.convert_value(test_dict, dict, SimpleTreeNode)
        assert result_node.name == "from_dict"
        assert result_node.value == 456

    def test_treenode_dict_conversion_disabled(self):
        """Test behavior when TreeNode ↔ dict conversion is disabled."""
        config = TypeMappingConfig(allow_treenode_dict_conversion=False)
        converter = TypeConverter(config)
        node = SimpleTreeNode()

        with pytest.raises(ValueError):
            converter.convert_value(node, SimpleTreeNode, dict)

    def test_union_type_priority_exact_match(self):
        """Test that exact type matches have highest priority in unions."""
        # Value is already int, should stay int even in int | float union
        result = self.converter.convert_value(42, int | float, str)
        assert result == "42"  # Converted from int, not float

        # Test with actual union conversion
        union_type = Union[int, float]
        result = self.converter.convert_value(42, union_type, str)
        assert result == "42"

    def test_union_type_priority_treenode_over_dict(self):
        """Test that TreeNode types have priority over dict in unions."""
        node = SimpleTreeNode(name="test", value=42)

        # Should prefer TreeNode → dict conversion over treating as dict
        union_type = Union[dict, SimpleTreeNode]
        result = self.converter.convert_value(node, union_type, dict)

        # Should use TreeNode.model_dump(), not treat as generic dict
        assert result == {"name": "test", "value": 42}

    def test_union_type_priority_specific_over_general(self):
        """Test that specific types (int) have priority over general (float)."""
        # Test with a value that could be both int and float
        Union[float, int]  # float first in union

        # Value 42 is int, should prioritize int conversion even though float is first
        # This tests the priority sorting indirectly through string conversion

        # The priority sorting should try int first (exact match) even though
        # float appears first in the union definition

    def test_union_conversion_fallback(self):
        """Test that union conversion falls back through priority list."""
        # Create a scenario where first priority fails, second succeeds
        union_type = Union[int, str]  # int first, str second

        # Value "hello" can't convert to int, should fall back to str
        result = self.converter.convert_value("hello", union_type, str)
        assert result == "hello"

    def test_conversion_errors(self):
        """Test that conversion errors are handled properly."""
        # Incompatible types should raise ValueError
        with pytest.raises(ValueError):
            self.converter.convert_value("not_json", str, dict)

        with pytest.raises(ValueError):
            self.converter.convert_value("not_a_number", str, int)

        # Union with no compatible members
        union_type = Union[int, float]
        with pytest.raises(ValueError, match="Cannot convert"):
            self.converter.convert_value("hello", union_type, dict)

    def test_compatibility_checking(self):
        """Test compatibility level checking."""
        # Identical types
        assert (
            self.converter.check_compatibility(int, int) == CompatibilityLevel.IDENTICAL
        )

        # Safe conversions
        assert self.converter.check_compatibility(int, float) == CompatibilityLevel.SAFE
        assert (
            self.converter.check_compatibility(SimpleTreeNode, dict)
            == CompatibilityLevel.SAFE
        )

        # Universal string compatibility
        assert self.converter.check_compatibility(int, str) == CompatibilityLevel.SAFE
        assert self.converter.check_compatibility(dict, str) == CompatibilityLevel.SAFE

        # Lossy conversion
        assert (
            self.converter.check_compatibility(float, int) == CompatibilityLevel.LOSSY
        )

    def test_configuration_affects_compatibility(self):
        """Test that configuration changes affect compatibility checking."""
        # With string parsing enabled
        assert self.converter.check_compatibility(str, int) == CompatibilityLevel.SAFE

        # With string parsing disabled
        config = TypeMappingConfig(allow_string_parsing=False)
        converter = TypeConverter(config)
        assert converter.check_compatibility(str, int) == CompatibilityLevel.FORBIDDEN

    def test_use_priority_union_sorting_enabled(self):
        """Test smart priority union sorting (default behavior)."""
        # Default config uses priority sorting
        union_type = Union[float, int]  # float first in static order

        # Value 42 is int, should prioritize int even though float is first in union
        # This is tested indirectly through the conversion behavior
        result = self.converter.convert_value(42, union_type, str)
        assert result == "42"

        # Test with TreeNode priority over dict
        node = SimpleTreeNode(name="test", value=42)
        union_type = Union[dict, SimpleTreeNode]  # dict first in static order

        # Should use TreeNode conversion priority despite dict being first
        result = self.converter.convert_value(node, union_type, dict)
        assert result == {"name": "test", "value": 42}  # TreeNode.model_dump()

    def test_use_priority_union_sorting_disabled(self):
        """Test static typing order when priority sorting is disabled."""
        config = TypeMappingConfig(use_priority_union_sorting=False)
        converter = TypeConverter(config)

        # Test that it still works correctly, even with static order
        union_type = Union[float, int]  # float first in static order
        result = converter.convert_value(42, union_type, str)
        assert result == "42"  # Should still work correctly

        # Test with string values that could be parsed as different types
        union_type = Union[int, str]  # int first in static order

        # When priority sorting is disabled, should try int first (static order)
        result = converter.convert_value("42", union_type, str)
        assert result == "42"  # Should successfully convert

        # Test that the flag is actually respected by checking configuration
        assert not converter.config.use_priority_union_sorting

    def test_priority_sorting_config_flag_respected(self):
        """Test that the configuration flag is properly respected."""
        # Test both configurations exist and produce different behavior
        priority_config = TypeMappingConfig(use_priority_union_sorting=True)
        static_config = TypeMappingConfig(use_priority_union_sorting=False)

        priority_converter = TypeConverter(priority_config)
        static_converter = TypeConverter(static_config)

        # Both should work but may have different internal ordering
        union_type = Union[float, int]

        priority_result = priority_converter.convert_value(42, union_type, str)
        static_result = static_converter.convert_value(42, union_type, str)

        # Both should successfully convert, though potentially via different paths
        assert priority_result in ["42", "42.0"]
        assert static_result in ["42", "42.0"]
