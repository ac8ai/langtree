"""
Tests for OutputFormatDescriptions configuration and integration.

This module tests the format description configuration system that allows
users to customize output format instructions for different field types.
"""

import tempfile
from pathlib import Path

import pytest

from langtree.structure.builder import RunStructure
from langtree.structure.format_descriptions import OutputFormatDescriptions


class TestOutputFormatDescriptions:
    """Test OutputFormatDescriptions dataclass and configuration."""

    def test_default_values(self):
        """Test that default descriptions are set correctly."""
        descriptions = OutputFormatDescriptions()

        assert descriptions.int == "Return only the integer value."
        assert descriptions.float == "Return only the numeric value."
        assert descriptions.bool == "Return exactly 'true' or 'false' (lowercase)."
        assert descriptions.str == "Return only the text content."
        assert descriptions.json == "Return valid JSON."
        assert descriptions.enum == "Return one of the following values: {values}"
        assert "Return formatted markdown content" in descriptions.markdown
        assert descriptions.include_format_sections is True

    def test_from_dict_partial_override(self):
        """Test creating from dict with partial overrides."""
        config = {
            "markdown": "Custom markdown instruction",
            "int": "Number only",
        }
        descriptions = OutputFormatDescriptions.from_dict(config)

        # Overridden values
        assert descriptions.markdown == "Custom markdown instruction"
        assert descriptions.int == "Number only"

        # Default values preserved
        assert descriptions.bool == "Return exactly 'true' or 'false' (lowercase)."
        assert descriptions.str == "Return only the text content."
        assert descriptions.include_format_sections is True

    def test_from_dict_all_fields(self):
        """Test creating from dict with all fields specified."""
        config = {
            "int": "Custom int",
            "float": "Custom float",
            "bool": "Custom bool",
            "str": "Custom str",
            "json": "Custom json",
            "enum": "Custom enum: {values}",
            "markdown": "Custom markdown",
            "include_format_sections": False,
        }
        descriptions = OutputFormatDescriptions.from_dict(config)

        assert descriptions.int == "Custom int"
        assert descriptions.float == "Custom float"
        assert descriptions.bool == "Custom bool"
        assert descriptions.str == "Custom str"
        assert descriptions.json == "Custom json"
        assert descriptions.enum == "Custom enum: {values}"
        assert descriptions.markdown == "Custom markdown"
        assert descriptions.include_format_sections is False

    def test_from_dict_ignores_unknown_fields(self):
        """Test that from_dict ignores fields not in the dataclass."""
        config = {
            "int": "Custom int",
            "unknown_field": "Should be ignored",
            "another_unknown": 123,
        }
        descriptions = OutputFormatDescriptions.from_dict(config)

        assert descriptions.int == "Custom int"
        assert not hasattr(descriptions, "unknown_field")
        assert not hasattr(descriptions, "another_unknown")

    def test_from_yaml(self):
        """Test creating from YAML file with partial overrides."""
        yaml_content = """
int: "Custom integer instruction"
markdown: "Custom markdown instruction"
include_format_sections: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            descriptions = OutputFormatDescriptions.from_yaml(yaml_path)

            # Overridden values
            assert descriptions.int == "Custom integer instruction"
            assert descriptions.markdown == "Custom markdown instruction"
            assert descriptions.include_format_sections is False

            # Default values preserved
            assert descriptions.bool == "Return exactly 'true' or 'false' (lowercase)."
            assert descriptions.str == "Return only the text content."
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_empty_file(self):
        """Test creating from empty YAML file uses all defaults."""
        yaml_content = ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            descriptions = OutputFormatDescriptions.from_yaml(yaml_path)

            # All defaults preserved
            assert descriptions.int == "Return only the integer value."
            assert descriptions.bool == "Return exactly 'true' or 'false' (lowercase)."
            assert descriptions.include_format_sections is True
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_path_object(self):
        """Test from_yaml accepts Path object."""
        yaml_content = """
int: "Path object test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            descriptions = OutputFormatDescriptions.from_yaml(yaml_path)
            assert descriptions.int == "Path object test"
        finally:
            yaml_path.unlink()

    def test_get_description_simple_types(self):
        """Test get_description for simple format types."""
        descriptions = OutputFormatDescriptions()

        assert descriptions.get_description("int") == "Return only the integer value."
        assert descriptions.get_description("float") == "Return only the numeric value."
        assert (
            descriptions.get_description("bool")
            == "Return exactly 'true' or 'false' (lowercase)."
        )
        assert descriptions.get_description("str") == "Return only the text content."
        assert descriptions.get_description("json") == "Return valid JSON."

    def test_get_description_enum_with_values(self):
        """Test get_description for enum with placeholder substitution."""
        descriptions = OutputFormatDescriptions()

        result = descriptions.get_description("enum", ["active", "inactive", "pending"])

        assert (
            result
            == 'Return one of the following values: "active", "inactive", "pending"'
        )

    def test_get_description_enum_without_values(self):
        """Test get_description for enum without values (no substitution)."""
        descriptions = OutputFormatDescriptions()

        result = descriptions.get_description("enum", None)

        # Placeholder remains unreplaced
        assert result == "Return one of the following values: {values}"

    def test_get_description_custom_enum_template(self):
        """Test get_description with custom enum template."""
        descriptions = OutputFormatDescriptions(enum="Choose from: {values} only")

        result = descriptions.get_description("enum", ["yes", "no"])

        assert result == 'Choose from: "yes", "no" only'

    def test_get_description_unknown_type(self):
        """Test get_description for unknown format type returns generic message."""
        descriptions = OutputFormatDescriptions()

        result = descriptions.get_description("custom_type")

        assert result == "Return output as custom_type."

    def test_get_description_with_custom_values(self):
        """Test custom descriptions are returned correctly."""
        descriptions = OutputFormatDescriptions(
            int="Custom int instruction",
            markdown="Custom markdown instruction",
        )

        assert descriptions.get_description("int") == "Custom int instruction"
        assert descriptions.get_description("markdown") == "Custom markdown instruction"
        # Defaults still work
        assert (
            descriptions.get_description("bool")
            == "Return exactly 'true' or 'false' (lowercase)."
        )


class TestRunStructureFormatDescriptionsIntegration:
    """Test OutputFormatDescriptions integration with RunStructure."""

    def test_runstructure_default_format_descriptions(self):
        """Test RunStructure creates default OutputFormatDescriptions."""
        structure = RunStructure()

        assert structure.format_descriptions is not None
        assert isinstance(structure.format_descriptions, OutputFormatDescriptions)
        assert structure.format_descriptions.int == "Return only the integer value."

    def test_runstructure_with_dict(self):
        """Test RunStructure with dict format_descriptions."""
        structure = RunStructure(
            format_descriptions={
                "int": "Custom int desc",
                "markdown": "Custom markdown desc",
            }
        )

        assert structure.format_descriptions.int == "Custom int desc"
        assert structure.format_descriptions.markdown == "Custom markdown desc"
        # Defaults preserved
        assert (
            structure.format_descriptions.bool
            == "Return exactly 'true' or 'false' (lowercase)."
        )

    def test_runstructure_with_outputformatdescriptions_instance(self):
        """Test RunStructure with OutputFormatDescriptions instance."""
        custom_descriptions = OutputFormatDescriptions(
            int="Instance int",
            bool="Instance bool",
        )
        structure = RunStructure(format_descriptions=custom_descriptions)

        assert structure.format_descriptions is custom_descriptions
        assert structure.format_descriptions.int == "Instance int"
        assert structure.format_descriptions.bool == "Instance bool"

    def test_runstructure_with_yaml_path_string(self):
        """Test RunStructure with YAML file path as string."""
        yaml_content = """
int: "YAML string path int"
float: "YAML string path float"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            structure = RunStructure(format_descriptions=yaml_path)

            assert structure.format_descriptions.int == "YAML string path int"
            assert structure.format_descriptions.float == "YAML string path float"
            # Defaults preserved
            assert structure.format_descriptions.str == "Return only the text content."
        finally:
            Path(yaml_path).unlink()

    def test_runstructure_with_yaml_path_object(self):
        """Test RunStructure with YAML file Path object."""
        yaml_content = """
int: "YAML Path object int"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            structure = RunStructure(format_descriptions=yaml_path)

            assert structure.format_descriptions.int == "YAML Path object int"
        finally:
            yaml_path.unlink()

    def test_runstructure_with_none(self):
        """Test RunStructure with None creates defaults."""
        structure = RunStructure(format_descriptions=None)

        assert structure.format_descriptions is not None
        assert isinstance(structure.format_descriptions, OutputFormatDescriptions)
        assert structure.format_descriptions.int == "Return only the integer value."

    def test_runstructure_with_invalid_type_raises_error(self):
        """Test RunStructure with invalid type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            RunStructure(format_descriptions=123)

        assert "format_descriptions must be" in str(exc_info.value)
        assert "OutputFormatDescriptions, dict, Path, str, or None" in str(
            exc_info.value
        )

    def test_runstructure_with_include_format_sections_false(self):
        """Test RunStructure with format sections disabled."""
        structure = RunStructure(format_descriptions={"include_format_sections": False})

        assert structure.format_descriptions.include_format_sections is False
