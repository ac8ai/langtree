"""
Output format description configuration for LangTree.

This module provides configuration for customizing the text shown in
"Output Format" sections of generated prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any


@dataclass
class OutputFormatDescriptions:
    """Configuration for output format instruction text.

    Can be created from dict, YAML, or Path with partial overrides.
    Only specified values override defaults.

    Examples:
        # All defaults
        descriptions = OutputFormatDescriptions()

        # Partial override from dict
        descriptions = OutputFormatDescriptions.from_dict({
            "markdown": "Brief markdown",
            "int": "Numeric value"
        })

        # From YAML file
        descriptions = OutputFormatDescriptions.from_yaml("config.yaml")

        # Disable format sections entirely
        descriptions = OutputFormatDescriptions(include_format_sections=False)
    """

    # Default descriptions for each format type
    int: str = "Return only the integer value."
    float: str = "Return only the numeric value."
    bool: str = "Return exactly 'true' or 'false' (lowercase)."
    str: str = "Return only the text content."
    json: str = "Return valid JSON."
    enum: str = "Return one of the following values: {values}"  # {values} placeholder
    markdown: str = (
        "Return formatted markdown content. "
        "You may use headings, lists, bold, italic, code blocks, and other markdown syntax. "
        "Ensure proper markdown structure with blank lines between sections."
    )

    # Toggle to include "Output Format" sections at all
    include_format_sections: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> OutputFormatDescriptions:
        """Create from dict, only overriding specified values.

        Args:
            config: Dictionary with partial overrides. Only keys matching
                   dataclass fields will be used.

        Returns:
            OutputFormatDescriptions instance with specified overrides

        Example:
            config = {"markdown": "Custom desc", "int": "Number only"}
            descriptions = OutputFormatDescriptions.from_dict(config)
        """
        # Only use keys that are valid dataclass fields
        valid_fields = {f.name for f in dataclass_fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> OutputFormatDescriptions:
        """Create from YAML file with partial overrides.

        Args:
            yaml_path: Path to YAML file containing configuration

        Returns:
            OutputFormatDescriptions instance with YAML overrides

        Example YAML:
            markdown: "Write detailed analysis"
            int: "Provide integer value"
            include_format_sections: false
        """
        import yaml

        path = Path(yaml_path)
        with path.open() as f:
            config = yaml.safe_load(f) or {}

        return cls.from_dict(config)

    def get_description(
        self, format_type: str, enum_values: list[str] | None = None
    ) -> str:
        """Get description for a format type, with placeholder substitution.

        Args:
            format_type: Type of format (int, str, enum, markdown, etc.)
            enum_values: List of enum values for placeholder substitution

        Returns:
            Format description with placeholders substituted

        Example:
            desc = descriptions.get_description("enum", ["active", "inactive"])
            # Returns: "Return one of the following values: "active", "inactive""
        """
        # Get description for this format type
        desc = getattr(self, format_type, None)

        if desc is None:
            # Fallback for unknown types
            return f"Return output as {format_type}."

        # Handle enum placeholder substitution
        if format_type == "enum" and enum_values and "{values}" in desc:
            values_str = ", ".join(f'"{v}"' for v in enum_values)
            return desc.replace("{values}", values_str)

        return desc
