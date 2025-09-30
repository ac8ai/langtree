"""
Type mapping and compatibility system for LangTree DSL variable mappings.

Implements deferred conversion strategy for union types - conversions are attempted
at runtime based on actual values, not pre-computed at configuration time.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin

from langtree import TreeNode


class CompatibilityLevel(Enum):
    """Type compatibility levels for variable mapping validation."""

    IDENTICAL = "identical"  # Same type, no conversion needed
    SAFE = "safe"  # Safe conversion (int -> float)
    LOSSY = "lossy"  # Potentially lossy (float -> int)
    UNSAFE = "unsafe"  # Risky conversion requiring explicit opt-in
    FORBIDDEN = "forbidden"  # Never allowed


class ValidationStrategy(Enum):
    """Validation strategies for type compatibility checking."""

    STRICT = "strict"  # Only IDENTICAL and SAFE allowed
    PERMISSIVE = "permissive"  # Allow LOSSY conversions with warnings
    DEVELOPMENT = "development"  # Allow UNSAFE for rapid prototyping


@dataclass
class TypeMappingConfig:
    """Configuration for type mapping behavior."""

    validation_strategy: ValidationStrategy = ValidationStrategy.STRICT
    allow_string_parsing: bool = True
    allow_treenode_dict_conversion: bool = True
    strict_bool_parsing: bool = True  # Only "true"/"false"/"1"/"0"
    enable_type_validation: bool = True  # Skip validation for performance
    use_priority_union_sorting: bool = True  # Use smart priority vs static typing order

    @classmethod
    def from_dict(cls, config: dict | None = None) -> "TypeMappingConfig":
        """Factory method to create config from dict with defaults."""
        if config is None:
            config = {}
        return cls(**config)


class TypeConverter:
    """Handles actual type conversions with deferred union resolution."""

    def __init__(self, config: TypeMappingConfig):
        self.config = config

    def convert_value(self, value: Any, source_type: Any, target_type: Any) -> Any:
        """
        Convert value from source_type to target_type.

        For union types, tries each member until one succeeds (deferred approach).
        """
        # Handle union source types - try each member
        if self._is_union(source_type):
            return self._convert_from_union(value, source_type, target_type)

        # Handle union target types - try each member
        if self._is_union(target_type):
            return self._convert_to_union(value, source_type, target_type)

        # Single type to single type conversion
        return self._convert_single(value, source_type, target_type)

    def check_compatibility(
        self, source_type: Any, target_type: Any
    ) -> CompatibilityLevel:
        """
        Check compatibility level between source and target types.

        For unions, returns the best possible compatibility level.
        """
        # Handle union types - return best possible compatibility
        if self._is_union(source_type) or self._is_union(target_type):
            return self._check_union_compatibility(source_type, target_type)

        return self._check_single_compatibility(source_type, target_type)

    def _sort_union_members_by_priority(self, members: tuple, value: Any) -> list:
        """
        Sort union members by conversion priority for deterministic type resolution.

        Priority order:
        1. Exact type matches (isinstance(value, member_type))
        2. TreeNode types (higher priority than dict for structured data)
        3. Specific numeric types (int, bool before float)
        4. General types (float, str)
        5. Structured types (dict, list)
        6. Remaining types

        This ensures lossless conversions are preferred and TreeNode conversion
        takes precedence over generic dict serialization.

        Params:
            members: Union member types to sort
            value: The actual value being converted (for type checking)

        Returns:
            Ordered list of types by conversion priority
        """
        members_list = list(members)

        # 1. Exact type matches first (no conversion needed)
        exact_matches = [m for m in members_list if isinstance(value, m)]

        # 2. TreeNode types (structured data priority)
        treenode_types = [
            m
            for m in members_list
            if self._is_treenode_type(m) and m not in exact_matches
        ]

        # 3. Specific numeric types (lossless priority)
        # Note: bool only prioritized for actual boolean values, not numeric 1/0
        specific_numeric = []
        if int in members_list and int not in exact_matches:
            specific_numeric.append(int)
        if (
            bool in members_list
            and bool not in exact_matches
            and isinstance(value, bool)
        ):
            specific_numeric.append(bool)

        # 4. General numeric and string types
        general_numeric = [
            m for m in members_list if m == float and m not in exact_matches
        ]
        bool_fallback = [
            m
            for m in members_list
            if m == bool and m not in exact_matches and m not in specific_numeric
        ]
        string_types = [m for m in members_list if m == str and m not in exact_matches]
        general_types = general_numeric + bool_fallback + string_types

        # 5. Structured types (dict, list, tuple, set)
        structured_types = [
            m
            for m in members_list
            if m in (dict, list, tuple, set) and m not in exact_matches
        ]

        # 6. All remaining types
        processed = set(
            exact_matches
            + treenode_types
            + specific_numeric
            + general_types
            + structured_types
        )
        remaining = [m for m in members_list if m not in processed]

        return (
            exact_matches
            + treenode_types
            + specific_numeric
            + general_types
            + structured_types
            + remaining
        )

    def _is_union(self, type_hint: Any) -> bool:
        """Check if type is a Union."""
        return get_origin(type_hint) is Union

    def _convert_from_union(
        self, value: Any, source_union: Any, target_type: Any
    ) -> Any:
        """Try converting from each union member until one succeeds."""
        union_members = get_args(source_union)

        # Choose between smart priority sorting or static typing order
        if self.config.use_priority_union_sorting:
            prioritized_members = self._sort_union_members_by_priority(
                union_members, value
            )
        else:
            prioritized_members = list(union_members)

        for member_type in prioritized_members:
            try:
                # Try to convert as if value is this union member type
                return self._convert_single(value, member_type, target_type)
            except (ValueError, TypeError):
                continue

        raise ValueError(
            f"Cannot convert {type(value).__name__} to {target_type} from union {source_union}"
        )

    def _convert_to_union(self, value: Any, source_type: Any, target_union: Any) -> Any:
        """Try converting to each union member until one succeeds."""
        union_members = get_args(target_union)

        # Choose between smart priority sorting or static typing order
        if self.config.use_priority_union_sorting:
            prioritized_members = self._sort_union_members_by_priority(
                union_members, value
            )
        else:
            prioritized_members = list(union_members)

        for member_type in prioritized_members:
            try:
                return self._convert_single(value, source_type, member_type)
            except (ValueError, TypeError):
                continue

        raise ValueError(f"Cannot convert {source_type} to union {target_union}")

    def _convert_single(self, value: Any, source_type: Any, target_type: Any) -> Any:
        """Convert between single (non-union) types."""
        # Check if conversion is needed
        if source_type == target_type:
            return value

        # Universal string compatibility for COLLECTED_CONTEXT
        if target_type == str:
            return self._convert_to_string(value)

        # String parsing if enabled
        if source_type == str and self.config.allow_string_parsing:
            return self._parse_from_string(value, target_type)

        # Check if string parsing is disabled but needed
        if source_type == str and not self.config.allow_string_parsing:
            raise ValueError(
                f"String parsing disabled: cannot convert {source_type} to {target_type}"
            )

        # TreeNode ↔ dict conversion
        if self.config.allow_treenode_dict_conversion:
            if self._is_treenode_type(source_type) and target_type == dict:
                return self._treenode_to_dict(value)
            elif source_type == dict and self._is_treenode_type(target_type):
                return self._dict_to_treenode(value, target_type)

        # Check if TreeNode conversion is disabled but needed
        if not self.config.allow_treenode_dict_conversion:
            if (self._is_treenode_type(source_type) and target_type == dict) or (
                source_type == dict and self._is_treenode_type(target_type)
            ):
                raise ValueError(
                    f"TreeNode ↔ dict conversion disabled: cannot convert {source_type} to {target_type}"
                )

        # Basic type conversions
        return self._convert_basic_types(value, source_type, target_type)

    def _convert_to_string(self, value: Any) -> str:
        """Universal string conversion for COLLECTED_CONTEXT assembly."""
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)
        elif isinstance(value, TreeNode):
            # TODO: Use COLLECTED_CONTEXT assembly function for hierarchical prompt text
            # This should build readable text for LLMs, not JSON
            # Will reuse the same function that creates COLLECTED_CONTEXT
            raise NotImplementedError(
                "TreeNode → string conversion for prompts not yet implemented"
            )
        else:
            return str(value)

    def _parse_from_string(self, value: str, target_type: Any) -> Any:
        """Parse string to target type with intelligent conversion."""
        if target_type == str:
            return value

        # JSON parsing for dict/list
        if target_type in (dict, list) or (
            hasattr(target_type, "__origin__")
            and target_type.__origin__ in (dict, list)
        ):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot parse '{value}' as JSON for {target_type}")

        # Bool parsing (strict)
        if target_type == bool:
            return self._parse_bool_strict(value)

        # Numeric parsing
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)

        raise ValueError(f"Cannot parse string '{value}' to {target_type}")

    def _parse_bool_strict(self, value: str) -> bool:
        """Parse bool from string with strict rules."""
        if not self.config.strict_bool_parsing:
            return bool(value)

        lower_val = value.lower()
        if lower_val in ("true", "false", "1", "0"):
            return lower_val in ("true", "1")
        else:
            raise ValueError(
                f"Cannot parse '{value}' as bool (only 'true'/'false'/'1'/'0' allowed)"
            )

    def _is_treenode_type(self, type_hint: Any) -> bool:
        """Check if type is a TreeNode subclass."""
        try:
            return isinstance(type_hint, type) and issubclass(type_hint, TreeNode)
        except TypeError:
            return False

    def _treenode_to_dict(self, value: TreeNode) -> dict:
        """Convert TreeNode to dict."""
        return value.model_dump()

    def _dict_to_treenode(self, value: dict, target_type: type) -> TreeNode:
        """Convert dict to TreeNode."""
        return target_type(**value)

    def _convert_basic_types(
        self, value: Any, source_type: Any, target_type: Any
    ) -> Any:
        """Handle basic type conversions (int ↔ float, etc.)."""
        if target_type == int and isinstance(value, float):
            return int(value)  # Lossy conversion
        elif target_type == float and isinstance(value, int):
            return float(value)  # Safe conversion
        else:
            # Attempt direct conversion
            return target_type(value)

    def _check_single_compatibility(
        self, source_type: Any, target_type: Any
    ) -> CompatibilityLevel:
        """Check compatibility between single (non-union) types."""
        if source_type == target_type:
            return CompatibilityLevel.IDENTICAL

        # Universal string compatibility
        if target_type == str:
            return CompatibilityLevel.SAFE

        # String parsing compatibility
        if source_type == str and self.config.allow_string_parsing:
            if target_type in (int, float, bool, dict, list):
                return CompatibilityLevel.SAFE

        # TreeNode ↔ dict
        if self.config.allow_treenode_dict_conversion:
            if (self._is_treenode_type(source_type) and target_type == dict) or (
                source_type == dict and self._is_treenode_type(target_type)
            ):
                return CompatibilityLevel.SAFE

        # Basic numeric conversions
        if source_type == int and target_type == float:
            return CompatibilityLevel.SAFE
        elif source_type == float and target_type == int:
            return CompatibilityLevel.LOSSY

        return CompatibilityLevel.FORBIDDEN

    def _check_union_compatibility(
        self, source_type: Any, target_type: Any
    ) -> CompatibilityLevel:
        """Check compatibility involving union types - return best possible level."""
        best_level = CompatibilityLevel.FORBIDDEN

        source_types = (
            get_args(source_type) if self._is_union(source_type) else [source_type]
        )
        target_types = (
            get_args(target_type) if self._is_union(target_type) else [target_type]
        )

        for src in source_types:
            for tgt in target_types:
                level = self._check_single_compatibility(src, tgt)
                if (
                    level.value < best_level.value
                ):  # Lower enum value = better compatibility
                    best_level = level

        return best_level


def create_type_mapping_config(
    config: TypeMappingConfig | dict | None = None,
) -> TypeMappingConfig:
    """
    Factory function for creating TypeMappingConfig with flexible input types.

    Args:
        config: TypeMappingConfig instance, dict to override defaults, or None for defaults

    Returns:
        TypeMappingConfig instance
    """
    if isinstance(config, TypeMappingConfig):
        return config
    elif isinstance(config, dict):
        return TypeMappingConfig.from_dict(config)
    else:
        return TypeMappingConfig()
