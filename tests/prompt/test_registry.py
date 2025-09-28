"""
Tests for registry module functionality.

This module tests the variable and pending target registry components:
- VariableRegistry: Variable registration, satisfaction tracking
- PendingTargetRegistry: Forward reference resolution
- Variable info tracking and validation
"""

from langtree.execution.scopes import get_scope
from langtree.parsing.parser import parse_command
from langtree.structure import (
    PendingTargetRegistry,
    RunStructure,
    VariableRegistry,
)


class TestVariableRegistry:
    """Test variable registry functionality."""

    def test_variable_registry_creation(self):
        """Test that RunStructure can be created with variable registry."""
        structure = RunStructure()
        # Should have registries after Phase 1 implementation
        assert hasattr(structure, "_variable_registry")
        assert hasattr(structure, "_pending_target_registry")
        assert isinstance(structure._variable_registry, VariableRegistry)
        assert isinstance(structure._pending_target_registry, PendingTargetRegistry)

    def test_relationship_types(self):
        """Test that variables track relationship types correctly."""
        registry = VariableRegistry()

        # 1:1 relationship (all command, no multiplicity)
        registry.register_variable(
            "title", get_scope("value"), "task.test", "all", False
        )

        # 1:n relationship (all command, has multiplicity)
        registry.register_variable(
            "results", get_scope("outputs"), "task.test", "all", True
        )

        # n:n relationship (each command, has multiplicity)
        registry.register_variable(
            "items", get_scope("value"), "task.test", "each", True
        )

    def test_satisfaction_source_tracking(self):
        """Test satisfaction source tracking and multiplicity."""
        registry = VariableRegistry()
        registry.register_variable("title", get_scope("value"), "task.test")

        # Add satisfaction sources
        registry.add_satisfaction_source(
            "title", get_scope("value"), "task.test", "sections.title"
        )
        registry.add_satisfaction_source(
            "title", get_scope("value"), "task.test", "document.title"
        )

        var_info = registry.variables["value.title"]
        assert var_info.is_satisfied()
        assert var_info.has_multiple_sources()
        satisfaction_sources = var_info.get_satisfaction_sources()
        assert "sections.title" in satisfaction_sources
        assert "document.title" in satisfaction_sources

    def test_unsatisfied_variables_detection(self):
        """Test detection of unsatisfied variables."""
        registry = VariableRegistry()
        registry.register_variable("title", get_scope("value"), "task.test")
        registry.register_variable("content", get_scope("prompt"), "task.test")

        # Only satisfy one variable
        registry.add_satisfaction_source(
            "title", get_scope("value"), "task.test", "sections.title"
        )

        unsatisfied = registry.get_unsatisfied_variables()
        assert len(unsatisfied) == 1
        assert unsatisfied[0].variable_path == "content"
        assert unsatisfied[0].get_scope_name() == "prompt"


class TestPendingTargetRegistry:
    """Test pending target registry and forward reference handling."""

    def test_pending_target_addition(self):
        """Test adding pending targets to registry."""
        registry = PendingTargetRegistry()

        command = parse_command("! @->task.future_target@{{prompt.data=*}}")
        registry.add_pending("task.future_target", command, "task.source")

        assert len(registry.pending_targets) == 1
        assert "task.future_target" in registry.pending_targets
        assert len(registry.pending_targets["task.future_target"]) == 1

    def test_pending_target_resolution(self):
        """Test resolution of pending targets."""
        registry = PendingTargetRegistry()

        command1 = parse_command("! @->task.target@{{prompt.data=*}}")
        command2 = parse_command("! @->task.target.subtask@{{value.x=y}}")

        registry.add_pending("task.target", command1, "task.source1")
        registry.add_pending("task.target.subtask", command2, "task.source2")

        # Resolve the parent target
        resolved = registry.resolve_pending("task.target")

        # Should resolve both the exact match and the child
        assert len(resolved) == 2
        assert "task.target" not in registry.pending_targets
        assert "task.target.subtask" not in registry.pending_targets
