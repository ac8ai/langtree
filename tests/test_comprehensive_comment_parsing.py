"""
Comprehensive test for comment parsing in all contexts.

Tests all comment patterns from user requirements:
1. Standalone comments in docstrings and Field descriptions
2. Inline comments after commands
3. Comments within multiline command mappings
4. Mixed comment scenarios

Author: Claude Code
"""

import pytest
from pydantic import Field

from langtree import TreeNode
from langtree.parsing.parser import CommentCommand, parse_command
from langtree.structure import RunStructure


class TestComprehensiveCommentParsing:
    """Test comprehensive comment parsing in all contexts."""

    def test_standalone_comment_parsing(self):
        """Test standalone comment commands are parsed correctly."""
        # Test with space after #
        cmd1 = parse_command("! # docstring comment solo")
        assert isinstance(cmd1, CommentCommand)
        assert cmd1.comment == "docstring comment solo"

        # Test without space after #
        cmd2 = parse_command("!# param comment before command")
        assert isinstance(cmd2, CommentCommand)
        assert cmd2.comment == "param comment before command"

        # Test with special characters
        cmd3 = parse_command("! # comment with symbols !@#$%^&*()")
        assert isinstance(cmd3, CommentCommand)
        assert cmd3.comment == "comment with symbols !@#$%^&*()"

    def test_inline_comment_parsing(self):
        """Test inline comments after commands work correctly."""
        # Test execution command with comment
        cmd1 = parse_command('! llm("gpt-4") # docstring comment after command')
        assert cmd1.comment == "docstring comment after command"

        # Test variable assignment with comment
        cmd2 = parse_command('! model="claude-3" # param comment solo')
        assert cmd2.comment == "param comment solo"

        # Test traditional command with comment
        cmd3 = parse_command(
            "! @each[items]->task.processor@{{value.processed=items}}* # processing comment"
        )
        assert cmd3.comment == "processing comment"

    def test_comprehensive_node_with_all_comment_patterns(self):
        """Test a TreeNode with all comment patterns from user requirements."""

        class TaskInsightGenerator(TreeNode):
            """
            !# docstring comment solo
            ! llm("gpt-4") #    docstring comment after command

            ## Insight Generation Phase

            Transform processed feedback into business recommendations.
            Create specific, measurable improvement suggestions.

            {PROMPT_SUBTREE}

            # Generated so far

            {COLLECTED_CONTEXT}
            """

            class Category(TreeNode):
                """
                Group related insights into logical business categories.
                Each category should address a specific operational area.
                """

                class Insight(TreeNode):
                    """
                    Generate one specific, actionable business recommendation.
                    Include implementation steps and expected outcomes.
                    """

                    recommendation: str = Field(
                        description="""
                        !# field description comment before command
                        ! llm("claude-3") # field description comment solo
                        ! # field description comment after command

                        Detailed business recommendation
                    """
                    )

                insights: list[Insight] = Field(
                    description="""
                    ! @each[insights]->task.order_processor@{{orders.feedback=insights.recommendation}}*

                    Insights for this category
                """
                )

            categories: list[Category] = Field(
                description="""
                ! llm('gpt5')

                Business insight categories
            """
            )
            final_report: str = Field(
                description="""
                ! @all->task.order_processor@{{prompt.final_report=final_report}}*

                Executive summary of all insights
            """
            )

        # Test that the node can be created and added to structure without errors
        structure = RunStructure()
        structure.add(TaskInsightGenerator)

        # Verify the node was added successfully
        node = structure.get_node("task.insight_generator")
        assert node is not None, "Node should be added successfully"

        # Verify nested classes are present
        category_node = structure.get_node("task.insight_generator.categories")
        assert category_node is not None, "Category nested node should be present"

        insight_node = structure.get_node("task.insight_generator.categories.insights")
        assert insight_node is not None, "Insight nested node should be present"

    def test_multiline_command_with_comments(self):
        """Test multiline commands with embedded comments."""

        class TaskWithMultilineComment(TreeNode):
            """Task demonstrating multiline commands with comments."""

            items: list[str] = Field(
                description="""
                ! @each[items]->task.processor@{{
                    # This is a comment within the mapping
                    value.processed_items=items, # Another comment on the same line
                    value.metadata=items.count # Yet another comment
                }}*

                Process these items with detailed mapping
            """
            )

        # Test that the node can be created successfully
        structure = RunStructure()
        structure.add(TaskWithMultilineComment)

        node = structure.get_node("task.with_multiline_comment")
        assert node is not None, (
            "Node with multiline comments should be created successfully"
        )

    def test_edge_case_comment_scenarios(self):
        """Test edge cases and boundary conditions for comments."""

        # Empty comment
        cmd1 = parse_command("! #")
        assert isinstance(cmd1, CommentCommand)
        assert cmd1.comment == ""

        # Comment with only whitespace
        cmd2 = parse_command("! #    ")
        assert isinstance(cmd2, CommentCommand)
        assert cmd2.comment == ""

        # Comment with hash symbols inside
        cmd3 = parse_command("! # This comment has # hash symbols")
        assert isinstance(cmd3, CommentCommand)
        assert cmd3.comment == "This comment has # hash symbols"

        # Comment with equals and arrows (command-like syntax)
        cmd4 = parse_command("! # fake=command->syntax@{{}}")
        assert isinstance(cmd4, CommentCommand)
        assert cmd4.comment == "fake=command->syntax@{{}}"

    def test_comments_do_not_break_existing_functionality(self):
        """Test that adding comment support doesn't break existing command parsing."""

        # All existing command types should still work
        from langtree.parsing.parser import (
            ExecutionCommand,
            NodeModifierCommand,
            ParsedCommand,
            ResamplingCommand,
            VariableAssignmentCommand,
        )

        # Variable assignment
        cmd1 = parse_command('! model="claude-3"')
        assert isinstance(cmd1, VariableAssignmentCommand)

        # Execution command
        cmd2 = parse_command('! llm("gpt-4")')
        assert isinstance(cmd2, ExecutionCommand)

        # Resampling command
        cmd3 = parse_command("! @resampled[scores]->mean")
        assert isinstance(cmd3, ResamplingCommand)

        # Node modifier
        cmd4 = parse_command("! @sequential")
        assert isinstance(cmd4, NodeModifierCommand)

        # Traditional command
        cmd5 = parse_command("! @each[items]->task.processor@{{value.data=items}}*")
        assert isinstance(cmd5, ParsedCommand)

    def test_comment_command_str_representation(self):
        """Test CommentCommand string representation."""
        comment_cmd = CommentCommand(comment="test comment")
        assert str(comment_cmd) == "# test comment"

        empty_comment_cmd = CommentCommand(comment="")
        assert str(empty_comment_cmd) == "# "


if __name__ == "__main__":
    pytest.main([__file__])
