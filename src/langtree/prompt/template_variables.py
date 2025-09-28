"""
Template variable processing for LangTree DSL framework.

This module handles the detection, validation, and processing of template variables
({PROMPT_SUBTREE} and {COLLECTED_CONTEXT}) in LangTree DSL docstrings and field descriptions.
"""

# Group 1: External direct imports (alphabetical)
import re

# Group 2: External from imports (alphabetical by source module)
from typing import TYPE_CHECKING, Optional

# Group 4: Internal from imports (alphabetical by source module)
from langtree.prompt.exceptions import (
    TemplateVariableConflictError,
    TemplateVariableNameError,
    TemplateVariableSpacingError,
)

if TYPE_CHECKING:
    from langtree.prompt.structure import RunStructure, StructureTreeNode


# Valid template variable names
VALID_TEMPLATE_VARIABLES = ("PROMPT_SUBTREE", "COLLECTED_CONTEXT")

# Template variable patterns
# These patterns specifically match single-brace syntax and exclude double-brace runtime variables
PROMPT_SUBTREE_PATTERN = re.compile(r"(?<!\{)\{PROMPT_SUBTREE\}(?!\})")
COLLECTED_CONTEXT_PATTERN = re.compile(r"(?<!\{)\{COLLECTED_CONTEXT\}(?!\})")
TEMPLATE_VARIABLES_PATTERN = re.compile(
    r"(?<!\{)\{(?:PROMPT_SUBTREE|COLLECTED_CONTEXT)\}(?!\})"
)

# Spacing validation patterns
INVALID_SPACING_PATTERN = re.compile(
    r"(?:(?<!\n\n)\{(?:PROMPT_SUBTREE|COLLECTED_CONTEXT)\}(?!\n\n))|"  # Missing newlines before/after
    r"(?:\S\{(?:PROMPT_SUBTREE|COLLECTED_CONTEXT)\})|"  # Text directly before
    r"(?:\{(?:PROMPT_SUBTREE|COLLECTED_CONTEXT)\}\S)"  # Text directly after
)


def detect_template_variables(content: str) -> dict[str, list[int]]:
    """
    Detect template variables in content and return their positions.

    Scans the provided content for PROMPT_SUBTREE and COLLECTED_CONTEXT
    template variables and tracks their character positions for validation
    and replacement processing.

    Params:
        content: Text content to scan for template variables

    Returns:
        Dictionary mapping variable names to lists of character positions
        where each variable is found in the content
    """
    if not content:
        return {}

    result = {}

    # Find PROMPT_SUBTREE occurrences
    subtree_matches = list(PROMPT_SUBTREE_PATTERN.finditer(content))
    if subtree_matches:
        result["PROMPT_SUBTREE"] = [match.start() for match in subtree_matches]

    # Find COLLECTED_CONTEXT occurrences
    context_matches = list(COLLECTED_CONTEXT_PATTERN.finditer(content))
    if context_matches:
        result["COLLECTED_CONTEXT"] = [match.start() for match in context_matches]

    return result


def validate_template_variable_names(content: str) -> list[str]:
    """
    Validate that only known template variables are used and syntax is correct.

    Ensures that only valid template variable names (PROMPT_SUBTREE, COLLECTED_CONTEXT)
    are used in the content. Unknown template variables and malformed syntax are considered errors.

    Params:
        content: Text content to validate for unknown template variables

    Returns:
        List of validation error messages, empty if no violations found
    """
    if not content:
        return []

    errors = []

    # First, check for malformed nested braces
    nested_brace_pattern = re.compile(r"\{[^}]*\{[^}]*\}[^}]*\}")
    nested_matches = nested_brace_pattern.finditer(content)
    for match in nested_matches:
        line_number = content[: match.start()].count("\n") + 1
        errors.append(
            f"Malformed template variable syntax with nested braces '{match.group()}' at line {line_number}. Template variables cannot be nested."
        )

    # Pattern to find any single-brace variable-like structures
    all_variable_pattern = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")

    # Find all potential template/runtime variables
    matches = all_variable_pattern.finditer(content)

    for match in matches:
        var_name = match.group(1)

        # Check if it looks like a template variable (more precise pattern matching)
        is_template_like = (
            var_name.isupper()  # All uppercase like INVALID_TEMPLATE
            or
            # Mixed case that looks like misspelled template variables
            (
                any(c.isupper() for c in var_name)
                and (
                    "subtree" in var_name.lower()
                    or "prompt" in var_name.lower()
                    or "collected" in var_name.lower()
                )
            )
            or
            # Exact misspellings of known template variables
            var_name.lower() in ("prompt_subtree", "collected_context")
        )

        if is_template_like and var_name not in VALID_TEMPLATE_VARIABLES:
            line_number = content[: match.start()].count("\n") + 1
            errors.append(
                f"Unknown template variable '{{{var_name}}}' at line {line_number}. Only {' and '.join(VALID_TEMPLATE_VARIABLES)} are supported."
            )

    # Check runtime variables for double underscore usage (reserved for system)
    runtime_matches = all_variable_pattern.finditer(content)
    for match in runtime_matches:
        var_name = match.group(1)
        # Skip template variables
        if var_name in VALID_TEMPLATE_VARIABLES:
            continue
        # Check for double underscore (reserved for system)
        if "__" in var_name:
            line_number = content[: match.start()].count("\n") + 1
            errors.append(
                f"Runtime variable '{{{var_name}}}' at line {line_number} contains double underscore '__' which is reserved for system use."
            )

    return errors


def validate_template_variable_spacing(content: str) -> list[str]:
    """
    Validate that template variables have proper spacing requirements.

    Ensures template variables are surrounded by empty lines according to
    LangTree DSL spacing rules. This maintains consistent formatting and readability
    in prompt templates.

    Params:
        content: Text content to validate for spacing violations

    Returns:
        List of validation error messages, empty if no violations found
    """
    if not content:
        return []

    errors = []

    # Find all template variable occurrences
    template_vars = list(TEMPLATE_VARIABLES_PATTERN.finditer(content))

    # Check for adjacent template variables first
    for i in range(len(template_vars) - 1):
        current_match = template_vars[i]
        next_match = template_vars[i + 1]

        # Check if template variables are directly adjacent
        if current_match.end() == next_match.start():
            current_line = content[: current_match.start()].count("\n") + 1
            next_line = content[: next_match.start()].count("\n") + 1
            errors.append(
                f"Template variables {current_match.group()} at line {current_line} and {next_match.group()} at line {next_line} must have empty lines between them"
            )

    for match in template_vars:
        var_name = match.group(0)
        start_pos = match.start()
        end_pos = match.end()

        # Calculate line number for error reporting
        line_number = content[:start_pos].count("\n") + 1

        # Check if this is the only content (special case)
        if content.strip() == var_name:
            # If template variable is the only content (possibly with surrounding whitespace),
            # this is acceptable since it's often the result of automatic addition
            # when docstring is empty or minimal
            continue

        # Check before the template variable
        before_valid = False
        if start_pos == 0:
            # Template variable is at the very start - acceptable
            before_valid = True
        else:
            # Check what comes before
            before_content = content[:start_pos]

            # Split into lines and check for empty lines
            lines = before_content.split("\n")
            if len(lines) <= 1:
                # No newlines before, so no empty line possible
                before_valid = False
            elif len(lines) == 2 and lines[1] == "":
                # Only one newline: ["text", ""] - this is just text\n{template}, not text\n\n{template}
                before_valid = False
            else:
                # Multiple lines - check if there's at least one empty line before the template variable
                # The last element after split is always "", so check the second-to-last
                # Use strict empty check - line must be completely empty, not just whitespace
                if len(lines) >= 3 and lines[-2] == "":
                    before_valid = True

        # Check after the template variable
        after_valid = False
        if end_pos == len(content):
            # Template variable is at the very end - acceptable
            after_valid = True
        else:
            # Check what comes after
            after_content = content[end_pos:]

            # Split into lines and check for empty lines
            lines = after_content.split("\n")
            if len(lines) <= 1:
                # No newlines after, only valid if it's just whitespace
                after_valid = after_content.strip() == ""
            elif len(lines) == 2 and lines[0] == "":
                # Only one newline: ["", "text"] - this is just {template}\ntext, not {template}\n\ntext
                after_valid = False
            else:
                # Multiple lines - check if there's at least one empty line after the template variable
                # The first element after split might be "", so check the second element
                # Use strict empty check - line must be completely empty, not just whitespace
                if len(lines) >= 3 and lines[1] == "":
                    after_valid = True  # Report violations
        if not before_valid or not after_valid:
            errors.append(
                f"Template variable {var_name} at line {line_number} requires empty lines before and after"
            )

    return errors


def add_automatic_prompt_subtree(content: str) -> str:
    """
    Add {PROMPT_SUBTREE} to docstring if not already present.

    Args:
        content: Docstring content

    Returns:
        Content with {PROMPT_SUBTREE} added if it wasn't present

    Note:
        Only adds {PROMPT_SUBTREE} to substantial content that looks like
        a main docstring, not to short field descriptions.
    """
    if not content:
        content = ""

    # Check if PROMPT_SUBTREE is already present
    if PROMPT_SUBTREE_PATTERN.search(content):
        return content

    # Don't add PROMPT_SUBTREE to short content (likely field descriptions)
    # Field descriptions are typically one line and shouldn't have subtrees
    if content.strip() and len(content.strip().split("\n")) <= 2:
        return content

    # Add PROMPT_SUBTREE at the end with proper spacing
    if content.strip():
        # Add with proper spacing if there's existing content
        if not content.endswith("\n\n"):
            content = content.rstrip() + "\n\n"
        content += "{PROMPT_SUBTREE}\n\n"
    else:
        # If content is empty, just add the template variable
        content = "{PROMPT_SUBTREE}"

    return content


def validate_template_variable_conflicts(
    content: str, assembly_variables: set[str]
) -> list[str]:
    """
    Validate that template variables don't conflict with Assembly Variables.

    Args:
        content: Text content to check
        assembly_variables: Set of Assembly Variable names defined in the node

    Returns:
        List of conflict errors (empty list if no conflicts)
    """
    errors = []

    # Check if any Assembly Variables use template variable names
    template_var_names = set(VALID_TEMPLATE_VARIABLES)

    for var_name in assembly_variables:
        if var_name in template_var_names:
            errors.append(
                f"Assembly Variable '{var_name}' conflicts with template variable {{{var_name}}}"
            )

    return errors


def get_assembly_variables_for_node(node: "StructureTreeNode") -> set[str]:
    """
    Get all Assembly Variable names available to a node.

    Assembly Variables are available from definition node through all descendant nodes,
    following the hierarchical scope rules defined in LANGUAGE_SPECIFICATION.md.

    Args:
        node: Structure tree node to get available variables for

    Returns:
        Set of Assembly Variable names available to this node
    """
    if not node:
        return set()

    # For now, implement a simple version that requires external context
    # The proper implementation would traverse the tree hierarchy to find
    # Assembly Variables available to this node based on scope rules

    # TODO: This function needs access to the RunStructure to get Assembly Variables
    # Currently there's no direct link from StructureTreeNode to RunStructure
    # The caller should provide the RunStructure context or we need architectural changes

    # Return empty set for now to prevent errors
    # The full implementation should:
    # 1. Get RunStructure from context
    # 2. Get Assembly Variable registry
    # 3. Filter variables by scope hierarchy (parent to child inheritance)
    # 4. Return set of variable names

    return set()


def get_assembly_variables_for_node_with_structure(
    node: "StructureTreeNode", run_structure: "RunStructure"
) -> set[str]:
    """
    Get all Assembly Variable names available to a node with RunStructure context.

    Assembly Variables are available from definition node through all descendant nodes,
    following the hierarchical scope rules defined in LANGUAGE_SPECIFICATION.md.

    Args:
        node: Structure tree node to get available variables for
        run_structure: RunStructure containing the Assembly Variable registry

    Returns:
        Set of Assembly Variable names available to this node
    """
    if not node or not run_structure:
        return set()

    # Get Assembly Variable registry from RunStructure
    assembly_registry = run_structure.get_assembly_variable_registry()

    # Get all Assembly Variables that are available to this node
    # For the initial implementation, return all variables
    # TODO: Implement proper scope filtering based on node hierarchy
    # Assembly Variables should be available from definition node through all descendant nodes
    all_variables = assembly_registry.list_variables()

    return {var.name for var in all_variables}


def field_name_to_title(field_name: str, heading_level: int = 1) -> str:
    """
    Convert a field name to a proper heading title.

    Args:
        field_name: Field name to convert (e.g., 'main_analysis')
        heading_level: Markdown heading level (1-6)

    Returns:
        Formatted heading (e.g., '# Main Analysis')
    """
    # Convert underscore to spaces and title case
    title = field_name.replace("_", " ").title()

    # Generate heading markdown
    heading_prefix = "#" * max(1, min(6, heading_level))

    return f"{heading_prefix} {title}"


def detect_heading_level(content: str, template_var_position: int) -> int:
    """
    Detect the appropriate heading level for template variable resolution.

    Args:
        content: Full docstring content
        template_var_position: Position of the template variable in content

    Returns:
        Heading level (1-6) based on context
    """
    # Find the preceding content up to the template variable
    preceding_content = content[:template_var_position]

    # Look for existing headings in the preceding content
    heading_pattern = re.compile(r"^(#{1,6})\s", re.MULTILINE)
    headings = list(heading_pattern.finditer(preceding_content))

    if not headings:
        # No existing headings, start at level 1
        return 1

    # Find the most recent heading
    last_heading = headings[-1]
    last_level = len(last_heading.group(1))

    # Return the next level down (for child content)
    return min(6, last_level + 1)


def strip_acl_commands(content: str) -> str:
    """
    Strip LangTree DSL command lines from content to avoid conflicts with template variable processing.

    LangTree DSL commands are lines starting with '!' and should be processed separately from
    template variables. This function removes them to create a clean prompt for
    template variable processing.

    Args:
        content: Raw docstring or field description content

    Returns:
        Clean content with LangTree DSL command lines removed
    """
    if not content:
        return content

    lines = content.split("\n")
    clean_lines = []

    for line in lines:
        stripped_line = line.lstrip()
        # Skip lines that start with LangTree DSL command prefix '!'
        if not stripped_line.startswith("!"):
            clean_lines.append(line)

    return "\n".join(clean_lines)


def process_template_variables(
    content: str, node: Optional["StructureTreeNode"] = None
) -> str:
    """
    Process template variables in content, applying automatic addition and validation.

    Strips LangTree DSL commands first to avoid conflicts with template variable processing,
    then processes template variables on the clean content.

    Args:
        content: Docstring or field description content
        node: Optional structure tree node for context

    Returns:
        Processed content with template variables handled

    Raises:
        TemplateVariableNameError: If unknown template variables are found
        TemplateVariableSpacingError: If spacing validation fails
        TemplateVariableConflictError: If template variables conflict with Assembly Variables
    """
    if not content:
        content = ""

    # Strip LangTree DSL commands first to avoid conflicts with template variable processing
    clean_content = strip_acl_commands(content)

    # Add automatic PROMPT_SUBTREE if not present
    clean_content = add_automatic_prompt_subtree(clean_content)

    # Validate template variable names on clean content
    name_errors = validate_template_variable_names(clean_content)
    if name_errors:
        raise TemplateVariableNameError(
            f"Template variable name errors: {'; '.join(name_errors)}"
        )

    # Validate spacing on clean content
    spacing_errors = validate_template_variable_spacing(clean_content)
    if spacing_errors:
        raise TemplateVariableSpacingError(
            f"Template variable spacing errors: {'; '.join(spacing_errors)}"
        )

    # Validate conflicts with Assembly Variables when node context is available
    if node:
        assembly_vars = get_assembly_variables_for_node(node)
        conflict_errors = validate_template_variable_conflicts(
            clean_content, assembly_vars
        )
        if conflict_errors:
            raise TemplateVariableConflictError(
                f"Template variable conflicts: {'; '.join(conflict_errors)}"
            )

    # Return original content with PROMPT_SUBTREE added - LangTree DSL commands preserved for later parsing
    return add_automatic_prompt_subtree(content)


def resolve_prompt_subtree(
    node: "StructureTreeNode", base_heading_level: int = 1
) -> str:
    """
    Resolve {PROMPT_SUBTREE} template variable for a given node.

    Args:
        node: Structure tree node to resolve subtree for
        base_heading_level: Base heading level for field titles

    Returns:
        Resolved content with field titles and descriptions
    """
    if not node or not node.field_type:
        return ""

    content_parts = []

    # Process each field in the node's type
    for field_name, field_def in node.field_type.model_fields.items():
        # Generate field title
        field_title = field_name_to_title(field_name, base_heading_level)
        content_parts.append(field_title)

        # Add field description if available
        if field_name in node.clean_field_descriptions:
            description = node.clean_field_descriptions[field_name]
            content_parts.append(description)
        elif field_def.description:
            # Use original description if clean version not available
            content_parts.append(field_def.description)

        # Add empty line after each field section
        content_parts.append("")

    # Join parts and clean up trailing empty lines
    result = "\n\n".join(content_parts).rstrip()
    return result


def resolve_collected_context(
    node: "StructureTreeNode", context_data: str | None = None
) -> str:
    """
    Resolve {COLLECTED_CONTEXT} template variable for a given node.

    Args:
        node: Structure tree node to resolve context for
        context_data: Optional context data to include

    Returns:
        Resolved context content
    """
    if context_data:
        return context_data

    # Implement basic context collection from node hierarchy
    if not node:
        return "# Context\n\n*No context available*"

    context_parts = []

    # Collect context from parent nodes (following hierarchical prompt assembly)
    # Check if node has parent attribute before accessing it
    if hasattr(node, "parent") and node.parent:
        current = node.parent
        while current:
            if hasattr(current, "clean_docstring") and current.clean_docstring:
                # Add parent context with appropriate heading
                parent_name = current.name.replace("_", " ").title()
                context_parts.append(f"## {parent_name}")
                context_parts.append(current.clean_docstring)
                context_parts.append("")  # Empty line separator
            current = current.parent if hasattr(current, "parent") else None

    # Collect context from processed field descriptions in current node
    if hasattr(node, "clean_field_descriptions") and node.clean_field_descriptions:
        context_parts.append("## Field Context")
        for field_name, description in node.clean_field_descriptions.items():
            field_title = field_name.replace("_", " ").title()
            context_parts.append(f"### {field_title}")
            context_parts.append(description)
            context_parts.append("")  # Empty line separator

    # If we have collected context, format it properly
    if context_parts:
        # Remove trailing empty line
        if context_parts and context_parts[-1] == "":
            context_parts.pop()
        return "\n".join(context_parts)

    # Default placeholder if no context collected
    return "# Context\n\n*Context data will be provided during execution*"


def resolve_template_variables_in_content(
    content: str, node: "StructureTreeNode"
) -> str:
    """
    Resolve all template variables in content.

    Args:
        content: Content containing template variables
        node: Structure tree node for context

    Returns:
        Content with template variables resolved
    """
    if not content:
        return ""

    result = content

    # Resolve PROMPT_SUBTREE
    def replace_prompt_subtree(match):
        # Detect heading level at this position
        position = match.start()
        heading_level = detect_heading_level(content, position)
        return resolve_prompt_subtree(node, heading_level)

    result = PROMPT_SUBTREE_PATTERN.sub(replace_prompt_subtree, result)

    # Resolve COLLECTED_CONTEXT
    def replace_collected_context(match):
        return resolve_collected_context(node)

    result = COLLECTED_CONTEXT_PATTERN.sub(replace_collected_context, result)

    return result
