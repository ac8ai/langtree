"""
Template variable processing for LangTree DSL framework.

This module handles the detection, validation, and processing of template variables
({PROMPT_SUBTREE} and {COLLECTED_CONTEXT}) in LangTree DSL docstrings and field descriptions.
"""

# Group 1: External direct imports (alphabetical)
import re
from dataclasses import dataclass

# Group 2: External from imports (alphabetical by source module)
from typing import TYPE_CHECKING, Optional

# Group 4: Internal from imports (alphabetical by source module)
from langtree.core.tree_node import TreeNode
from langtree.exceptions import (
    TemplateVariableConflictError,
    TemplateVariableNameError,
    TemplateVariableSpacingError,
)
from langtree.templates.utils import ExtractedCommand

if TYPE_CHECKING:
    from langtree.structure.builder import RunStructure, StructureTreeNode


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


@dataclass(frozen=True)
class ProcessedDocstring:
    """
    Processed docstring from a single class in the inheritance chain.

    Params:
        commands: Extracted DPCL commands with line numbers
        clean: Clean docstring content (stripped, commands removed)
        line_offset: Number of lines removed from top of original docstring
        source_class: Name of the class this docstring came from
        merged_start: Line number where this segment starts in merged content (set during merge)
    """

    commands: tuple[ExtractedCommand, ...]
    clean: str
    line_offset: int
    source_class: str
    merged_start: int = 0


@dataclass(frozen=True)
class LineMapping:
    """
    Line number mapping from merged docstring back to original class docstring.

    Enables error reporting to pinpoint exact class and line number where
    an error occurred in the original source.

    Params:
        class_name: Name of the class this segment came from
        merged_start: Line number where this segment starts in merged docstring
        merged_end: Line number where this segment ends in merged docstring (exclusive)
        original_offset: Lines removed from top of original class docstring
    """

    class_name: str
    merged_start: int
    merged_end: int
    original_offset: int


def process_class_docstring(content: str, class_name: str) -> ProcessedDocstring:
    """
    Process a single class docstring with line tracking.

    Extracts DPCL commands from the top of the docstring and validates that
    each template variable appears at most once. Tracks line numbers for
    accurate error reporting.

    Params:
        content: Raw docstring content from a single class
        class_name: Name of the class (for error reporting)

    Returns:
        ProcessedDocstring with commands, clean content, and line offset

    Raises:
        TemplateVariableNameError: If template variable appears more than once
    """
    from langtree.templates.utils import extract_commands

    if not content or not content.strip():
        return ProcessedDocstring(
            commands=(),
            clean="",
            line_offset=0,
            source_class=class_name,
            merged_start=0,
        )

    # Extract commands (they appear at top before content)
    commands, clean_unstripped = extract_commands(content)

    # Calculate line offset BEFORE stripping
    # This is where clean content starts in the original
    if clean_unstripped:
        clean_start_pos = content.index(clean_unstripped)
        line_offset = content[:clean_start_pos].count("\n")
    else:
        # No clean content (all commands or empty)
        line_offset = content.count("\n")

    # Now strip the clean content
    clean_stripped = clean_unstripped.strip()

    # Validate no duplicate template variables in this class
    duplicate_errors = validate_no_duplicate_template_variables(clean_stripped)
    if duplicate_errors:
        raise TemplateVariableNameError(
            f"In class {class_name}: {'; '.join(duplicate_errors)}"
        )

    return ProcessedDocstring(
        commands=tuple(commands),
        clean=clean_stripped,
        line_offset=line_offset,
        source_class=class_name,
        merged_start=0,
    )


def merge_processed_docstrings(
    processed_docs: list[ProcessedDocstring],
) -> tuple[str, list[LineMapping]]:
    """
    Merge multiple processed docstrings with line number tracking.

    Concatenates clean docstring content from multiple classes and creates
    line mappings to enable reverse lookup from merged line numbers back to
    original class and line numbers.

    Params:
        processed_docs: List of processed docstrings from inheritance chain

    Returns:
        Tuple of (merged_content, line_mappings) where line_mappings enable
        finding original class and line number for any line in merged content
    """
    if not processed_docs:
        return "", []

    merged_parts = []
    line_mappings = []
    current_line = 0

    for doc in processed_docs:
        if not doc.clean:
            # Skip empty docstrings
            continue

        segment_start = current_line
        line_count = doc.clean.count("\n") + 1  # +1 because last line has no \n

        # Create line mapping for this segment
        line_mappings.append(
            LineMapping(
                class_name=doc.source_class,
                merged_start=segment_start,
                merged_end=segment_start + line_count,
                original_offset=doc.line_offset,
            )
        )

        # Add clean content
        merged_parts.append(doc.clean)
        current_line += line_count

    # Join with double newlines between segments
    merged = "\n\n".join(merged_parts)

    # Adjust line mappings to account for separator lines
    # Each separator adds 1 line (the blank line between segments)
    if len(line_mappings) > 1:
        # Adjust all mappings after the first one
        separator_count = 0
        adjusted_mappings = []

        for i, mapping in enumerate(line_mappings):
            if i > 0:
                separator_count += 1  # One blank line before this segment

            adjusted_mappings.append(
                LineMapping(
                    class_name=mapping.class_name,
                    merged_start=mapping.merged_start + separator_count,
                    merged_end=mapping.merged_end + separator_count,
                    original_offset=mapping.original_offset,
                )
            )

        return merged, adjusted_mappings

    return merged, line_mappings


def collect_inherited_docstrings(tree_node_class: type[TreeNode]) -> str:
    """
    Collect docstrings from Python class inheritance chain up to TreeNode.

    Walks the class inheritance hierarchy (MRO) and collects docstrings from
    all parent classes up to (but not including) TreeNode itself. Docstrings
    are concatenated in MRO order (child last, most distant parent first).

    Params:
        tree_node_class: The TreeNode subclass to collect docstrings for

    Returns:
        Concatenated docstrings from inheritance chain, or empty string if none
    """
    from textwrap import dedent

    if not tree_node_class or not issubclass(tree_node_class, TreeNode):
        return ""

    docstrings = []

    # Walk MRO (Method Resolution Order) - skip first (self) and last (object)
    # Stop when we hit TreeNode itself
    for base_class in tree_node_class.__mro__[:-1]:  # Exclude 'object'
        # Stop at TreeNode - don't include its docstring
        if base_class is TreeNode:
            break

        # Collect docstring if present, using dedent to remove indentation
        if base_class.__doc__:
            # Use textwrap.dedent to properly dedent docstrings
            # Unlike inspect.cleandoc, this treats all lines uniformly including the first line
            cleaned = dedent(base_class.__doc__).strip()
            if cleaned:  # Only add if non-empty after cleaning
                docstrings.append(cleaned)

    # Reverse to get parent-first order (most distant parent → child)
    docstrings.reverse()

    # Join with double newlines to maintain markdown separation
    return "\n\n".join(docstrings) if docstrings else ""


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
    matches = list(all_variable_pattern.finditer(content))

    for match in matches:
        var_name = match.group(1)

        # Check if this looks like a misspelled template variable
        # Only check for case variations and typos of the actual template variables
        is_likely_template_error = False
        error_suggestion = None

        # Check for case-insensitive match with template variables
        if var_name.upper() == "PROMPT_SUBTREE" and var_name != "PROMPT_SUBTREE":
            is_likely_template_error = True
            error_suggestion = f"'{{{var_name}}}' appears to be a misspelled template variable. Did you mean '{{PROMPT_SUBTREE}}'?"
        elif (
            var_name.upper() == "COLLECTED_CONTEXT" and var_name != "COLLECTED_CONTEXT"
        ):
            is_likely_template_error = True
            error_suggestion = f"'{{{var_name}}}' appears to be a misspelled template variable. Did you mean '{{COLLECTED_CONTEXT}}'?"
        # Check for common typos with high similarity
        elif var_name in ["PROMPT_TREE", "PROMPTSUBTREE", "PROMPT", "SUBTREE"]:
            is_likely_template_error = True
            error_suggestion = f"'{{{var_name}}}' appears to be a misspelled template variable. Did you mean '{{PROMPT_SUBTREE}}'?"
        elif var_name in [
            "COLLECTED",
            "CONTEXT",
            "COLLECT_CONTEXT",
            "COLLECTEDCONTEXT",
        ]:
            is_likely_template_error = True
            error_suggestion = f"'{{{var_name}}}' appears to be a misspelled template variable. Did you mean '{{COLLECTED_CONTEXT}}'?"

        if is_likely_template_error and error_suggestion:
            line_number = content[: match.start()].count("\n") + 1
            errors.append(f"{error_suggestion} at line {line_number}")

    # Check for variables without lowercase letters (reserved for template variables)
    for match in matches:
        var_name = match.group(1)

        # Skip if it's a valid template variable
        if var_name in VALID_TEMPLATE_VARIABLES:
            continue

        # Check if variable has no lowercase letters (reserved namespace)
        if not any(c.islower() for c in var_name):
            line_number = content[: match.start()].count("\n") + 1
            errors.append(
                f"Variable '{{{var_name}}}' at line {line_number} uses naming without lowercase letters "
                f"which is reserved for template variables. "
                f"Use lowercase ('{{{var_name.lower()}}}'), mixed case, or add lowercase letters."
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

            # CRITICAL: First check if there's non-whitespace text on the same line
            # If lines[0] has any non-whitespace content, that's a spacing violation
            if lines and lines[0].strip() != "":
                # There's text on the same line as the template variable - this is invalid
                after_valid = False
            elif len(lines) <= 1:
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


def validate_no_duplicate_template_variables(content: str) -> list[str]:
    """
    Validate that template variables don't appear multiple times.

    Checks that each template variable (PROMPT_SUBTREE, COLLECTED_CONTEXT)
    appears at most once in the content. Multiple occurrences indicate a
    configuration error.

    Params:
        content: Text content to validate

    Returns:
        List of error messages (empty if no duplicates found)
    """
    errors = []

    # Detect all template variables
    detected = detect_template_variables(content)

    # Check for duplicates
    for var_name, positions in detected.items():
        if len(positions) > 1:
            errors.append(
                f"Template variable {{{var_name}}} appears {len(positions)} times "
                f"(positions: {positions}). Each template variable must appear at most once."
            )

    return errors


def add_automatic_template_variables(
    content: str, is_field_description: bool = False
) -> str:
    """
    Add {COLLECTED_CONTEXT} and {PROMPT_SUBTREE} to docstring if not already present.

    Adds template variables in the correct order:
    1. {COLLECTED_CONTEXT} - for sibling field values (added first)
    2. {PROMPT_SUBTREE} - for current node's field structure (added second)

    Args:
        content: Docstring or field description content
        is_field_description: If True, template variables will NOT be added.
                             Template variables should only appear in node docstrings,
                             not in field descriptions.

    Returns:
        Content with template variables added if they weren't present (unless is_field_description=True)
    """
    if not content:
        content = ""

    # Never add template variables to field descriptions
    # Template variables only belong in node docstrings
    if is_field_description:
        return content

    has_collected_context = COLLECTED_CONTEXT_PATTERN.search(content)
    has_prompt_subtree = PROMPT_SUBTREE_PATTERN.search(content)

    # If both are already present, return as-is
    if has_collected_context and has_prompt_subtree:
        return content

    # Prepare content with proper spacing
    if content.strip():
        if not content.endswith("\n\n"):
            content = content.rstrip() + "\n\n"
    else:
        # Empty content - add both template variables
        return "{COLLECTED_CONTEXT}\n\n{PROMPT_SUBTREE}"

    # Add missing template variables in order: COLLECTED_CONTEXT first, PROMPT_SUBTREE second
    if not has_collected_context:
        content += "{COLLECTED_CONTEXT}\n\n"

    if not has_prompt_subtree:
        content += "{PROMPT_SUBTREE}\n\n"

    return content


def add_automatic_prompt_subtree(content: str) -> str:
    """
    Add {PROMPT_SUBTREE} to docstring if not already present.

    DEPRECATED: Use add_automatic_template_variables() instead which adds both
    {COLLECTED_CONTEXT} and {PROMPT_SUBTREE} in the correct order.

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

    # Note: For access to RunStructure, use get_assembly_variables_for_node_with_structure()
    # which provides the full implementation with proper Assembly Variable registry access

    # Return empty set for now to prevent errors
    # The full implementation should:
    # 1. Get RunStructure from context (available via _with_structure variant)
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

    # Assembly Variables are available from definition node through all descendant nodes
    # Walk up the parent chain and collect variables defined at each ancestor
    available_vars = set()
    current = node
    while current:
        # Get variables defined at this node
        for var in assembly_registry.list_variables():
            if var.source_node_tag == current.name:
                available_vars.add(var.name)
        current = current.parent

    return available_vars


def field_name_to_title(field_name: str, heading_level: int = 1) -> str:
    """
    Convert a field name to a proper heading title.

    Handles underscore_case, camelCase, and numbers in field names.

    Args:
        field_name: Field name to convert (e.g., 'main_analysis', 'fieldName2')
        heading_level: Markdown heading level (1-6+)

    Returns:
        Formatted heading (e.g., '# Main Analysis', '## Field Name 2')
    """
    import re

    # Insert space before uppercase letters (camelCase)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name)

    # Insert space before numbers
    spaced = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", spaced)

    # Insert space after numbers when followed by letters
    spaced = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", spaced)

    # Replace underscores with spaces
    spaced = spaced.replace("_", " ")

    # Title case each word
    title = spaced.title()

    # Generate heading markdown
    # Allow levels > 6 for deep nesting (invalid markdown but LLMs handle it)
    heading_prefix = "#" * max(1, heading_level)

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
    # Allow any number of # for deep nesting (not just 1-6)
    heading_pattern = re.compile(r"^(#+)\s", re.MULTILINE)
    headings = list(heading_pattern.finditer(preceding_content))

    if not headings:
        # No existing headings, start at level 1
        return 1

    # Find the most recent heading
    last_heading = headings[-1]
    last_level = len(last_heading.group(1))

    # Return the next level down (for child content)
    # Allow levels > 6 for deep nesting
    return last_level + 1


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
    content: str,
    node: Optional["StructureTreeNode"] = None,
    is_field_description: bool = False,
) -> str:
    """
    Process template variables in content, applying automatic addition and validation.

    Strips LangTree DSL commands first to avoid conflicts with template variable processing,
    then processes template variables on the clean content.

    Args:
        content: Docstring or field description content
        node: Optional structure tree node for context
        is_field_description: If True, template variables will NOT be automatically added.
                             Template variables should only appear in node docstrings.

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

    # Add automatic template variables (COLLECTED_CONTEXT and PROMPT_SUBTREE) if not present
    # Only for docstrings, not for field descriptions
    clean_content = add_automatic_template_variables(
        clean_content, is_field_description=is_field_description
    )

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

    # Validate no duplicate template variables
    duplicate_errors = validate_no_duplicate_template_variables(clean_content)
    if duplicate_errors:
        raise TemplateVariableNameError(
            f"Duplicate template variables: {'; '.join(duplicate_errors)}"
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


# OLD STRING-BASED RESOLUTION FUNCTIONS REMOVED
# These have been replaced with element-based resolution:
# - resolve_prompt_subtree() → use resolve_prompt_subtree_elements()
# - resolve_collected_context() → use resolve_collected_context_elements()
# - resolve_template_variables_in_content() → use node.get_prompt(previous_values={...})
# See StructureTreeNode.get_prompt() in langtree/structure/builder.py for production API
