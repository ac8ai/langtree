"""High-level chain construction helpers for LLM prompt execution.

This module provides helper functions that assemble LangChain style runnable
pipelines used throughout the project. Responsibilities are intentionally
limited to:
    - Constructing a prompt template with consistent section ordering
    - Applying optional structured output parsing (Pydantic) when requested
    - Injecting sampling signatures and sample blocks for re‑prompting flows

Design notes:
    - Functions return `Runnable` objects rather than executing eagerly so callers
        can compose additional operators (e.g. mapping, logging) before invocation.
    - Validation is minimal; upstream code is expected to supply coherent prompt
        fragments (system/context/task/output/input) already sanitized.
    - Sampling utilities wrap an existing chain instead of duplicating logic.
"""

import random
from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel
from uuid import uuid4

from langtree.dynamic import describe_model
from langtree.models import LLMProvider

_llm_provider = LLMProvider()

def prepare_chain(
    llm_name: str,
    prompt_system: str,
    prompt_context: str | None,
    prompt_task: str,
    prompt_output: str | None,
    prompt_input: str | None,
    parse_as_string: bool = False,
    structured_output: type[BaseModel] | None = None,
    for_sampling: bool = False,
    samples_template: str | None = None,
) -> Runnable:
    """
    Assemble a runnable LLM chain with standardized prompt sections.

    Constructs a LangChain runnable pipeline with consistent prompt structure
    used throughout the project. Handles both string and structured output
    parsing with proper LangChain API integration.

    Sections are rendered in the following order (omitting any optional context):
      1. System message (high level behavioral instructions)
      2. Human message containing: Context, Task, Output spec, (optional model schema), Input, (optional Samples)

    When `structured_output` is provided, uses LangChain's modern
    `.with_structured_output()` API for Pydantic model binding. If 
    `parse_as_string` is True, a `StrOutputParser` is appended (mutually 
    exclusive with structured output binding).

    Params:
        llm_name: Registered model identifier from LLMProvider
        prompt_system: System prompt segment with behavioral instructions
        prompt_context: Optional context block (None omits block entirely)
        prompt_task: Task instructions block defining the work to be done
        prompt_output: Output expectations block specifying format requirements
        prompt_input: User input/payload section with actual data to process
        parse_as_string: Append raw string parser (disallowed with structured output)
        structured_output: Optional Pydantic model class for structured response parsing
        for_sampling: If True, injects signature placeholder for sampling workflows
        samples_template: Optional pre-rendered samples block for resampling flows

    Returns:
        A composed `Runnable` ready for additional mapping or direct invocation

    Raises:
        ValueError: If both `structured_output` and `parse_as_string` are requested
    """
    llm = _llm_provider.get_llm(llm_name)
    if structured_output is not None:
        llm = llm.with_structured_output(structured_output)
    human_message = ''.join((
        f'# Context\n\n{prompt_context.strip()}\n' if prompt_context is not None else "",
        f'\n# Task\n\n{prompt_task.strip()}\n',
        f'\n# Output\n\n{prompt_output.strip()}\n' if prompt_output is not None else "",
        describe_model(structured_output(), describe_fields=True) if structured_output is not None else "",
        f'\n# Input\n\n{prompt_input}\n' if prompt_input else "",
        f"\n## Samples\n\n{samples_template}\n" if samples_template else "",
        "\n<ignore>{signature}</ignore>\n" if for_sampling else ""
    ))
    messages = [
        SystemMessagePromptTemplate.from_template(prompt_system),
        HumanMessagePromptTemplate.from_template(human_message)
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt_template | llm
    if parse_as_string:
        if structured_output is not None:
            raise ValueError("Structured output is not allowed in this chain.")
        chain = chain | StrOutputParser()
    return chain

def sample_mapper(input_data: dict[str, str]) -> dict[str, str]:
    """Inject a unique signature marker into input mapping for sampling runs.

    The injected `<ignore>` signature enables downstream prompts to distinguish
    and safely ignore per-sample metadata while preserving traceability.

    Params:
        input_data: Original mapping passed into the chain invocation.

    Returns:
        A shallow copy of `input_data` with an added `signature` key whose value
        is a UUID wrapped in an `<ignore>` tag.
    """
    output_data = input_data.copy()
    output_data['signature'] = f'<ignore>{str(uuid4())}</ignore>'
    return output_data
    

def sample_chain(
    chain: Runnable,
    n_samples: int,
    original_task: str,
    prompts: dict[str, str],
    skip_context_prompt: bool = False,
    output_as_string: bool = False,
) -> Runnable:
    """Wrap an existing chain to produce N independently sampled executions.

    A lightweight resampling prompt (summarization LLM) is constructed that
    consumes individual sample outputs as templated placeholders. Each sample
    is produced by piping the original `chain` through a UUID signature mapper.

    Params:
        chain: Base runnable to execute for each sample slot.
        n_samples: Number of samples to generate.
        original_task: Task text injected into the resampling template.
        prompts: Dictionary of prompt sections (system, context, task, output).
        skip_context_prompt: If True, omit context block in sampling template.
        output_as_string: If True, final aggregated resample output is parsed to raw string.

    Returns:
        Runnable producing aggregated re‑prompt output (string or model-bound response).
    """
    if skip_context_prompt:
        prompts['prompt_context'] = None
    prompts['prompt_task'] = prompts['prompt_task'].format(original_task=original_task)
    samples_template = "\n\n".join(
        "<sample>\n{sample_" + str(i + 1) + "}\n</sample>"
        for i in range(n_samples)
    )
    sampling_chain = prepare_chain(
        llm_name='summarization',
        **prompts,
        samples_template=samples_template,
    )
    signed_chain = RunnableLambda(sample_mapper) | chain
    samples = {
        f'sample_{i + 1}': signed_chain
        for i in range(n_samples)
    }
    resample = samples | sampling_chain
    if output_as_string:
        resample = resample | StrOutputParser()
    return resample
