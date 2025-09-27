"""Runtime construction & description of Pydantic models from JSON Schema fragments.

Primary features:
    - Resolve primitive or custom referenced types (including simple union syntax `a | b | c`).
    - Convert JSON Schema (subset) into concrete Pydantic model classes preserving field
        descriptions for prompt embedding.
    - Provide lightweight natural language description helpers for models and fields.

Limitations / TODO:
    - TODO: Support nested array/object constraints beyond simple `properties` copying.
    - TODO: Support `required` lists to distinguish optional vs mandatory fields.
    - TODO: Expand `$ref` handling to detect circular references gracefully.
"""

import builtins
from types import UnionType
from typing import Any, TypeAlias, Union, cast

from pydantic import BaseModel, Field, create_model

NestedStrDict: TypeAlias = dict[str, Union[str, "NestedStrDict", list["NestedStrDict"]]]

_type_mapping = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
    "none": "None",
    "object": dict,
    "array": list,
}


def get_type(name: str, custom_types: dict[str, type] = {}) -> UnionType | type | None:
    """Resolve a primitive or custom type name (supports simple union syntax).

    Supports a mini DSL where unions are expressed with the first '|' delimiting
    the remainder (recursively parsed). Mapping uses `_type_mapping` for schema
    aliases (e.g. 'integer' -> 'int').

    Params:
        name: Type name or pipe-delimited union specification.
        custom_types: Registry of previously generated dynamic model classes.

    Returns:
        Concrete Python `type` or `types.UnionType` (PEP604) representing the resolved type.

    Raises:
        ValueError: If any referenced type token cannot be resolved.
    """
    rest_type = None
    rest_defined = False
    if "|" in name:
        name, rest = name.split("|", 1)
        rest_type = get_type(rest.strip(), custom_types)
        rest_defined = True
    name = name.strip()
    name = _type_mapping.get(name, name)
    if hasattr(builtins, name):
        t = getattr(builtins, name)
        t = t if t is not None else type(None)
        if not isinstance(t, type):
            raise ValueError(f"{name} is not a type.")
    elif name in custom_types:
        t = custom_types[name]
    else:
        raise ValueError(f"Type {name} is not defined in builtins or custom types.")

    if rest_defined:
        t = t | rest_type
    return t


def resolve_type(
    field: NestedStrDict, custom_types: dict[str, type] = {}
) -> UnionType | type | None:
    """Derive a Python type (or union) from a JSON Schema field fragment.

    Recognized keys (precedence order):
      - 'type': Single primitive or mapped alias.
      - 'anyOf': Union of multiple schema fragments each containing a 'type'.
      - '$ref': Reference to a previously constructed definition in `custom_types`.

    Params:
        field: JSON-like dict describing the field.
        custom_types: Registry of already generated model classes.

    Returns:
        Concrete Python `type` or union.

    Raises:
        ValueError: If schema is unsupported / reference not found.
    """
    if "type" in field:
        raw_type = field.get("type")
        if not isinstance(raw_type, str):
            raise ValueError(
                f"Invalid 'type' value; expected string, got: {type(raw_type)}"
            )
        t = get_type(raw_type, custom_types)
    elif "anyOf" in field:
        any_of_raw = field.get("anyOf")
        if not isinstance(any_of_raw, list):
            raise ValueError("'anyOf' must be a list of schema fragments")
        any_of_list: list[NestedStrDict] = cast(list[NestedStrDict], any_of_raw)
        if not any_of_list:
            raise ValueError("'anyOf' list cannot be empty")
        first_type_fragment = any_of_list[0]
        first_type_name = first_type_fragment.get("type")
        if not isinstance(first_type_name, str):
            raise ValueError("Each 'anyOf' fragment must contain a string 'type'")
        t = get_type(first_type_name, custom_types)
        for frag in any_of_list[1:]:
            frag_type_name = frag.get("type")
            if not isinstance(frag_type_name, str):
                raise ValueError("Each 'anyOf' fragment must contain a string 'type'")
            frag_type = get_type(frag_type_name, custom_types)
            if frag_type is None:
                raise ValueError(
                    f"Could not resolve union fragment type '{frag_type_name}'"
                )
            if t is None:
                t = frag_type
            else:
                t = t | frag_type  # type: ignore[operator]
    elif "$ref" in field:
        ref_raw = field.get("$ref")
        if not isinstance(ref_raw, str):
            raise ValueError("'$ref' must be a string")
        type_name = ref_raw.split("/")[-1]
        t = custom_types.get(type_name, None)
        if t is None:
            raise ValueError(f"Type {type_name} is not defined in custom types.")
    else:
        raise ValueError(
            "Field does not have a `type` or `anyOf` defined. Field json:\n"
            + str(field)
        )
    return t


def describe_datafield_type(field: NestedStrDict) -> str:
    """Render a concise inline code formatted description of a field's type.

    Example outputs: `int`, `str | None`, `CustomModel`.

    Params:
        field: JSON Schema fragment.

    Returns:
        Markdown-ready backticked type description.

    Raises:
        ValueError: If schema lacks required keys or produces invalid type text.
    """
    if "type" in field:
        raw_type = field.get("type")
        if not isinstance(raw_type, str):
            raise ValueError("'type' must be a string")
        field_type = _type_mapping.get(raw_type, raw_type)
        type_descr = f"`{field_type}`"
    elif "anyOf" in field:
        any_of_raw = field.get("anyOf")
        if not isinstance(any_of_raw, list):
            raise ValueError("'anyOf' must be a list")
        collected: list[str] = []
        for frag in any_of_raw:
            if not isinstance(frag, dict):
                raise ValueError("Each 'anyOf' fragment must be a dict")
            frag_type = frag.get("type")
            if not isinstance(frag_type, str):
                raise ValueError("Each 'anyOf' fragment must have string 'type'")
            collected.append(_type_mapping.get(frag_type, frag_type))
        type_descr = f"`{' | '.join(collected)}`"
    elif "$ref" in field:
        ref_raw = field.get("$ref")
        if not isinstance(ref_raw, str):
            raise ValueError("'$ref' must be a string")
        type_name = ref_raw.split("/")[-1]
        type_descr = f"`{type_name}`"
    else:
        raise ValueError(
            "Field does not have a type or anyOf defined. Field json:\n" + str(field)
        )
    if not isinstance(type_descr, str):
        raise ValueError(
            f"Field type is not a string: {type_descr}. Field json:\n{field}"
        )
    return type_descr


def describe_model(model: BaseModel, describe_fields: bool = False) -> str:
    """Generate a human-readable description of a Pydantic model.

    Params:
        model: Pydantic model class instance with schema metadata.
        describe_fields: Include per-field sections when True.

    Returns:
        Markdown string listing model description and optionally field details.

    Raises:
        ValueError: If model exposes no properties (invalid schema).
    """
    model_schema = model.model_json_schema()
    name = model_schema.get("title", model.__name__)
    docstring = model_schema.get("description", "")
    properties = model_schema.get("properties", None)
    if properties is None:
        raise ValueError("Model has no properties defined.")
    model_description = f"## Model\n\nModel is defined by a Pydantic shema {name}.\n{docstring}\n{name} fields are: {', '.join(properties)}.\n"
    if not describe_fields:
        return model_description
    field_descriptions = [
        f"\n### Field `{name}`\n\n{property['description']} {name.title()} is of a type {describe_datafield_type(property)}."
        for name, property in properties.items()
    ]
    field_description = "## Fields\n" + "\n".join(field_descriptions)
    return model_description + "\n" + field_description


def schema_to_model(
    name: str,
    schema: NestedStrDict,
    custom_types: dict[str, type] = {},
    return_full: bool = False,
    replace_existing: bool = False,
    new_types: set[str] = set(),
) -> BaseModel | tuple[BaseModel, dict[str, type], set[str]]:
    """Create (and optionally register) a Pydantic model from JSON Schema.

    Params:
        name: Target model class name.
        schema: JSON Schema fragment containing `properties` and optional `$defs`.
        custom_types: Registry of existing types for `$ref` resolution.
        return_full: When True, also return updated registry + newly added names.
        replace_existing: Overwrite previously generated type with same name.
        new_types: Accumulator of names generated in this invocation branch.

    Returns:
        Model class or tuple with registry details when `return_full` is True.
    """
    if name in custom_types:
        if not replace_existing or name in new_types:
            return custom_types[name]  # type: ignore
    custom_types = custom_types.copy()
    new_types = new_types.copy()
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})
    for type_name, type_schema in defs.items():  # type: ignore[assignment]
        if (
            type_name in custom_types and not replace_existing
        ) or type_name in new_types:
            continue
        if not isinstance(type_schema, dict):
            raise ValueError(
                f"Definition for {type_name} must be a dict schema fragment"
            )
        nested_model, custom_types, new_types = schema_to_model(
            type_name,
            type_schema,  # type: ignore[arg-type]
            custom_types=custom_types,
            return_full=True,
            replace_existing=replace_existing,
            new_types=new_types,
        )
    if not isinstance(properties, dict):
        raise ValueError("'properties' must be a dict")
    # Relax typing for dynamic model synthesis; Pydantic will validate at runtime.
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, type_schema in properties.items():
        if not isinstance(type_schema, dict):
            raise ValueError(f"Property '{field_name}' schema must be a dict")
        resolved = resolve_type(type_schema, custom_types)
        description_val = type_schema.get("description", None)
        if not (isinstance(description_val, str) or description_val is None):
            description_val = str(description_val)
        fields[field_name] = (
            resolved or (str | None),
            Field(description=description_val),
        )  # type: ignore
    doc_raw = schema.get("description", None)
    if not (isinstance(doc_raw, str) or doc_raw is None):
        doc_raw = str(doc_raw)
    output_model = create_model(name, __doc__=doc_raw, **fields)  # type: ignore[arg-type]
    custom_types[name] = output_model
    new_types.add(name)
    if return_full:
        return output_model, custom_types, new_types
    return output_model
