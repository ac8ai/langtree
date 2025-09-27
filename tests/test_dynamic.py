from typing import get_args
from pydantic import BaseModel
import pytest

from langtree.dynamic import describe_datafield_type, get_type, resolve_type, schema_to_model


def test_get_type_simple_types():
    assert get_type("string") is str, "Expected 'string' to resolve to str"
    assert get_type("str") is str, "Expected 'str' to resolve to str"
    assert get_type("integer") is int, "Expected 'integer' to resolve to int"
    assert get_type("int") is int, "Expected 'int' to resolve to int"
    assert get_type("number") is float, "Expected 'number' to resolve to float"
    assert get_type("float") is float, "Expected 'float' to resolve to float"
    assert get_type("boolean") is bool, "Expected 'boolean' to resolve to bool"
    assert get_type("bool") is bool, "Expected 'bool' to resolve to bool"
    assert get_type("null") is type(None), "Expected 'null' to resolve to NoneType"
    assert get_type("none") is type(None), "Expected 'none' to resolve to NoneType"


def test_get_type_union_types():
    t = get_type("int | None")
    assert get_args(t) == (int, type(None)), "Expected 'int | None' to resolve to a union of int and NoneType"

class SimpleModel(BaseModel):
    name: str
    age: int

class ComplexModel(BaseModel):
    person: SimpleModel
    is_active: bool
    name: str | None

def test_get_type_pydantic_types():
    custom_types = {
        "SimpleModel": SimpleModel,
        "ComplexModel": ComplexModel,
    }
    assert get_type("SimpleModel", custom_types) is SimpleModel, "Expected 'SimpleModel' to resolve to SimpleModel"
    assert get_type("ComplexModel", custom_types) is ComplexModel, "Expected 'ComplexModel' to resolve to ComplexModel"

def test_resolve_type_simple():
    field = {"type": "string"}
    t = resolve_type(field)
    assert t is str, "Expected 'string' type to resolve to str"

def test_resolve_type_union():
    field = {"type": "int | None"}
    t = resolve_type(field)
    assert get_args(t) == (int, type(None)), "Expected 'int | None' to resolve to a union of int and NoneType"

def test_resolve_type_any_of():
    field = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"}
        ]
    }
    t = resolve_type(field)
    assert get_args(t) == (str, type(None)), "Expected 'anyOf' to resolve to a union of str and NoneType"

def test_resolve_type_pydantic():
    field = {"type": "SimpleModel"}
    custom_types = {
        "SimpleModel": SimpleModel,
        "ComplexModel": ComplexModel,
    }
    t = resolve_type(field, custom_types)
    assert t is SimpleModel, "Expected 'SimpleModel' to resolve to SimpleModel"

def test_resolve_type_pydantic_any_of():
    field = {
        "anyOf": [
            {"type": "SimpleModel"},
            {"type": "ComplexModel"}
        ]
    }
    custom_types = {
        "SimpleModel": SimpleModel,
        "ComplexModel": ComplexModel,
    }
    t = resolve_type(field, custom_types)
    assert get_args(t) == (SimpleModel, ComplexModel), "Expected 'anyOf' to resolve to a union of SimpleModel and ComplexModel"

def test_resolve_type_invalid():
    field = {"type": "unknown_type"}
    with pytest.raises(ValueError):
        resolve_type(field)
    
def test_describe_datafield_type():
    field = {"type": "string"}
    description = describe_datafield_type(field)
    assert description == "`str`", "Expected description for 'str' type to be '`string`'. Got: " + description

    field = {"type": "int | None"}
    description = describe_datafield_type(field)
    assert description == "`int | None`", "Expected description for 'int | None' to be '`int | None`'. Got: " + description

    field = {
        "anyOf": [
            {"type": "string"},
            {"type": "int | None"}
        ]
    }
    description = describe_datafield_type(field)
    assert description == "`str | int | None`", "Expected description for 'anyOf' to be '`str | int | None`. Got: " + description

def test_schema_to_model_simple():
    schema = SimpleModel.model_json_schema()
    print(schema)
    model = schema_to_model("SimpleModel", schema)
    assert issubclass(model, BaseModel), "Expected model to be a subclass of BaseModel"
    assert schema == model.model_json_schema(), "Expected schema to match the model's JSON schema"

def test_schema_to_model_complex():
    schema = ComplexModel.model_json_schema()
    custom_types = {
        "SimpleModel": SimpleModel,
    }
    model = schema_to_model("ComplexModel", schema, custom_types=custom_types)
    assert issubclass(model, BaseModel), "Expected model to be a subclass of BaseModel"
    assert schema == model.model_json_schema(), "Expected schema to match the model's JSON schema"
