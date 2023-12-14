import capnp
from typing import Union, List, Dict, Any


def generate_type(type_walker, schema_walker) -> Union[str, List[Any], Dict[str, Any]]:
  data_type = next(type_walker)
  if data_type.which() == 'struct':
    return generate_struct(next(schema_walker))
  elif data_type.which() == 'list':
    _ = next(schema_walker)
    return [generate_type(type_walker, schema_walker)]
  elif data_type.which() == 'enum':
    return "text"
  else:
    return str(data_type.which())


def generate_struct(schema: capnp.lib.capnp._StructSchema) -> Dict[str, Any]:
  return {field: generate_field(schema.fields[field]) for field in schema.fields if not field.endswith("DEPRECATED")}


def generate_field(field: capnp.lib.capnp._StructSchemaField) -> Union[str, List[Any], Dict[str, Any]]:
  def schema_walker(field):
    yield field.schema

    s = field.schema
    while hasattr(s, 'elementType'):
      s = s.elementType
      yield s

  def type_walker(field):
    yield field.proto.slot.type

    t = field.proto.slot.type
    while hasattr(getattr(t, t.which()), 'elementType'):
      t = getattr(t, t.which()).elementType
      yield t

  if field.proto.which() == "slot":
    schema_gen, type_gen = schema_walker(field), type_walker(field)
    return generate_type(type_gen, schema_gen)
  else:
    return generate_struct(field.schema)
