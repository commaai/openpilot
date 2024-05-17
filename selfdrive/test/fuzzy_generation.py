import capnp
import hypothesis.strategies as st
from typing import Any
from collections.abc import Callable

from cereal import log

DrawType = Callable[[st.SearchStrategy], Any]


class FuzzyGenerator:
  def __init__(self, draw: DrawType, real_floats: bool):
    self.draw = draw
    self.real_floats = real_floats

  def generate_native_type(self, field: str) -> st.SearchStrategy[bool | int | float | str | bytes]:
    def floats(**kwargs) -> st.SearchStrategy[float]:
      allow_nan = not self.real_floats
      allow_infinity = not self.real_floats
      return st.floats(**kwargs, allow_nan=allow_nan, allow_infinity=allow_infinity)

    if field == 'bool':
      return st.booleans()
    elif field == 'int8':
      return st.integers(min_value=-2**7, max_value=2**7-1)
    elif field == 'int16':
      return st.integers(min_value=-2**15, max_value=2**15-1)
    elif field == 'int32':
      return st.integers(min_value=-2**31, max_value=2**31-1)
    elif field == 'int64':
      return st.integers(min_value=-2**63, max_value=2**63-1)
    elif field == 'uint8':
      return st.integers(min_value=0, max_value=2**8-1)
    elif field == 'uint16':
      return st.integers(min_value=0, max_value=2**16-1)
    elif field == 'uint32':
      return st.integers(min_value=0, max_value=2**32-1)
    elif field == 'uint64':
      return st.integers(min_value=0, max_value=2**64-1)
    elif field == 'float32':
      return floats(width=32)
    elif field == 'float64':
      return floats(width=64)
    elif field == 'text':
      return st.text(max_size=1000)
    elif field == 'data':
      return st.binary(max_size=1000)
    elif field == 'anyPointer':
      return st.text()
    else:
      raise NotImplementedError(f'Invalid type : {field}')

  def generate_field(self, field: capnp.lib.capnp._StructSchemaField) -> st.SearchStrategy:
    def rec(field_type: capnp.lib.capnp._DynamicStructReader) -> st.SearchStrategy:
      if field_type.which() == 'struct':
        return self.generate_struct(field.schema.elementType if base_type == 'list' else field.schema)
      elif field_type.which() == 'list':
        return st.lists(rec(field_type.list.elementType))
      elif field_type.which() == 'enum':
        schema = field.schema.elementType if base_type == 'list' else field.schema
        return st.sampled_from(list(schema.enumerants.keys()))
      else:
        return self.generate_native_type(field_type.which())

    if 'slot' in field.proto.to_dict():
      base_type = field.proto.slot.type.which()
      return rec(field.proto.slot.type)
    else:
      return self.generate_struct(field.schema)

  def generate_struct(self, schema: capnp.lib.capnp._StructSchema, event: str = None) -> st.SearchStrategy[dict[str, Any]]:
    full_fill: list[str] = list(schema.non_union_fields)
    single_fill: list[str] = [event] if event else [self.draw(st.sampled_from(schema.union_fields))] if schema.union_fields else []
    return st.fixed_dictionaries({field: self.generate_field(schema.fields[field]) for field in full_fill + single_fill})

  @classmethod
  def get_random_msg(cls, draw: DrawType, struct: capnp.lib.capnp._StructModule, real_floats: bool = False) -> dict[str, Any]:
    fg = cls(draw, real_floats=real_floats)
    data: dict[str, Any] = draw(fg.generate_struct(struct.schema))
    return data

  @classmethod
  def get_random_event_msg(cls, draw: DrawType, events: list[str], real_floats: bool = False) -> list[dict[str, Any]]:
    fg = cls(draw, real_floats=real_floats)
    return [draw(fg.generate_struct(log.Event.schema, e)) for e in sorted(events)]
