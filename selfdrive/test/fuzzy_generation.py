import capnp
import hypothesis.strategies as st
from typing import Any
from collections.abc import Callable
from functools import cache

from cereal import log

DrawType = Callable[[st.SearchStrategy], Any]


class FuzzyGenerator:
  def __init__(self, draw: DrawType, real_floats: bool):
    self.draw = draw
    self.native_type_map = FuzzyGenerator._get_native_type_map(real_floats)

  def generate_native_type(self, field: str) -> st.SearchStrategy[bool | int | float | str | bytes]:
    value_func = self.native_type_map.get(field)
    if value_func is not None:
      return value_func
    else:
      raise NotImplementedError(f'Invalid type: {field}')

  def generate_field(self, field: capnp.lib.capnp._StructSchemaField) -> st.SearchStrategy:
    def rec(field_type: capnp.lib.capnp._DynamicStructReader) -> st.SearchStrategy:
      type_which = field_type.which()
      if type_which == 'struct':
        return self.generate_struct(field.schema.elementType if base_type == 'list' else field.schema)
      elif type_which == 'list':
        return st.lists(rec(field_type.list.elementType))
      elif type_which == 'enum':
        schema = field.schema.elementType if base_type == 'list' else field.schema
        return st.sampled_from(list(schema.enumerants.keys()))
      else:
        return self.generate_native_type(type_which)

    try:
      if hasattr(field.proto, 'slot'):
        slot_type =  field.proto.slot.type
        base_type = slot_type.which()
        return rec(slot_type)
      else:
        return self.generate_struct(field.schema)
    except capnp.lib.capnp.KjException:
      return self.generate_struct(field.schema)

  def generate_struct(self, schema: capnp.lib.capnp._StructSchema, event: str = None) -> st.SearchStrategy[dict[str, Any]]:
    single_fill: tuple[str, ...] = (event,) if event else (self.draw(st.sampled_from(schema.union_fields)),) if schema.union_fields else ()
    fields_to_generate = schema.non_union_fields + single_fill
    return st.fixed_dictionaries({field: self.generate_field(schema.fields[field]) for field in fields_to_generate if not field.endswith('DEPRECATED')})

  @staticmethod
  @cache
  def _get_native_type_map(real_floats: bool) -> dict[str, st.SearchStrategy]:
    return {
      'bool': st.booleans(),
      'int8': st.integers(min_value=-2**7, max_value=2**7-1),
      'int16': st.integers(min_value=-2**15, max_value=2**15-1),
      'int32': st.integers(min_value=-2**31, max_value=2**31-1),
      'int64': st.integers(min_value=-2**63, max_value=2**63-1),
      'uint8': st.integers(min_value=0, max_value=2**8-1),
      'uint16': st.integers(min_value=0, max_value=2**16-1),
      'uint32': st.integers(min_value=0, max_value=2**32-1),
      'uint64': st.integers(min_value=0, max_value=2**64-1),
      'float32': st.floats(width=32, allow_nan=not real_floats, allow_infinity=not real_floats),
      'float64': st.floats(width=64, allow_nan=not real_floats, allow_infinity=not real_floats),
      'text': st.text(max_size=1000),
      'data': st.binary(max_size=1000),
      'anyPointer': st.text(),  # Note: No need to define a separate function for anyPointer
    }

  @classmethod
  def get_random_msg(cls, draw: DrawType, struct: capnp.lib.capnp._StructModule, real_floats: bool = False) -> dict[str, Any]:
    fg = cls(draw, real_floats=real_floats)
    data: dict[str, Any] = draw(fg.generate_struct(struct.schema))
    return data

  @classmethod
  def get_random_event_msg(cls, draw: DrawType, events: list[str], real_floats: bool = False) -> list[dict[str, Any]]:
    fg = cls(draw, real_floats=real_floats)
    return [draw(fg.generate_struct(log.Event.schema, e)) for e in sorted(events)]
