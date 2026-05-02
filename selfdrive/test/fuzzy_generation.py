import math
import random
import struct
from typing import Any
from functools import cache

import capnp

from cereal import log

EDGE_CASE_PROB = 0.05
MAX_LIST_LEN = 10

INT_RANGES = {
  'int8': (-2**7, 2**7 - 1),
  'int16': (-2**15, 2**15 - 1),
  'int32': (-2**31, 2**31 - 1),
  'int64': (-2**63, 2**63 - 1),
  'uint8': (0, 2**8 - 1),
  'uint16': (0, 2**16 - 1),
  'uint32': (0, 2**32 - 1),
  'uint64': (0, 2**64 - 1),
}

FLOAT32_EDGES = (0.0, -0.0, 1.0, -1.0, 3.4028235e38, -3.4028235e38, 1.1754944e-38)
FLOAT64_EDGES = (0.0, -0.0, 1.0, -1.0, 1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308)
NON_REAL_FLOAT_EDGES = (math.nan, math.inf, -math.inf)


def _rand_int(min_value: int, max_value: int) -> int:
  if random.random() < EDGE_CASE_PROB:
    return random.choice((min_value, max_value, 0, -1, 1))
  return random.randint(min_value, max_value)


def _rand_float(width: int, real_floats: bool) -> float:
  edges = FLOAT32_EDGES if width == 32 else FLOAT64_EDGES
  if not real_floats and random.random() < EDGE_CASE_PROB:
    return random.choice(NON_REAL_FLOAT_EDGES)
  if random.random() < EDGE_CASE_PROB:
    return random.choice(edges)
  if width == 32:
    while True:
      f = struct.unpack('<f', struct.pack('<I', random.getrandbits(32)))[0]
      if real_floats and (math.isnan(f) or math.isinf(f)):
        continue
      return f
  while True:
    f = struct.unpack('<d', struct.pack('<Q', random.getrandbits(64)))[0]
    if real_floats and (math.isnan(f) or math.isinf(f)):
      continue
    return f


@cache
def _native_generators(real_floats: bool):
  return {
    'bool': lambda: random.random() < 0.5,
    'int8': lambda: _rand_int(*INT_RANGES['int8']),
    'int16': lambda: _rand_int(*INT_RANGES['int16']),
    'int32': lambda: _rand_int(*INT_RANGES['int32']),
    'int64': lambda: _rand_int(*INT_RANGES['int64']),
    'uint8': lambda: _rand_int(*INT_RANGES['uint8']),
    'uint16': lambda: _rand_int(*INT_RANGES['uint16']),
    'uint32': lambda: _rand_int(*INT_RANGES['uint32']),
    'uint64': lambda: _rand_int(*INT_RANGES['uint64']),
    'float32': lambda: _rand_float(32, real_floats),
    'float64': lambda: _rand_float(64, real_floats),
    'text': lambda: ''.join(chr(random.randint(0x20, 0x7e)) for _ in range(random.randint(0, 1000))),
    'data': lambda: random.randbytes(random.randint(0, 1000)),
    'anyPointer': lambda: ''.join(chr(random.randint(0x20, 0x7e)) for _ in range(random.randint(0, 1000))),
  }


class FuzzyGenerator:
  def __init__(self, real_floats: bool):
    self.real_floats = real_floats
    self.native = _native_generators(real_floats)

  def generate_native_type(self, field: str) -> Any:
    gen = self.native.get(field)
    if gen is None:
      raise NotImplementedError(f'Invalid type: {field}')
    return gen()

  def generate_field(self, field: capnp.lib.capnp._StructSchemaField) -> Any:
    def rec(field_type: capnp.lib.capnp._DynamicStructReader) -> Any:
      type_which = field_type.which()
      if type_which == 'struct':
        return self.generate_struct(field.schema.elementType if base_type == 'list' else field.schema)
      elif type_which == 'list':
        n = random.randint(0, MAX_LIST_LEN)
        return [rec(field_type.list.elementType) for _ in range(n)]
      elif type_which == 'enum':
        schema = field.schema.elementType if base_type == 'list' else field.schema
        return random.choice(list(schema.enumerants.keys()))
      else:
        return self.generate_native_type(type_which)

    try:
      if hasattr(field.proto, 'slot'):
        slot_type = field.proto.slot.type
        base_type = slot_type.which()
        return rec(slot_type)
      else:
        return self.generate_struct(field.schema)
    except capnp.lib.capnp.KjException:
      return self.generate_struct(field.schema)

  def generate_struct(self, schema: capnp.lib.capnp._StructSchema, event: str | None = None) -> dict[str, Any]:
    single_fill: tuple[str, ...] = (event,) if event else (random.choice(schema.union_fields),) if schema.union_fields else ()
    fields_to_generate = [f for f in schema.non_union_fields + single_fill if not f.endswith('DEPRECATED') and f != 'deprecated']
    return {field: self.generate_field(schema.fields[field]) for field in fields_to_generate}

  @classmethod
  def get_random_msg(cls, struct: capnp.lib.capnp._StructModule, real_floats: bool = False) -> dict[str, Any]:
    return cls(real_floats=real_floats).generate_struct(struct.schema)

  @classmethod
  def get_random_event_msg(cls, events: list[str], real_floats: bool = False) -> list[dict[str, Any]]:
    fg = cls(real_floats=real_floats)
    return [fg.generate_struct(log.Event.schema, e) for e in sorted(events)]
