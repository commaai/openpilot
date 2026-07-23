import math
import os
import random
import secrets
import struct
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

import capnp


T = TypeVar("T")

_EDGE_SLOTS = 16
_MINIMAL_EXAMPLES = 10
_INTEGER_RANGES = {
  "int8": (-2**7, 2**7 - 1),
  "int16": (-2**15, 2**15 - 1),
  "int32": (-2**31, 2**31 - 1),
  "int64": (-2**63, 2**63 - 1),
  "uint8": (0, 2**8 - 1),
  "uint16": (0, 2**16 - 1),
  "uint32": (0, 2**32 - 1),
  "uint64": (0, 2**64 - 1),
}

# One seed is shared by the whole test process. Individual tests derive their seed
# from their unittest ID, so FUZZ_SEED is reproducible under pytest-xdist too.
FUZZ_SEED = int(os.environ.get("FUZZ_SEED", secrets.randbits(64)))


class Fuzzy:
  """Fast, deterministic data generator with systematic boundary coverage."""

  def __init__(self, seed: int | str, example_index: int):
    self.example_index = example_index
    self._random = random.Random(seed)
    self._draw_index = 0

  def _draw(self, edges: Sequence[T], random_value: Callable[[], T]) -> T:
    draw_index = self._draw_index
    self._draw_index += 1

    # Preserve the cheap minimal prefix Hypothesis produced, then interleave
    # systematic boundaries and random values at every draw site.
    if self.example_index < _MINIMAL_EXAMPLES:
      return edges[0]
    search_example = self.example_index - _MINIMAL_EXAMPLES
    if search_example < _EDGE_SLOTS * 2 and search_example % 2 == 0:
      return edges[(search_example // 2 + draw_index) % len(edges)]
    if self._random.randrange(4) == 0:
      return self._random.choice(edges)
    return random_value()

  def boolean(self) -> bool:
    return self._draw((False, True), lambda: bool(self._random.getrandbits(1)))

  def choice(self, values: Sequence[T]) -> T:
    if not values:
      raise ValueError("cannot choose from an empty sequence")
    return self._draw(values, lambda: self._random.choice(values))

  def integer(self, min_value: int, max_value: int) -> int:
    if min_value > max_value:
      raise ValueError(f"{min_value=} must not exceed {max_value=}")

    edges = [
      0, 1, -1, min_value, max_value,
      min_value + 1, max_value - 1,
    ]
    edges.extend(1 << bit for bit in range(max_value.bit_length()))
    edges.extend(-(1 << bit) for bit in range((-min_value).bit_length()))
    valid_edges = tuple(dict.fromkeys(v for v in edges if min_value <= v <= max_value))
    return self._draw(valid_edges, lambda: self._random.randint(min_value, max_value))

  def floating(self, width: int = 64, *, allow_nan: bool = True, allow_infinity: bool = True) -> float:
    if width not in (32, 64):
      raise ValueError("float width must be 32 or 64")

    if width == 32:
      unpack_format = "!f"
      finite_edges = (
        0.0, -0.0, 1.0, -1.0,
        struct.unpack(unpack_format, b"\x00\x00\x00\x01")[0],
        struct.unpack(unpack_format, b"\x80\x00\x00\x01")[0],
        struct.unpack(unpack_format, b"\x7f\x7f\xff\xff")[0],
        struct.unpack(unpack_format, b"\xff\x7f\xff\xff")[0],
        struct.unpack(unpack_format, b"\x00\x80\x00\x00")[0],
        struct.unpack(unpack_format, b"\x80\x80\x00\x00")[0],
      )
    else:
      unpack_format = "!d"
      finite_edges = (
        0.0, -0.0, 1.0, -1.0,
        math.ulp(0.0), -math.ulp(0.0),
        float.fromhex("0x1.fffffffffffffp+1023"), -float.fromhex("0x1.fffffffffffffp+1023"),
        float.fromhex("0x1p-1022"), -float.fromhex("0x1p-1022"),
      )

    edges = list(finite_edges)
    if allow_infinity:
      edges.extend((math.inf, -math.inf))
    if allow_nan:
      edges.append(math.nan)

    def random_float() -> float:
      while True:
        value = struct.unpack(unpack_format, self._random.randbytes(width // 8))[0]
        if (allow_nan or not math.isnan(value)) and (allow_infinity or not math.isinf(value)):
          return value

    return self._draw(tuple(edges), random_float)

  def _length(self, min_length: int, max_length: int | None) -> int:
    if min_length < 0:
      raise ValueError("minimum length must be non-negative")
    if max_length is not None and min_length > max_length:
      raise ValueError(f"{min_length=} must not exceed {max_length=}")
    if max_length == min_length:
      return min_length

    offsets = (0, 1, 2, 4, 8, 16, 32)
    edges = tuple(min_length + offset for offset in offsets if max_length is None or min_length + offset <= max_length)

    def random_length() -> int:
      # A geometric tail keeps ordinary examples small without placing an
      # artificial ceiling on an unbounded list.
      length = min_length
      while max_length is None or length < max_length:
        if self._random.randrange(8) == 0:
          break
        length += 1
      return length

    return self._draw(edges, random_length)

  def binary(self, min_size: int = 0, max_size: int | None = None) -> bytes:
    size = self._length(min_size, max_size)
    patterns = (
      bytes(size),
      b"\xff" * size,
      (b"\xaa\x55" * ((size + 1) // 2))[:size],
      bytes(i & 0xff for i in range(size)),
    )
    return self._draw(patterns, lambda: self._random.randbytes(size))

  def text(self, min_size: int = 0, max_size: int | None = None) -> str:
    size = self._length(min_size, max_size)

    def scalar() -> str:
      value = self._random.randrange(0x110000 - 0x800)
      if value >= 0xd800:
        value += 0x800
      return chr(value)

    patterns = (
      "",
      "a" * size,
      "\0" * size,
      "\U0010ffff" * size,
    )
    valid_patterns = tuple(value for value in patterns if len(value) == size)
    return self._draw(valid_patterns, lambda: "".join(scalar() for _ in range(size)))

  def list(self, generate: Callable[[], T], min_size: int = 0, max_size: int | None = None) -> list[T]:
    return [generate() for _ in range(self._length(min_size, max_size))]


def fuzzy_test(max_examples: int) -> Callable[[Callable[..., None]], Callable[..., None]]:
  """Run a unittest method repeatedly with independent, reproducible fuzzy data."""
  max_examples = int(os.environ.get("MAX_EXAMPLES", max_examples))
  assert max_examples >= 1

  def decorator(fn: Callable[..., None]) -> Callable[..., None]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
      test_seed = f"{FUZZ_SEED}:{args[0].id()}"
      selected_example = os.environ.get("FUZZ_EXAMPLE")
      examples = [int(selected_example, 0)] if selected_example is not None else range(max_examples)

      for example_index in examples:
        if not 0 <= example_index < max_examples:
          raise ValueError(f"FUZZ_EXAMPLE={example_index} is outside [0, {max_examples})")
        try:
          fn(*args, **kwargs, fuzzy=Fuzzy(f"{test_seed}:{example_index}", example_index))
        except Exception as exc:
          exc.add_note(f"reproduce with FUZZ_SEED={FUZZ_SEED} FUZZ_EXAMPLE={example_index}")
          raise

    return wrapper
  return decorator


def capnp_random_dict(fuzzy: Fuzzy, schema: Any, event: str | None = None, *, real_floats: bool = False) -> dict[str, Any]:
  """Generate a dictionary accepted by a pycapnp struct constructor."""

  def native(type_name: str) -> bool | int | float | str | bytes:
    if type_name == "bool":
      return fuzzy.boolean()
    if type_name in _INTEGER_RANGES:
      return fuzzy.integer(*_INTEGER_RANGES[type_name])
    if type_name in ("float32", "float64"):
      return fuzzy.floating(width=int(type_name[-2:]), allow_nan=not real_floats, allow_infinity=not real_floats)
    if type_name == "text":
      return fuzzy.text(max_size=1000)
    if type_name == "anyPointer":
      return fuzzy.text()
    if type_name == "data":
      return fuzzy.binary(max_size=1000)
    raise NotImplementedError(f"invalid Cap'n Proto type: {type_name}")

  def generate_field(field: Any) -> Any:
    def rec(field_type: Any, base_type: str) -> Any:
      type_name = field_type.which()
      if type_name == "struct":
        struct_schema = field.schema.elementType if base_type == "list" else field.schema
        return capnp_random_dict(fuzzy, struct_schema, real_floats=real_floats)
      if type_name == "list":
        return fuzzy.list(lambda: rec(field_type.list.elementType, "list"))
      if type_name == "enum":
        enum_schema = field.schema.elementType if base_type == "list" else field.schema
        return fuzzy.choice(tuple(enum_schema.enumerants))
      return native(type_name)

    try:
      if hasattr(field.proto, "slot"):
        slot_type = field.proto.slot.type
        return rec(slot_type, slot_type.which())
      return capnp_random_dict(fuzzy, field.schema, real_floats=real_floats)
    except capnp.lib.capnp.KjException:
      return capnp_random_dict(fuzzy, field.schema, real_floats=real_floats)

  union_field = event or (fuzzy.choice(tuple(schema.union_fields)) if schema.union_fields else None)
  fields = schema.non_union_fields + ((union_field,) if union_field else ())
  return {
    field_name: generate_field(schema.fields[field_name])
    for field_name in fields
    if not field_name.endswith("DEPRECATED") and field_name != "deprecated"
  }
