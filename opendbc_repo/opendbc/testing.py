import functools
import os
import random
import secrets
import sys
from collections.abc import Callable, Sequence
from typing import Any, TypeVar


T = TypeVar("T")

_EDGE_EXAMPLES = 16
_MINIMAL_EXAMPLES = 2

# One seed per test process. Test IDs and example indexes make the generated
# inputs independent of test ordering and reproducible under parallel runners.
FUZZ_SEED = int(os.environ.get("FUZZ_SEED", secrets.randbits(64)))


class Fuzzy:
  """Small deterministic data generator with systematic boundary coverage."""

  def __init__(self, seed: int | str, example_index: int):
    self.example_index = example_index
    self._random = random.Random(seed)
    self._draw_index = 0

  def _draw(self, edges: Sequence[T], random_value: Callable[[], T]) -> T:
    draw_index = self._draw_index
    self._draw_index += 1

    if self.example_index < _MINIMAL_EXAMPLES:
      return edges[0]

    search_index = self.example_index - _MINIMAL_EXAMPLES
    if search_index < _EDGE_EXAMPLES * 2 and search_index % 2 == 0:
      return edges[(search_index // 2 + draw_index) % len(edges)]
    if self._random.randrange(8) == 0:
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

    edges = [0, 1, -1, min_value, max_value, min_value + 1, max_value - 1]
    edges.extend(1 << bit for bit in range(max_value.bit_length()))
    edges.extend(-(1 << bit) for bit in range((-min_value).bit_length()))
    valid_edges = tuple(dict.fromkeys(value for value in edges if min_value <= value <= max_value))
    return self._draw(valid_edges, lambda: self._random.randint(min_value, max_value))

  def _length(self, min_size: int, max_size: int | None) -> int:
    if min_size < 0:
      raise ValueError("minimum size must be non-negative")
    if max_size is not None and min_size > max_size:
      raise ValueError(f"{min_size=} must not exceed {max_size=}")
    if min_size == max_size:
      return min_size

    # The larger unbounded edges cover inputs that are rare with a geometric
    # distribution while keeping bounded draws inside their requested range.
    offsets = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
    edges = tuple(min_size + offset for offset in offsets if max_size is None or min_size + offset <= max_size)

    def random_length() -> int:
      size = min_size
      while max_size is None or size < max_size:
        size += 1
        if self._random.randrange(20) == 0:
          break
      return size

    return self._draw(edges, random_length)

  def binary(self, min_size: int = 0, max_size: int | None = None) -> bytes:
    size = self._length(min_size, max_size)
    patterns = (
      bytes(size),
      b"\xff" * size,
      (b"\xaa\x55" * ((size + 1) // 2))[:size],
      bytes(value & 0xff for value in range(size)),
    )
    return self._draw(patterns, lambda: self._random.randbytes(size))

  def list(self, generate: Callable[[], T], min_size: int = 0, max_size: int | None = None) -> list[T]:
    return [generate() for _ in range(self._length(min_size, max_size))]


def fuzzy_test(max_examples: int) -> Callable[[Callable[..., None]], Callable[..., None]]:
  """Repeat a unittest with reproducible fuzzy data.

  MAX_EXAMPLES overrides the decorator default. A failure can be replayed with
  the FUZZ_SEED and FUZZ_EXAMPLE values included in its exception note.
  """
  max_examples = int(os.environ.get("MAX_EXAMPLES", max_examples))
  if max_examples < 1:
    raise ValueError("max_examples must be at least one")

  def decorator(func: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
      test_seed = f"{FUZZ_SEED}:{args[0].id()}"
      selected_example = os.environ.get("FUZZ_EXAMPLE")
      examples = [int(selected_example, 0)] if selected_example is not None else range(max_examples)

      for example_index in examples:
        if not 0 <= example_index < max_examples:
          raise ValueError(f"FUZZ_EXAMPLE={example_index} is outside [0, {max_examples})")
        try:
          func(*args, **kwargs, fuzzy=Fuzzy(f"{test_seed}:{example_index}", example_index))
        except Exception as exc:
          exc.add_note(f"reproduce with FUZZ_SEED={FUZZ_SEED} FUZZ_EXAMPLE={example_index}")
          raise

    return wrapper
  return decorator


def parameterized(argnames, argvalues):
  """Method decorator that runs a test once per parameter set using subTest.

  Usage:
    @parameterized("x, y", [(1, 2), (3, 4)])
    def test_add(self, x, y): ...

    @parameterized("car_model, fingerprints", FINGERPRINTS.items())
    def test_fw(self, car_model, fingerprints): ...
  """
  if isinstance(argnames, str):
    argnames = [a.strip() for a in argnames.split(',')]

  def decorator(func):

    @functools.wraps(func)
    def wrapper(self):
      for values in argvalues:
        if not isinstance(values, (tuple, list)):
          values = (values,)
        kwargs = dict(zip(argnames, values, strict=True))
        with self.subTest(**kwargs):
          func(self, **kwargs)

    return wrapper

  return decorator


def parameterized_class(attrs, values=None):
  """Class decorator that generates subclasses with different class attributes.

  Usage:
    @parameterized_class([{"x": 1}, {"x": 2}])
    @parameterized_class('x', [(1,), (2,)])
  """
  if isinstance(attrs, str):
    attrs = [attrs]
    params = [dict(zip(attrs, v, strict=True)) for v in values]
  else:
    params = attrs

  def decorator(cls):
    module = sys.modules[cls.__module__]
    for param_set in params:
      name = f"{cls.__name__}_{'_'.join(str(v) for v in param_set.values())}"
      new_cls = type(name, (cls,), param_set)
      new_cls.__qualname__ = name
      new_cls.__module__ = cls.__module__
      new_cls.__test__ = True
      setattr(module, name, new_cls)
    cls.__test__ = False
    return cls

  return decorator
