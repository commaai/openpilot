"""Minimal dict diff implementation for process replay log comparison.

Replaces the dictdiffer PyPI package. Only implements the diff() function
with the ignore parameter, which is all that openpilot uses.

Output format matches dictdiffer exactly: iterator of (action, path, value) where
  - action: 'add', 'remove', or 'change'
  - path: dotted string (e.g. 'a.b') or list when path contains non-strings/dots
  - value: list of (key, val) pairs for add/remove, (old, new) tuple for change
"""

import math
import sys
from collections.abc import Iterable, MutableMapping, MutableSequence, MutableSet
from copy import deepcopy

EPSILON = sys.float_info.epsilon

ADD = 'add'
REMOVE = 'remove'
CHANGE = 'change'


def _are_different(first, second, tolerance):
  if first == second:
    return False
  # NaN: two NaNs are considered equal (consistent with dictdiffer)
  first_nan = bool(first != first)
  second_nan = bool(second != second)
  if first_nan or second_nan:
    return not (first_nan and second_nan)
  if isinstance(first, (int, float)) and isinstance(second, (int, float)):
    return not math.isclose(first, second, rel_tol=tolerance, abs_tol=0)
  return True


def _dotted(node):
  """Return dotted string if all keys are plain strings, else list."""
  if all(isinstance(k, str) and '.' not in k for k in node):
    return '.'.join(node)
  return list(node)


def diff(first, second, node=None, ignore=None, tolerance=EPSILON, **_kwargs):
  """Compare two nested dicts/lists/sets and yield (action, path, value) tuples.

  Args:
    first: Original dict, list, or set.
    second: New dict, list, or set.
    node: Starting node (used internally for recursion).
    ignore: Iterable of keys/paths to skip. Strings are treated as dotted paths.
    tolerance: Relative float comparison tolerance (default: sys.float_info.epsilon).
    **_kwargs: Accepted but ignored (expand, path_limit, absolute_tolerance, dot_notation).
  """
  # Build ignore set - strings stay as strings (dot_notation=True behaviour)
  ignore_set: set | None = None
  if ignore is not None and isinstance(ignore, Iterable):
    ignore_set = set()
    for v in ignore:
      if isinstance(v, int):
        ignore_set.add((v,))
      elif isinstance(v, list):
        ignore_set.add(tuple(v))
      else:
        ignore_set.add(v)  # str stays str

  def _check(path_node, key):
    """Return True if this key should be included (not ignored)."""
    if ignore_set is None:
      return True
    full = path_node + [key]
    return _dotted(full) not in ignore_set and tuple(full) not in ignore_set

  def _diff(a, b, path):
    dotted_path = _dotted(path)

    if isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
      intersection = [k for k in a if k in b and _check(path, k)]
      addition = [k for k in b if k not in a and _check(path, k)]
      deletion = [k for k in a if k not in b and _check(path, k)]

      for k in intersection:
        yield from _diff(a[k], b[k], path + [k])

      if addition:
        yield ADD, dotted_path, [(k, deepcopy(b[k])) for k in addition]
      if deletion:
        yield REMOVE, dotted_path, [(k, deepcopy(a[k])) for k in deletion]

    elif isinstance(a, MutableSequence) and isinstance(b, MutableSequence):
      n = min(len(a), len(b))
      for i in range(n):
        yield from _diff(a[i], b[i], path + [i])
      if len(b) > n:
        yield ADD, dotted_path, [(i, deepcopy(b[i])) for i in range(n, len(b))]
      if len(a) > n:
        yield REMOVE, dotted_path, [(i, deepcopy(a[i])) for i in reversed(range(n, len(a)))]

    elif isinstance(a, MutableSet) and isinstance(b, MutableSet):
      added = b - a
      if added:
        yield ADD, dotted_path, [(0, added)]
      removed = a - b
      if removed:
        yield REMOVE, dotted_path, [(0, removed)]

    else:
      if _are_different(a, b, tolerance):
        yield CHANGE, dotted_path, (deepcopy(a), deepcopy(b))

  return _diff(first, second, list(node) if node is not None else [])
