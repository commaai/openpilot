# Partition boundaries from the process's own source: numeric constants in comparisons, min/max/clip/interp calls and
# module-level constants, resolved through the module's globals (so imported names like V_CRUISE_UNSET count).
# A constant is tied to a field when the expression it's compared with mentions the field's name (vEgo, dRel, ...);
# aliases (v_ego for carState.vEgo) are missed by design, Spec.bounds adds those.
import ast
import functools
import importlib
import math
from collections import defaultdict

CALLS = {'min', 'max', 'clip', 'interp', 'np.clip', 'np.interp', 'np.minimum', 'np.maximum'}


def numbers(v, out):
  if isinstance(v, bool):
    return
  if isinstance(v, (int, float)) and math.isfinite(v):
    out.add(float(v))
  elif isinstance(v, (list, tuple)) and len(v) < 64:
    for x in v:
      numbers(x, out)
  elif hasattr(v, 'tolist') and isinstance(getattr(v, 'size', None), int) and v.size < 64:  # not numpy scalar types (np.float32)
    numbers(v.tolist(), out)


def evaluate(node, g):
  # constant-fold names/attributes/arithmetic against the module's globals; anything else is not a constant
  if any(isinstance(n, ast.Call) for n in ast.walk(node)) or any(isinstance(n, ast.Name) and n.id not in g for n in ast.walk(node)):
    return None
  try:
    return eval(compile(ast.Expression(node), '<c>', 'eval'), dict(g))
  except Exception:
    return None


def names(node):
  return {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(node) if isinstance(n, (ast.Attribute, ast.Name))}


@functools.cache
def harvest(files: tuple[str, ...]):
  glob, by_name = set(), defaultdict(set)
  for f in files:
    mod = importlib.import_module(module_of(f))
    g = vars(mod)
    for k, v in g.items():  # module-level constants, also imported ones
      if k.isupper():
        numbers(v, glob)
    for node in ast.walk(ast.parse(open(mod.__file__).read())):
      groups = []
      if isinstance(node, ast.Compare):
        groups = [[node.left, *node.comparators]]
      elif isinstance(node, ast.Call) and ast.unparse(node.func) in CALLS:
        groups = [node.args]
      for operands in groups:
        vals = [evaluate(o, g) for o in operands]
        named = set().union(*(names(o) for o, v in zip(operands, vals, strict=True) if v is None))
        for v in vals:
          s = set()
          numbers(v, s)
          glob |= s
          for n in named:
            by_name[n] |= s
      if isinstance(node, ast.Constant):
        numbers(node.value, glob)
  return frozenset(glob), dict(by_name)


def module_of(f):
  # 'selfdrive/controls/plannerd.py' -> 'openpilot.selfdrive.controls.plannerd', 'opendbc/car/x.py' -> 'opendbc.car.x'
  m = f.removesuffix('.py').replace('/', '.')
  return m if m.startswith('opendbc.') else 'openpilot.' + m
