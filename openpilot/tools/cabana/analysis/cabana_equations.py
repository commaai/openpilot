"""Python layout equations and migration of the bundled PlotJuggler equations."""
import hashlib
import json
import math
import textwrap
from functools import lru_cache
from pathlib import Path


def compile_equation(globals_code, function_code, input_count):
  namespace = {"math": math}
  exec(compile(globals_code, "<layout globals>", "exec"), namespace)
  args = ", ".join(["time", "value"] + [f"v{i + 1}" for i in range(input_count)])
  source = f"def calc({args}):\n" + textwrap.indent(function_code, "  ") + "\n"
  exec(compile(source, "<layout equation>", "exec"), namespace)
  return namespace["calc"]


@lru_cache(maxsize=1)
def legacy_equations():
  ports = {}
  for path in (Path(__file__).resolve().parent.parent / "layouts").glob("*.json"):
    for equation in json.loads(path.read_text()).get("equations", []):
      if key := equation.get("legacy_lua_hash"):
        ports[key] = equation["globals"], equation["function"]
  return ports


def port_equation(globals_code, function_code):
  key = hashlib.sha256((globals_code.strip() + "\0" + function_code.strip()).encode()).hexdigest()
  try:
    return legacy_equations()[key]
  except KeyError:
    raise ValueError("This Lua equation has no Python port. Rewrite it in Python and set language to python.") from None
