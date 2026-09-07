"""Python layout equations."""
import builtins
import math
import textwrap

# Pure computation only: no filesystem, network, or sleeping in a chart worker thread.
ALLOWED_MODULES = {"math", "cmath", "statistics", "itertools", "functools", "operator", "collections",
                   "bisect", "heapq", "random", "fractions", "decimal", "numbers", "string", "re", "json"}


def _import(name, *args, **kwargs):
  if name.partition(".")[0] not in ALLOWED_MODULES:
    raise ImportError(f"{name} is not available in layout equations")
  return builtins.__import__(name, *args, **kwargs)


def compile_equation(globals_code, function_code, input_count):
  safe_builtins = {k: v for k, v in vars(builtins).items() if k not in ("open", "input", "breakpoint", "exit", "quit")}
  safe_builtins["__import__"] = _import
  namespace = {"math": math, "__builtins__": safe_builtins}
  exec(compile(globals_code, "<layout globals>", "exec"), namespace)
  args = ", ".join(["time", "value"] + [f"v{i + 1}" for i in range(input_count)])
  source = f"def calc({args}):\n" + textwrap.indent(function_code, "  ") + "\n"
  exec(compile(source, "<layout equation>", "exec"), namespace)
  return namespace["calc"]
