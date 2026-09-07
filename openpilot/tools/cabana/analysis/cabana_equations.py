"""Python layout equations."""
import math
import textwrap


def compile_equation(globals_code, function_code, input_count):
  namespace = {"math": math}
  exec(compile(globals_code, "<layout globals>", "exec"), namespace)
  args = ", ".join(["time", "value"] + [f"v{i + 1}" for i in range(input_count)])
  source = f"def calc({args}):\n" + textwrap.indent(function_code, "  ") + "\n"
  exec(compile(source, "<layout equation>", "exec"), namespace)
  return namespace["calc"]
