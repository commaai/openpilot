"""Small, fail-closed numeric Python subset for the bundled layout equations."""
import ast
import math

MAX_SOURCE_BYTES, MAX_NODES, MAX_DEPTH = 8192, 512, 32
MAX_INPUTS, MAX_VARIABLES, MAX_ITEMS = 32, 64, 16
MATH_FUNCTIONS = {"sin", "cos", "sqrt", "atan2", "radians"}
MATH_CONSTANTS = {"pi", "e", "tau", "inf", "nan"}
HELPERS = {"abs", "min", "max", "int", "map"}
ALLOWED = {
  ast.Module, ast.Assign, ast.AugAssign, ast.If, ast.Global, ast.Return, ast.Pass,
  ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Call, ast.Tuple,
  ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Not,
  ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
}


class EquationError(ValueError):
  pass


def _number(value):
  if type(value) not in (int, float, bool):
    raise EquationError("Only numbers are allowed")
  try:
    return float(value)
  except OverflowError as e:
    raise EquationError("Number is too large") from e


def _is_call(node, name):
  return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _validate(source, initialization=False, globals_names=()):
  if type(source) is not str or len(source) > MAX_SOURCE_BYTES or len(source.encode()) > MAX_SOURCE_BYTES:
    raise EquationError("Code exceeds 8 KiB")
  tree = ast.parse(source)
  stack, count, assigned = [(tree, None, 0)], 0, set()
  while stack:
    node, parent, depth = stack.pop()
    count += 1
    if type(node) not in ALLOWED or count > MAX_NODES or depth > MAX_DEPTH:
      raise EquationError(f"Unsupported syntax or excessive complexity: {type(node).__name__}")
    if isinstance(node, ast.Name):
      if node.id.startswith('_') or len(node.id) > 64:
        raise EquationError("Private or oversized names are not allowed")
      if isinstance(node.ctx, ast.Store):
        assigned.add(node.id)
      if node.id in HELPERS | {"math"}:
        if not ((isinstance(parent, ast.Call) and parent.func is node) or
                (isinstance(parent, ast.Attribute) and parent.value is node and node.id == "math")):
          raise EquationError("Math names cannot be assigned or used as values")
    if isinstance(node, ast.Attribute):
      if not (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id == "math"):
        raise EquationError("Only approved math attributes are allowed")
      if node.attr not in MATH_CONSTANTS:
        if node.attr not in MATH_FUNCTIONS or not (
          (isinstance(parent, ast.Call) and parent.func is node) or (_is_call(parent, "map") and parent.args[0] is node)
        ):
          raise EquationError("Only approved math functions may be called")
    if isinstance(node, ast.Call):
      if node.keywords or not 1 <= len(node.args) <= MAX_ITEMS:
        raise EquationError("Only bounded positional calls are allowed")
      if isinstance(node.func, ast.Name):
        if node.func.id not in HELPERS:
          raise EquationError("Only numeric helpers may be called")
        if node.func.id == "map" and not (
          isinstance(parent, ast.Assign) and isinstance(parent.targets[0], ast.Tuple) and len(node.args) == 2 and
          isinstance(node.args[0], ast.Attribute) and isinstance(node.args[1], ast.Tuple)
        ):
          raise EquationError("map is only supported for math tuple unpacking")
      elif not (isinstance(node.func, ast.Attribute) and node.func.attr in MATH_FUNCTIONS):
        raise EquationError("Dynamic calls are not allowed")
    if isinstance(node, ast.Assign):
      if len(node.targets) != 1 or not isinstance(node.targets[0], (ast.Name, ast.Tuple)):
        raise EquationError("Assign only to numeric variables")
      if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple) and len(node.targets[0].elts) != len(node.value.elts):
        raise EquationError("Unpacking requires matching tuple sizes")
    if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
      raise EquationError("Assign only to numeric variables")
    if isinstance(node, ast.Tuple):
      unpack = isinstance(parent, ast.Assign) and isinstance(parent.targets[0], ast.Tuple)
      valid = (unpack or (isinstance(parent, ast.Return) and len(node.elts) == 2) or
               (_is_call(parent, "map") and parent.args[1] is node) or
               (any(_is_call(parent, name) for name in ("min", "max")) and len(parent.args) == 1))
      if not valid or not 1 <= len(node.elts) <= MAX_ITEMS or any(isinstance(e, ast.Tuple) for e in node.elts):
        raise EquationError("Tuples are limited to numeric unpacking, min/max, and (time, value) returns")
      if isinstance(node.ctx, ast.Store) and not all(isinstance(e, ast.Name) for e in node.elts):
        raise EquationError("Unpack only into variable names")
    if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
      # Layouts only need small integer powers. This also excludes complex results.
      exponent = node.right if isinstance(node, ast.BinOp) else node.value
      if not (isinstance(exponent, ast.Constant) and type(exponent.value) in (int, float) and
              0 <= exponent.value <= 16 and exponent.value == int(exponent.value)):
        raise EquationError("Powers require an integer literal between 0 and 16")
    if isinstance(node, ast.Constant):
      node.value = _number(node.value)  # Prevent arbitrary-size integer arithmetic, including constant folding.
    if isinstance(node, ast.Global) and (initialization or not set(node.names) <= set(globals_names)):
      raise EquationError("Global state must be initialized in global code")
    if isinstance(node, ast.Return) and initialization:
      raise EquationError("Global code cannot return")
    stack.extend((child, node, depth + 1) for child in ast.iter_child_nodes(node))
  if len(assigned) > MAX_VARIABLES:
    raise EquationError("Too many variables")
  return tree, assigned


class _NumericBools(ast.NodeTransformer):
  def generic_visit(self, node):
    node = super().generic_visit(node)
    # bool + bool creates an int in Python. Keep those results numeric floats too,
    # so repeated arithmetic cannot grow arbitrary-size integers from comparisons.
    if isinstance(node, ast.Compare) or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
      return ast.copy_location(ast.Call(ast.Name('_number', ast.Load()), [node], []), node)
    return node


def compile_numeric_equation(globals_code, function_code, input_count):
  """Validated function for native callers that supply floats and check numeric results."""
  if type(input_count) is not int or not 0 <= input_count <= MAX_INPUTS:
    raise EquationError("At most 32 additional inputs are allowed")
  inputs = ["time", "value"] + [f"v{i + 1}" for i in range(input_count)]
  initialization, state = _validate(globals_code, initialization=True)
  body, _ = _validate(function_code, globals_names=state)
  if state & set(inputs):
    raise EquationError("Global state cannot replace signal inputs")
  # Build the wrapper from a fixed template, never interpolated user source.
  wrapper = ast.parse('def _calc():\n  pass')
  wrapper.body[0].args.args = [ast.arg(arg=name) for name in inputs]
  wrapper.body[0].body = body.body or [ast.Pass()]
  program = ast.fix_missing_locations(_NumericBools().visit(ast.Module(body=initialization.body + wrapper.body, type_ignores=[])))
  namespace = {"__builtins__": {}, "math": math, "abs": abs, "min": min, "max": max,
               "int": lambda x: float(int(x)), "map": map, "_number": _number}
  # Only the fully validated, float-normalized AST crosses this execution boundary.
  exec(compile(program, '<layout equation>', 'exec'), namespace)
  return namespace['_calc']


def compile_equation(globals_code, function_code, input_count):
  calc = compile_numeric_equation(globals_code, function_code, input_count)

  def evaluate(*args):
    if len(args) != input_count + 2:
      raise EquationError("Incorrect number of signal inputs")
    result = calc(*(_number(v) for v in args))
    if type(result) is tuple and len(result) == 2:
      return tuple(_number(v) for v in result)
    return _number(result)

  return evaluate
