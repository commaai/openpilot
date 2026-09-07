import math
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from openpilot.tools.cabana.analysis.cabana_equations import (
  EquationError, MAX_DEPTH, MAX_INPUTS, MAX_ITEMS, MAX_NODES, MAX_SOURCE_BYTES, MAX_VARIABLES, compile_numeric_equation,
)


class TestEquationRestrictions(unittest.TestCase):
  def test_math_and_numeric_helpers(self):
    cases = {
      'math.sqrt(value) + math.sin(math.pi / 2)': 4,
      'min(8, max(value, 4))': 8,
      'min((value, 4, 6))': 4,
      'int(-value / 2)': -4,
      'value ** 2': 81,
      'abs(-value) if 0 < value <= 10 else 0': 9,
    }
    for expression, expected in cases.items():
      with self.subTest(expression=expression):
        result = compile_numeric_equation('', 'return ' + expression, 0)(0.0, 9.0)
        self.assertIs(type(result), float)
        self.assertAlmostEqual(result, expected)
    self.assertTrue(math.isnan(compile_numeric_equation('', 'return math.nan', 0)(0.0, 1.0)))

  def test_numeric_state_and_reset(self):
    code = 'global total\nif value > 0:\n  total += value\nelse:\n  total -= v1\nreturn total'
    calc = compile_numeric_equation('total = 0', code, 1)
    self.assertEqual(calc(0.0, 2.0, 1.0), 2)
    self.assertEqual(calc(1.0, 3.0, 1.0), 5)
    self.assertEqual(calc(2.0, -1.0, 1.0), 4)
    self.assertEqual(compile_numeric_equation('total = 0', code, 1)(0.0, 2.0, 1.0), 2)
    # Local variables retain Python's scope rules and do not persist between samples.
    self.assertEqual(compile_numeric_equation('offset = 3', 'offset = value\nreturn offset', 0)(0.0, 5.0), 5)

  def test_unpacking_and_timestamp_return(self):
    calc = compile_numeric_equation('', 'a, b = math.radians(value), math.radians(v1)\nx, y = (a, b)\nreturn time + 1, x + y', 1)
    time, value = calc(10.0, 90.0, 90.0)
    self.assertEqual(time, 11)
    self.assertAlmostEqual(value, math.pi)

  def test_forbidden_syntax_and_escape_paths(self):
    cases = [
      'import math\nreturn value', 'import os\nreturn value', 'from math import sin\nreturn sin(value)',
      'import statistics\nreturn statistics.mean((value, 1))',
      'return __builtins__', 'return __import__("os")',
      'return __import__.__globals__["builtins"]', 'return math.sin.__globals__',
      'return ().__class__.__base__.__subclasses__()', 'return value.__class__',
      'return math.__dict__', 'return math.__loader__', 'return math.sin.__call__(value)',
      'return getattr(math, "sin")(value)', 'return vars(math)', 'return globals()', 'return locals()',
      'return eval("1")', 'return exec("pass")', 'return open("/dev/null")',
      'return compile("1", "x", "eval")', 'return type(value)',
      'return math', 'return math.sin', 'return abs', 'fn = math.sin\nreturn fn(value)',
      'return (math.sin if value else math.cos)(value)', 'return (lambda: value)()',
      'def f():\n  return f()\nreturn f()', 'class C:\n  pass\nreturn C()',
      'try:\n  return 1 / 0\nexcept Exception as e:\n  return e.__traceback__',
      'with value:\n  pass\nreturn 0', 'raise value', 'assert value\nreturn value',
      'for x in range(100):\n  pass\nreturn value', 'while True:\n  pass',
      'return [x for x in value]', 'return (x for x in value)', 'return {x: x for x in value}',
      'return await value', 'yield value', 'return (x := value)',
      'return "text"', 'return b"text"', 'return f"{value}"', 'return None', 'return 1j',
      'return [value]', 'return {value}', 'return {1: value}', 'return value[0]',
      'return value ** 0.5', 'return value ** value', 'return value ** 17',
      'return value << 1000', 'return value & 1', 'return value @ value', 'return value is math',
      'return round(value)', 'return float(value)', 'return math.floor(value)', 'return math.pow(value, 2)',
      'return math.factorial(1000000)', 'return math.comb(1000000, 100)', 'return math.prod(value)',
      'return min(*value)', 'return math.sin(x=value)', 'return map(math.sin, (1, 2))',
      'x = (1, 2)\nreturn x', 'return (1, (2, 3))', 'return (1, 2) * 1000000000',
      'x, y = map(abs, (1, 2))\nreturn x', 'x, y = map(math.sin, (1, 2))\nreturn x', 'x, y = (1,)\nreturn x',
      'x, *y = (1, 2)\nreturn x', 'return (1, 2, 3)',
      'math.pi = 3\nreturn value', 'math = 3\nreturn value', 'abs = 3\nreturn value',
      'value[0] = 3\nreturn value', 'del value\nreturn 0',
      '_power = 3\nreturn value', 'return _calc(value)', 'return __debug__',
      'global unknown\nunknown = 1\nreturn unknown',
      'if False:\n  import os\nreturn 1', 'return 1\nimport os',
    ]
    for code in cases:
      with self.subTest(code=code), self.assertRaises(EquationError):
        compile_numeric_equation('', code, 0)

  def test_globals_are_equally_restricted(self):
    for code in ('import math', 'x = ().__class__', 'while True:\n  pass', 'x = [1]', 'math.pi = 2',
                 'value = 0', 'time = 0', 'v1 = 0', 'return 0'):
      with self.subTest(code=code), self.assertRaises(EquationError):
        compile_numeric_equation(code, 'return value', 1)

  def test_validates_everything_before_executing_initialization(self):
    # The division would raise first if initialization ran before validating the body.
    with self.assertRaises(EquationError):
      compile_numeric_equation('x = 1 / 0', 'return math.__dict__', 0)
    with patch('builtins.exec', side_effect=AssertionError('unvalidated exec')):
      with self.assertRaises(EquationError):
        compile_numeric_equation('import os', 'return value', 0)

  def test_structural_limits(self):
    cases = [
      ('#' * (MAX_SOURCE_BYTES + 1), 'return value', 0),
      ('#' + 'é' * (MAX_SOURCE_BYTES // 2), 'return value', 0),
      ('', 'return value\n' + 'pass\n' * MAX_NODES, 0),
      ('', 'return ' + 'not ' * MAX_DEPTH + 'value', 0),
      ('', 'return min(' + ','.join(['1'] * (MAX_ITEMS + 1)) + ')', 0),
      ('', 'return min((' + ','.join(['1'] * (MAX_ITEMS + 1)) + '))', 0),
      ('', '\n'.join(f'x{i} = 1' for i in range(MAX_VARIABLES + 1)) + '\nreturn value', 0),
      ('', 'return value', MAX_INPUTS + 1), ('', 'return value', -1), ('', 'return value', True),
    ]
    for globals_code, code, count in cases:
      with self.subTest(code=code[:80], count=count), self.assertRaises(EquationError):
        compile_numeric_equation(globals_code, code, count)

  def test_no_state_leakage(self):
    compile_numeric_equation('state = 4', 'return state', 0)(0.0, 1.0)
    with self.assertRaises(NameError):
      compile_numeric_equation('', 'return state', 0)(0.0, 1.0)

  def test_resource_attacks_in_limited_process(self):
    # Keep this regression safe even if the language restrictions accidentally regress.
    script = '''
import math
import resource
from openpilot.tools.cabana.analysis.cabana_equations import compile_numeric_equation
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
attacks = ['return 10 ** (10 ** 10)', 'x = 10\\nx **= 10 ** 10\\nreturn x',
           'return (0, 0) * 1000000000', 'return "x" * 1000000000',
           'return math.factorial(1000000000)', 'while True:\\n  pass',
           'return round(value, 1000000000)']
for code in attacks:
  try:
    compile_numeric_equation('', code, 0)(0.0, 1.0)
  except (ValueError, OverflowError):
    pass
  else:
    raise AssertionError(code)
# Numeric helper results must not reintroduce arbitrary-size integer multiplication.
for expression in ('int(1e100)', 'abs(1e100)', '(value > 0) + (value > 0)', '(not 0) + (not 0)'):
  code = f'x = {expression}\\n' + 'x *= x\\n' * 40 + 'return x'
  result = compile_numeric_equation('', code, 0)(0.0, 1.0)
  assert type(result) is float and math.isinf(result)
print('resource limits passed')
'''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)], capture_output=True, text=True, timeout=5)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    assert 'resource limits passed' in result.stdout
