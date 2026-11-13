"""Migration coverage for all shipped PlotJuggler XML layouts."""

import ctypes.util
import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET
import pytest

repo = Path(__file__).resolve().parents[4]
core = repo / 'openpilot/tools/cabana/core'
spec = importlib.util.spec_from_file_location('layout_convert', core / 'layout_convert.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_bundled_legacy_layouts():
  count = plots = curves = derivatives = scales = 0
  for path in sorted((repo / 'openpilot/tools/cabana/layouts').glob('*.xml')):
    document = ET.parse(path).getroot()
    converted = converter.convert(path)
    flattened = []

    def visit(node, flattened=flattened):
      if 'children' in node:
        for child in node['children']:
          visit(child)
      else:
        flattened.append(node)

    for tab in converted['tabs']:
      visit(tab['root'])
    original = list(document.iter('plot'))
    assert len(flattened) == len(original), path
    for before, after in zip(original, flattened, strict=True):
      assert len(before.findall('curve')) == len(after['curves']), path
      for source, result in zip(before.findall('curve'), after['curves'], strict=True):
        assert source.get('name') == result['name'], path
        assert source.get('color', '#0072b2') == result['color'], path
        transform = source.find('transform')
        if transform is not None:
          if transform.get('name') == 'Derivative':
            assert result['transform'] == 'derivative', path
            derivatives += 1
          elif transform.get('name') == 'Scale/Offset':
            assert result['transform'] == 'scale', path
            scales += 1
      curves += len(after['curves'])
    plots += len(flattened)
    count += 1
  assert count == 14 and plots == 89 and derivatives == 25 and scales == 3
  print(f'Legacy layouts: {count}; plots: {plots}; curves: {curves}; derivatives: {derivatives}; scales: {scales}')


def test_legacy_lua_initialization():
  if not any(ctypes.util.find_library(name) for name in ('lua5.4', 'lua5.3')):
    pytest.skip('Lua 5.3/5.4 runtime is required for XML equations')
  spec = importlib.util.spec_from_file_location('math_eval', core / 'math_eval.py')
  evaluator = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(evaluator)
  program = evaluator.LuaEvaluator('total=0', 'total=total+value; return total+v1', 1)
  try:
    assert program.sample(0, 2, 10) == 12
    assert program.sample(1, 3, 10) == 15
  finally:
    program.close()
  print('Legacy Lua initialization and state continuity check passed.')
