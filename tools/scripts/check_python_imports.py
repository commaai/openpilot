#!/usr/bin/env python3
import ast
import importlib
import os

DIRS = ['selfdrive', 'common', 'tools', 'system', 'scripts', 'release']
IGNORED_IMPORTS = ['PyNvCodec',
                   'openpilot.third_party.acados.acados_template',
                   'openpilot.selfdrive.modeld.runners.thneedmodel_pyx',
                   'openpilot.system.hardware.tici.casync']

def try_import(file_path, module):
  if module in IGNORED_IMPORTS:
    return True
  try:
    importlib.import_module(module)
    return True
  except ImportError:
    print(f"Can't import {module} from {file_path}. Is this package missing from pyproject.toml?")
    return False

def find_python_files(dirs):
  python_files = []
  for folder in dirs:
    for root, _, files in os.walk(folder):
      python_files.extend([os.path.join(root, x) for x in filter(lambda x: x.endswith('.py'), files)])
  return python_files

def extract_imports(file_path):
  with open(file_path) as file:
    tree = ast.parse(file.read(), filename=file_path)
    imports = []
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        module = ''.join([x.name for x in node.names])
        imports.append((file_path, module))
      elif isinstance(node, ast.ImportFrom):
        imports.append((file_path, node.module))
    return all(try_import(*x) for x in imports)

if __name__ == '__main__':
  quit(not all(list(map(extract_imports, find_python_files(DIRS)))))
