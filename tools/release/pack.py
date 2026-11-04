#!/usr/bin/env python3

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipapp
from argparse import ArgumentParser
from pathlib import Path

from openpilot.common.basedir import BASEDIR


# source dir -> package name in the zipapp
DIRS = {
  'openpilot': 'openpilot',
  'opendbc_repo/opendbc': 'opendbc',  # openpilot.cereal imports it
}
EXTS = ['.png', '.py', '.ttf', '.capnp', '.json', '.fnt', '.mo', '.po']
EXCLUDE = ['openpilot/selfdrive/assets/training']
INTERPRETER = '/usr/bin/env python3'


# Zipapps run without an openpilot checkout (e.g. AGNOS setup after a factory reset), so the
# entrypoint must import using only the zipapp and installed packages.
CHECK = """
import importlib, os, site, sys
zipapp, module = sys.argv[1:]
sys.path.insert(0, zipapp)
importlib.import_module(module)
allowed = {os.path.realpath(p) for p in (zipapp, sys.prefix, sys.base_prefix, *site.getsitepackages())}
outside = sorted(name for name, m in list(sys.modules.items())
                 if getattr(m, '__file__', None) and not any(os.path.realpath(m.__file__).startswith(a + os.sep) for a in allowed))
if outside:
  sys.exit('imported from outside the zipapp: ' + ', '.join(outside))
"""


def copy(src, dest):
  if any(src.endswith(ext) for ext in EXTS) and not any(exc in src for exc in EXCLUDE):
    shutil.copy2(src, dest, follow_symlinks=True)


def check(output, module):
  with tempfile.TemporaryDirectory() as cwd:
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'} | {'PWD': cwd}
    result = subprocess.run([sys.executable, '-I', '-c', CHECK, os.path.realpath(output), module], cwd=cwd, env=env,
                            capture_output=True, text=True)
  if result.returncode != 0:
    os.remove(output)
    sys.exit(f'{output} is not self-contained:\n{result.stderr.strip()}')


if __name__ == '__main__':
  parser = ArgumentParser(prog='pack.py', description="package script into a portable executable", epilog='comma.ai')
  parser.add_argument('-e', '--entrypoint', help="function to call in module, default is 'main'", default='main')
  parser.add_argument('-o', '--output', help='output file')
  parser.add_argument('module', help="the module to target, e.g. 'openpilot.system.ui.spinner'")
  args = parser.parse_args()

  print('WARNING: copying all files! make sure to run scons and git tree is clean')

  if not args.output:
    args.output = args.module

  try:
    mod = importlib.import_module(args.module)
  except ModuleNotFoundError:
    print(f'{args.module} not found, typo?')
    sys.exit(1)

  if not hasattr(mod, args.entrypoint):
    print(f'{args.module} does not have a {args.entrypoint}() function, typo?')
    sys.exit(1)

  with tempfile.TemporaryDirectory() as tmp:
    for src, package in DIRS.items():
      shutil.copytree(BASEDIR + '/' + src, tmp + '/' + package, symlinks=False, dirs_exist_ok=True, copy_function=copy)
    entry = f'{args.module}:{args.entrypoint}'
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER, main=entry)

  check(args.output, args.module)

  print(f'created executable {Path(args.output).resolve()}')
