#!/usr/bin/env python3

import importlib
import importlib.metadata
import importlib.util
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
VENDOR_DIR = '_zipapp_vendor'
# CFFI is needed by pyray; nanobind's runtime is compiled into raylib_nb.
RAYLIB_MODULES = ('pyray', 'raylib', 'raylib_nb', 'cffi', '_cffi_backend', 'pycparser')


BOOTSTRAP = """
import importlib
import shutil
import signal
import sys
import tempfile
import zipfile
from pathlib import Path

def terminate(signum, frame):
  raise SystemExit(128 + signum)

with tempfile.TemporaryDirectory(prefix='openpilot-zipapp-') as tmp:
  signal.signal(signal.SIGTERM, terminate)
  with zipfile.ZipFile(__loader__.archive) as archive:
    for member in archive.infolist():
      path = Path(member.filename)
      if path.parts and path.parts[0] == '_zipapp_vendor':
        relative = Path(*path.parts[1:])
        if path.is_absolute() or '..' in path.parts:
          raise ValueError('invalid bundled module path')
        if not member.is_dir():
          target = Path(tmp) / relative
          target.parent.mkdir(parents=True, exist_ok=True)
          with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
  sys.path.insert(0, tmp)
  module = importlib.import_module({module!r})
  getattr(module, {entrypoint!r})()
"""


def bundle_raylib(destination):
  destination.mkdir(parents=True, exist_ok=True)
  for name in RAYLIB_MODULES:
    spec = importlib.util.find_spec(name)
    if spec is None:
      raise RuntimeError(f'{name} is missing; install the pinned Raylib and CFFI packages before packing')
    if spec.submodule_search_locations:
      source = Path(spec.origin).parent
      # Keep runtime libraries and licenses, but omit static libraries, headers and bytecode.
      shutil.copytree(source, destination / name,
                      ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.a', '*.h', 'include'))
    else:
      shutil.copy2(spec.origin, destination / Path(spec.origin).name)
  for name in ('comma-deps-raylib', 'cffi', 'pycparser'):
    distribution = importlib.metadata.distribution(name)
    for file in distribution.files or ():
      if file.parts[0].endswith('.dist-info'):
        target = destination / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(distribution.locate_file(file), target)


# Zipapps run without an openpilot checkout (e.g. AGNOS setup after a factory reset), so the
# entrypoint must import using only the zipapp and installed packages.
CHECK = """
import importlib, os, site, sys, tempfile, zipfile
zipapp, module = sys.argv[1:]
baseline = set(sys.modules)
with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zipapp) as archive:
    for member in archive.namelist():
      if member.startswith('_zipapp_vendor/'):
        archive.extract(member, tmp)
  vendor = os.path.join(tmp, '_zipapp_vendor')
  sys.path[:0] = [vendor, zipapp]
  importlib.import_module(module)
  for name in ('pyray', 'raylib', 'raylib_nb', 'cffi', '_cffi_backend', 'pycparser'):
    m = importlib.import_module(name)
    if not os.path.realpath(m.__file__).startswith(vendor + os.sep):
      sys.exit('loaded system copy of ' + name)
  allowed = {os.path.realpath(p) for p in (vendor, zipapp, sys.prefix, sys.base_prefix, *site.getsitepackages())}
  outside = sorted(name for name, m in list(sys.modules.items())
                   if name not in baseline and getattr(m, '__file__', None) and not any(os.path.realpath(m.__file__).startswith(a + os.sep) for a in allowed))
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
  print('Bundling Raylib from this environment; build with the target architecture and Python version.')

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
    bundle_raylib(Path(tmp) / VENDOR_DIR)
    (Path(tmp) / '__main__.py').write_text(BOOTSTRAP.format(module=args.module, entrypoint=args.entrypoint))
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER)

  check(args.output, args.module)

  print(f'created executable {Path(args.output).resolve()}')
