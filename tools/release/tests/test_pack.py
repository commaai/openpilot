import json
import os
import subprocess
import sys
import zipapp
import zipfile
from pathlib import Path

import pytest

from tools.release.pack import BOOTSTRAP, VENDOR_DIR, bundle_raylib, check


@pytest.mark.parametrize('binding', ['pyray', 'raylib_nb'])
def test_bundled_raylib(tmp_path, monkeypatch, binding):
  monkeypatch.setenv("RAYLIB_BACKEND", "headless")
  source = tmp_path / 'source'
  source.mkdir()
  bundle_raylib(source / VENDOR_DIR)
  (source / 'app.py').write_text(f'''
import json
import {binding} as rl
import _cffi_backend

def main():
  rl.init_window(64, 64, "bundled raylib")
  rl.begin_drawing()
  rl.clear_background(rl.RED)
  rl.end_drawing()
  rl.close_window()
  print("BUNDLE_PATHS=" + json.dumps([rl.__file__, _cffi_backend.__file__]))
''')
  (source / '__main__.py').write_text(BOOTSTRAP.format(module='app', entrypoint='main'))
  archive = tmp_path / 'app.pyz'
  zipapp.create_archive(source, archive)
  # A broken system/site package must never override the bundled binding.
  poison = tmp_path / 'poison'
  poison.mkdir()
  (poison / f'{binding}.py').write_text('raise RuntimeError("system binding loaded")')
  env = os.environ | {'RAYLIB_BACKEND': 'headless', 'PYTHONPATH': str(poison)}
  # -S removes site-packages altogether: this must work with only Python + the archive.
  result = subprocess.run([sys.executable, '-S', str(archive)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30, check=True)
  paths = json.loads(next(line.removeprefix('BUNDLE_PATHS=') for line in result.stdout.splitlines()
                          if line.startswith('BUNDLE_PATHS=')))
  assert all('openpilot-zipapp-' in path for path in paths)
  assert not Path(paths[0]).exists()  # Private extraction is cleaned after normal exit.
  with zipfile.ZipFile(archive) as z:
    assert not any(name.endswith(('.a', '.pyc', '.h')) for name in z.namelist())
  check(archive, 'app')
