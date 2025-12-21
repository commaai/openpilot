import os
import sys
import platform
from openpilot.selfdrive.controls.lib.acados_setup import get_acados_lib_path, get_acados_dir

def test_get_acados_lib_path_linux_x86_64(monkeypatch):
  monkeypatch.setattr(sys, 'platform', 'linux')
  monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
  expected = os.path.join(get_acados_dir(), 'x86_64', 'lib')
  assert get_acados_lib_path() == expected

def test_get_acados_lib_path_linux_aarch64(monkeypatch):
  monkeypatch.setattr(sys, 'platform', 'linux')
  monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
  expected = os.path.join(get_acados_dir(), 'larch64', 'lib')
  assert get_acados_lib_path() == expected

def test_get_acados_lib_path_darwin(monkeypatch):
  monkeypatch.setattr(sys, 'platform', 'darwin')
  expected = os.path.join(get_acados_dir(), 'Darwin', 'lib')
  assert get_acados_lib_path() == expected
