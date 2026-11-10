from datetime import datetime, UTC
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess


def environment():
  versions = {}
  for package in ('torch', 'transformers', 'numpy', 'tinygrad', 'flash-attn', 'triton', 'torchao'):
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      pass
  chestnuts = []
  for path in Path('/sys/bus/usb/devices').glob('*'):
    try:
      if (path / 'idVendor').read_text().strip() in ('3801', 'add1') and (path / 'idProduct').read_text().strip() == '0001':
        chestnuts.append({key: (path / key).read_text().strip() for key in ('product', 'serial', 'speed')})
    except FileNotFoundError:
      pass
  commit = subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True)
  return {'recorded_at': datetime.now(UTC).isoformat(), 'python': platform.python_version(),
          'packages': versions, 'openpilot_commit': commit.stdout.strip(), 'chestnut_usb': chestnuts,
          'runtime_environment': {key: os.environ[key] for key in ('DEV', 'CUDA_VISIBLE_DEVICES', 'TC_OPT', 'BEAM') if key in os.environ}}
