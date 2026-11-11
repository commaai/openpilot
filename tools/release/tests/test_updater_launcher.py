import concurrent.futures
import os
import subprocess
import tarfile
from pathlib import Path


def test_updater_launcher(tmp_path):
  launcher = (Path(__file__).resolve().parents[3] / 'openpilot/common/hardware/comma/updater').read_text()
  runtime = tmp_path / 'source/agnos-apps'
  runtime.mkdir(parents=True)
  binary = runtime / 'agnos-apps'
  binary.write_text('#!/bin/sh\nprintf "%s|%s|%s\\n" "${0##*/}" "$1" "$2"\n')
  binary.chmod(0o755)
  (runtime / 'updater').symlink_to('agnos-apps')
  app = tmp_path / 'updater'
  app.write_text(launcher.replace('CACHE=/data/tmp/agnos-updater', f'CACHE="{tmp_path}/cache"'))
  app.chmod(0o755)
  payload = tmp_path / 'updater.tar.xz'
  with tarfile.open(payload, 'w:xz') as archive:
    archive.add(runtime, arcname='agnos-apps')

  def launch(_):
    result = subprocess.run([str(app), 'helper with spaces', 'manifest.json'], env={'PATH': os.defpath},
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout == 'updater|helper with spaces|manifest.json\n'

  # Two simultaneous first launches must both work, followed by a cached launch.
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(launch, range(2)))
  launch(0)
  cache = tmp_path / 'cache'
  assert not list(cache.glob('.unpack.*'))
  assert len(list(cache.iterdir())) == 1

  payload.write_bytes(b'broken')
  result = subprocess.run([str(app)], capture_output=True, timeout=10)
  assert result.returncode != 0
  assert not list(cache.glob('.unpack.*'))
