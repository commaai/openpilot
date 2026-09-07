import os
import subprocess
import tempfile
from pathlib import Path

from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase


class TestDownloader(OpenpilotTestCase):
  @parameterized.expand(["ok", "fail", "abort", "missing"])
  def test_downloader_spawn(self, mode):
    tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
    if mode != "missing":
      launcher = tmp_path / "python3"
      launcher.write_text("""#!/bin/sh
set -eu
[ "$1" = "-m" ]
[ "$2" = "openpilot.tools.lib.file_downloader" ]
[ "$3" = "download" ]
[ -z "${OPENPILOT_PREFIX+x}" ]
[ ! -t 0 ]
if read -r line; then exit 9; fi
case "$4" in
  'url with spaces & literal $value')
    printf 'PROGRESS:42:100\\n' >&2
    printf 'downloaded path\\n'
    ;;
  fail) exit 7 ;;
  abort) exec /bin/sleep 10 ;;
  *) exit 8 ;;
esac
""")
      launcher.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + (os.pathsep + os.environ["PATH"] if mode != "missing" else ""),
               OPENPILOT_PREFIX="cabana-spawn-test")
    result = subprocess.run([str(Path(__file__).with_name("test_cabana")), "--check-downloader", mode],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
