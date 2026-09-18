import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from PIL import ImageGrab
from openpilot.cereal import log

REPO = Path(__file__).resolve().parents[4]


def run_replay(output: Path, no_can: bool):
  repo = REPO
  route = '2026-09-17--12-00-00'
  segment = output / (route + '--0')
  segment.mkdir(parents=True, exist_ok=True)
  with (segment / 'rlog').open('wb') as stream:
    for i in range(600):
      event = log.Event.new_message(logMonoTime=10**12 + i * 100_000_000, valid=True)
      state = event.init('carState')
      state.vEgo = 15 + 5 * math.sin(i / 30)
      state.aEgo = math.cos(i / 30)
      state.steeringPressed = i % 100 > 50
      event.write(stream)
      if no_can:
        continue
      event = log.Event.new_message(logMonoTime=10**12 + i * 100_000_000, valid=True)
      frames = event.init('can', 1)
      frames[0].address = 123
      frames[0].src = 0
      frames[0].dat = bytes([i % 256] + [0] * 7)
      event.write(stream)

  dbc = output / 'test.dbc'
  dbc.write_text('VERSION ""\nNS_ :\nBS_:\nBU_: TEST\nBO_ 123 TEST: 8 TEST\n SG_ VALUE : 0|8@1+ (1,0) [0|255] "" TEST\n')
  config = output / 'config'
  config.mkdir(exist_ok=True)
  (config / 'cabana.json').write_text(json.dumps({
    'recent_dbc_file': '' if no_can else str(dbc),
    'active_charts': ['cereal|/carState/vEgo,cereal|/carState/aEgo' + ('' if no_can else ',0:7B|VALUE'), 'cereal|/carState/steeringPressed'],
  }))
  environment = dict(os.environ, XDG_CONFIG_HOME=str(config), LIBGL_ALWAYS_SOFTWARE='1')
  environment.pop('WAYLAND_DISPLAY', None)
  environment['XDG_SESSION_TYPE'] = 'x11'
  with (output / 'app.log').open('w') as app_log:
    args = [str(repo / 'openpilot/tools/cabana/_cabana_ui'), '--data_dir', str(output), '--no-vipc', route]
    if not no_can:
      args += ['--dbc', str(dbc)]
    process = subprocess.Popen(args, cwd=repo, env=environment,
                 stdout=app_log, stderr=subprocess.STDOUT)
    try:
      time.sleep(5)
      assert process.poll() is None, (output / 'app.log').read_text()
      screenshot = ImageGrab.grab(xdisplay=os.environ['DISPLAY'])
      assert screenshot.getbbox(), 'Application did not render on the test display'
      screenshot.save(output / 'cabana.png')
      process.send_signal(signal.SIGTERM)
      assert process.wait(timeout=20) == 0, (output / 'app.log').read_text()
    finally:
      if process.poll() is None:
        process.kill()
        process.wait()
  saved = json.loads((config / 'cabana.json').read_text())
  assert len(saved['active_charts']) == 2, saved['active_charts']
  assert any('cereal|/carState/vEgo,cereal|/carState/aEgo' in chart for chart in saved['active_charts'])
  if not no_can:
    assert any('0:7B|VALUE' in chart for chart in saved['active_charts'])


@unittest.skipUnless(shutil.which("xvfb-run"), "Requires xvfb-run for the headless GUI")
class TestCerealReplay(unittest.TestCase):
  def test_replay_and_restore(self):
    for no_can in (False, True):
      with self.subTest(no_can=no_can), tempfile.TemporaryDirectory(prefix="cabana-replay-") as directory:
        command = ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24", sys.executable,
                   str(Path(__file__).resolve()), "--worker", directory]
        if no_can:
          command.append("--no-can")
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  if "--worker" in sys.argv:
    run_replay(Path(sys.argv[sys.argv.index("--worker") + 1]), "--no-can" in sys.argv)
  else:
    unittest.main()
