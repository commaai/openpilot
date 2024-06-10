import pytest
import sys
import subprocess

import openpilot.common.proc_title_pyx as proc_title

def _run(script: str):
  result = subprocess.run(
    [sys.executable],
    input=script,
    capture_output=True,
    text=True,
    shell=True,
  )

  return result.stdout


@pytest.mark.skipif(sys.platform not in ('linux'), reason="only supported on linux")
class TestProcTitle:
  def test_set_proc_title(self):
    title = 'test'
    script = f'''
from openpilot.common.proc_title_pyx import set_proc_title
import os

set_proc_title("{title}")
print(os.getpid())
print(os.popen("ps -x -o pid,command 2> /dev/null").read())
'''
    out = _run(script)
    lines = [line for line in out.splitlines() if line]
    test_pid = lines.pop(0)
    pids = dict([line.strip().split(None, 1) for line in lines])
    assert pids[test_pid] == title


  def test_truncate_long_title(self):
    title_len = 1000
    script = f'''
from openpilot.common.proc_title_pyx import set_proc_title
import os

set_proc_title(''.join(['a' for _ in range({title_len})]))
print(os.getpid())
print(os.popen("ps -x -o pid,command 2> /dev/null").read())
'''
    out = _run(script)
    lines = [line for line in out.splitlines() if line]
    test_pid = lines.pop(0)
    pids = dict([line.strip().split(None, 1) for line in lines])
    assert len(pids[test_pid]) < title_len


  def test_set_and_get(self):
    proc_title.set_proc_title('abc')
    assert proc_title.get_proc_title() == 'abc'
    proc_title.set_proc_title('again')
    assert proc_title.get_proc_title() == 'again'
