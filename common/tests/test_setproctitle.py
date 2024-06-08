from openpilot.common.setproctitle import setproctitle, getproctitle, LINUX

class TestProcTitle:
  def test_set_get_proctitle(self):
    if LINUX:
      name = 'TESTING'
      setproctitle(name)
      assert name == getproctitle()
