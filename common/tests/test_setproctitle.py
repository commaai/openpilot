from openpilot.common.setproctitle import setproctitle, getproctitle
class TestProcTitle:
  def test_set_get_proctitle(self):
    name = 'TESTING'
    setproctitle(name)
    assert name == getproctitle()

if __name__ == '__main__':
  t = TestProcTitle()
  t.test_set_get_proctitle()
