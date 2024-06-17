from openpilot.common.threadname import setthreadname, getthreadname, LINUX

class TestThreadName:
  def test_set_get_threadname(self):
    if LINUX:
      name = 'TESTING'
      setthreadname(name)
      assert name == getthreadname()
