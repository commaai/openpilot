import unittest
from unittest.mock import MagicMock
from collections.abc import Callable


class MockDevice:
  def __init__(self):
    self._interactive_timeout_callbacks: list[Callable] = []
    self._interaction_time = 0
    self._prev_timed_out = False

  def add_interactive_timeout_callback(self, callback: Callable):
    self._interactive_timeout_callbacks.append(callback)

  def remove_interactive_timeout_callback(self, callback: Callable):
    while callback in self._interactive_timeout_callbacks:
      self._interactive_timeout_callbacks.remove(callback)

  def trigger_timeout(self):
    for callback in list(self._interactive_timeout_callbacks):
      callback()


class TestInteractiveTimeoutCallbacks(unittest.TestCase):
  def test_add_and_trigger(self):
    dev = MockDevice()
    cb1 = MagicMock()
    cb2 = MagicMock()

    dev.add_interactive_timeout_callback(cb1)
    dev.add_interactive_timeout_callback(cb2)
    self.assertEqual(len(dev._interactive_timeout_callbacks), 2)

    dev.trigger_timeout()
    cb1.assert_called_once()
    cb2.assert_called_once()

  def test_remove_callback(self):
    dev = MockDevice()
    cb1 = MagicMock()
    cb2 = MagicMock()

    dev.add_interactive_timeout_callback(cb1)
    dev.add_interactive_timeout_callback(cb2)
    dev.remove_interactive_timeout_callback(cb1)

    self.assertEqual(len(dev._interactive_timeout_callbacks), 1)
    dev.trigger_timeout()
    cb1.assert_not_called()
    cb2.assert_called_once()

  def test_remove_duplicates(self):
    dev = MockDevice()
    cb1 = MagicMock()

    dev.add_interactive_timeout_callback(cb1)
    dev.add_interactive_timeout_callback(cb1)
    self.assertEqual(len(dev._interactive_timeout_callbacks), 2)

    dev.remove_interactive_timeout_callback(cb1)
    self.assertEqual(len(dev._interactive_timeout_callbacks), 0)

    dev.trigger_timeout()
    cb1.assert_not_called()

  def test_remove_during_iteration(self):
    dev = MockDevice()
    cb2 = MagicMock()

    def self_removing_cb():
      dev.remove_interactive_timeout_callback(self_removing_cb)

    dev.add_interactive_timeout_callback(self_removing_cb)
    dev.add_interactive_timeout_callback(cb2)

    # Should not raise RuntimeError: list changed size during iteration
    dev.trigger_timeout()
    cb2.assert_called_once()
    self.assertEqual(len(dev._interactive_timeout_callbacks), 1)
    self.assertNotIn(self_removing_cb, dev._interactive_timeout_callbacks)


if __name__ == '__main__':
  unittest.main()
