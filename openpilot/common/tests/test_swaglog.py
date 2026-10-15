import logging
import sys

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.logging_extra import SwagLogger


class ListHandler(logging.Handler):
  def __init__(self):
    super().__init__()
    self.records = []

  def emit(self, record):
    self.records.append(record)


class TestSwagLogger(OpenpilotTestCase):
  def setup_method(self):
    self.log = SwagLogger()
    self.handler = ListHandler()
    self.log.addHandler(self.handler)

  def test_find_caller(self):
    def call_site(record):
      return record.pathname, record.lineno, record.funcName

    for level in ("debug", "info", "warning", "error", "exception"):
      line = sys._getframe().f_lineno + 1
      getattr(self.log, level)("msg")
      assert call_site(self.handler.records[-1]) == (__file__, line, "test_find_caller"), level

    line = sys._getframe().f_lineno + 1
    self.log.event("evt", x=1)
    assert call_site(self.handler.records[-1]) == (__file__, line, "test_find_caller")

    def helper():
      self.log.warning("from helper", stacklevel=2)
    line = sys._getframe().f_lineno + 1
    helper()
    assert call_site(self.handler.records[-1]) == (__file__, line, "test_find_caller")
