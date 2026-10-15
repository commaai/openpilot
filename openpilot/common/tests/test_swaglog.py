import logging
import sys
import time

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.logging_extra import SwagLogger, RATE_LIMIT_BURST, RATE_LIMIT_PER_S


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


class TestCallSiteRateLimiter(OpenpilotTestCase):
  def setup_method(self):
    self.log = SwagLogger()
    self.handler = ListHandler()
    self.log.addHandler(self.handler)
    self.now = 100.0

  @property
  def messages(self):
    return [r.getMessage() for r in self.handler.records]

  def warn(self, *args):
    self.log.warning(*args)  # one call site shared by everything logged through here

  def test_burst_then_one_per_second(self, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: self.now)
    for i in range(RATE_LIMIT_BURST + 5):
      self.warn("msg %d", i)
    assert self.messages == [f"msg {i}" for i in range(RATE_LIMIT_BURST)]

    # not refilled yet
    self.now += 0.9 / RATE_LIMIT_PER_S
    self.warn("msg late")
    assert len(self.messages) == RATE_LIMIT_BURST

    # one token refilled, the record that passes carries the suppressed count
    self.now += 0.1 / RATE_LIMIT_PER_S
    self.warn("msg %d", 99)
    assert self.messages[-1] == "msg 99 (suppressed 6 repeats)"
    self.warn("dropped")
    assert len(self.messages) == RATE_LIMIT_BURST + 1

    # the count resets once reported
    self.now += 1 / RATE_LIMIT_PER_S
    self.warn("msg ok")
    assert self.messages[-1] == "msg ok (suppressed 1 repeats)"
    self.now += 1 / RATE_LIMIT_PER_S
    self.warn("msg ok")
    assert self.messages[-1] == "msg ok"

  def test_call_sites_independent(self, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: self.now)
    for _ in range(RATE_LIMIT_BURST + 1):
      self.log.warning("a")
      self.log.warning("b")
    assert self.messages.count("a") == RATE_LIMIT_BURST
    assert self.messages.count("b") == RATE_LIMIT_BURST
    assert len({r.lineno for r in self.handler.records}) == 2

  def test_same_line_is_one_call_site(self, monkeypatch):
    # the bucket is per line, not per message: 3 different messages from one line share it
    monkeypatch.setattr(time, "monotonic", lambda: self.now)
    for msg in ["a", "b", "c"] * RATE_LIMIT_BURST + ["d"]:
      if msg == "d":
        self.now += 1 / RATE_LIMIT_PER_S
      self.log.warning(msg)
    assert len(self.messages) == RATE_LIMIT_BURST + 1
    assert self.messages[-1] == f"d (suppressed {2 * RATE_LIMIT_BURST} repeats)"

  def test_exception_keeps_traceback(self, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: self.now)
    for i in range(RATE_LIMIT_BURST + 2):
      if i == RATE_LIMIT_BURST + 1:
        self.now += 1 / RATE_LIMIT_PER_S
      try:
        raise ValueError("boom")
      except ValueError:
        self.log.exception("oops")
    assert len(self.messages) == RATE_LIMIT_BURST + 1
    assert self.messages[-1] == "oops (suppressed 1 repeats)"
    for record in self.handler.records:
      assert record.exc_info[0] is ValueError
      assert "ValueError: boom" in logging.Formatter().format(record)

  def test_event_suppressed_count_is_structured(self, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: self.now)
    for i in range(RATE_LIMIT_BURST + 2):
      if i == RATE_LIMIT_BURST + 1:
        self.now += 1 / RATE_LIMIT_PER_S
      self.log.event("evt", x=1)
    assert len(self.handler.records) == RATE_LIMIT_BURST + 1
    assert self.handler.records[-1].msg == {"event": "evt", "x": 1, "suppressed": 1}
    assert "suppressed" not in self.handler.records[0].msg
