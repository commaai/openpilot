from pathlib import Path

import numpy as np

from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.tools.lib.logreader import LogReader


FAKEDATA_ROOT = Path(__file__).resolve().parents[3] / "selfdrive/test/process_replay/fakedata"


class TestLogTimeSeries:
  def test_include_types(self):
    log_path = next(FAKEDATA_ROOT.glob("*controlsd*.zst"))
    msgs = list(LogReader(str(log_path)))

    full = msgs_to_time_series(msgs)
    filtered = msgs_to_time_series(msgs, include_types={"controlsState"})

    assert set(filtered) == {"controlsState"}
    np.testing.assert_equal(full["controlsState"]["t"], filtered["controlsState"]["t"])
    np.testing.assert_equal(full["controlsState"]["curvature"], filtered["controlsState"]["curvature"])
    np.testing.assert_equal(full["controlsState"]["_valid"], filtered["controlsState"]["_valid"])
