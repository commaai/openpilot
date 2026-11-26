import tempfile
import unittest
from pathlib import Path

from openpilot.common.hardware.comma.hardware import HardwareComma


class TestGpuUsageFromClockStats(unittest.TestCase):
  def setUp(self):
    self.hw = HardwareComma.__new__(HardwareComma)   # the method needs no device state
    self.dir = tempfile.TemporaryDirectory()
    self.f = Path(self.dir.name) / "gpu_clock_stats"

  def tearDown(self):
    self.dir.cleanup()

  def read(self, text, t_us):
    if text is not None:
      self.f.write_text(text)
    return self.hw.get_gpu_usage_percent(path=str(self.f), now_us=t_us)

  def test_first_sample_is_zero_then_deltas(self):
    self.assertEqual(self.read("0 1000 0", 0), 0)
    self.assertEqual(self.read("0 501000 0", 1_000_000), 50.0)
    self.assertEqual(self.read("100000 501000 400000", 2_000_000), 50.0)   # summed across power levels
    self.assertEqual(self.read("100000 501000 400000", 3_000_000), 0.0)    # an idle second is a real zero

  def test_clamp_to_100(self):
    self.read("0 0 0", 0)
    self.assertEqual(self.read("0 9000000 0", 1_000_000), 100.0)

  def test_counter_reset_or_wrap_restarts_the_window(self):
    self.read("0 5000000 0", 0)
    self.assertEqual(self.read("0 10 0", 1_000_000), 0)
    self.assertEqual(self.read("0 250010 0", 2_000_000), 25.0)

  def test_missing_or_garbage_file_is_zero_and_restarts(self):
    self.read("0 0 0", 0)
    self.assertEqual(self.read("garbage", 1_000_000), 0)
    self.f.unlink()
    self.assertEqual(self.read(None, 2_000_000), 0)
    self.assertEqual(self.read("0 100 0", 3_000_000), 0)                   # first sample again
    self.assertEqual(self.read("0 500100 0", 4_000_000), 50.0)

  def test_rapid_reads_repeat_the_last_value(self):
    self.read("0 0 0", 0)
    self.assertEqual(self.read("0 250000 0", 500_000), 50.0)
    self.assertEqual(self.read("0 400000 0", 600_000), 50.0)
    self.assertEqual(self.read(None, 1_000_000), 30.0)

  def test_deviceState_value_rounds_like_hardwared(self):
    self.read("0 0 0", 0)
    self.assertEqual(int(round(self.read("0 633000 0", 1_000_000))), 63)

  def test_governor_and_clock_lines_untouched(self):
    src = Path(__file__).parent.parent.joinpath("hardware.py").read_text()
    for line in ('sudo_write("performance", "/sys/class/kgsl/kgsl-3d0/devfreq/governor")',
                 'sudo_write("710", "/sys/class/kgsl/kgsl-3d0/max_clock_mhz")',
                 'sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_clk_on")'):
      self.assertTrue(line in src, line)


if __name__ == "__main__":
  unittest.main()
