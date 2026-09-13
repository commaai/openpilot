import asyncio
from collections.abc import Callable
import os
from typing import Any

from openpilot.common.params import Params


class LivestreamBitrateController:
  bitrates = [500_000, 1_500_000, int(os.environ.get("STREAM_BITRATE", 5_000_000))]
  label_to_bitrate = { "high": bitrates[2], "med": bitrates[1], "low": bitrates[0]}
  sample_interval = 0.2
  high_level = 0.1 # drop immediately
  med_level = 0.05 # drop after # of samples
  low_level = 0 # raise after # of samples
  down_samples = 5
  param_name = "LivestreamEncoderBitrate"

  def __init__(self, get_stats: Callable[[], dict[str, Any]], params: Params, enabled: bool = True):
    self.get_stats = get_stats
    self.params = params

    self.level = 2
    self._publish(self.bitrates[self.level])
    self.prev_stats: tuple[Any, ...] | None = None
    self.counter = 0
    self.up_samples = 5 # 1s
    self._auto = True
    self._enabled = enabled

  def enable(self, enable: bool):
    self._enabled = enable

  async def run(self):
    while True:
      await asyncio.sleep(self.sample_interval)
      if not self._enabled:
        continue
      if not self._auto:
        continue

      loss_rate = self._sample()
      if loss_rate is None:
        continue
      if loss_rate >= self.med_level and self.level > 0:
        self.counter += 1
        if self.counter >= self.down_samples or loss_rate >= self.high_level:
          self.level -= 1
          self.up_samples *= 2 # exponential backoff before raising again
          self.counter = 0
          self._publish(self.bitrates[self.level])
      elif loss_rate <= self.low_level and self.level < len(self.bitrates) - 1:
        self.counter -= 1
        if -self.counter >= self.up_samples:
          self.level += 1
          self.counter = 0
          self._publish(self.bitrates[self.level])

  def _sample(self) -> float | None:
    report = next(iter(self.get_stats().values()), None)
    if report is None:
      return None

    current = (report.ssrc, report.fraction_lost, report.packets_lost, report.highest_seq_no, report.jitter, report.lsr, report.dlsr)
    if self.prev_stats == current:
      return None
    self.prev_stats = current

    loss_rate = report.fraction_lost / 256
    return loss_rate

  def _publish(self, bitrate: float):
    self.params.put(self.param_name, bitrate)

  def set_quality(self, quality):
    if quality in self.label_to_bitrate:
      self._publish(self.label_to_bitrate[quality])
      self._auto = False
    elif quality == "auto":
      self._auto = True
