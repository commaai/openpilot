from __future__ import absolute_import

from kafka.metrics.measurable_stat import AbstractMeasurableStat


class Total(AbstractMeasurableStat):
    """An un-windowed cumulative total maintained over all time."""
    def __init__(self, value=0.0):
        self._total = value

    def record(self, config, value, now):
        self._total += value

    def measure(self, config, now):
        return float(self._total)
