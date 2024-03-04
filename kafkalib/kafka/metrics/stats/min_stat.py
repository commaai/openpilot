from __future__ import absolute_import

import sys

from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class Min(AbstractSampledStat):
    """An AbstractSampledStat that gives the min over its samples."""
    def __init__(self):
        super(Min, self).__init__(float(sys.maxsize))

    def update(self, sample, config, value, now):
        sample.value = min(sample.value, value)

    def combine(self, samples, config, now):
        if not samples:
            return float(sys.maxsize)
        return float(min(sample.value for sample in samples))
