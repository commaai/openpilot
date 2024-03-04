from __future__ import absolute_import

from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class Max(AbstractSampledStat):
    """An AbstractSampledStat that gives the max over its samples."""
    def __init__(self):
        super(Max, self).__init__(float('-inf'))

    def update(self, sample, config, value, now):
        sample.value = max(sample.value, value)

    def combine(self, samples, config, now):
        if not samples:
            return float('-inf')
        return float(max(sample.value for sample in samples))
