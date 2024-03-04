from __future__ import absolute_import

from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class Count(AbstractSampledStat):
    """
    An AbstractSampledStat that maintains a simple count of what it has seen.
    """
    def __init__(self):
        super(Count, self).__init__(0.0)

    def update(self, sample, config, value, now):
        sample.value += 1.0

    def combine(self, samples, config, now):
        return float(sum(sample.value for sample in samples))
