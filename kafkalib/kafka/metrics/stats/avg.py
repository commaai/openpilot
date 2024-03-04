from __future__ import absolute_import

from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class Avg(AbstractSampledStat):
    """
    An AbstractSampledStat that maintains a simple average over its samples.
    """
    def __init__(self):
        super(Avg, self).__init__(0.0)

    def update(self, sample, config, value, now):
        sample.value += value

    def combine(self, samples, config, now):
        total_sum = 0
        total_count = 0
        for sample in samples:
            total_sum += sample.value
            total_count += sample.event_count
        if not total_count:
            return 0
        return float(total_sum) / total_count
