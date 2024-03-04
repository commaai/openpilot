from __future__ import absolute_import

from kafka.metrics import AnonMeasurable, NamedMeasurable
from kafka.metrics.compound_stat import AbstractCompoundStat
from kafka.metrics.stats import Histogram
from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class BucketSizing(object):
    CONSTANT = 0
    LINEAR = 1


class Percentiles(AbstractSampledStat, AbstractCompoundStat):
    """A compound stat that reports one or more percentiles"""
    def __init__(self, size_in_bytes, bucketing, max_val, min_val=0.0,
                 percentiles=None):
        super(Percentiles, self).__init__(0.0)
        self._percentiles = percentiles or []
        self._buckets = int(size_in_bytes / 4)
        if bucketing == BucketSizing.CONSTANT:
            self._bin_scheme = Histogram.ConstantBinScheme(self._buckets,
                                                           min_val, max_val)
        elif bucketing == BucketSizing.LINEAR:
            if min_val != 0.0:
                raise ValueError('Linear bucket sizing requires min_val'
                                 ' to be 0.0.')
            self.bin_scheme = Histogram.LinearBinScheme(self._buckets, max_val)
        else:
            ValueError('Unknown bucket type: %s' % (bucketing,))

    def stats(self):
        measurables = []

        def make_measure_fn(pct):
            return lambda config, now: self.value(config, now,
                                                  pct / 100.0)

        for percentile in self._percentiles:
            measure_fn = make_measure_fn(percentile.percentile)
            stat = NamedMeasurable(percentile.name, AnonMeasurable(measure_fn))
            measurables.append(stat)
        return measurables

    def value(self, config, now, quantile):
        self.purge_obsolete_samples(config, now)
        count = sum(sample.event_count for sample in self._samples)
        if count == 0.0:
            return float('NaN')
        sum_val = 0.0
        quant = float(quantile)
        for b in range(self._buckets):
            for sample in self._samples:
                assert type(sample) is self.HistogramSample
                hist = sample.histogram.counts
                sum_val += hist[b]
                if sum_val / count > quant:
                    return self._bin_scheme.from_bin(b)
        return float('inf')

    def combine(self, samples, config, now):
        return self.value(config, now, 0.5)

    def new_sample(self, time_ms):
        return Percentiles.HistogramSample(self._bin_scheme, time_ms)

    def update(self, sample, config, value, time_ms):
        assert type(sample) is self.HistogramSample
        sample.histogram.record(value)

    class HistogramSample(AbstractSampledStat.Sample):
        def __init__(self, scheme, now):
            super(Percentiles.HistogramSample, self).__init__(0.0, now)
            self.histogram = Histogram(scheme)
