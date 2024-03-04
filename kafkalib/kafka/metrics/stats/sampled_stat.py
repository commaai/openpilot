from __future__ import absolute_import

import abc

from kafka.metrics.measurable_stat import AbstractMeasurableStat


class AbstractSampledStat(AbstractMeasurableStat):
    """
    An AbstractSampledStat records a single scalar value measured over
    one or more samples. Each sample is recorded over a configurable
    window. The window can be defined by number of events or elapsed
    time (or both, if both are given the window is complete when
    *either* the event count or elapsed time criterion is met).

    All the samples are combined to produce the measurement. When a
    window is complete the oldest sample is cleared and recycled to
    begin recording the next sample.

    Subclasses of this class define different statistics measured
    using this basic pattern.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_value):
        self._initial_value = initial_value
        self._samples = []
        self._current = 0

    @abc.abstractmethod
    def update(self, sample, config, value, time_ms):
        raise NotImplementedError

    @abc.abstractmethod
    def combine(self, samples, config, now):
        raise NotImplementedError

    def record(self, config, value, time_ms):
        sample = self.current(time_ms)
        if sample.is_complete(time_ms, config):
            sample = self._advance(config, time_ms)
        self.update(sample, config, float(value), time_ms)
        sample.event_count += 1

    def new_sample(self, time_ms):
        return self.Sample(self._initial_value, time_ms)

    def measure(self, config, now):
        self.purge_obsolete_samples(config, now)
        return float(self.combine(self._samples, config, now))

    def current(self, time_ms):
        if not self._samples:
            self._samples.append(self.new_sample(time_ms))
        return self._samples[self._current]

    def oldest(self, now):
        if not self._samples:
            self._samples.append(self.new_sample(now))
        oldest = self._samples[0]
        for sample in self._samples[1:]:
            if sample.last_window_ms < oldest.last_window_ms:
                oldest = sample
        return oldest

    def purge_obsolete_samples(self, config, now):
        """
        Timeout any windows that have expired in the absence of any events
        """
        expire_age = config.samples * config.time_window_ms
        for sample in self._samples:
            if now - sample.last_window_ms >= expire_age:
                sample.reset(now)

    def _advance(self, config, time_ms):
        self._current = (self._current + 1) % config.samples
        if self._current >= len(self._samples):
            sample = self.new_sample(time_ms)
            self._samples.append(sample)
            return sample
        else:
            sample = self.current(time_ms)
            sample.reset(time_ms)
            return sample

    class Sample(object):

        def __init__(self, initial_value, now):
            self.initial_value = initial_value
            self.event_count = 0
            self.last_window_ms = now
            self.value = initial_value

        def reset(self, now):
            self.event_count = 0
            self.last_window_ms = now
            self.value = self.initial_value

        def is_complete(self, time_ms, config):
            return (time_ms - self.last_window_ms >= config.time_window_ms or
                    self.event_count >= config.event_window)
