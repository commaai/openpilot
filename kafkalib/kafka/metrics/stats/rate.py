from __future__ import absolute_import

from kafka.metrics.measurable_stat import AbstractMeasurableStat
from kafka.metrics.stats.sampled_stat import AbstractSampledStat


class TimeUnit(object):
    _names = {
        'nanosecond': 0,
        'microsecond': 1,
        'millisecond': 2,
        'second': 3,
        'minute': 4,
        'hour':  5,
        'day': 6,
    }

    NANOSECONDS = _names['nanosecond']
    MICROSECONDS = _names['microsecond']
    MILLISECONDS = _names['millisecond']
    SECONDS = _names['second']
    MINUTES = _names['minute']
    HOURS = _names['hour']
    DAYS = _names['day']

    @staticmethod
    def get_name(time_unit):
        return TimeUnit._names[time_unit]


class Rate(AbstractMeasurableStat):
    """
    The rate of the given quantity. By default this is the total observed
    over a set of samples from a sampled statistic divided by the elapsed
    time over the sample windows. Alternative AbstractSampledStat
    implementations can be provided, however, to record the rate of
    occurrences (e.g. the count of values measured over the time interval)
    or other such values.
    """
    def __init__(self, time_unit=TimeUnit.SECONDS, sampled_stat=None):
        self._stat = sampled_stat or SampledTotal()
        self._unit = time_unit

    def unit_name(self):
        return TimeUnit.get_name(self._unit)

    def record(self, config, value, time_ms):
        self._stat.record(config, value, time_ms)

    def measure(self, config, now):
        value = self._stat.measure(config, now)
        return float(value) / self.convert(self.window_size(config, now))

    def window_size(self, config, now):
        # purge old samples before we compute the window size
        self._stat.purge_obsolete_samples(config, now)

        """
        Here we check the total amount of time elapsed since the oldest
        non-obsolete window. This give the total window_size of the batch
        which is the time used for Rate computation. However, there is
        an issue if we do not have sufficient data for e.g. if only
        1 second has elapsed in a 30 second window, the measured rate
        will be very high. Hence we assume that the elapsed time is
        always N-1 complete windows plus whatever fraction of the final
        window is complete.

        Note that we could simply count the amount of time elapsed in
        the current window and add n-1 windows to get the total time,
        but this approach does not account for sleeps. AbstractSampledStat
        only creates samples whenever record is called, if no record is
        called for a period of time that time is not accounted for in
        window_size and produces incorrect results.
        """
        total_elapsed_time_ms = now - self._stat.oldest(now).last_window_ms
        # Check how many full windows of data we have currently retained
        num_full_windows = int(total_elapsed_time_ms / config.time_window_ms)
        min_full_windows = config.samples - 1

        # If the available windows are less than the minimum required,
        # add the difference to the totalElapsedTime
        if num_full_windows < min_full_windows:
            total_elapsed_time_ms += ((min_full_windows - num_full_windows) *
                                      config.time_window_ms)

        return total_elapsed_time_ms

    def convert(self, time_ms):
        if self._unit == TimeUnit.NANOSECONDS:
            return time_ms * 1000.0 * 1000.0
        elif self._unit == TimeUnit.MICROSECONDS:
            return time_ms * 1000.0
        elif self._unit == TimeUnit.MILLISECONDS:
            return time_ms
        elif self._unit == TimeUnit.SECONDS:
            return time_ms / 1000.0
        elif self._unit == TimeUnit.MINUTES:
            return time_ms / (60.0 * 1000.0)
        elif self._unit == TimeUnit.HOURS:
            return time_ms / (60.0 * 60.0 * 1000.0)
        elif self._unit == TimeUnit.DAYS:
            return time_ms / (24.0 * 60.0 * 60.0 * 1000.0)
        else:
            raise ValueError('Unknown unit: %s' % (self._unit,))


class SampledTotal(AbstractSampledStat):
    def __init__(self, initial_value=None):
        if initial_value is not None:
            raise ValueError('initial_value cannot be set on SampledTotal')
        super(SampledTotal, self).__init__(0.0)

    def update(self, sample, config, value, time_ms):
        sample.value += value

    def combine(self, samples, config, now):
        return float(sum(sample.value for sample in samples))
