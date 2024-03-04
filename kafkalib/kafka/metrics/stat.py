from __future__ import absolute_import

import abc


class AbstractStat(object):
    """
    An AbstractStat is a quantity such as average, max, etc that is computed
    off the stream of updates to a sensor
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def record(self, config, value, time_ms):
        """
        Record the given value

        Arguments:
            config (MetricConfig): The configuration to use for this metric
            value (float): The value to record
            timeMs (int): The POSIX time in milliseconds this value occurred
        """
        raise NotImplementedError
