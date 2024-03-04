from __future__ import absolute_import

import abc

from kafka.metrics.measurable import AbstractMeasurable
from kafka.metrics.stat import AbstractStat


class AbstractMeasurableStat(AbstractStat, AbstractMeasurable):
    """
    An AbstractMeasurableStat is an AbstractStat that is also
    an AbstractMeasurable (i.e. can produce a single floating point value).
    This is the interface used for most of the simple statistics such
    as Avg, Max, Count, etc.
    """
    __metaclass__ = abc.ABCMeta
