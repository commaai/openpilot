from __future__ import absolute_import

import math


class Histogram(object):
    def __init__(self, bin_scheme):
        self._hist = [0.0] * bin_scheme.bins
        self._count = 0.0
        self._bin_scheme = bin_scheme

    def record(self, value):
        self._hist[self._bin_scheme.to_bin(value)] += 1.0
        self._count += 1.0

    def value(self, quantile):
        if self._count == 0.0:
            return float('NaN')
        _sum = 0.0
        quant = float(quantile)
        for i, value in enumerate(self._hist[:-1]):
            _sum += value
            if _sum / self._count > quant:
                return self._bin_scheme.from_bin(i)
        return float('inf')

    @property
    def counts(self):
        return self._hist

    def clear(self):
        for i in range(self._hist):
            self._hist[i] = 0.0
        self._count = 0

    def __str__(self):
        values = ['%.10f:%.0f' % (self._bin_scheme.from_bin(i), value) for
                  i, value in enumerate(self._hist[:-1])]
        values.append('%s:%s' % (float('inf'), self._hist[-1]))
        return '{%s}' % ','.join(values)

    class ConstantBinScheme(object):
        def __init__(self, bins, min_val, max_val):
            if bins < 2:
                raise ValueError('Must have at least 2 bins.')
            self._min = float(min_val)
            self._max = float(max_val)
            self._bins = int(bins)
            self._bucket_width = (max_val - min_val) / (bins - 2)

        @property
        def bins(self):
            return self._bins

        def from_bin(self, b):
            if b == 0:
                return float('-inf')
            elif b == self._bins - 1:
                return float('inf')
            else:
                return self._min + (b - 1) * self._bucket_width

        def to_bin(self, x):
            if x < self._min:
                return 0
            elif x > self._max:
                return self._bins - 1
            else:
                return int(((x - self._min) / self._bucket_width) + 1)

    class LinearBinScheme(object):
        def __init__(self, num_bins, max_val):
            self._bins = num_bins
            self._max = max_val
            self._scale = max_val / (num_bins * (num_bins - 1) / 2)

        @property
        def bins(self):
            return self._bins

        def from_bin(self, b):
            if b == self._bins - 1:
                return float('inf')
            else:
                unscaled = (b * (b + 1.0)) / 2.0
                return unscaled * self._scale

        def to_bin(self, x):
            if x < 0.0:
                raise ValueError('Values less than 0.0 not accepted.')
            elif x > self._max:
                return self._bins - 1
            else:
                scaled = x / self._scale
                return int(-0.5 + math.sqrt(2.0 * scaled + 0.25))
