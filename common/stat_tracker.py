import numpy as np

_DESC_FMT = """
{} (n={}):
MEAN={}
VAR={}
MIN={}
MAX={}
"""

class StatTracker():
  def __init__(self, name):
    self._name = name
    self._mean = 0.
    self._var = 0.
    self._n = 0
    self._min = -float("-inf")
    self._max = -float("inf")

  @property
  def mean(self):
    return self._mean

  @property
  def var(self):
    return (self._n * self._var) / (self._n - 1.)

  @property
  def min(self):
    return self._min

  @property
  def max(self):
    return self._max

  def update(self, samples):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    data = samples.reshape(-1)
    n_a = data.size
    mean_a = np.mean(data)
    var_a = np.var(data, ddof=0)

    n_b = self._n
    mean_b = self._mean

    delta = mean_b - mean_a
    m_a = var_a * (n_a - 1)
    m_b = self._var * (n_b - 1)
    m2 = m_a + m_b + delta**2 * n_a * n_b / (n_a + n_b)

    self._var = m2 / (n_a + n_b)
    self._mean = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    self._n = n_a + n_b

    self._min = min(self._min, np.min(data))
    self._max = max(self._max, np.max(data))

  def __str__(self):
    return _DESC_FMT.format(self._name, self._n, self._mean, self.var, self._min,
                            self._max)

# FIXME(mgraczyk): The variance computation does not work with 1 sample batches.
class VectorStatTracker(StatTracker):
  def __init__(self, name, dim):
    self._name = name
    self._mean = np.zeros((dim, ))
    self._var = np.zeros((dim, dim))
    self._n = 0
    self._min = np.full((dim, ), -float("-inf"))
    self._max = np.full((dim, ), -float("inf"))

  @property
  def cov(self):
    return self.var

  def update(self, samples):
    n_a = samples.shape[0]
    mean_a = np.mean(samples, axis=0)
    var_a = np.cov(samples, ddof=0, rowvar=False)

    n_b = self._n
    mean_b = self._mean

    delta = mean_b - mean_a
    m_a = var_a * (n_a - 1)
    m_b = self._var * (n_b - 1)
    m2 = m_a + m_b + delta**2 * n_a * n_b / (n_a + n_b)

    self._var = m2 / (n_a + n_b)
    self._mean = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    self._n = n_a + n_b

    self._min = np.minimum(self._min, np.min(samples, axis=0))
    self._max = np.maximum(self._max, np.max(samples, axis=0))
