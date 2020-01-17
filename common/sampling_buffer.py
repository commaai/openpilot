import os
import numpy as np
import random

class SamplingBuffer():
  def __init__(self, fn, size, write=False):
    self._fn = fn
    self._write = write
    if self._write:
      self._f = open(self._fn, "ab")
    else:
      self._f = open(self._fn, "rb")
    self._size = size
    self._refresh()

  def _refresh(self):
    self.cnt = os.path.getsize(self._fn) / self._size

  @property
  def count(self):
    self._refresh()
    return self.cnt

  def _fetch_one(self, x):
    assert self._write == False
    self._f.seek(x*self._size)
    return self._f.read(self._size)

  def sample(self, count, indices = None):
    if indices == None:
      cnt = self.count
      assert cnt != 0
      indices = map(lambda x: random.randint(0, cnt-1), range(count))
    return map(self._fetch_one, indices)

  def write(self, dat):
    assert self._write == True
    assert (len(dat) % self._size) == 0
    self._f.write(dat)
    self._f.flush()

class NumpySamplingBuffer():
  def __init__(self, fn, size, dtype, write=False):
    self._size = size
    self._dtype = dtype
    self._buf = SamplingBuffer(fn, len(np.zeros(size, dtype=dtype).tobytes()), write)

  @property
  def count(self):
    return self._buf.count

  def write(self, dat):
    self._buf.write(dat.tobytes())

  def sample(self, count, indices = None):
    return np.fromstring(''.join(self._buf.sample(count, indices)), dtype=self._dtype).reshape([count]+list(self._size))

# TODO: n IOPS needed where n is the Multi
class MultiNumpySamplingBuffer():
  def __init__(self, fn, npa, write=False):
    self._bufs = []
    for i,n in enumerate(npa):
      self._bufs.append(NumpySamplingBuffer(fn + ("_%d" % i), n[0], n[1], write))

  def write(self, dat):
    for b,x in zip(self._bufs, dat):
      b.write(x)

  @property
  def count(self):
    return min(map(lambda x: x.count, self._bufs))

  def sample(self, count):
    cnt = self.count
    assert cnt != 0
    indices = map(lambda x: random.randint(0, cnt-1), range(count))
    return map(lambda x: x.sample(count, indices), self._bufs)

