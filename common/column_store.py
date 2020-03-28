import os
import numpy as np
import collections
from contextlib import closing

from common.file_helpers import mkdirs_exists_ok

class ColumnStoreReader():
  def __init__(self, path, mmap=False, allow_pickle=False, direct_io=False):
    if not (path and os.path.isdir(path)):
      raise ValueError("Not a column store: {}".format(path))

    self._path = os.path.realpath(path)
    self._keys = os.listdir(self._path)
    self._mmap = mmap
    self._allow_pickle = allow_pickle
    self._direct_io = direct_io

  @property
  def path(self):
    return self._path

  def close(self):
    pass

  def get(self, key):
    try:
      return self[key]
    except KeyError:
      return None

  def keys(self):
    return list(self._keys)

  def iteritems(self):
    for k in self:
      yield (k, self[k])

  def itervalues(self):
    for k in self:
      yield self[k]

  def get_npy_path(self, key):
    """Gets a filesystem path for an npy file containing the specified array,
       or none if the column store does not contain key.
    """
    if key in self:
      return os.path.join(self._path, key)
    else:
      return None

  def __getitem__(self, key):
    try:
      path = os.path.join(self._path, key)

      # TODO(mgraczyk): This implementation will need to change for zip.
      if os.path.isdir(path):
        return ColumnStoreReader(path)
      else:
        if self._mmap:
          # note that direct i/o does nothing for mmap since file read/write interface is not used
          ret = np.load(path, mmap_mode='r', allow_pickle=self._allow_pickle, fix_imports=False)
        else:
          if self._direct_io:
            opener = lambda path, flags: os.open(path, os.O_RDONLY | os.O_DIRECT)
            with open(path, 'rb', buffering=0, opener=opener) as f:
              ret = np.load(f, allow_pickle=self._allow_pickle, fix_imports=False)
          else:
            ret = np.load(path, allow_pickle=self._allow_pickle, fix_imports=False)
        if type(ret) == np.lib.npyio.NpzFile:
          # if it's saved as compressed, it has arr_0 only in the file. deref this
          return ret['arr_0']
        else:
          return ret
    except IOError:
      raise KeyError(key)

  def __contains__(self, item):
    try:
      self[item]
      return True
    except KeyError:
      return False

  def __len__(self):
    return len(self._keys)

  def __bool__(self):
    return bool(self._keys)

  def __iter__(self):
    return iter(self._keys)

  def __str__(self):
    return "ColumnStoreReader({})".format(str({k: "..." for k in self._keys}))

  def __enter__(self): return self
  def __exit__(self, type, value, traceback): self.close()

class ColumnStoreWriter():
  def __init__(self, path, allow_pickle=False):
    self._path = path
    self._allow_pickle = allow_pickle
    mkdirs_exists_ok(self._path)

  def map_column(self, path, dtype, shape):
    npy_path = os.path.join(self._path, path)
    mkdirs_exists_ok(os.path.dirname(npy_path))
    return np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=shape)

  def add_column(self, path, data, dtype=None, compression=False, overwrite=False):
    npy_path = os.path.join(self._path, path)
    mkdirs_exists_ok(os.path.dirname(npy_path))

    if overwrite:
      f = open(npy_path, "wb")
    else:
      f = os.fdopen(os.open(npy_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb")

    with closing(f) as f:
      data2 = np.array(data, copy=False, dtype=dtype)
      if compression:
        np.savez_compressed(f, data2)
      else:
        np.save(f, data2, allow_pickle=self._allow_pickle, fix_imports=False)

  def add_group(self, group_name):
    # TODO(mgraczyk): This implementation will need to change if we add zip or compression.
    return ColumnStoreWriter(os.path.join(self._path, group_name))

  def close(self):
    pass

  def __enter__(self): return self
  def __exit__(self, type, value, traceback): self.close()


def _save_dict_as_column_store(values, writer, compression):
  for k, v in values.items():
    if isinstance(v, collections.Mapping):
      _save_dict_as_column_store(v, writer.add_group(k), compression)
    else:
      writer.add_column(k, v, compression=compression)


def save_dict_as_column_store(values, output_path, compression=False):
  with ColumnStoreWriter(output_path) as writer:
    _save_dict_as_column_store(values, writer, compression)


