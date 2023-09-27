#!/usr/bin/env python3
import os
import bz2
import urllib.parse
import subprocess
import tqdm
import glob
from tempfile import TemporaryDirectory
import capnp

from tools.lib.logreader import FileReader, LogReader
from cereal import log as capnp_log


class RobustLogReader(LogReader):
  def __init__(self, fn, canonicalize=True, only_union_types=False, sort_by_time=False):  # pylint: disable=super-init-not-called
    data_version = None
    _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
    with FileReader(fn) as f:
      dat = f.read()

    if ext == "":
      pass
    elif ext == ".bz2":
      try:
        dat = bz2.decompress(dat)
      except ValueError:
        print("Failed to decompress, falling back to bzip2recover")
        with TemporaryDirectory() as directory:
          # Run bzip2recovery on log
          with open(os.path.join(directory, 'out.bz2'), 'wb') as f:
            f.write(dat)
          subprocess.check_call(["bzip2recover", "out.bz2"], cwd=directory)

          # Decompress and concatenate parts
          dat = b""
          for n in sorted(glob.glob(f"{directory}/rec*.bz2")):
            print(f"Decompressing {n}")
            with open(n, 'rb') as f:
              dat += bz2.decompress(f.read())
    else:
      raise Exception(f"unknown extension {ext}")

    progress = None
    while True:
      try:
        ents = capnp_log.Event.read_multiple_bytes(dat)
        self._ents = list(sorted(ents, key=lambda x: x.logMonoTime) if sort_by_time else ents)
        break
      except capnp.lib.capnp.KjException:
        if progress is None:
          progress = tqdm.tqdm(total=len(dat))

        # Cut off bytes at the end until capnp is able to read
        dat = dat[:-1]
        progress.update(1)

    self._ts = [x.logMonoTime for x in self._ents]
    self.data_version = data_version
    self._only_union_types = only_union_types
