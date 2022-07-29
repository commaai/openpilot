#!/usr/bin/env python3
import os
import unittest
import tempfile
import subprocess

import system.hardware.tici.casync as casync


class TestCasync(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmpdir = tempfile.TemporaryDirectory()
    print(cls.tmpdir.name)

    # Build example contents
    contents = []
    for _ in range(128):
      contents += [i for i in range(256)]
    contents += [0] * 1024
    cls.contents = bytes(contents)

    # Write to file
    cls.orig_fn = os.path.join(cls.tmpdir.name, 'orig.bin')
    with open(cls.orig_fn, 'wb') as f:
      f.write(cls.contents)

    # Create casync files
    cls.manifest_fn = os.path.join(cls.tmpdir.name, 'orig.caibx')
    cls.store_fn = os.path.join(cls.tmpdir.name, 'store')
    subprocess.check_output(["casync", "make", "--compression=xz", "--store", cls.store_fn, cls.manifest_fn, cls.orig_fn])

  def test_simple_extract(self):
    target = casync.parse_caibx(self.manifest_fn)
    sources = [('store', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    with tempfile.NamedTemporaryFile() as target_fn:
      stats = casync.extract(target, sources, target_fn.name)
      self.assertEqual(target_fn.read(), self.contents)

    self.assertEqual(stats['store'], len(self.contents))


if __name__ == "__main__":
  unittest.main()
