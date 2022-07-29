#!/usr/bin/env python3
import os
import unittest
import tempfile
import subprocess

import system.hardware.tici.casync as casync

LOOPBACK = os.environ.get('LOOPBACK', None)


class TestCasync(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmpdir = tempfile.TemporaryDirectory()

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

  def setUp(self):
    if LOOPBACK is not None:
      self.target_lo = LOOPBACK
      with open(self.target_lo, 'wb') as f:
        f.write(b"0" * len(self.contents))

    self.target_fn = os.path.join(self.tmpdir.name, next(tempfile._get_candidate_names()))

  def tearDown(self):
    try:
      os.unlink(self.target_fn)
    except FileNotFoundError:
      pass

  def test_simple_extract(self):
    target = casync.parse_caibx(self.manifest_fn)
    sources = [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]
    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as target_f:
      self.assertEqual(target_f.read(), self.contents)

    self.assertEqual(stats['remote'], len(self.contents))

  @unittest.skipUnless(LOOPBACK, "requires loopback device")
  def test_lo_already_done(self):
    target = casync.parse_caibx(self.manifest_fn)
    sources = []

    with open(self.target_lo, 'wb') as f:
      f.write(self.contents)

    sources += [('target', casync.FileChunkReader(self.target_lo), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_lo)

    with open(self.target_lo, 'rb') as f:
      self.assertEqual(f.read(len(self.contents)), self.contents)

    self.assertEqual(stats['target'], len(self.contents))


if __name__ == "__main__":
  unittest.main()
