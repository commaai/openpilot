#!/usr/bin/env python3
import os
import unittest
import tempfile
import subprocess

import openpilot.system.hardware.tici.casync as casync

# dd if=/dev/zero of=/tmp/img.raw bs=1M count=2
# sudo losetup -f /tmp/img.raw
# losetup -a | grep img.raw
LOOPBACK = os.environ.get('LOOPBACK', None)


class TestCasync(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmpdir = tempfile.TemporaryDirectory()

    # Build example contents
    chunk_a = [i % 256 for i in range(1024)] * 512
    chunk_b = [(256 - i) % 256 for i in range(1024)] * 512
    zeroes = [0] * (1024 * 128)
    contents = chunk_a + chunk_b + zeroes + chunk_a

    cls.contents = bytes(contents)

    # Write to file
    cls.orig_fn = os.path.join(cls.tmpdir.name, 'orig.bin')
    with open(cls.orig_fn, 'wb') as f:
      f.write(cls.contents)

    # Create casync files
    cls.manifest_fn = os.path.join(cls.tmpdir.name, 'orig.caibx')
    cls.store_fn = os.path.join(cls.tmpdir.name, 'store')
    subprocess.check_output(["casync", "make", "--compression=xz", "--store", cls.store_fn, cls.manifest_fn, cls.orig_fn])

    target = casync.parse_caibx(cls.manifest_fn)
    hashes = [c.sha.hex() for c in target]

    # Ensure we have chunk reuse
    assert len(hashes) > len(set(hashes))

  def setUp(self):
    # Clear target_lo
    if LOOPBACK is not None:
      self.target_lo = LOOPBACK
      with open(self.target_lo, 'wb') as f:
        f.write(b"0" * len(self.contents))

    self.target_fn = os.path.join(self.tmpdir.name, next(tempfile._get_candidate_names()))
    self.seed_fn = os.path.join(self.tmpdir.name, next(tempfile._get_candidate_names()))

  def tearDown(self):
    for fn in [self.target_fn, self.seed_fn]:
      try:
        os.unlink(fn)
      except FileNotFoundError:
        pass

  def test_simple_extract(self):
    target = casync.parse_caibx(self.manifest_fn)

    sources = [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]
    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as target_f:
      self.assertEqual(target_f.read(), self.contents)

    self.assertEqual(stats['remote'], len(self.contents))

  def test_seed(self):
    target = casync.parse_caibx(self.manifest_fn)

    # Populate seed with half of the target contents
    with open(self.seed_fn, 'wb') as seed_f:
      seed_f.write(self.contents[:len(self.contents) // 2])

    sources = [('seed', casync.FileChunkReader(self.seed_fn), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]
    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as target_f:
      self.assertEqual(target_f.read(), self.contents)

    self.assertGreater(stats['seed'], 0)
    self.assertLess(stats['remote'], len(self.contents))

  def test_already_done(self):
    """Test that an already flashed target doesn't download any chunks"""
    target = casync.parse_caibx(self.manifest_fn)

    with open(self.target_fn, 'wb') as f:
      f.write(self.contents)

    sources = [('target', casync.FileChunkReader(self.target_fn), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as f:
      self.assertEqual(f.read(), self.contents)

    self.assertEqual(stats['target'], len(self.contents))

  def test_chunk_reuse(self):
    """Test that chunks that are reused are only downloaded once"""
    target = casync.parse_caibx(self.manifest_fn)

    # Ensure target exists
    with open(self.target_fn, 'wb'):
      pass

    sources = [('target', casync.FileChunkReader(self.target_fn), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as f:
      self.assertEqual(f.read(), self.contents)

    self.assertLess(stats['remote'], len(self.contents))

  @unittest.skipUnless(LOOPBACK, "requires loopback device")
  def test_lo_simple_extract(self):
    target = casync.parse_caibx(self.manifest_fn)
    sources = [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_lo)

    with open(self.target_lo, 'rb') as target_f:
      self.assertEqual(target_f.read(len(self.contents)), self.contents)

    self.assertEqual(stats['remote'], len(self.contents))

  @unittest.skipUnless(LOOPBACK, "requires loopback device")
  def test_lo_chunk_reuse(self):
    """Test that chunks that are reused are only downloaded once"""
    target = casync.parse_caibx(self.manifest_fn)

    sources = [('target', casync.FileChunkReader(self.target_lo), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_lo)

    with open(self.target_lo, 'rb') as f:
      self.assertEqual(f.read(len(self.contents)), self.contents)

    self.assertLess(stats['remote'], len(self.contents))


if __name__ == "__main__":
  unittest.main()
