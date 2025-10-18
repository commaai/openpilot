import pytest
import os
import pathlib
import tempfile
import subprocess

from openpilot.system.updated.casync import casync
from openpilot.system.updated.casync import tar

# dd if=/dev/zero of=/tmp/img.raw bs=1M count=2
# sudo losetup -f /tmp/img.raw
# losetup -a | grep img.raw
LOOPBACK = os.environ.get('LOOPBACK', None)


@pytest.mark.skip("not used yet")
class TestCasync:
  @classmethod
  def setup_class(cls):
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

  def setup_method(self):
    # Clear target_lo
    if LOOPBACK is not None:
      self.target_lo = LOOPBACK
      with open(self.target_lo, 'wb') as f:
        f.write(b"0" * len(self.contents))

    self.target_fn = os.path.join(self.tmpdir.name, next(tempfile._get_candidate_names()))
    self.seed_fn = os.path.join(self.tmpdir.name, next(tempfile._get_candidate_names()))

  def teardown_method(self):
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
      assert target_f.read() == self.contents

    assert stats['remote'] == len(self.contents)

  def test_seed(self):
    target = casync.parse_caibx(self.manifest_fn)

    # Populate seed with half of the target contents
    with open(self.seed_fn, 'wb') as seed_f:
      seed_f.write(self.contents[:len(self.contents) // 2])

    sources = [('seed', casync.FileChunkReader(self.seed_fn), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]
    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as target_f:
      assert target_f.read() == self.contents

    assert stats['seed'] > 0
    assert stats['remote'] < len(self.contents)

  def test_already_done(self):
    """Test that an already flashed target doesn't download any chunks"""
    target = casync.parse_caibx(self.manifest_fn)

    with open(self.target_fn, 'wb') as f:
      f.write(self.contents)

    sources = [('target', casync.FileChunkReader(self.target_fn), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_fn)

    with open(self.target_fn, 'rb') as f:
      assert f.read() == self.contents

    assert stats['target'] == len(self.contents)

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
      assert f.read() == self.contents

    assert stats['remote'] < len(self.contents)

  @pytest.mark.skipif(not LOOPBACK, reason="requires loopback device")
  def test_lo_simple_extract(self):
    target = casync.parse_caibx(self.manifest_fn)
    sources = [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_lo)

    with open(self.target_lo, 'rb') as target_f:
      assert target_f.read(len(self.contents)) == self.contents

    assert stats['remote'] == len(self.contents)

  @pytest.mark.skipif(not LOOPBACK, reason="requires loopback device")
  def test_lo_chunk_reuse(self):
    """Test that chunks that are reused are only downloaded once"""
    target = casync.parse_caibx(self.manifest_fn)

    sources = [('target', casync.FileChunkReader(self.target_lo), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract(target, sources, self.target_lo)

    with open(self.target_lo, 'rb') as f:
      assert f.read(len(self.contents)) == self.contents

    assert stats['remote'] < len(self.contents)


@pytest.mark.skip("not used yet")
class TestCasyncDirectory:
  """Tests extracting a directory stored as a casync tar archive"""

  NUM_FILES = 16

  @classmethod
  def setup_cache(cls, directory, files=None):
    if files is None:
      files = range(cls.NUM_FILES)

    chunk_a = [i % 256 for i in range(1024)] * 512
    chunk_b = [(256 - i) % 256 for i in range(1024)] * 512
    zeroes = [0] * (1024 * 128)
    cls.contents = chunk_a + chunk_b + zeroes + chunk_a
    cls.contents = bytes(cls.contents)

    for i in files:
      with open(os.path.join(directory, f"file_{i}.txt"), "wb") as f:
        f.write(cls.contents)

      os.symlink(f"file_{i}.txt", os.path.join(directory, f"link_{i}.txt"))

  @classmethod
  def setup_class(cls):
    cls.tmpdir = tempfile.TemporaryDirectory()

    # Create casync files
    cls.manifest_fn = os.path.join(cls.tmpdir.name, 'orig.caibx')
    cls.store_fn = os.path.join(cls.tmpdir.name, 'store')

    cls.directory_to_extract = tempfile.TemporaryDirectory()
    cls.setup_cache(cls.directory_to_extract.name)

    cls.orig_fn = os.path.join(cls.tmpdir.name, 'orig.tar')
    tar.create_tar_archive(cls.orig_fn, pathlib.Path(cls.directory_to_extract.name))

    subprocess.check_output(["casync", "make", "--compression=xz", "--store", cls.store_fn, cls.manifest_fn, cls.orig_fn])

  @classmethod
  def teardown_class(cls):
    cls.tmpdir.cleanup()
    cls.directory_to_extract.cleanup()

  def setup_method(self):
    self.cache_dir = tempfile.TemporaryDirectory()
    self.working_dir = tempfile.TemporaryDirectory()
    self.out_dir = tempfile.TemporaryDirectory()

  def teardown_method(self):
    self.cache_dir.cleanup()
    self.working_dir.cleanup()
    self.out_dir.cleanup()

  def run_test(self):
    target = casync.parse_caibx(self.manifest_fn)

    cache_filename = os.path.join(self.working_dir.name, "cache.tar")
    tmp_filename = os.path.join(self.working_dir.name, "tmp.tar")

    sources = [('cache', casync.DirectoryTarChunkReader(self.cache_dir.name, cache_filename), casync.build_chunk_dict(target))]
    sources += [('remote', casync.RemoteChunkReader(self.store_fn), casync.build_chunk_dict(target))]

    stats = casync.extract_directory(target, sources, pathlib.Path(self.out_dir.name), tmp_filename)

    with open(os.path.join(self.out_dir.name, "file_0.txt"), "rb") as f:
      assert f.read() == self.contents

    with open(os.path.join(self.out_dir.name, "link_0.txt"), "rb") as f:
      assert f.read() == self.contents
      assert os.readlink(os.path.join(self.out_dir.name, "link_0.txt")) == "file_0.txt"

    return stats

  def test_no_cache(self):
    self.setup_cache(self.cache_dir.name, [])
    stats = self.run_test()
    assert stats['remote'] > 0
    assert stats['cache'] == 0

  def test_full_cache(self):
    self.setup_cache(self.cache_dir.name, range(self.NUM_FILES))
    stats = self.run_test()
    assert stats['remote'] == 0
    assert stats['cache'] > 0

  def test_one_file_cache(self):
    self.setup_cache(self.cache_dir.name, range(1))
    stats = self.run_test()
    assert stats['remote'] > 0
    assert stats['cache'] > 0
    assert stats['cache'] < stats['remote']

  def test_one_file_incorrect_cache(self):
    self.setup_cache(self.cache_dir.name, range(self.NUM_FILES))
    with open(os.path.join(self.cache_dir.name, "file_0.txt"), "wb") as f:
      f.write(b"1234")

    stats = self.run_test()
    assert stats['remote'] > 0
    assert stats['cache'] > 0
    assert stats['cache'] > stats['remote']

  def test_one_file_missing_cache(self):
    self.setup_cache(self.cache_dir.name, range(self.NUM_FILES))
    os.unlink(os.path.join(self.cache_dir.name, "file_12.txt"))

    stats = self.run_test()
    assert stats['remote'] > 0
    assert stats['cache'] > 0
    assert stats['cache'] > stats['remote']
