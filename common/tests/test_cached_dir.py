import os
import tempfile
import time

from openpilot.common.cached_dir import CachedDir, dir_cache


class TestCachedDir:
  def setup_method(self):
    self.tmp_dir = tempfile.mkdtemp()

  def teardown_method(self):
    for f in os.listdir(self.tmp_dir):
      os.remove(os.path.join(self.tmp_dir, f))
    os.rmdir(self.tmp_dir)

  def test_cache_populated_on_first_access(self):
    with open(os.path.join(self.tmp_dir, "file1.txt"), "w") as f:
      f.write("Test file")

    listing = CachedDir.listdir(self.tmp_dir)

    assert "file1.txt" in listing
    assert self.tmp_dir in dir_cache
    assert dir_cache[self.tmp_dir].mtime == os.path.getmtime(self.tmp_dir)

  def test_cache_refresh_on_change(self):
    listing = CachedDir.listdir(self.tmp_dir)  # Initial access

    time.sleep(1)
    # Add a new file and wait for the mtime to update
    with open(os.path.join(self.tmp_dir, "file2.txt"), "w") as f:
      f.write("Another test file")
    time.sleep(1)

    # Check if the new file appears
    updated_listing = CachedDir.listdir(self.tmp_dir)
    assert(listing != updated_listing)
    assert "file2.txt" in updated_listing
    assert dir_cache[self.tmp_dir].mtime == os.path.getmtime(self.tmp_dir)

  def test_cache_updates_on_file_removal(self):
    with open(os.path.join(self.tmp_dir, "file1.txt"), "w") as f:
      f.write("File 1")
    with open(os.path.join(self.tmp_dir, "file2.txt"), "w") as f:
      f.write("File 2")

    CachedDir.listdir(self.tmp_dir)  # Initial access
    time.sleep(1)
    # Remove file and update cache
    os.remove(os.path.join(self.tmp_dir, "file1.txt"))
    time.sleep(1)

    updated_listing = CachedDir.listdir(self.tmp_dir)
    assert "file1.txt" not in updated_listing
    assert "file2.txt" in updated_listing
