import os
import random
import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs

  def start_thread(self):
    self.end_event = threading.Event()
    self.del_thread = threading.Thread(target=deleter.deleter_thread, args=[self.end_event])
    self.del_thread.daemon = True
    self.del_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.del_thread.join()

  def test_delete(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, 1)

    self.start_thread()

    try:
      with Timeout(2, "Timeout waiting for file to be deleted"):
        while f_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def assertDeleteOrder(self, f_paths: Sequence[Path], timeout: int = 5) -> None:
    deleted_order = []

    self.start_thread()
    try:
      with Timeout(timeout, "Timeout waiting for files to be deleted"):
        while True:
          for f in f_paths:
            if not f.exists() and f not in deleted_order:
              deleted_order.append(f)
          if len(deleted_order) == len(f_paths):
            break
          time.sleep(0.01)
    except TimeoutException:
      print("Not deleted:", [f for f in f_paths if f not in deleted_order])
      raise
    finally:
      self.join_thread()

    assert deleted_order == f_paths, "Files not deleted in expected order"

  def test_delete_order(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
    ])

  def test_delete_many_preserved(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data(self.seg_format.format(2), self.f_type),
    ] + [
      self.make_file_with_data(self.seg_format2.format(i), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE)
      for i in range(5)
    ])

  def test_delete_last(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(0), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data("boot", self.seg_format[:-4]),
      self.make_file_with_data("crash", self.seg_format2[:-4]),
    ])

  def test_no_delete_when_available_space(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)

    block_size = 4096
    available = (10 * 1024 * 1024 * 1024) / block_size  # 10GB free
    self.fake_stats = Stats(f_bavail=available, f_blocks=10, f_frsize=block_size)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted with available space"

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted when locked"

  def wait_for_empty_root(self, root: Path, timeout: int = 5) -> None:
    self.start_thread()
    try:
      with Timeout(timeout, "Timeout waiting for log root cleanup"):
        while list(root.iterdir()):
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_stray_files(self):
    root = Path(Paths.log_root())

    # stray files at log root level
    (root / "data.tar.gz").write_bytes(os.urandom(64))
    (root / "tmp_upload.part").write_bytes(os.urandom(128))
    (root / "logfile.txt").write_text("stale log data")

    # also create a normal segment so the test has something for the dir path too
    self.make_file_with_data(self.seg_format.format(0), self.f_type)

    self.wait_for_empty_root(root)

  def test_delete_broken_symlinks(self):
    root = Path(Paths.log_root())

    root.mkdir(parents=True, exist_ok=True)
    (root / "broken_link").symlink_to("/nonexistent/path")
    (root / "another_broken").symlink_to("/tmp/does_not_exist_12345")

    self.make_file_with_data(self.seg_format.format(0), self.f_type)

    self.wait_for_empty_root(root)

  def test_delete_hidden_and_empty_files(self):
    root = Path(Paths.log_root())

    root.mkdir(parents=True, exist_ok=True)
    (root / ".hidden_config").write_bytes(b"secret")
    (root / ".DS_Store").write_bytes(b"\x00\x00")
    (root / "empty_file").touch()

    self.make_file_with_data(self.seg_format.format(0), self.f_type)

    self.wait_for_empty_root(root)

  def test_delete_stray_files_with_segments(self):
    root = Path(Paths.log_root())

    # mix of stray files and normal segments
    (root / "stray.dat").write_bytes(os.urandom(64))

    self.make_file_with_data(self.seg_format.format(0), self.f_type)
    self.make_file_with_data(self.seg_format.format(1), self.f_type)

    (root / "another_stray.log").write_text("data")

    self.wait_for_empty_root(root)

  def test_delete_random_file_structures(self):
    """Stress test with lots of random files and directories with different structures."""
    root = Path(Paths.log_root())
    root.mkdir(parents=True, exist_ok=True)

    # create many random stray files at root level with varied content
    extensions = ['.dat', '.log', '.tmp', '.tar.gz', '.hevc', '.txt', '.part', '']
    for i in range(random.randint(15, 30)):
      ext = random.choice(extensions)
      size = random.randint(0, 256)
      f = root / f"stray_{i}{ext}"
      f.write_bytes(os.urandom(size))

    # create random non-segment directories with nested files
    for i in range(random.randint(3, 8)):
      d = root / f"random_dir_{i}"
      d.mkdir(exist_ok=True)
      for j in range(random.randint(0, 6)):
        (d / f"file_{j}.dat").write_bytes(os.urandom(random.randint(0, 128)))
        # random nested subdirectories
        if random.random() < 0.5:
          sub = d / f"sub_{j}"
          sub.mkdir(exist_ok=True)
          (sub / "nested.dat").write_bytes(os.urandom(random.randint(0, 64)))
          if random.random() < 0.3:
            deep = sub / "deeper"
            deep.mkdir(exist_ok=True)
            (deep / "deep_file").write_bytes(os.urandom(16))

    # broken symlinks
    for i in range(random.randint(1, 4)):
      try:
        (root / f"broken_sym_{i}").symlink_to(f"/nonexistent_{random.randint(0, 99999)}")
      except OSError:
        pass

    # empty files
    for i in range(random.randint(1, 5)):
      (root / f"empty_{i}").touch()

    # hidden dotfiles
    for i in range(random.randint(1, 3)):
      (root / f".hidden_{i}").write_bytes(os.urandom(random.randint(0, 32)))

    # also create some normal segments so the deleter has dirs to process too
    for i in range(random.randint(1, 4)):
      self.make_file_with_data(self.seg_format.format(i), self.f_type)

    # everything should be deleted without crashing
    self.wait_for_empty_root(root, timeout=10)
