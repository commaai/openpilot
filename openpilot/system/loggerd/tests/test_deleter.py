from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  # Deletion behavior is independent of file size; use smaller files to keep these tests fast.
  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .001, lock: bool = False,
                          upload_xattr: bytes | None = None, preserve_xattr: bytes | None = None) -> Path:
    return super().make_file_with_data(f_dir, fn, size_mb, lock, upload_xattr, preserve_xattr)

  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().openpilot_setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs  # ty: ignore[invalid-assignment]  # test double

  def test_delete(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)
    assert deleter.deleter_step() == (True, str(f_path.parent))
    assert not f_path.exists()

  def assertDeleteOrder(self, f_paths: Sequence[Path]) -> None:
    deleted_order = []
    for _ in f_paths:
      out_of_space, deleted_path = deleter.deleter_step()
      assert out_of_space and deleted_path is not None
      deleted_order.append(next(f for f in f_paths if f.parent == Path(deleted_path)))

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

    assert deleter.deleter_step() == (False, None)
    assert f_path.exists(), "File deleted with available space"

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    assert deleter.deleter_step() == (True, None)
    assert f_path.exists(), "File deleted when locked"
    assert self.params.get(deleter.CLEANUP_ALERT) is None

  def test_cleanup_alert_on_failed_deletion(self, monkeypatch, mocker):
    failed = self.make_file_with_data(self.seg_format.format(0), self.f_type)
    deletable = self.make_file_with_data(self.seg_format.format(1), self.f_type)
    original = deleter.shutil.rmtree
    alerts = mocker.patch.object(deleter, "set_offroad_alert")
    monkeypatch.setattr(deleter, "_cleanup_alert_shown", False)

    def rmtree(path):
      if path == str(failed.parent):
        raise PermissionError("cannot delete directory")
      return original(path)

    monkeypatch.setattr(deleter.shutil, "rmtree", rmtree)
    assert deleter.deleter_step() == (True, str(deletable.parent))
    assert failed.exists()
    assert not deletable.exists()
    alerts.assert_called_once_with(deleter.CLEANUP_ALERT, True)

  def test_preservation_inspected_only_once(self, monkeypatch, mocker):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE)
    original = deleter.getxattr
    getxattr = mocker.Mock(side_effect=[original(str(f_path.parent), deleter.PRESERVE_ATTR_NAME), PermissionError("second read failed")])
    monkeypatch.setattr(deleter, "getxattr", getxattr)
    assert deleter.deleter_step() == (True, str(f_path.parent))
    assert getxattr.call_count == 1

  def test_unreadable_preservation_is_skipped(self, monkeypatch, mocker):
    unreadable = self.make_file_with_data(self.seg_format.format(0), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE)
    deletable = self.make_file_with_data(self.seg_format.format(1), self.f_type)
    original = deleter.getxattr
    monkeypatch.setattr(deleter, "_cleanup_alert_shown", False)

    def getxattr(path, attr):
      if path == str(unreadable.parent):
        raise PermissionError("cannot inspect xattr")
      return original(path, attr)

    monkeypatch.setattr(deleter, "getxattr", getxattr)
    assert deleter.deleter_step() == (True, str(deletable.parent))
    assert unreadable.exists()
    assert not deletable.exists()
    assert self.params.get(deleter.CLEANUP_ALERT)["severity"] == 1

    # An unknown preservation status must not become a fallback deletion candidate.
    assert deleter.deleter_step() == (True, None)
    assert unreadable.exists()
    monkeypatch.setattr(deleter, "getxattr", original)
    assert deleter.deleter_step() == (True, str(unreadable.parent))
    assert self.params.get(deleter.CLEANUP_ALERT) is not None  # Low storage persists.

    self.fake_stats = Stats(f_bavail=10 * 1024 * 1024 * 1024 / 4096, f_blocks=10 * 1024 * 1024, f_frsize=4096)
    assert deleter.deleter_step() == (False, None)
    assert self.params.get(deleter.CLEANUP_ALERT) is None

  def test_unreadable_directory_is_skipped(self, monkeypatch, mocker):
    unreadable = self.make_file_with_data(self.seg_format.format(0), self.f_type)
    deletable = self.make_file_with_data(self.seg_format.format(1), self.f_type)
    original = deleter.os.listdir
    log_exception = mocker.patch.object(deleter.cloudlog, "exception")

    def listdir(path):
      if path == str(unreadable.parent):
        raise PermissionError("cannot inspect directory")
      return original(path)

    monkeypatch.setattr(deleter.os, "listdir", listdir)
    assert deleter.deleter_step() == (True, str(deletable.parent))
    assert unreadable.exists()
    assert not deletable.exists()
    log_exception.assert_called_once_with(f"issue deleting {unreadable.parent}")

    assert deleter.deleter_step() == (True, None)
    assert unreadable.exists()
