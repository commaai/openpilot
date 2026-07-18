import os

from openpilot.system.loggerd import xattr_cache


ATTR_NAME = "user.openpilot_test"


def test_get_and_set_xattr(tmp_path):
  path = tmp_path / "file"
  path.touch()

  assert xattr_cache.getxattr(str(path), ATTR_NAME) is None

  xattr_cache.setxattr(str(path), ATTR_NAME, b"value")
  assert xattr_cache.getxattr(str(path), ATTR_NAME) == b"value"
  assert os.getxattr(path, ATTR_NAME) == b"value"


def test_empty_and_binary_xattr_values(tmp_path):
  path = tmp_path / "file"
  path.touch()

  for value in (b"", b"a\0b"):
    xattr_cache.setxattr(str(path), ATTR_NAME, value)
    assert xattr_cache.getxattr(str(path), ATTR_NAME) == value


def test_getxattr_caches_value(tmp_path):
  path = tmp_path / "file"
  path.touch()
  os.setxattr(path, ATTR_NAME, b"first")

  assert xattr_cache.getxattr(str(path), ATTR_NAME) == b"first"
  os.setxattr(path, ATTR_NAME, b"second")
  assert xattr_cache.getxattr(str(path), ATTR_NAME) == b"first"


def test_getxattr_propagates_unexpected_errors(tmp_path):
  path = tmp_path / "missing"

  try:
    xattr_cache.getxattr(str(path), ATTR_NAME)
  except FileNotFoundError:
    pass
  else:
    raise AssertionError("getxattr did not propagate FileNotFoundError")
