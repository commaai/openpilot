import os
import tempfile
import shutil
import unittest

from common.xattr import getxattr, setxattr, listxattr, removexattr

class TestParams(unittest.TestCase):
  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()
    self.tmpfn = os.path.join(self.tmpdir, 'test.txt')
    open(self.tmpfn, 'w').close()
    #print("using", self.tmpfn)

  def tearDown(self):
    shutil.rmtree(self.tmpdir)

  def test_getxattr_none(self):
    a = getxattr(self.tmpfn, 'user.test')
    assert a is None

  def test_listxattr_none(self):
    l = listxattr(self.tmpfn)
    assert l == []

  def test_setxattr(self):
    setxattr(self.tmpfn, 'user.test', b'123')
    a = getxattr(self.tmpfn, 'user.test')
    assert a == b'123'

  def test_listxattr(self):
    setxattr(self.tmpfn, 'user.test1', b'123')
    setxattr(self.tmpfn, 'user.test2', b'123')
    l = listxattr(self.tmpfn)
    assert l == ['user.test1', 'user.test2']

  def test_removexattr(self):
    setxattr(self.tmpfn, 'user.test', b'123')
    a = getxattr(self.tmpfn, 'user.test')
    assert a == b'123'
    removexattr(self.tmpfn, 'user.test')
    a = getxattr(self.tmpfn, 'user.test')
    assert a is None

if __name__ == "__main__":
  unittest.main()
