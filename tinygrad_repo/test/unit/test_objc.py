import sys, unittest

class TestObjCMetaSpec(unittest.TestCase):
  @unittest.skipUnless(sys.platform == "darwin", "objc runtime only on macOS")
  def test_classmethods_are_classmethods(self):
    from tinygrad.runtime.support.objc import Spec, id_

    #_classmethods_ must include classmethod descriptors
    class ObjCTest(Spec):
      _methods_ = [("foo", id_, [])]
      _classmethods_ = [("bar", id_, [])]

    self.assertNotIsInstance(ObjCTest.__dict__["foo"], classmethod)
    self.assertIsInstance(ObjCTest.__dict__["bar"], classmethod)

if __name__ == "__main__":
  unittest.main()
