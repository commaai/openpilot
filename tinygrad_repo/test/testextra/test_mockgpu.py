from tinygrad.helpers import getenv
import unittest, importlib

@unittest.skipUnless(getenv("MOCKGPU"), 'Testing mockgpu')
class TestMockGPU(unittest.TestCase):
  # https://github.com/tinygrad/tinygrad/pull/7627
  def test_import_typing_extensions(self):
    import test.mockgpu.mockgpu # noqa: F401  # pylint: disable=unused-import
    import typing_extensions
    importlib.reload(typing_extensions) # pytest imports typing_extension before mockgpu

if __name__ == '__main__':
  unittest.main()