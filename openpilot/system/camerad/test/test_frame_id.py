import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class TestCameraFrameCounter(unittest.TestCase):
  def test_sof_gaps_and_drift(self):
    compiler = shutil.which('c++')
    self.assertIsNotNone(compiler)
    header = Path(__file__).resolve().parents[1] / 'cameras/frame_id.h'
    source = r'''
#include <cassert>
#include "frame_id.h"
int main() {
  CameraFrameCounter narrow, wide;
  // Captured UP05 failure: narrow loses a physical period with consecutive IFE IDs.
  assert(narrow.update(8639646558ULL, 8589643693ULL) == 0);
  assert(wide.update(8639640152ULL, 8589649579ULL) == 0);
  assert(wide.update(8689624891ULL, 0) == 1);
  assert(narrow.update(8739648797ULL, 0) == 2);
  assert(wide.update(8739640099ULL, 0) == 2);
  assert(narrow.update(8789648797ULL, 0) == 3);
  assert(narrow.update(8939648797ULL, 0) == 6); // repeated/multiple lost periods
  assert(!narrow.update(8939648797ULL, 0));    // duplicate, no state advance
  assert(!narrow.update(8839648797ULL, 0));    // backwards
  assert(!narrow.update(8939648798ULL, 0));    // implausibly short interval
  assert(narrow.update(8989648797ULL, 0) == 7);
  CameraFrameCounter initial;
  assert(initial.update(1'100'000'000, 1'000'000'000) == 1); // missing first frame
  CameraFrameCounter drift;
  uint64_t sof = 1'000'000'000;
  uint64_t sync = sof;
  for (uint32_t i = 0; i < 100'000; ++i) {
    sof += 50'010'000 + (i % 2 ? 100'000 : -100'000);
    assert(drift.update(sof, sync) == i); // no accumulated epoch rounding
  }
}
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      (path / 'test.cc').write_text(source)
      subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(header.parent),
                      str(path / 'test.cc'), '-o', str(path / 'test')], check=True)
      subprocess.run([str(path / 'test')], check=True)


if __name__ == '__main__':
  unittest.main()
