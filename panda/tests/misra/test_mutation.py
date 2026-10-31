#!/usr/bin/env python3
import os
import glob
import unittest
import shutil
import subprocess
import tempfile
import random
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.join(HERE, "../../")

# skip mutating these paths
IGNORED_PATHS = (
  'board/obj',
  'board/body',
  'board/stm32h7/inc',
  'board/fake_stm.h',

  # bootstub only files
  'board/flasher.h',
  'board/bootstub.c',
  'board/bootstub_declarations.h',
  'board/stm32h7/llflash.h',
)

mutations = [
  (None, None, False),  # no mods, should pass
  ("board/stm32h7/llfdcan.h", "s/return ret;/if (true) { return ret; } else { return false; }/g", True),
]

patterns = [
  # misra-c2012-13.3
  "$a void test(int tmp) { int tmp2 = tmp++ + 2; if (tmp2) {;}}",
  # misra-c2012-13.4
  "$a int test(int x, int y) { return (x=2) && (y=2); }",
  # misra-c2012-13.5
  "$a void test(int tmp) { if (true && tmp++) {;} }",
  # misra-c2012-13.6
  "$a void test(int tmp) { if (sizeof(tmp++)) {;} }",
  # misra-c2012-14.1
  "$a void test(float len) { for (float j = 0; j < len; j++) {;} }",
  # misra-c2012-14.4
  "$a void test(int len) { if (len - 8) {;} }",
  # misra-c2012-16.4
  r"$a void test(int temp) {switch (temp) { case 1: ; }}\n",
  # misra-c2012-17.8
  "$a void test(int cnt) { for (cnt=0;;cnt++) {;} }",
  # misra-c2012-20.4
  r"$a #define auto 1\n",
  # misra-c2012-20.5
  r"$a #define TEST 1\n#undef TEST\n",
]

all_files = glob.glob('board/**', root_dir=ROOT, recursive=True)
files = sorted(f for f in all_files if f.endswith(('.c', '.h')) and not f.startswith(IGNORED_PATHS))
assert len(files) > 50, all(d in files for d in ('board/main.c', 'board/stm32h7/llfdcan.h'))

# fixed seed for reproducible mutation selection
rng = random.Random(len(files))
for p in patterns:
  mutations.append((rng.choice(files), p, True))

# sample to keep CI fast, but always include the no-mutation case
mutations = [mutations[0]] + rng.sample(mutations[1:], min(2, len(mutations) - 1))

def run_mutation(fn, patch, should_fail):
  with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT, tmp + "/panda", ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))

    # apply patch
    if fn is not None:
      fpath = os.path.join(tmp, "panda", fn)
      with open(fpath) as f:
        content = f.read()
      if patch.startswith("s/"):
        old, new = patch[2:].rsplit("/g", 1)[0].split("/", 1)
        content = content.replace(old, new)
      elif patch.startswith("$a "):
        content += patch[3:].replace(r"\n", "\n")
      with open(fpath, "w") as f:
        f.write(content)

    return subprocess.run("SKIP_TABLES_DIFF=1 panda/tests/misra/test_misra.sh", cwd=tmp, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class TestMisraMutation(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Each mutation has its own checkout and subprocess; keep the expensive checks parallel.
    executor = ProcessPoolExecutor(max_workers=len(mutations))
    cls.addClassCleanup(executor.shutdown)
    cls.results = [executor.submit(run_mutation, *mutation) for mutation in mutations]


def mutation_test(index):
  def test(self):
    result = self.results[index].result()
    self.assertEqual(result.returncode != 0, mutations[index][2], result.stdout)
  test.__doc__ = f"MISRA mutation: {mutations[index]}"
  return test


for index in range(len(mutations)):
  setattr(TestMisraMutation, f"test_misra_mutation_{index}", mutation_test(index))
