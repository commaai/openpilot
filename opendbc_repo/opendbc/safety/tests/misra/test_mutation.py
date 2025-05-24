#!/usr/bin/env python3
import os
import glob
import pytest
import shutil
import subprocess
import tempfile
import random

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.join(HERE, "../../../../")

IGNORED_PATHS = (
  'opendbc/safety/tests/',
  'opendbc/safety/board/',
)

mutations = [
  # default
  (None, None, False),
  # general safety
  ("opendbc/safety/modes/toyota.h", "s/if (addr == 0x260) {/if (addr == 1 || addr == 2) {/g", True),
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

all_files = glob.glob('opendbc/safety/**', root_dir=ROOT, recursive=True)
files = [f for f in all_files if f.endswith(('.c', '.h')) and not f.startswith(IGNORED_PATHS)]
assert len(files) > 20, files

for p in patterns:
  mutations.append((random.choice(files), p, True))

@pytest.mark.parametrize("fn, patch, should_fail", mutations)
def test_misra_mutation(fn, patch, should_fail):
  with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT, tmp, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(tmp, '.venv'), ignore_errors=True)

    # apply patch
    if fn is not None:
      r = os.system(f"cd {tmp} && sed -i '{patch}' {fn}")
      assert r == 0

    # run test
    r = subprocess.run("SKIP_TABLES_DIFF=1 SKIP_BUILD=1 opendbc/safety/tests/misra/test_misra.sh", cwd=tmp, shell=True)
    failed = r.returncode != 0
    assert failed == should_fail
