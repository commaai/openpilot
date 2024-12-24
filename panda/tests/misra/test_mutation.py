#!/usr/bin/env python3
import os
import glob
import pytest
import shutil
import subprocess
import tempfile
import random

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.join(HERE, "../../")

IGNORED_PATHS = (
  'board/obj',
  'board/jungle',
  'board/stm32h7/inc',
  'board/stm32f4/inc',
  'board/fake_stm.h',

  # bootstub only files
  'board/flasher.h',
  'board/bootstub.c',
  'board/bootstub_declarations.h',
  'board/stm32h7/llflash.h',
  'board/stm32f4/llflash.h',
)

mutations = [
  # default
  (None, None, False),
  # F4 only
  ("board/stm32f4/llbxcan.h", "s/1U/1/g", True),
  # H7 only
  ("board/stm32h7/llfdcan.h", "s/return ret;/if (true) { return ret; } else { return false; }/g", True),
  # general safety
  ("board/safety/safety_toyota.h", "s/is_lkas_msg =.*;/is_lkas_msg = addr == 1 || addr == 2;/g", True),
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
files = [f for f in all_files if f.endswith(('.c', '.h')) and not f.startswith(IGNORED_PATHS)]
assert len(files) > 70, all(d in files for d in ('board/main.c', 'board/stm32f4/llbxcan.h', 'board/stm32h7/llfdcan.h', 'board/safety/safety_toyota.h'))

for p in patterns:
  mutations.append((random.choice(files), p, True))

@pytest.mark.parametrize("fn, patch, should_fail", mutations)
def test_misra_mutation(fn, patch, should_fail):
  with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT, tmp, dirs_exist_ok=True)

    # apply patch
    if fn is not None:
      r = os.system(f"cd {tmp} && sed -i '{patch}' {fn}")
      assert r == 0

    # run test
    r = subprocess.run("SKIP_TABLES_DIFF=1 tests/misra/test_misra.sh", cwd=tmp, shell=True)
    failed = r.returncode != 0
    assert failed == should_fail
