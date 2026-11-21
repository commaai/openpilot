#!/usr/bin/env python3
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
# RM0468 4.5.1: FLASH_CR and FLASH_OPTCR only unlock with these keys
FLASH_KEYS = (0x45670123, 0xCDEF89AB)
OPT_KEYS = (0x08192A3B, 0x4C5D6E7F)


def word(w):
  return struct.pack("<I", w)


class TestBuild(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # debug and release build in copies, board/obj is left alone
    files = [f for f in subprocess.check_output(["git", "ls-files"], cwd=ROOT, encoding="utf8").split() if os.path.isfile(os.path.join(ROOT, f))]
    env = {k: v for k, v in os.environ.items() if k not in ("RELEASE", "CERT")}
    cls.tmp = {}
    for release in (False, True):
      d = cls.tmp[release] = tempfile.mkdtemp()
      for f in files:
        os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
        shutil.copy2(f"{ROOT}/{f}", f"{d}/{f}")
      release_env = {"RELEASE": "1", "CERT": f"{d}/board/crypto/certs/debug"} if release else {}
      subprocess.check_call(["scons", "-u", "board/obj/panda_h7.bin.signed", "board/obj/bootstub.panda_h7.bin"], cwd=d, env=env | release_env,
                            stdout=subprocess.DEVNULL)

  @classmethod
  def tearDownClass(cls):
    for d in cls.tmp.values():
      shutil.rmtree(d)

  def _read(self, release, fn):
    with open(f"{self.tmp[release]}/board/obj/{fn}", "rb") as f:
      return f.read()

  def _has_symbol(self, release, elf, name):
    # the pip toolchain has no nm, symbol names are null terminated in the ELF string table
    return f"\0{name}\0".encode() in self._read(release, f"panda_h7/{elf}")

  def test_release_has_no_debug_features(self):
    # each is only compiled with ALLOW_DEBUG, the debug build checks the name still exists
    for sym in ("self_test_inject", "relay_readback_inject", "stop_mode_requested"):
      assert self._has_symbol(False, "main.elf", sym), sym
      assert not self._has_symbol(True, "main.elf", sym), sym

  def test_only_bootstub_programs_flash(self):
    for release in (False, True):
      for func in ("flash_unlock", "flash_erase_sector", "flash_write_word"):
        assert self._has_symbol(release, "bootstub.elf", func), func
        assert not self._has_symbol(release, "main.elf", func), func
      assert not any(word(k) in self._read(release, "panda_h7/main.bin") for k in FLASH_KEYS)

  def test_option_bytes_never_unlocked(self):
    for release in (False, True):
      for fn in ("panda_h7/main.bin", "bootstub.panda_h7.bin"):
        assert not any(word(k) in self._read(release, fn) for k in OPT_KEYS), (release, fn)


if __name__ == "__main__":
  unittest.main()
