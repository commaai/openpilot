#!/usr/bin/env python3
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]

# the safety code, without the libsafety harness and ignition.h
SRC = """
#include <stddef.h>
#include <stdint.h>
uint32_t microsecond_timer_get(void);
#include "opendbc/safety/can.h"
#include "opendbc/safety/safety.h"
"""


class TestGlobals(unittest.TestCase):
  # everything a safety decision depends on is in the two copies, so the dual computation covers it
  def test_only_dual_layer_is_writable(self):
    for debug in (["-DALLOW_DEBUG"], []):
      with self.subTest(debug=debug), tempfile.NamedTemporaryFile(suffix=".o") as obj:
        cmd = ["cc", "-std=gnu11", "-fno-pic", "-fno-common", *debug, "-I", ROOT, "-x", "c", "-c", "-", "-o", obj.name]
        subprocess.run(cmd, input=SRC, text=True, check=True)
        symbols = subprocess.check_output(["objdump", "-t", obj.name], text=True)

        # ELF and Mach-O writable sections, ltmp are Mach-O section labels
        names = re.findall(r" O (?:\.data|\.bss|__DATA,__data|__DATA,__bss|__DATA,__common)\s.*?(\S+)$", symbols, re.MULTILINE)
        writable = {n.removeprefix("_") for n in names if not n.startswith("ltmp")}
        self.assertEqual(writable, {"safety_a", "safety_b", "current_copy", "safety_ts", "safety_redundancy_fault"})


if __name__ == "__main__":
  unittest.main()
