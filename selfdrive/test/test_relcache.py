#!/usr/bin/env python3
import sys
import os
import unittest
OUTPUT_FILE = "build.log"
WORKSPACE = os.getcwd()
class Test_RelCached(unittest.TestCase):

  def test_relcache(self):
    valid_start_strings = ('Compiling /', '[1/1] Cythonizing', 'Scanning directory ', 'clang++ -o tools/cabana')
    with open(OUTPUT_FILE, 'r') as f:
      for index, line in enumerate(f):
        if line.startswith(valid_start_strings):
          continue
        else:
          words = line.split(" ")
          for wrd in words:
            if WORKSPACE in wrd:
                self.fail(f"Contains absolute path ({wrd}) in line '{line}' at {index+1}\n")

if __name__ == "__main__":
  del sys.argv[1:]
  unittest.main()
