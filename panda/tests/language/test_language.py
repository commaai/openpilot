#!/usr/bin/env python

import subprocess
import sys

checked_ext = ["h", "c", "py", "pyx", "cpp", "hpp", "md", "mk"]

if __name__ == "__main__":
  with open("list.txt", 'r') as handle:

    suffix_cmd = " "
    for i in checked_ext:
      suffix_cmd +=  "--include \*." + i + " "

    found_bad_language = False
    for line in handle:
      line = line.rstrip('\n').rstrip(" ")
      try:
        cmd = "cd ../../; grep -R -i -w " + suffix_cmd + " '" + line + "'"
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        print res
        found_bad_language = True
      except subprocess.CalledProcessError as e:
        pass
  if found_bad_language:
    sys.exit("Failed: found bad language")
  else:
    print "Success"
