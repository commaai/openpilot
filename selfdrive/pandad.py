#!/usr/bin/env python
# simple boardd wrapper that updates the panda first
import os
from panda import ensure_st_up_to_date

def main(gctx=None):
  ensure_st_up_to_date()

  os.chdir("boardd")
  os.execvp("./boardd", ["./boardd"])

if __name__ == "__main__":
    main()
