#!/usr/bin/env python3

import sys

from openpilot.system.loggerd import encoderd


def main() -> int:
  sys.argv = ["stream_encoderd", "--stream"]
  return encoderd.main()


if __name__ == "__main__":
  raise SystemExit(main())
