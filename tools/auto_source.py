#!/usr/bin/env python3
import sys
from openpilot.tools.lib.logreader import LogReader


def main():
  if len(sys.argv) != 2:
    print("Usage: python auto_source.py <log_path>")
    sys.exit(1)

  log_path = sys.argv[1]
  lr = LogReader(log_path, sort_by_time=True)
  print("\n".join(lr.logreader_identifiers))


if __name__ == "__main__":
  main()
