#!/usr/bin/env python3
from openpilot.system.loggerd import encoderd


def main():
  encoderd.main(stream=True)


if __name__ == "__main__":
  main()
