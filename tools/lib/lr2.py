#!/usr/bin/env python3
import re
from tools.lib.helpers import RE
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator


class LogReader2(MultiLogIterator):
  def __init__(self, identifier: str):
    m = re. search(RE.SEGMENT_RANGE, identifier)
    if m is None:
      raise ValueError("Not a valid route or segment range: repr(identifier)")

    route = Route(m.group('dongle_id') + '|' + m.group('timestamp'))
    start = int(m.group('start')) if m.group('start') else None
    end = int(m.group('end')) if m.group('end') else None
    super().__init__(route.log_paths()[start:end])


if __name__ == "__main__":
  # all of these are valid inputs
  inputs = [
    "7a224f93464d0add|2023-04-22--14-56-28",
    "7a224f93464d0add/2023-04-22--14-56-28",
    "7a224f93464d0add/2023-04-22--14-56-28/0",
    "7a224f93464d0add|2023-04-22--14-56-28/0/10",
    "7a224f93464d0add|2023-04-22--14-56-28/0/-1",
  ]

  # and eventually, these will be valid
  # "7a224f93464d0add|2023-04-22--14-56-28/67s/100s",

  for i in inputs:
    lr = LogReader2(i)
