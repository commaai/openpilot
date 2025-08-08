#!/usr/bin/env python3
from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass

from openpilot.tools.lib.logreader import LogReader

TEST_SEGMENT = "4019fff6e54cf1c7|00000123--4bc0d95ef6/5"

class Assert(Enum):
  ENGAGED = 0
  SOFT_DISABLE = 1

@dataclass
class Scenario:
  name: str
  get_msgs: Callable
  asserts: list[Assert]

scenarios = [
  Scenario("all good", lambda x: x, [Assert.ENGAGED, ]),
  Scenario("bad CAN msgs", lambda x: x, [Assert.ENGAGED, Assert.SOFT_DISABLE]),
  Scenario("missing road cam", lambda x: x, [Assert.ENGAGED, Assert.SOFT_DISABLE]),
  Scenario("DM red alert, user not disengaging", lambda x: x, [Assert.ENGAGED, Assert.SOFT_DISABLE]),
]

def run(msgs):
  # TODO: need to process replay the whole stack
  return msgs

if __name__ == "__main__":
  lr = list(LogReader(TEST_SEGMENT))

  for s in scenarios:
    inputs = s.get_msgs(lr)
    output = run(inputs)
