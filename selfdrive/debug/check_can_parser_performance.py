#!/usr/bin/env python3
import numpy as np
import time
from tqdm import tqdm

from cereal import car
from openpilot.selfdrive.car.tests.routes import CarTestRoute
from openpilot.selfdrive.car.tests.test_models import TestCarModelBase
from openpilot.tools.plotjuggler.juggle import DEMO_ROUTE

N_RUNS = 10


class CarModelTestCase(TestCarModelBase):
  test_route = CarTestRoute(DEMO_ROUTE, None)
  ci = False


if __name__ == '__main__':
  # Get CAN messages and parsers
  tm = CarModelTestCase()
  tm.setUpClass()
  tm.setUp()

  CC = car.CarControl.new_message()
  ets = []
  for _ in tqdm(range(N_RUNS)):
    start_t = time.process_time_ns()
    for msg in tm.can_msgs:
      for cp in tm.CI.can_parsers:
        if cp is not None:
          cp.update_strings((msg.as_builder().to_bytes(),))
    ets.append((time.process_time_ns() - start_t) * 1e-6)

  print(f'{len(tm.can_msgs)} CAN packets, {N_RUNS} runs')
  print(f'{np.mean(ets):.2f} mean ms, {max(ets):.2f} max ms, {min(ets):.2f} min ms, {np.std(ets):.2f} std ms')
  print(f'{np.mean(ets) / len(tm.can_msgs):.4f} mean ms / CAN packet')
