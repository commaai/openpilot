import numpy as np
import time
from tqdm import tqdm

from selfdrive.car.tests.routes import CarTestRoute
from selfdrive.car.tests.test_models import TestCarModelBase
from tools.plotjuggler.juggle import DEMO_ROUTE

N_RUNS = 5


class TestCarModelDemo(TestCarModelBase):
  car_model = None
  test_route = CarTestRoute(DEMO_ROUTE, 'TOYOTA COROLLA TSS2 2019')
  ci = False


if __name__ == '__main__':
  tm = TestCarModelDemo()
  tm.setUpClass()

  can_kb = sum([len(m.as_builder().to_bytes()) for m in tm.can_msgs]) * 1e-3

  ets = []
  for _ in tqdm(range(N_RUNS)):
    tm.setUp()

    start_t = time.process_time_ns()
    tm.test_car_interface()
    ets.append((time.process_time_ns() - start_t) * 1e-6)
  print(f'{len(tm.can_msgs)} CAN packets, {round(can_kb)} CAN kb, {N_RUNS} runs')
  print(f'{np.mean(ets):.2f} mean ms, {max(ets):.2f} max ms, {min(ets):.2f} min ms, {np.std(ets):.2f} std ms')
  print(f'{np.mean(ets) / len(tm.can_msgs):.4f} mean ms / CAN packet')
