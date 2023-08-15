import argparse
import sys
import time
import unittest
import numpy as np
from tqdm import tqdm
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route
from selfdrive.car.tests.test_models import TestCarModelBase
from selfdrive.car.tests.routes import CarTestRoute
from tools.plotjuggler.juggle import DEMO_ROUTE
from selfdrive.debug.test_car_model import create_test_models_suite


# @parameterized_class(('car_model', 'test_route'), get_test_cases())
class TestCarModelDemo(TestCarModelBase):
  # car_model = 'COMMA BODY'
  # car_model = 'TOYOTA COROLLA TSS2 2019'
  car_model = 'KIA EV6 2022'
  # test_route = CarTestRoute(DEMO_ROUTE, 'TOYOTA COROLLA TSS2 2019')
  # test_route = CarTestRoute('efdf9af95e71cd84|2022-05-13--19-03-31', None)
  test_route = CarTestRoute('d545129f3ca90f28|2022-10-19--09-22-54', None)
  ci = False


if __name__ == '__main__':
  tm = TestCarModelDemo()
  tm.setUpClass()

  can_kb = sum([len(m.as_builder().to_bytes()) for m in tm.can_msgs]) * 1e-3

  ets = []
  for _ in tqdm(range(2)):
    tm.setUp()

    start_t = time.process_time_ns()
    tm.test_car_interface()
    ets.append((time.process_time_ns() - start_t) * 1e-6)
  print(f'{len(tm.can_msgs)} CAN packets, {round(can_kb)} CAN kb')
  print(f'{np.mean(ets):.2f} mean ms, {max(ets):.2f} max ms, {min(ets):.2f} min ms, {np.std(ets):.2f} std ms')
  print(f'{np.mean(ets) / len(tm.can_msgs):.4f} mean ms / CAN packet')
  print(f'{np.mean(ets) / can_kb:.4f} mean ms / CAN kb')
  print(ets)

  # print(tm.CP)
  # print('canmsgs', len(tm.can_msgs))

  # test_suite = create_test_models_suite([CarTestRoute(DEMO_ROUTE, None)], test_filter=('test_car_interface', ))
  # test_suite._tests[0]()
  # unittest.TextTestRunner().run(test_suite)
  # tests.run()

  # lr = MultiLogIterator(Route(DEMO_ROUTE).log_paths())
  #
  #
  #
  # parser = argparse.ArgumentParser()
  # parser.add_argument("route_or_segment_name", help="Specify route to run tests on")
  # args = parser.parse_args()
  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit()
  #
  # route_or_segment_name = SegmentName(args.route_or_segment_name.strip(), allow_route_name=True)
  # segment_num = route_or_segment_name.segment_num if route_or_segment_name.segment_num != -1 else None
  # test_route = CarTestRoute(route_or_segment_name.route_name.canonical_name, args.car, segment=segment_num)
  # test_suite = create_test_models_suite([test_route], ci=args.ci)
  #
  # unittest.TextTestRunner().run(test_suite)
