#!/usr/bin/env python3
import os
import unittest
import requests
from cereal import car

from tools.lib.logreader import LogReader

from opendbc.can.parser import CANParser

from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.honda.interface import CarInterface as HondaCarInterface
from selfdrive.car.honda.carcontroller import CarController as HondaCarController
from selfdrive.car.honda.radar_interface import RadarInterface as HondaRadarInterface

from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.toyota.interface import CarInterface as ToyotaCarInterface
from selfdrive.car.toyota.carcontroller import CarController as ToyotaCarController
from selfdrive.car.toyota.radar_interface import RadarInterface as ToyotaRadarInterface

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

def run_route(route, car_name, CarInterface, CarController):
  lr = LogReader("/tmp/"+route + ".bz2")
  print(lr)

  cps = []
  def CANParserHook(dbc_name, signals, checks=None, bus=0, sendcan=False, tcp_addr="127.0.0.1", timeout=-1):
    cp = CANParser(dbc_name, signals, checks, bus, sendcan, "", timeout)
    cps.append(cp)
    return cp

  params = CarInterface.get_params(car_name)
  CI = CarInterface(params, CarController, CANParserHook)
  print(CI)

  i = 0
  last_monotime = 0
  for msg in lr:
    if msg.which() == 'can':
      msg_bytes = msg.as_builder().to_bytes()
      monotime = msg.logMonoTime
      for x in cps:
        x.update_string(monotime, msg_bytes)

    if (monotime-last_monotime) > 0.01:
      control = car.CarControl.new_message()
      CS = CI.update(control)
      if i % 100 == 0:
        print('\033[2J\033[H'+str(CS))
      last_monotime = monotime
      i += 1

  return True

def run_route_radar(route, car_name, RadarInterface, CarInterface):
  lr = LogReader("/tmp/"+route + ".bz2")
  print(lr)

  cps = []
  def CANParserHook(dbc_name, signals, checks=None, bus=0, sendcan=False, tcp_addr="127.0.0.1", timeout=-1):
    cp = CANParser(dbc_name, signals, checks, bus, sendcan, "", timeout)
    print(signals)
    cps.append(cp)
    return cp

  params = CarInterface.get_params(car_name)
  RI = RadarInterface(params, CANParserHook)

  i = 0
  updated_messages = set()
  for msg in lr:
    if msg.which() == 'can':
      msg_bytes = msg.as_builder().to_bytes()
      _, vls = cps[0].update_string(msg.logMonoTime, msg_bytes)
      updated_messages.update(vls)
      if RI.trigger_msg in updated_messages:
        ret = RI._update(updated_messages)
        if i % 10 == 0:
          print('\033[2J\033[H'+str(ret))
        updated_messages = set()
        i += 1

  return True


# TODO: make this generic
class TestCarInterface(unittest.TestCase):
  def setUp(self):
    self.routes = {
      HONDA.CIVIC: "b0c9d2329ad1606b|2019-05-30--20-23-57",
      HONDA.ACCORD: "0375fdf7b1ce594d|2019-05-21--20-10-33",
      TOYOTA.PRIUS: "38bfd238edecbcd7|2019-06-07--10-15-25",
      TOYOTA.RAV4: "02ec6bea180a4d36|2019-04-17--11-21-35"
    }

    for route in self.routes.values():
      route_filename = route + ".bz2"
      if not os.path.isfile("/tmp/"+route_filename):
        with open("/tmp/"+route + ".bz2", "w") as f:
          f.write(requests.get(BASE_URL + route_filename).content)

  def test_parser_civic(self):
    #self.assertTrue(run_route(self.routes[HONDA.CIVIC], HONDA.CIVIC, HondaCarInterface, HondaCarController))
    pass

  def test_parser_accord(self):
    # one honda
    #self.assertTrue(run_route(self.routes[HONDA.ACCORD], HONDA.ACCORD, HondaCarInterface, HondaCarController))
    pass

  def test_parser_prius(self):
    #self.assertTrue(run_route(self.routes[TOYOTA.PRIUS], TOYOTA.PRIUS, ToyotaCarInterface, ToyotaCarController))
    pass

  def test_parser_rav4(self):
    # hmm, rav4 is broken
    #self.assertTrue(run_route(self.routes[TOYOTA.RAV4], TOYOTA.RAV4, ToyotaCarInterface, ToyotaCarController))
    pass

  def test_radar_civic(self):
    #self.assertTrue(run_route_radar(self.routes[HONDA.CIVIC], HONDA.CIVIC, HondaRadarInterface, HondaCarInterface))
    pass

  def test_radar_prius(self):
    self.assertTrue(run_route_radar(self.routes[TOYOTA.PRIUS], TOYOTA.PRIUS, ToyotaRadarInterface, ToyotaCarInterface))
    pass


if __name__ == "__main__":
  unittest.main()

