#!/usr/bin/env python2

import os
import unittest

import requests

import selfdrive.messaging as messaging
from selfdrive.can.parser import CANParser as CANParserNew
from selfdrive.can.tests.parser_old import CANParser as CANParserOld
from selfdrive.car.honda.carstate import get_can_signals
from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR, DBC
from selfdrive.services import service_list
from tools.lib.logreader import LogReader

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"
DT = int(0.01 * 1e9)  # ns


def dict_keys_differ(dict1, dict2):
  keys1 = set(dict1.keys())
  keys2 = set(dict2.keys())

  if keys1 != keys2:
    return True

  for k in keys1:
    keys1 = set(dict1[k].keys())
    keys2 = set(dict2[k].keys())

    if keys1 != keys2:
      return True

  return False

def dicts_vals_differ(dict1, dict2):
  for k_outer in dict1.keys():
    for k_inner in dict1[k_outer].keys():
      if dict1[k_outer][k_inner] != dict2[k_outer][k_inner]:
        return True

  return False

def run_route(route):
  can = messaging.pub_sock(service_list['can'].port)

  CP = CarInterface.get_params(CAR.CIVIC, {})
  signals, checks = get_can_signals(CP)
  parser_old = CANParserOld(DBC[CP.carFingerprint]['pt'], signals, checks, 0, timeout=-1)
  parser_new = CANParserNew(DBC[CP.carFingerprint]['pt'], signals, checks, 0, timeout=-1)

  if dict_keys_differ(parser_old.vl, parser_new.vl):
    return False

  lr = LogReader(route + ".bz2")

  route_ok = True

  t = 0
  for msg in lr:
    if msg.which() == 'can':
      t += DT
      can.send(msg.as_builder().to_bytes())

      _, updated_old = parser_old.update(t, True)
      _, updated_new = parser_new.update(t, True)

      if updated_old != updated_new:
        route_ok = False
        print(t, "Diff in seen")

      if dicts_vals_differ(parser_old.vl, parser_new.vl):
        print(t, "Diff in dict")
        route_ok = False

  return route_ok

class TestCanParser(unittest.TestCase):
  def setUp(self):
    self.routes = {
      CAR.CIVIC: "b0c9d2329ad1606b|2019-05-30--20-23-57"
    }

    for route in self.routes.values():
      route_filename = route + ".bz2"
      if not os.path.isfile(route_filename):
        with open(route + ".bz2", "w") as f:
          f.write(requests.get(BASE_URL + route_filename).content)

  def test_parser_civic(self):
    self.assertTrue(run_route(self.routes[CAR.CIVIC]))


if __name__ == "__main__":
  unittest.main()
