#!/usr/bin/env python3
import sys
import unittest
from collections import Counter

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, settings

from cereal import log
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.test.process_replay.process_replay import (CONFIGS,
                                                          replay_process)


def get_process_config(process):
  return [cfg for cfg in CONFIGS if cfg.proc_name == process][0]


def get_event_union_strategy(r, name):
  return st.fixed_dictionaries({
    'valid': st.booleans(),
    'logMonoTime': st.integers(min_value=0, max_value=2**64-1),
    name: r[name[0].upper() + name[1:]],
  })


def get_strategy_for_events(event_types, finite=False):
  # TODO: generate automatically based on capnp definitions
  def floats(**kwargs):
    allow_nan = False if finite else None
    allow_infinity = False if finite else None
    return st.floats(**kwargs, allow_nan=allow_nan, allow_infinity=allow_infinity)

  r = {}
  r['liveLocationKalman.Measurement'] = st.fixed_dictionaries({
    'value': st.lists(floats(), min_size=3, max_size=3),
    'std': st.lists(floats(), min_size=3, max_size=3),
    'valid': st.booleans(),
  })
  r['LiveLocationKalman'] = st.fixed_dictionaries({
    'angularVelocityCalibrated': r['liveLocationKalman.Measurement'],
    'inputsOK': st.booleans(),
    'posenetOK': st.booleans(),
  })
  r['CarState'] = st.fixed_dictionaries({
    'vEgo': floats(width=32),
    'vEgoRaw': floats(width=32),
    'steeringPressed': st.booleans(),
    'steeringAngleDeg': floats(width=32),
  })
  r['CameraOdometry'] = st.fixed_dictionaries({
    'frameId': st.integers(min_value=0, max_value=2**32-1),
    'timestampEof': st.integers(min_value=0, max_value=2**64-1),
    'trans': st.lists(floats(width=32), min_size=3, max_size=3),
    'rot': st.lists(floats(width=32), min_size=3, max_size=3),
    'transStd': st.lists(floats(width=32), min_size=3, max_size=3),
    'rotStd': st.lists(floats(width=32), min_size=3, max_size=3),
  })
  r['SensorEventData.SensorVec'] = st.fixed_dictionaries({
    'v': st.lists(floats(width=32), min_size=3, max_size=3),
    'status': st.integers(min_value=0, max_value=1),
  })
  r['SensorEventData_gyro'] = st.fixed_dictionaries({
    'version': st.just(1),
    'sensor': st.just(5),
    'type': st.just(16),  # BMX055
    'timestamp': st.integers(min_value=0, max_value=2**63-1),
    'source': st.just(8),
    'gyroUncalibrated': r['SensorEventData.SensorVec'],
  })
  r['SensorEventData_accel'] = st.fixed_dictionaries({
    'version': st.just(1),
    'sensor': st.just(1),
    'type': st.just(1),  # BMX055
    'timestamp': st.integers(min_value=0, max_value=2**63-1),
    'source': st.just(8),
    'acceleration': r['SensorEventData.SensorVec'],
  })
  r['SensorEvents'] = st.lists(st.one_of(r['SensorEventData_gyro'], r['SensorEventData_accel']), min_size=1)
  r['GpsLocationExternal'] = st.fixed_dictionaries({
    'flags': st.integers(min_value=0, max_value=1),
    'latitude': floats(),
    'longitude': floats(),
    'altitude': floats(),
    'speed': floats(width=32),
    'bearingDeg': floats(width=32),
    'accuracy': floats(width=32),
    'timestamp': st.integers(min_value=0, max_value=2**63-1),
    'source': st.just(6),  # Ublox
    'vNED': st.lists(floats(width=32), min_size=3, max_size=3),
    'verticalAccuracy': floats(width=32),
    'bearingAccuracyDeg': floats(width=32),
    'speedAccuracy': floats(width=32),
  })
  r['LiveCalibration'] = st.fixed_dictionaries({
    'calStatus': st.integers(min_value=0, max_value=1),
    'rpyCalib': st.lists(floats(width=32), min_size=3, max_size=3),
  })

  return st.lists(st.one_of(*[get_event_union_strategy(r, n) for n in event_types]))


def get_strategy_for_process(process, finite=False):
  return get_strategy_for_events(get_process_config(process).pub_sub.keys(), finite)


def convert_to_lr(msgs):
  return [log.Event.new_message(**m).as_reader() for m in msgs]


def assume_all_services_present(cfg, lr):
  tps = Counter([m.which() for m in lr])
  for p in cfg.pub_sub:
    assume(tps[p] > 0)


def is_finite(d, exclude=[], prefix=""):  # pylint: disable=dangerous-default-value
  ret = True
  for k, v in d.items():
    name = prefix + f"{k}"
    if name in exclude:
      continue

    if isinstance(v, dict):
      ret = ret and is_finite(v, exclude, name + ".")
    else:
      try:
        ret = ret and np.isfinite(v).all()
      except TypeError:
        pass

  return ret


def test_process(dat, name):
  cfg = get_process_config(name)
  lr = convert_to_lr(dat)
  assume_all_services_present(cfg, lr)
  return replay_process(cfg, lr, TOYOTA.COROLLA_TSS2)


class TestFuzzy(unittest.TestCase):
  @given(get_strategy_for_process('paramsd'))
  @settings(deadline=1000)
  def test_paramsd(self, dat):
    for r in test_process(dat, 'paramsd'):
      lp = r.liveParameters.to_dict()
      assert is_finite(lp)

  @given(get_strategy_for_process('locationd', finite=True))
  @settings(deadline=10000)
  def test_locationd(self, dat):
    exclude = [
      'positionGeodetic.std',
      'velocityNED.std',
      'orientationNED.std',
      'calibratedOrientationECEF.std',
    ]
    for r in test_process(dat, 'locationd'):
      lp = r.liveLocationKalman.to_dict()
      assert is_finite(lp, exclude)


if __name__ == "__main__":
  procs = {
    'locationd': TestFuzzy().test_locationd,
    'paramsd': TestFuzzy().test_paramsd,
  }

  if len(sys.argv) != 2:
    print("Usage: ./test_fuzzy.py <process name>")
    sys.exit(0)

  proc = sys.argv[1]
  if proc not in procs:
    print(f"{proc} not available")
    sys.exit(0)
  else:
    procs[proc]()
