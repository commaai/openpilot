#!/usr/bin/env python3
import math
from collections import Counter

import hypothesis.strategies as st
from hypothesis import assume, given, settings, seed

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


def get_strategy_for_events(event_types):
  # TODO: generate automatically based on capnp definitions
  r = {}
  r['liveLocationKalman.Measurement'] = st.fixed_dictionaries({
    'value': st.lists(st.floats(), min_size=3, max_size=3),
    'std': st.lists(st.floats(), min_size=3, max_size=3),
    'valid': st.booleans(),
  })
  r['LiveLocationKalman'] = st.fixed_dictionaries({
    'angularVelocityCalibrated': r['liveLocationKalman.Measurement'],
    'inputsOK': st.booleans(),
    'posenetOK': st.booleans(),
  })
  r['CarState'] = st.fixed_dictionaries({
    'vEgo': st.floats(),
    'steeringPressed': st.booleans(),
    'steeringAngleDeg': st.floats(),
  })
  r['CameraOdometry'] = st.fixed_dictionaries({
    'trans': st.lists(st.floats(), min_size=3, max_size=3),
    'rot': st.lists(st.floats(), min_size=3, max_size=3),
    'transStd': st.lists(st.floats(), min_size=3, max_size=3),
    'rotStd': st.lists(st.floats(), min_size=3, max_size=3),
    'frameId': st.integers(min_value=0, max_value=2**32-1),
    'timestampEof': st.integers(min_value=0, max_value=2**64-1),
  })
  r['SensorEventData.SensorVec'] = st.fixed_dictionaries({
    'v': st.lists(st.floats(), min_size=3, max_size=3),
    'status': st.integers(min_value=0, max_value=1),
  })
  r['SensorEventData_gyro'] = st.fixed_dictionaries({
    'version': st.just(1),
    'sensor': st.just(5),
    'type': st.just(16),  # BMX055
    'timestamp': st.integers(min_value=0, max_value=2**64-1),
    'source': st.just(8),
    'gyroUncalibrated': r['SensorEventData.SensorVec'],
  })
  r['SensorEventData_accel'] = st.fixed_dictionaries({
    'version': st.just(1),
    'sensor': st.just(1),
    'type': st.just(1),  # BMX055
    'timestamp': st.integers(min_value=0, max_value=2**64-1),
    'source': st.just(8),
    'acceleration': r['SensorEventData.SensorVec'],
  })
  r['SensorEvents'] = st.lists(st.one_of(r['SensorEventData_gyro'], r['SensorEventData_accel']), min_size=1)
  r['GpsLocationExternal'] = st.fixed_dictionaries({
    'flags': st.integers(min_value=0, max_value=1),
    'latitude': st.floats(),
    'longitude': st.floats(),
    'altitude': st.floats(),
    'speed': st.floats(),
    'bearingDeg': st.floats(),
    'accuracy': st.floats(),
    'timestamp': st.integers(min_value=0, max_value=2**63-1),
    'source': st.just(6),  # Ublox
    'vNED': st.lists(st.floats(), min_size=3, max_size=3),
    'verticalAccuracy': st.floats(),
    'bearingAccuracyDeg': st.floats(),
    'speedAccuracy': st.floats(),
  })
  r['LiveCalibration'] = st.fixed_dictionaries({
    'calStatus': st.integers(min_value=0, max_value=1),
    'rpyCalib': st.lists(st.floats(), min_size=3, max_size=3),
  })

  return st.lists(st.one_of(*[get_event_union_strategy(r, n) for n in event_types]))


def get_strategy_for_process(process):
  return get_strategy_for_events(get_process_config(process).pub_sub.keys())


def convert_to_lr(msgs):
  return [log.Event.new_message(**m).as_reader() for m in msgs]


def assume_all_services_present(cfg, lr):
  tps = Counter([m.which() for m in lr])
  for p in cfg.pub_sub:
    assume(tps[p] > 0)


@given(get_strategy_for_process('paramsd'))
@settings(deadline=1000)
@seed(260777467434450485154004373463592546383)
def test_paramsd(dat):
  cfg = get_process_config('paramsd')
  lr = convert_to_lr(dat)
  assume_all_services_present(cfg, lr)
  results = replay_process(cfg, lr, TOYOTA.COROLLA_TSS2)

  for r in results:
    lp = r.liveParameters.to_dict()
    assert all(map(math.isfinite, lp.values()))


@given(get_strategy_for_process('locationd'))
@settings(deadline=10000)
def test_locationd(dat):
  cfg = get_process_config('locationd')
  lr = convert_to_lr(dat)
  assume_all_services_present(cfg, lr)
  results = replay_process(cfg, lr, TOYOTA.COROLLA_TSS2)

  for r in results:
    lp = r.liveLocationKalman.to_dict()
    assert all(map(math.isfinite, lp.values()))


if __name__ == "__main__":
  test_locationd()  # pylint: disable=no-value-for-parameter
  # test_paramsd()  # pylint: disable=no-value-for-parameter
