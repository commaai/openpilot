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


def get_strategy_for_events(event_types):
  # TODO: generate automatically based on capnp definitions

  def get_std_event(r, name):
    return st.fixed_dictionaries({
      'valid': st.booleans(),
      'logMonoTime': st.integers(min_value=0, max_value=2**64-1),
      name: r[name[0].upper() + name[1:]],
    })

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
  })

  a = st.one_of(*[get_std_event(r, n) for n in event_types])
  return st.lists(a)


def get_strategy_for_process(process):
  cfg = [cfg for cfg in CONFIGS if cfg.proc_name == process][0]
  return get_strategy_for_events(cfg.pub_sub.keys())


def convert_to_lr(data):
  r = []
  for m in data:
    r.append(log.Event.new_message(**m).as_reader())
  return r


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
    assert not any(map(math.isnan, lp.values()))


if __name__ == "__main__":
  test_paramsd()  # pylint: disable=no-value-for-parameter
