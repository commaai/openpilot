#!/usr/bin/env python3
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from opendbc.car.car_helpers import get_demo_car_params
from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.hardware import HARDWARE
from openpilot.common.mock import mock_messages
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.modeld.helpers import WORLDMODEL_DIR, chestnut_present
from openpilot.selfdrive.test.helpers import processes_context, log_collector
from openpilot.system.manager.process import launcher
from openpilot.system.manager.process_config import managed_processes

TEST_DURATION = 25


def worldmodel_launcher(module, name):
  from tinygrad.device import Compiler

  with mock.patch.object(Compiler, 'compile_cached', side_effect=AssertionError("Worldmodel attempted runtime compilation")):
    launcher(module, name)


@unittest.skipUnless(HARDWARE.get_device_type() == "mici", "requires MICI")
class TestWorldModelOnroad(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  @mock_messages(['deviceMotion'])
  def test_camera_models(self, subtests):
    assert WORLDMODEL_DIR and (Path(WORLDMODEL_DIR) / 'model.pkl').is_file()
    assert chestnut_present()
    Params().put("CarParams", get_demo_car_params().to_bytes(), block=True)
    services = ['narrowRoadCameraState', 'wideRoadCameraState', 'cabinCameraState', 'modelV2', 'driverStateV2', 'worldModelPlan']
    sm = messaging.SubMaster(services)
    pm = messaging.PubMaster(['deviceState', 'lateralDelay'])
    device_state = messaging.new_message('deviceState')
    device_state.deviceState.deviceType = HARDWARE.get_device_type()
    lateral_delay = messaging.new_message('lateralDelay')
    lateral_delay.lateralDelay.lateralDelay = .2
    device_state_bytes, lateral_delay_bytes = device_state.to_bytes(), lateral_delay.to_bytes()

    def update():
      pm.send('deviceState', device_state_bytes)
      pm.send('lateralDelay', lateral_delay_bytes)
      sm.update(100)
      assert all(p.proc.is_alive() for p in processes), "Camera/model process exited"

    names = ['camerad', 'calibrationd', 'modeld', 'dmonitoringmodeld', 'worldmodeld']
    with mock.patch.object(managed_processes['worldmodeld'], 'launcher', worldmodel_launcher), processes_context(names) as processes:
      with Timeout(90, "worldmodel didn't start and become active"):
        while not all(sm.seen.values()) or not sm.valid['worldModelPlan'] or not sm.valid['modelV2'] or not sm['modelV2'].big:
          update()
      assert not Params().get_bool('ChestnutActive'), "Stock big model is using the worldmodel GPU"
      with log_collector(services) as (logs, _):
        end = time.monotonic() + TEST_DURATION
        while time.monotonic() < end:
          update()

    msgs = {s: [m for m in logs if m.which() == s] for s in services}
    for service, messages in msgs.items():
      with subtests.test(service=service):
        expected = TEST_DURATION * SERVICE_LIST[service].frequency
        assert np.isclose(len(messages), expected, rtol=.05, atol=2), f"{service}: expected {expected}, got {len(messages)}"
        assert all(m.valid for m in messages), f"{service}: invalid predictions after warmup"
        assert np.all(np.diff([getattr(m, service).frameId for m in messages]) > 0), service

    plans = msgs['worldModelPlan']
    period = 1 / SERVICE_LIST['worldModelPlan'].frequency
    times = [m.worldModelPlan.modelExecutionTime for m in plans]
    ages = [(m.logMonoTime - m.worldModelPlan.timestampEof) / 1e9 for m in plans]
    print(f"worldmodel: {len(plans)} frames, median {np.median(times)*1000:.2f} ms, max {max(times)*1000:.2f} ms")
    print(f"worldmodel capture-to-publish: median {np.median(ages)*1000:.2f} ms, max {max(ages)*1000:.2f} ms")
    assert max(times) < period, f"Worldmodel exceeded {period}s inference budget: {max(times)}"
    assert all(0 <= age < 2 * period for age in ages), f"Worldmodel published stale plans: {max(ages)}"
    for m in plans:
      plan = m.worldModelPlan
      assert (len(plan.plan), len(plan.action), len(plan.actionT)) == (990, 4, 2)
      assert np.isfinite([*plan.plan, *plan.action, *plan.actionT]).all()
    fallback_frames = [m.modelV2.frameId for m in msgs['modelV2'] if not m.modelV2.big]
    if fallback_frames:
      publication_times = [m.logMonoTime for m in plans]
      samples = []
      for m in msgs['modelV2']:
        index = np.searchsorted(publication_times, m.logMonoTime, side='right') - 1
        if not m.modelV2.big and 0 <= index < len(plans) - 1:
          age_ms = (m.logMonoTime - plans[index].worldModelPlan.timestampEof) / 1e6
          until_next_ms = (publication_times[index + 1] - m.logMonoTime) / 1e6
          samples.append((m.modelV2.frameId, round(age_ms, 2), round(until_next_ms, 2)))
      print(f"fallback (frame, latest plan age ms, next plan in ms): {samples[:10]}")
    assert not fallback_frames, f"Modeld fell back instead of using worldmodel plans on frames: {fallback_frames}"
    assert all(np.isfinite(m.modelV2.position.x).all() for m in msgs['modelV2'])


if __name__ == '__main__':
  unittest.main()
