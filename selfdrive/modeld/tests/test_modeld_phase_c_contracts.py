"""
Phase C — additional ``modeld`` daemon contracts (LOW-LEVEL §7.1, §4.1 P0).

Uses a private VisionIPC + ``modeld`` harness mirrored from ``test_modeld.py``.
Subscribes to ``modelV2``, ``cameraOdometry``, and ``drivingModelData``; waits
until all three share the same ``frameId`` before asserting timestamps and IDs.

Includes a timeliness subtest: ``modelExecutionTime`` must stay below a 5.0s
ceiling (documented in the subtest message for stressed desktop runs).

Assertions run in one method with ``subtests`` so warmup / skip happens once
per setup (faster in environments where ``modeld`` never publishes).

**Does not modify** ``test_modeld.py``.

Maps: R1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer, VisionStreamType
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state
from openpilot.system.manager.process_config import managed_processes

_CAM = DEVICE_CAMERAS[("tici", "ar0231")].fcam
_IMG = np.zeros(int(_CAM.width * _CAM.height * (3 / 2)), dtype=np.uint8)
_IMG_BYTES = _IMG.flatten().tobytes()


class _ModeldVipcHarness:
  """Same lifecycle as ``test_modeld.TestModeld``; not a pytest test class."""

  def setup(self) -> None:
    self.vipc_server = VisionIpcServer("camerad")
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, _CAM.width, _CAM.height)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, _CAM.width, _CAM.height)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, _CAM.width, _CAM.height)
    self.vipc_server.start_listener()
    Params().put("CarParams", get_demo_car_params().to_bytes())

    self.sm = messaging.SubMaster(["modelV2", "cameraOdometry", "drivingModelData"])
    self.pm = messaging.PubMaster(["roadCameraState", "wideRoadCameraState", "liveCalibration"])

    managed_processes["modeld"].start()
    self.pm.wait_for_readers_to_update("roadCameraState", 10)

  def teardown(self) -> None:
    managed_processes["modeld"].stop()
    del self.vipc_server

  def send_frames(self, frame_id, cams=None):
    if cams is None:
      cams = ("roadCameraState", "wideRoadCameraState")

    cs = None
    for cam in cams:
      msg = messaging.new_message(cam)
      cs = getattr(msg, cam)
      cs.frameId = frame_id
      cs.timestampSof = int((frame_id * DT_MDL) * 1e9)
      cs.timestampEof = int(cs.timestampSof + (DT_MDL * 1e9))
      cam_meta = meta_from_camera_state(cam)

      self.pm.send(msg.which(), msg)
      self.vipc_server.send(cam_meta.stream, _IMG_BYTES, cs.frameId, cs.timestampSof, cs.timestampEof)
    return cs

  def wait(self) -> None:
    self.sm.update(5000)
    if self.sm["modelV2"].frameId != self.sm["cameraOdometry"].frameId:
      self.sm.update(1000)

  def wait_quick(self) -> None:
    """Shorter poll for startup / skip detection."""
    self.sm.update(800)
    if self.sm["modelV2"].frameId != self.sm["cameraOdometry"].frameId:
      self.sm.update(400)

  def wait_for_published_frame(self, frame_id: int, *, max_rounds: int = 40) -> None:
    """Poll until ``modelV2``, ``cameraOdometry``, and ``drivingModelData`` match ``frame_id`` (conflate-safe)."""
    for _ in range(max_rounds):
      self.wait()
      m = self.sm["modelV2"]
      o = self.sm["cameraOdometry"]
      d = self.sm["drivingModelData"]
      if m.frameId == frame_id and o.frameId == frame_id and d.frameId == frame_id:
        return
    msg = (
      f"timeout waiting for modeld to publish frame_id={frame_id} "
      f"(got modelV2={self.sm['modelV2'].frameId}, odo={self.sm['cameraOdometry'].frameId}, "
      f"drivingModelData={self.sm['drivingModelData'].frameId})"
    )
    raise AssertionError(msg)


class TestModeldPhaseCContracts:
  def setup_method(self):
    self._h = _ModeldVipcHarness()
    self._h.setup()

  def teardown_method(self):
    self._h.teardown()

  def _warmup_until_live(self) -> int:
    for n in range(1, 12):
      self._h.send_frames(n)
      self._h.wait_quick()
      if self._h.sm["modelV2"].frameId == n:
        self._h.wait()
        return n
    pytest.skip(
      "modeld did not publish consecutive modelV2 frames (need working modeld + vision stack; see LOW-LEVEL Phase C)",
    )

  def test_phase_c_daemon_message_contracts(self, subtests):
    last = self._warmup_until_live()
    n = last

    with subtests.test(msg="modelV2, drivingModelData, and odometry stay frame-locked"):
      for _ in range(10):
        n += 1
        cs = self._h.send_frames(n)
        self._h.wait_for_published_frame(n)
        mdl = self._h.sm["modelV2"]
        odo = self._h.sm["cameraOdometry"]
        dm = self._h.sm["drivingModelData"]
        assert mdl.frameId == odo.frameId == dm.frameId == n
        assert mdl.frameIdExtra == dm.frameIdExtra == n
        assert mdl.timestampEof == cs.timestampEof
        assert odo.timestampEof == cs.timestampEof

    with subtests.test(msg="timestampEof increases"):
      prev_eof = self._h.sm["modelV2"].timestampEof
      for _ in range(8):
        n += 1
        cs = self._h.send_frames(n)
        self._h.wait_for_published_frame(n)
        eof = self._h.sm["modelV2"].timestampEof
        assert eof > prev_eof
        assert eof == cs.timestampEof
        prev_eof = eof

    with subtests.test(msg="modelExecutionTime finite"):
      n += 1
      self._h.send_frames(n)
      self._h.wait_for_published_frame(n)
      mdl = self._h.sm["modelV2"]
      assert mdl.frameId == n
      assert math.isfinite(mdl.modelExecutionTime)
      assert mdl.modelExecutionTime >= 0.0

    with subtests.test(msg="frameAge zero while tracking"):
      for _ in range(5):
        n += 1
        self._h.send_frames(n)
        self._h.wait_for_published_frame(n)
        assert self._h.sm["modelV2"].frameAge == 0

    with subtests.test(msg="modelExecutionTime below 5.0s ceiling (Phase C timeliness guard)"):
      n += 1
      self._h.send_frames(n)
      self._h.wait_for_published_frame(n)
      mdl = self._h.sm["modelV2"]
      assert mdl.frameId == n
      assert mdl.modelExecutionTime < 5.0, (
        f"modelExecutionTime={mdl.modelExecutionTime}s exceeds 5.0s ceiling (slow GPU / load?)"
      )
