"""
``fill_model_msg`` frame id / age / drop-percent contracts (STP modeld path).

Uses shared fixtures only; does not modify upstream ``test_fill_model_msg.py``.

Aligns with ``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 (correctness) and **R1**.

Maps: R1.
"""

from __future__ import annotations

import pytest

from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_frame_age_zero_when_model_frame_not_ahead_of_vipc():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=10.0,
    delay=0.0,
    publish_state=state,
    vipc_frame_id=100,
    vipc_frame_id_extra=100,
    frame_id=50,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  assert ext.modelV2.frameAge == 0


def test_frame_age_matches_positive_gap():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=10.0,
    delay=0.0,
    publish_state=state,
    vipc_frame_id=40,
    vipc_frame_id_extra=41,
    frame_id=44,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  assert ext.modelV2.frameAge == 4


def test_frame_drop_percent_scaled_to_percentage():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=10.0,
    delay=0.0,
    publish_state=state,
    vipc_frame_id=1,
    vipc_frame_id_extra=1,
    frame_id=1,
    frame_drop=0.37,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  assert ext.modelV2.frameDropPerc == pytest.approx(37.0)
  assert base.drivingModelData.frameDropPerc == pytest.approx(37.0)
