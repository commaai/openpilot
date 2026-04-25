"""
``fill_model_msg`` FCW / ``hardBrakePredicted`` path (rolling ``PublishState`` buffers).

Maps: R1.
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.constants import Meta, ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_hard_brake_predicted_after_buffers_fill_with_high_meta_probs():
  """``prev_brake_*`` ring buffers must fill before ``hardBrakePredicted`` can go true."""
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  hb5 = np.full(5, 0.99, dtype=np.float32)
  hb3 = np.full(5, 0.99, dtype=np.float32)
  outs["meta"][0, Meta.HARD_BRAKE_5] = hb5
  outs["meta"][0, Meta.HARD_BRAKE_3] = hb3
  state = PublishState()

  for _ in range(max(ModelConstants.FCW_5MS2_PROBS_WIDTH, ModelConstants.FCW_3MS2_PROBS_WIDTH) + 1):
    fill_model_msg(
      base_msg=base,
      extended_msg=ext,
      net_output_data=outs,
      v_ego=1.0,
      delay=0.0,
      publish_state=state,
      vipc_frame_id=40,
      vipc_frame_id_extra=40,
      frame_id=40,
      frame_drop=0.0,
      timestamp_eof=1,
      model_execution_time=0.01,
      valid=True,
    )

  assert ext.modelV2.meta.hardBrakePredicted is True
