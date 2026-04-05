"""Tests for LaneDepartureWarning blinker cooldown timing.

Validates that recent_blinker uses DT_MDL (model rate, 20Hz)
instead of DT_CTRL (control rate, 100Hz), since sm.frame in
plannerd advances at the model rate.
"""
import types

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning


# ---------------------------------------------------------------------------
# Helper: build minimal mock objects that LDW.update() expects
# ---------------------------------------------------------------------------
def _make_cs(v_ego: float = 20.0, left_blinker: bool = False,
             right_blinker: bool = False):
  return types.SimpleNamespace(
    vEgo=v_ego,
    leftBlinker=left_blinker,
    rightBlinker=right_blinker,
  )


def _make_cc(lat_active: bool = False):
  return types.SimpleNamespace(latActive=lat_active)


def _make_model(l_change_prob: float = 0.0, r_change_prob: float = 0.0):
  """Build a modelV2-like mock with lane lines and desire prediction.

  Desire enum indices: none=0, turnLeft=1, turnRight=2,
  laneChangeLeft=3, laneChangeRight=4, keepLeft=5, keepRight=6.
  """
  desire_prediction = [0.0, 0.0, 0.0, l_change_prob, r_change_prob, 0.0, 0.0]

  left_line = types.SimpleNamespace(y=[0.0])    # > -(1.08 + 0.04) => close
  right_line = types.SimpleNamespace(y=[0.5])   # < (1.08 - 0.04) => close
  dummy_line = types.SimpleNamespace(y=[0.0])

  return types.SimpleNamespace(
    meta=types.SimpleNamespace(desirePrediction=desire_prediction),
    laneLineProbs=[0.0, 0.9, 0.9, 0.0],
    laneLines=[dummy_line, left_line, right_line, dummy_line],
  )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRecentBlinkerTiming:
  """Verify that the blinker cooldown uses the correct time step (DT_MDL)."""

  def test_dt_mdl_is_used_not_dt_ctrl(self):
    """Core regression test: DT_MDL (0.05) must be the multiplier, not DT_CTRL (0.01).

    With DT_MDL=0.05, 5s cooldown = 100 frames.
    With DT_CTRL=0.01, 5s cooldown would be 500 frames.

    At frame 101 after blinker:
      DT_MDL:  101 * 0.05 = 5.05s >= 5.0 => cooldown expired => LDW triggers
      DT_CTRL: 101 * 0.01 = 1.01s <  5.0 => cooldown active  => LDW suppressed (BUG)
    """
    ldw = LaneDepartureWarning()
    cs_blinker_on = _make_cs(left_blinker=True)
    cs_no_blinker = _make_cs()
    cc = _make_cc()
    model = _make_model(l_change_prob=0.5, r_change_prob=0.0)

    # Frame 0: blinker is on -> records last_blinker_frame
    ldw.update(0, model, cs_blinker_on, cc)
    assert ldw.last_blinker_frame == 0

    # At frame 101 (just past DT_MDL cooldown): blinker cooldown should have expired
    ldw.update(101, model, cs_no_blinker, cc)
    assert ldw.left is True, (
      "LDW should trigger after 101 model frames (~5.05s with DT_MDL); "
      + "if this fails, DT_CTRL is likely still in use"
    )

  def test_blinker_still_active_within_cooldown(self):
    """Within the 5s cooldown window, LDW must be suppressed."""
    ldw = LaneDepartureWarning()
    cs_blinker_on = _make_cs(left_blinker=True)
    cs_no_blinker = _make_cs()
    cc = _make_cc()
    model = _make_model(l_change_prob=0.5)

    # Activate blinker at frame 0
    ldw.update(0, model, cs_blinker_on, cc)

    # At frame 50: (50 - 0) * 0.05 = 2.5s < 5.0s => still in cooldown
    ldw.update(50, model, cs_no_blinker, cc)
    assert ldw.left is False, "LDW should be suppressed during blinker cooldown"

  def test_blinker_cooldown_expires_at_boundary(self):
    """At exactly 100 frames (5.0s with DT_MDL), the < comparison means cooldown is over."""
    ldw = LaneDepartureWarning()
    cs_blinker_on = _make_cs(left_blinker=True)
    cs_no_blinker = _make_cs()
    cc = _make_cc()
    model = _make_model(l_change_prob=0.5)

    # Blinker at frame 0
    ldw.update(0, model, cs_blinker_on, cc)

    # Frame 99: 99 * 0.05 = 4.95s < 5.0 => still cooling down
    ldw.update(99, model, cs_no_blinker, cc)
    assert ldw.left is False, "At 4.95s, cooldown should still be active"

    # Frame 100: 100 * 0.05 = 5.0s, NOT < 5.0 => cooldown expired
    ldw.update(100, model, cs_no_blinker, cc)
    assert ldw.left is True, "At exactly 5.0s, cooldown should have expired"

  def test_blinker_cooldown_math_dt_ctrl_vs_dt_mdl(self):
    """Numerical demonstration: DT_CTRL produces incorrect elapsed time at model frame rate."""
    elapsed_with_ctrl = 101 * DT_CTRL  # 1.01s
    elapsed_with_mdl = 101 * DT_MDL    # 5.05s

    assert elapsed_with_ctrl < 5.0, "DT_CTRL gives 1.01s at 101 frames - cooldown incorrectly active"
    assert elapsed_with_mdl >= 5.0, "DT_MDL gives 5.05s at 101 frames - cooldown correctly expired"

  def test_right_blinker_cooldown(self):
    """Right blinker should also respect the DT_MDL-based cooldown."""
    ldw = LaneDepartureWarning()
    cs_blinker_on = _make_cs(right_blinker=True)
    cs_no_blinker = _make_cs()
    cc = _make_cc()
    model = _make_model(r_change_prob=0.5)

    ldw.update(0, model, cs_blinker_on, cc)

    # Within cooldown: 50 * 0.05 = 2.5s < 5.0
    ldw.update(50, model, cs_no_blinker, cc)
    assert ldw.right is False

    # After cooldown: 101 * 0.05 = 5.05s >= 5.0
    ldw.update(101, model, cs_no_blinker, cc)
    assert ldw.right is True

  def test_no_ldw_when_speed_too_low(self):
    """LDW should not trigger when vehicle speed is below minimum (~13.86 m/s)."""
    ldw = LaneDepartureWarning()
    cs_slow = _make_cs(v_ego=5.0)
    cc = _make_cc()
    model = _make_model(l_change_prob=0.5)

    ldw.update(200, model, cs_slow, cc)
    assert ldw.left is False
    assert ldw.right is False

  def test_no_ldw_when_lat_active(self):
    """LDW should not trigger when lateral control is active."""
    ldw = LaneDepartureWarning()
    cs = _make_cs()
    cc = _make_cc(lat_active=True)
    model = _make_model(l_change_prob=0.5)

    ldw.update(200, model, cs, cc)
    assert ldw.left is False
    assert ldw.right is False
