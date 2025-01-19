import pytest
import numpy as np
from openpilot.common.params import Params

from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import WMACConstants, SNG_State
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController, TRAJECTORY_SIZE, STOP_AND_GO_FRAME

class MockInterp:
  def __call__(self, x, xp, fp):
    return np.interp(x, xp, fp)

class MockCarState:
  def __init__(self, v_ego=0., standstill=False, left_blinker=False, right_blinker=False):
    self.vEgo = v_ego
    self.standstill = standstill
    self.leftBlinker = left_blinker
    self.rightBlinker = right_blinker

class MockLeadOne:
  def __init__(self, status=False, d_rel=0):
    self.status = status
    self.dRel = d_rel

class MockModelData:
  def __init__(self, x_vals=None, positions=None):
    self.orientation = type('Orientation', (), {'x': x_vals})()
    self.position = type('Position', (), {'x': positions})()

class MockControlState:
  def __init__(self, v_cruise=0):
    self.vCruise = v_cruise

@pytest.fixture
def interp(monkeypatch):
  mock_interp = MockInterp()
  monkeypatch.setattr('openpilot.common.numpy_fast.interp', mock_interp)
  return mock_interp

@pytest.fixture
def controller(interp):
  params = Params()
  params.put_bool("DynamicExperimentalControl", True)
  return DynamicExperimentalController()

def test_initial_state(controller):
  """Test initial state of the controller"""
  assert controller._mode == 'acc'
  assert not controller._has_lead
  assert not controller._has_standstill
  assert controller._sng_state == SNG_State.off
  assert not controller._has_lead_filtered
  assert not controller._has_slow_down
  assert not controller._has_dangerous_ttc
  assert not controller._has_mpc_fcw

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_standstill_detection(controller, has_radar):
  """Test standstill detection and state transitions"""
  car_state = MockCarState(standstill=True)
  lead_one = MockLeadOne()
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[150] * TRAJECTORY_SIZE)
  controls_state = MockControlState()

  # Test transition to standstill
  controller.update(not has_radar, car_state, lead_one, md, controls_state)
  assert controller._sng_state == SNG_State.stopped
  assert controller.get_mpc_mode() == 'blended'

  # Test transition from standstill to moving
  car_state.standstill = False
  controller.update(not has_radar, car_state, lead_one, md, controls_state)
  assert controller._sng_state == SNG_State.going

  # Test complete transition to normal driving
  for _ in range(STOP_AND_GO_FRAME + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)
  assert controller._sng_state == SNG_State.off

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_lead_detection(controller, has_radar):
  """Test lead vehicle detection and filtering"""
  car_state = MockCarState(v_ego=20)  # 72 kph
  lead_one = MockLeadOne(status=True, d_rel=50)  # Safe distance
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[150] * TRAJECTORY_SIZE)
  controls_state = MockControlState(v_cruise=72)

  # Let moving average stabilize
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_lead_filtered
  expected_mode = 'acc' if has_radar else 'blended'
  assert controller.get_mpc_mode() == expected_mode

  # Test lead loss detection
  lead_one.status = False
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert not controller._has_lead_filtered

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_slow_down_detection(controller, has_radar):
  """Test slow down detection based on trajectory"""
  car_state = MockCarState(v_ego=10/3.6)  # 10 kph
  lead_one = MockLeadOne()
  x_vals = [0] * TRAJECTORY_SIZE
  positions = [20] * TRAJECTORY_SIZE  # Position within slow down threshold
  md = MockModelData(x_vals=x_vals, positions=positions)
  controls_state = MockControlState(v_cruise=30)

  # Test slow down detection
  for _ in range(WMACConstants.SLOW_DOWN_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_slow_down
  assert controller.get_mpc_mode() == 'blended'

  # Test slow down recovery
  positions = [200] * TRAJECTORY_SIZE  # Position outside slow down threshold
  md = MockModelData(x_vals=x_vals, positions=positions)
  for _ in range(WMACConstants.SLOW_DOWN_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert not controller._has_slow_down

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_dangerous_ttc_detection(controller, has_radar):
  """Test Time-To-Collision detection and handling"""
  car_state = MockCarState(v_ego=10)  # 36 kph
  lead_one = MockLeadOne(status=True)
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[150] * TRAJECTORY_SIZE)
  controls_state = MockControlState(v_cruise=36)

  # First establish normal conditions with lead
  lead_one.dRel = 100  # Safe distance
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):  # First establish lead detection
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_lead_filtered  # Verify lead is detected

  # Now test dangerous TTC detection
  lead_one.dRel = 10  # 10m distance - should trigger dangerous TTC
  # TTC = dRel/vEgo = 10/10 = 1s (which is less than DANGEROUS_TTC = 2.3s)

  # Need to update multiple times to allow the weighted average to stabilize
  for _ in range(WMACConstants.DANGEROUS_TTC_WINDOW_SIZE * 2):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_dangerous_ttc, "TTC of 1s should be considered dangerous"
  expected_mode = 'acc' if has_radar else 'blended'
  assert controller.get_mpc_mode() == expected_mode, f"Should be in [{expected_mode}] mode with dangerous TTC"

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_mode_transitions(controller, has_radar):
  """Test comprehensive mode transitions under different conditions"""
  # Initialize with normal driving conditions
  car_state = MockCarState(v_ego=25)  # 90 kph
  lead_one = MockLeadOne(status=False)
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[200] * TRAJECTORY_SIZE)
  controls_state = MockControlState(v_cruise=100)

  def stabilize_filters():
    """Helper to let all moving averages stabilize"""
    for _ in range(max(WMACConstants.LEAD_WINDOW_SIZE, WMACConstants.SLOW_DOWN_WINDOW_SIZE,
                       WMACConstants.DANGEROUS_TTC_WINDOW_SIZE, WMACConstants.MPC_FCW_WINDOW_SIZE) + 1):
      controller.update(not has_radar, car_state, lead_one, md, controls_state)

  # Test 1: Normal driving -> ACC mode
  stabilize_filters()
  assert controller.get_mpc_mode() == 'acc', "Should be in ACC mode under normal driving conditions"

  # Test 2: Standstill -> Blended mode
  car_state.standstill = True
  controller.update(not has_radar, car_state, lead_one, md, controls_state)
  assert controller.get_mpc_mode() == 'blended', "Should be in blended mode during standstill"

  # Test 3: Lead car appears -> ACC mode
  car_state = MockCarState(v_ego=20)  # Reset car state
  lead_one.status = True
  lead_one.dRel = 50  # Safe distance
  stabilize_filters()
  assert not controller._has_dangerous_ttc, "Should not have dangerous TTC"
  assert controller.get_mpc_mode() == 'acc', "Should be in ACC mode with safe lead distance"

  # Test 4: Dangerous TTC -> Blended mode
  car_state = MockCarState(v_ego=20)  # 72 kph
  lead_one.status = True
  lead_one.dRel = 50  # First establish normal lead detection

  # First establish lead detection
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_lead_filtered  # Verify lead is detected

  # Now create dangerous TTC condition
  lead_one.dRel = 20  # This creates a TTC of 1s, well below DANGEROUS_TTC

  for _ in range(WMACConstants.DANGEROUS_TTC_WINDOW_SIZE * 2):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_dangerous_ttc, "Should detect dangerous TTC condition"
  expected_mode = 'acc' if has_radar else 'blended'
  assert controller.get_mpc_mode() == expected_mode, f"Should be in [{expected_mode}] mode with dangerous TTC"

@pytest.mark.parametrize("has_radar", [True, False], ids=["with_radar", "without_radar"])
def test_mpc_fcw_handling(controller, has_radar):
  """Test MPC FCW crash count handling and mode transitions"""
  car_state = MockCarState(v_ego=20)
  lead_one = MockLeadOne()
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[150] * TRAJECTORY_SIZE)
  controls_state = MockControlState(v_cruise=72)

  # Test FCW activation
  controller.set_mpc_fcw_crash_cnt(5)
  for _ in range(WMACConstants.MPC_FCW_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert controller._has_mpc_fcw
  assert controller.get_mpc_mode() == 'blended'

  # Test FCW recovery
  controller.set_mpc_fcw_crash_cnt(0)
  for _ in range(WMACConstants.MPC_FCW_WINDOW_SIZE + 1):
    controller.update(not has_radar, car_state, lead_one, md, controls_state)

  assert not controller._has_mpc_fcw

def test_radar_unavailable_handling(controller):
  """Test behavior transitions between radar available and unavailable states"""
  car_state = MockCarState(v_ego=27.78)  # 100 kph
  lead_one = MockLeadOne(status=True, d_rel=50)
  md = MockModelData(x_vals=[0] * TRAJECTORY_SIZE, positions=[150] * TRAJECTORY_SIZE)
  controls_state = MockControlState(v_cruise=100)

  # Test with radar available
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):
    controller.update(False, car_state, lead_one, md, controls_state)
  radar_mode = controller.get_mpc_mode()

  # Test with radar unavailable
  for _ in range(WMACConstants.LEAD_WINDOW_SIZE + 1):
    controller.update(True, car_state, lead_one, md, controls_state)
  radarless_mode = controller.get_mpc_mode()

  assert radar_mode is not None
  assert radarless_mode is not None
