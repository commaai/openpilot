#!/usr/bin/env python3
"""
Test to verify that removing the deprecated migration function doesn't break paramsd functionality
"""
from cereal import car
from openpilot.selfdrive.locationd.paramsd import retrieve_initial_vehicle_params, VehicleParamsLearner
from openpilot.selfdrive.locationd.models.car_kf import States


class Mock:
    """Simple mock class to replace unittest.mock.Mock for params test"""
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        # Return the default value that was set up for the test
        # In the test we want get() to return None to simulate no saved parameters
        return getattr(self, 'get_return_value', default)

    def put(self, key, value):
        self._data[key] = value

    def remove(self, key):
        pass  # Mock implementation doesn't need to do anything


def test_retrieve_initial_vehicle_params():
  """Test that the parameter retrieval still works correctly without the migration function"""

  # Create a mock CP (CarParams)
  CP = car.CarParams()
  CP.carFingerprint = "TEST_CAR"
  CP.steerRatio = 15.0
  CP.mass = 1500.0
  CP.rotationalInertia = 2000.0
  CP.centerToFront = 1.2
  CP.wheelbase = 2.7
  CP.tireStiffnessFront = 80000.0
  CP.tireStiffnessRear = 80000.0

  # Create mock params object
  mock_params = Mock()
  mock_params.get_return_value = None  # Simulate no saved parameters (should use defaults)

  # Test that retrieve_initial_vehicle_params returns default values when no saved params exist
  steer_ratio, stiffness_factor, angle_offset_deg, p_initial = retrieve_initial_vehicle_params(
    mock_params, CP, replay=False, debug=False
  )

  # Verify defaults are returned
  assert steer_ratio == CP.steerRatio  # Should be 15.0
  assert stiffness_factor == 1.0  # Should be reset to 1.0 when not in replay mode
  assert angle_offset_deg == 0.0  # Should be 0.0 when no saved params exist
  assert p_initial is None  # Should be None when no saved params exist


def test_vehicule_params_learner_initialization():
  """Test that VehicleParamsLearner initializes correctly with default values"""

  # Create a mock CP (CarParams)
  CP = car.CarParams()
  CP.carFingerprint = "TEST_CAR"
  CP.steerRatio = 15.0
  CP.mass = 1500.0
  CP.rotationalInertia = 2000.0
  CP.centerToFront = 1.2
  CP.wheelbase = 2.7
  CP.tireStiffnessFront = 80000.0
  CP.tireStiffnessRear = 80000.0

  # Test initialization with default values
  learner = VehicleParamsLearner(CP, CP.steerRatio, 1.0, 0.0)

  # Verify the learner was created successfully
  assert learner is not None
  assert abs(learner.x_initial[States.STEER_RATIO] - CP.steerRatio) < 0.001  # Check steer ratio is set correctly


if __name__ == "__main__":
  test_retrieve_initial_vehicle_params()
  test_vehicule_params_learner_initialization()
  print("All tests passed!")
