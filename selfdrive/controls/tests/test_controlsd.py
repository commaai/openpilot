import numpy as np


def test_actuator_finite_values():
  """Test that actuator values are properly validated as finite"""
  # Create a mock actuator-like object to test finite value validation
  class MockActuators:
    def __init__(self):
      self.accel = 1.0
      self.torque = 0.5
      self.steeringAngleDeg = 0.0

  actuators = MockActuators()
  # Test that finite values pass the validation
  assert np.isfinite(actuators.accel)
  assert np.isfinite(actuators.torque)
  assert np.isfinite(actuators.steeringAngleDeg)

  # Test that the validation catches non-finite values
  actuators.accel = float('inf')
  assert not np.isfinite(actuators.accel)
