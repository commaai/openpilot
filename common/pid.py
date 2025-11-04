import numpy as np
from numbers import Number

class PIDController:
  """A PID controller with speed-dependent gains and integrator anti-windup."""
  
  # Default limits for control output
  DEFAULT_POSITIVE_LIMIT = 1e308
  DEFAULT_NEGATIVE_LIMIT = -1e308

  def __init__(self, proportional_gain, integral_gain, derivative_gain=0., 
               positive_limit=1e308, negative_limit=-1e308, rate=100):
    """
    Initialize the PID controller.
    
    Args:
        proportional_gain: Proportional gain (can be a scalar or [speed_points, gains] pair)
        integral_gain: Integral gain (can be a scalar or [speed_points, gains] pair) 
        derivative_gain: Derivative gain (can be a scalar or [speed_points, gains] pair)
        positive_limit: Maximum output limit
        negative_limit: Minimum output limit
        rate: Update rate in Hz
    """
    self._proportional_gain = self._normalize_gain_format(proportional_gain)
    self._integral_gain = self._normalize_gain_format(integral_gain)
    self._derivative_gain = self._normalize_gain_format(derivative_gain)

    self.set_output_limits(positive_limit, negative_limit)

    self.integration_time_step = 1.0 / rate
    self.current_speed = 0.0

    self.reset()

  @staticmethod
  def _normalize_gain_format(gain):
    """Normalize gain format to be either a scalar or a [speed_points, gains] pair."""
    if isinstance(gain, Number):
      return [[0], [gain]]
    return gain

  @property
  def proportional_gain_at_current_speed(self):
    """Get the proportional gain interpolated at the current speed."""
    return np.interp(self.current_speed, self._proportional_gain[0], self._proportional_gain[1])

  @property
  def integral_gain_at_current_speed(self):
    """Get the integral gain interpolated at the current speed."""
    return np.interp(self.current_speed, self._integral_gain[0], self._integral_gain[1])

  @property
  def derivative_gain_at_current_speed(self):
    """Get the derivative gain interpolated at the current speed."""
    return np.interp(self.current_speed, self._derivative_gain[0], self._derivative_gain[1])

  def reset(self):
    """Reset the internal state of the PID controller."""
    self.proportional_term = 0.0
    self.integral_term = 0.0
    self.derivative_term = 0.0
    self.feedforward_term = 0.0
    self.output = 0

  def set_output_limits(self, positive_limit, negative_limit):
    """Set the output limits for the controller."""
    self.positive_limit = positive_limit
    self.negative_limit = negative_limit

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False):
    """
    Update the PID controller with new error values.
    
    Args:
        error: Current error value
        error_rate: Rate of change of error (derivative term input)
        speed: Current speed (used for gain scheduling)
        feedforward: Feedforward control component
        freeze_integrator: Whether to freeze the integral term integration
        
    Returns:
        The control output after applying all PID terms and limits
    """
    self.current_speed = speed
    
    # Calculate individual PID terms
    self.proportional_term = self.proportional_gain_at_current_speed * float(error)
    self.derivative_term = self.derivative_gain_at_current_speed * error_rate
    self.feedforward_term = feedforward

    # Update integral term with anti-windup protection
    if not freeze_integrator:
      new_integral = self.integral_term + self.integral_gain_at_current_speed * self.integration_time_step * error

      # Anti-windup: don't allow windup when output is already at limit
      proposed_output = self.proportional_term + new_integral + self.derivative_term + self.feedforward_term
      integral_upper_bound = self.integral_term if proposed_output > self.positive_limit else self.positive_limit
      integral_lower_bound = self.integral_term if proposed_output < self.negative_limit else self.negative_limit
      self.integral_term = np.clip(new_integral, integral_lower_bound, integral_upper_bound)
    
    # Calculate final control output
    control_output = (self.proportional_term + self.integral_term + 
                     self.derivative_term + self.feedforward_term)
    
    # Apply output limits
    self.output = np.clip(control_output, self.negative_limit, self.positive_limit)
    return self.output
