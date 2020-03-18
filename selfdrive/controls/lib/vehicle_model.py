#!/usr/bin/env python3
import numpy as np
from numpy.linalg import solve

"""
Dynamic bycicle model from "The Science of Vehicle Dynamics (2014), M. Guiggiani"

The state is x = [v, r]^T
with v lateral speed [m/s], and r rotational speed [rad/s]

The input u is the steering angle [rad]

The system is defined by
x_dot = A*x + B*u

A depends on longitudinal speed, u [m/s], and vehicle parameters CP
"""


def create_dyn_state_matrices(u, VM):
  """Returns the A and B matrix for the dynamics system

  Args:
    u: Vehicle speed [m/s]
    VM: Vehicle model

  Returns:
    A tuple with the 2x2 A matrix, and 2x1 B matrix

  Parameters in the vehicle model:
    cF: Tire stiffnes Front [N/rad]
    cR: Tire stiffnes Front [N/rad]
    aF: Distance from CG to front wheels [m]
    aR: Distance from CG to rear wheels [m]
    m: Mass [kg]
    j: Rotational inertia [kg m^2]
    sR: Steering ratio [-]
    chi: Steer ratio rear [-]
  """
  A = np.zeros((2, 2))
  B = np.zeros((2, 1))
  A[0, 0] = - (VM.cF + VM.cR) / (VM.m * u)
  A[0, 1] = - (VM.cF * VM.aF - VM.cR * VM.aR) / (VM.m * u) - u
  A[1, 0] = - (VM.cF * VM.aF - VM.cR * VM.aR) / (VM.j * u)
  A[1, 1] = - (VM.cF * VM.aF**2 + VM.cR * VM.aR**2) / (VM.j * u)
  B[0, 0] = (VM.cF + VM.chi * VM.cR) / VM.m / VM.sR
  B[1, 0] = (VM.cF * VM.aF - VM.chi * VM.cR * VM.aR) / VM.j / VM.sR
  return A, B


def kin_ss_sol(sa, u, VM):
  """Calculate the steady state solution at low speeds
  At low speeds the tire slip is undefined, so a kinematic
  model is used.

  Args:
    sa: Steering angle [rad]
    u: Speed [m/s]
    VM: Vehicle model

  Returns:
    2x1 matrix with steady state solution
  """
  K = np.zeros((2, 1))
  K[0, 0] = VM.aR / VM.sR / VM.l * u
  K[1, 0] = 1. / VM.sR / VM.l * u
  return K * sa


def dyn_ss_sol(sa, u, VM):
  """Calculate the steady state solution when x_dot = 0,
  Ax + Bu = 0 => x = A^{-1} B u

  Args:
    sa: Steering angle [rad]
    u: Speed [m/s]
    VM: Vehicle model

  Returns:
    2x1 matrix with steady state solution
  """
  A, B = create_dyn_state_matrices(u, VM)
  return -solve(A, B) * sa


def calc_slip_factor(VM):
  """The slip factor is a measure of how the curvature changes with speed
  it's positive for Oversteering vehicle, negative (usual case) otherwise.
  """
  return VM.m * (VM.cF * VM.aF - VM.cR * VM.aR) / (VM.l**2 * VM.cF * VM.cR)


class VehicleModel():
  def __init__(self, CP):
    """
    Args:
      CP: Car Parameters
    """
    # for math readability, convert long names car params into short names
    self.m = CP.mass
    self.j = CP.rotationalInertia
    self.l = CP.wheelbase
    self.aF = CP.centerToFront
    self.aR = CP.wheelbase - CP.centerToFront
    self.chi = CP.steerRatioRear

    self.cF_orig = CP.tireStiffnessFront
    self.cR_orig = CP.tireStiffnessRear
    self.update_params(1.0, CP.steerRatio)

  def update_params(self, stiffness_factor, steer_ratio):
    """Update the vehicle model with a new stiffness factor and steer ratio"""
    self.cF = stiffness_factor * self.cF_orig
    self.cR = stiffness_factor * self.cR_orig
    self.sR = steer_ratio

  def steady_state_sol(self, sa, u):
    """Returns the steady state solution.

    If the speed is too small we can't use the dynamic model (tire slip is undefined),
    we then have to use the kinematic model

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]

    Returns:
      2x1 matrix with steady state solution (lateral speed, rotational speed)
    """
    if u > 0.1:
      return dyn_ss_sol(sa, u, self)
    else:
      return kin_ss_sol(sa, u, self)

  def calc_curvature(self, sa, u):
    """Returns the curvature. Multiplied by the speed this will give the yaw rate.

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]

    Returns:
      Curvature factor [1/m]
    """
    return self.curvature_factor(u) * sa / self.sR

  def curvature_factor(self, u):
    """Returns the curvature factor.
    Multiplied by wheel angle (not steering wheel angle) this will give the curvature.

    Args:
      u: Speed [m/s]

    Returns:
      Curvature factor [1/m]
    """
    sf = calc_slip_factor(self)
    return (1. - self.chi) / (1. - sf * u**2) / self.l

  def get_steer_from_curvature(self, curv, u):
    """Calculates the required steering wheel angle for a given curvature

    Args:
      curv: Desired curvature [1/m]
      u: Speed [m/s]

    Returns:
      Steering wheel angle [rad]
    """

    return curv * self.sR * 1.0 / self.curvature_factor(u)

  def get_steer_from_yaw_rate(self, yaw_rate, u):
    """Calculates the required steering wheel angle for a given yaw_rate

    Args:
      yaw_rate: Desired yaw rate [rad/s]
      u: Speed [m/s]

    Returns:
      Steering wheel angle [rad]
    """
    curv = yaw_rate / u
    return self.get_steer_from_curvature(curv, u)

  def yaw_rate(self, sa, u):
    """Calculate yaw rate

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]

    Returns:
      Yaw rate [rad/s]
    """
    return self.calc_curvature(sa, u) * u


if __name__ == '__main__':
  import math
  from selfdrive.car.honda.interface import CarInterface
  from selfdrive.car.honda.values import CAR

  CP = CarInterface.get_params(CAR.CIVIC)
  VM = VehicleModel(CP)
  print(VM.yaw_rate(math.radians(20), 10.))
