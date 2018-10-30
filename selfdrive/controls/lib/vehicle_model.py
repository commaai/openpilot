#!/usr/bin/env python
import numpy as np
from numpy.linalg import solve

# dynamic bycicle model from "The Science of Vehicle Dynamics (2014), M. Guiggiani"##
# Xdot = A*X + B*U
# where X = [v, r], with v and r lateral speed and rotational speed, respectively
# and U is the steering angle (controller input)
#
# A depends on longitudinal speed, u, and vehicle parameters CP


def create_dyn_state_matrices(u, VM):
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
  # kinematic solution, useful when speed ~ 0
  K = np.zeros((2, 1))
  K[0, 0] = VM.aR / VM.sR / VM.l * u
  K[1, 0] = 1. / VM.sR / VM.l * u
  return K * sa


def dyn_ss_sol(sa, u, VM):
  # Dynamic solution, useful when speed > 0
  A, B = create_dyn_state_matrices(u, VM)
  return - solve(A, B) * sa


def calc_slip_factor(VM):
  # the slip factor is a measure of how the curvature changes with speed
  # it's positive for Oversteering vehicle, negative (usual case) otherwise
  return VM.m * (VM.cF * VM.aF - VM.cR * VM.aR) / (VM.l**2 * VM.cF * VM.cR)


class VehicleModel(object):
  def __init__(self, CP, init_state=np.asarray([[0.], [0.]])):
    self.dt = 0.1
    lookahead = 2.    # s
    self.steps = int(lookahead / self.dt)
    self.update_state(init_state)
    self.state_pred = np.zeros((self.steps, self.state.shape[0]))
    self.CP = CP
    # for math readability, convert long names car params into short names
    self.m = CP.mass
    self.j = CP.rotationalInertia
    self.l = CP.wheelbase
    self.aF = CP.centerToFront
    self.aR = CP.wheelbase - CP.centerToFront
    self.cF = CP.tireStiffnessFront
    self.cR = CP.tireStiffnessRear
    self.sR = CP.steerRatio
    self.chi = CP.steerRatioRear

  def update_state(self, state):
    self.state = state

  def steady_state_sol(self, sa, u):
    # if the speed is too small we can't use the dynamic model
    # (tire slip is undefined), we then use the kinematic model
    if u > 0.1:
      return dyn_ss_sol(sa, u, self)
    else:
      return kin_ss_sol(sa, u, self)

  def calc_curvature(self, sa, u):
    # this formula can be derived from state equations in steady state conditions
    return self.curvature_factor(u) * sa / self.sR

  def curvature_factor(self, u):
    sf = calc_slip_factor(self)
    return (1. - self.chi)/(1. - sf * u**2) / self.l

  def get_steer_from_curvature(self, curv, u):
    return curv * self.sR * 1.0 / self.curvature_factor(u)

  def state_prediction(self, sa, u):
    # U is the matrix of the controls
    # u is the long speed
    A, B = create_dyn_state_matrices(u, self)
    return np.matmul((A * self.dt + np.eye(2)), self.state) + B * sa * self.dt

  def yaw_rate(self, sa, u):
    return self.calc_curvature(sa, u) * u

if __name__ == '__main__':
  from selfdrive.car.honda.interface import CarInterface
  CP = CarInterface.get_params("HONDA CIVIC 2016 TOURING", {})
  #from selfdrive.car.hyundai.interface import CarInterface
  #CP = CarInterface.get_params("HYUNDAI SANTA FE UNLIMITED 2019", {})
  #print CP
  VM = VehicleModel(CP)
  #print VM.steady_state_sol(.1, 0.15)
  #print calc_slip_factor(VM)
  print VM.yaw_rate(1.*np.pi/180, 32.) * 180./np.pi
  #print VM.curvature_factor(32)
