#!/usr/bin/env python3
import numpy as np
from selfdrive.controls.lib.vehicle_model import VehicleModel, calc_slip_factor
from selfdrive.car.honda.interface import CarInterface

def mpc_path_prediction(sa, u, psi_0, dt, VM):
  # sa and u needs to be numpy arrays
  sa_w = sa * np.pi / 180. / VM.CP.steerRatio
  x = np.zeros(len(sa))
  y = np.zeros(len(sa))
  psi = np.ones(len(sa)) * psi_0

  for i in range(0, len(sa)-1):
    x[i+1] = x[i] + np.cos(psi[i]) * u[i] * dt
    y[i+1] = y[i] + np.sin(psi[i]) * u[i] * dt
    psi[i+1] = psi[i] + sa_w[i] * u[i] * dt * VM.curvature_factor(u[i])

  return x, y, psi


def model_path_prediction(sa, u, psi_0, dt, VM):
  # steady state solution
  sa_r = sa * np.pi / 180.
  x = np.zeros(len(sa))
  y = np.zeros(len(sa))
  psi = np.ones(len(sa)) * psi_0
  for i in range(0, len(sa)-1):

    out = VM.steady_state_sol(sa_r[i], u[i])

    x[i+1] = x[i] + np.cos(psi[i]) * u[i] * dt - np.sin(psi[i]) * out[0] * dt
    y[i+1] = y[i] + np.sin(psi[i]) * u[i] * dt + np.cos(psi[i]) * out[0] * dt
    psi[i+1] = psi[i] + out[1] * dt

  return x, y, psi

if __name__ == "__main__":
  CP = CarInterface.get_params("HONDA CIVIC 2016 TOURING")
  print(CP)
  VM = VehicleModel(CP)
  print(VM.steady_state_sol(.1, 0.15))
  print(calc_slip_factor(VM))
  print("Curv", VM.curvature_factor(30.))

  dt = 0.05
  st = 20
  u = np.ones(st) * 1.
  sa = np.ones(st) * 1.

  out = mpc_path_prediction(sa, u, dt, VM)
  out_model = model_path_prediction(sa, u, dt, VM)

  print("mpc", out)
  print("model", out_model)
