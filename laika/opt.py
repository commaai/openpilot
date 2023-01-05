import sympy
import numpy as np
from typing import List

from .constants import EARTH_ROTATION_RATE, SPEED_OF_LIGHT
from .helpers import ConstellationId
from .raw_gnss import GNSSMeasurement


def gauss_newton(fun, b, xtol=1e-8, max_n=25):
  for _ in range(max_n):
    # Compute function and jacobian on current estimate
    r, J = fun(b)

    # Update estimate
    delta = np.linalg.pinv(J) @ r
    b -= delta

    # Check step size for stopping condition
    if np.linalg.norm(delta) < xtol:
      break
  return b


def calc_pos_fix(measurements, posfix_functions=None, x0=None, no_weight=False, signal='C1C', min_measurements=6):
  '''
  Calculates gps fix using gauss newton method
  To solve the problem a minimal of 4 measurements are required.
    If Glonass is included 5 are required to solve for the additional free variable.
  returns:
  0 -> list with positions
  1 -> pseudorange errs
  '''
  if x0 is None:
    x0 = [0, 0, 0, 0, 0]

  if len(measurements) < min_measurements:
    return [],[]

  Fx_pos = pr_residual(measurements, posfix_functions, signal=signal, no_weight=no_weight, no_nans=True)
  x = gauss_newton(Fx_pos, x0)
  residual, _ = Fx_pos(x, no_weight=True)
  return x.tolist(), residual.tolist()


def calc_vel_fix(measurements, est_pos, velfix_function=None, v0=None, no_weight=False, signal='D1C', min_measurements=6):
  '''
  Calculates gps velocity fix using gauss newton method
  returns:
  0 -> list with velocities
  1 -> pseudorange_rate errs
  '''
  if v0 is None:
    v0 = [0, 0, 0, 0]

  if len(measurements) < min_measurements:
    return [], []

  Fx_vel = prr_residual(measurements, est_pos, velfix_function, signal=signal, no_weight=no_weight, no_nans=True)
  v = gauss_newton(Fx_vel, v0)
  residual, _ = Fx_vel(v, no_weight=True)
  return v.tolist(), residual.tolist()


def get_posfix_sympy_fun(constellation):
  # Unknowns
  x, y, z = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')
  bc = sympy.Symbol('bc')
  bg = sympy.Symbol('bg')
  zero_theta = sympy.Symbol('zero_theta')
  var = [x, y, z, bc, bg]

  # Knowns
  pr = sympy.Symbol('pr')
  sat_x, sat_y, sat_z = sympy.Symbol('sat_x'), sympy.Symbol('sat_y'), sympy.Symbol('sat_z')
  weight = sympy.Symbol('weight')

  theta = (EARTH_ROTATION_RATE * (pr - bc) / SPEED_OF_LIGHT)*zero_theta
  val = sympy.sqrt(
    (sat_x * sympy.cos(theta) + sat_y * sympy.sin(theta) - x) ** 2 +
    (sat_y * sympy.cos(theta) - sat_x * sympy.sin(theta) - y) ** 2 +
    (sat_z - z) ** 2
  )

  if constellation == ConstellationId.GLONASS:
    res = weight * (val - (pr - bc - bg))
  elif constellation == ConstellationId.GPS:
    res = weight * (val - (pr - bc))
  else:
    raise NotImplementedError(f"Constellation {constellation} not supported")

  res = [res] + [sympy.diff(res, v) for v in var]

  return sympy.lambdify([x, y, z, bc, bg, pr, zero_theta, sat_x, sat_y, sat_z, weight], res, modules=["numpy"])


def get_velfix_sympy_func():
  # implementing this without sympy.Matrix gives a 2x speedup at generation

  # knowns, receiver position, satellite position, satellite velocity
  ep_x, ep_y, ep_z = sympy.Symbol('ep_x'), sympy.Symbol('ep_y'), sympy.Symbol('ep_z')
  est_pos = np.array([ep_x, ep_y, ep_z])
  sp_x, sp_y, sp_z = sympy.Symbol('sp_x'), sympy.Symbol('sp_y'), sympy.Symbol('sp_z')
  sat_pos = np.array([sp_x, sp_y, sp_z])
  sv_x, sv_y, sv_z = sympy.Symbol('sv_x'), sympy.Symbol('sv_y'), sympy.Symbol('sv_z')
  sat_vel = np.array([sv_x, sv_y, sv_z])
  observables = sympy.Symbol('observables')
  weight = sympy.Symbol('weight')

  # unknown, receiver velocity
  v_x, v_y, v_z = sympy.Symbol('v_x'), sympy.Symbol('v_y'), sympy.Symbol('v_z')
  vel = np.array([v_x, v_y, v_z])
  vel_o = sympy.Symbol('vel_o')

  loss = sat_pos - est_pos
  loss /= sympy.sqrt(loss.dot(loss))

  nv = loss.dot(sat_vel - vel)
  ov = (observables - vel_o)
  res = (nv - ov)*weight

  res = [res] + [sympy.diff(res, v) for v in [v_x, v_y, v_z, vel_o]]

  return sympy.lambdify([
      ep_x, ep_y, ep_z, sp_x, sp_y, sp_z,
      sv_x, sv_y, sv_z, observables, weight,
      v_x, v_y, v_z, vel_o
    ],
    res, modules=["numpy"])


def pr_residual(measurements: List[GNSSMeasurement], posfix_functions=None, signal='C1C', no_weight=False, no_nans=False):

  if posfix_functions is None:
    posfix_functions = {constellation: get_posfix_sympy_fun(constellation) for constellation in (ConstellationId.GPS, ConstellationId.GLONASS)}

  def Fx_pos(inp, no_weight=no_weight):
    vals, gradients = [], []

    for meas in measurements:
      if signal in meas.observables_final and np.isfinite(meas.observables_final[signal]):
        pr = meas.observables_final[signal]
        sat_pos = meas.sat_pos_final
        zero_theta = 0
      elif signal in meas.observables and np.isfinite(meas.observables[signal]) and meas.processed:
        pr = meas.observables[signal]
        pr += meas.sat_clock_err * SPEED_OF_LIGHT
        sat_pos = meas.sat_pos
        zero_theta = 1
      else:
        if not no_nans:
          vals.append(np.nan)
          gradients.append(np.nan)
        continue

      w = 1.0 if no_weight or meas.observables_std[signal] == 0 else (1 / meas.observables_std[signal])
      val, *gradient = posfix_functions[meas.constellation_id](*inp, pr, zero_theta, *sat_pos, w)
      vals.append(val)
      gradients.append(gradient)
    return np.asarray(vals), np.asarray(gradients)
  return Fx_pos


def prr_residual(measurements: List[GNSSMeasurement], est_pos, velfix_function=None, signal='D1C', no_weight=False, no_nans=False):

  if velfix_function is None:
    velfix_function = get_velfix_sympy_func()

  def Fx_vel(vel, no_weight=no_weight):
    vals, gradients = [], []

    for meas in measurements:
      if signal not in meas.observables or not np.isfinite(meas.observables[signal]):
        if not no_nans:
          vals.append(np.nan)
          gradients.append(np.nan)
        continue

      sat_pos = meas.sat_pos_final if meas.corrected else meas.sat_pos
      weight = 1.0 if no_weight or meas.observables_std[signal] == 0 else (1 / meas.observables_std[signal])

      val, *gradient = velfix_function(est_pos[0], est_pos[1], est_pos[2],
                                       sat_pos[0], sat_pos[1], sat_pos[2],
                                       meas.sat_vel[0], meas.sat_vel[1], meas.sat_vel[2],
                                       meas.observables[signal], weight,
                                       vel[0], vel[1], vel[2], vel[3])
      vals.append(val)
      gradients.append(gradient)

    return np.asarray(vals), np.asarray(gradients)
  return Fx_vel
