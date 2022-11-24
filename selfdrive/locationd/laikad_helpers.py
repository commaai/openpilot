import numpy as np
import sympy

from laika.constants import EARTH_ROTATION_RATE, SPEED_OF_LIGHT
from laika.helpers import ConstellationId
from laika.raw_gnss import prr_residual


def calc_pos_fix_gauss_newton(measurements, posfix_functions, x0=None, signal='C1C', min_measurements=6):
  '''
  Calculates gps fix using gauss newton method
  To solve the problem a minimal of 4 measurements are required.
    If Glonass is included 5 are required to solve for the additional free variable.
  returns:
  0 -> list with positions
  '''
  if x0 is None:
    x0 = [0, 0, 0, 0, 0]
  n = len(measurements)
  if n < min_measurements:
    return [], []

  Fx_pos = pr_residual(measurements, posfix_functions, signal=signal)
  x = gauss_newton(Fx_pos, x0)
  residual, _ = Fx_pos(x, weight=1.0)
  return x.tolist(), residual.tolist()


def pr_residual(measurements, posfix_functions, signal='C1C'):
  def Fx_pos(inp, weight=None):
    vals, gradients = [], []

    for meas in measurements:
      pr = meas.observables[signal]
      pr += meas.sat_clock_err * SPEED_OF_LIGHT

      w = (1 / meas.observables_std[signal]) if weight is None else weight

      val, *gradient = posfix_functions[meas.constellation_id](*inp, pr, *meas.sat_pos, w)
      vals.append(val)
      gradients.append(gradient)
    return np.asarray(vals), np.asarray(gradients)

  return Fx_pos


def get_prr_sympy_func(no_weight):
  # knowns, receiver position, satellite position, satellite velocity
  est_pos_x = sympy.Symbol('est_pos_x')
  est_pos_y = sympy.Symbol('est_pos_y')
  est_pos_z = sympy.Symbol('est_pos_z')

  sat_pos_x = sympy.Symbol('sat_pos_x')
  sat_pos_y = sympy.Symbol('sat_pos_y')
  sat_pos_z = sympy.Symbol('sat_pos_z')

  sat_vel_x = sympy.Symbol('sat_vel_x')
  sat_vel_y = sympy.Symbol('sat_vel_y')
  sat_vel_z = sympy.Symbol('sat_vel_z')

  observables = sympy.Symbol('observables')

  # unknown, receiver velocity
  vel_x = sympy.Symbol('vel_x')
  vel_y = sympy.Symbol('vel_y')
  vel_z = sympy.Symbol('vel_z')
  vel_o = sympy.Symbol('vel_o')

  var = [vel_x, vel_y, vel_z, vel_o]

  '''
  loss_vector = (sat_pos - est_pos)# /  # np.linalg.norm(sat_pos - est_pos)
  print(f"sat_vel: {type(sat_vel)}")
  nv = (sat_vel - vel_v).dot(loss_vector)
  ov = (observables - vel_o)
  '''
  loss_x = sat_pos_x - est_pos_x
  loss_y = sat_pos_y - est_pos_y
  loss_z = sat_pos_z - est_pos_z

  # normalize loss vector
  d = sympy.sqrt(loss_x**2 + loss_y**2 + loss_z**2)
  loss_x /= d
  loss_y /= d
  loss_z /= d

  dx = sat_vel_x - vel_x
  dy = sat_vel_y - vel_y
  dz = sat_vel_z - vel_z

  nv = loss_x*dx + loss_y*dy + loss_z*dz # dot product
  ov = (observables - vel_o)

  if no_weight:
    res = (nv - ov)
  else:
    res = (nv - ov)/observables

  res = [res] +  [sympy.diff(res, v) for v in var]

  return sympy.lambdify([
      est_pos_x, est_pos_y, est_pos_z,
      sat_pos_x, sat_pos_y, sat_pos_z,
      sat_vel_x, sat_vel_y, sat_vel_z, observables,
      vel_x, vel_y, vel_z, vel_o
    ],
    res, modules=["numpy"])


def prr_residual_h(measurements, est_pos, no_weight=False):
  signal='D1C'

  def Fx_vel(vel, no_weight=no_weight):
    vals, gradients = [], []

    for meas in measurements:
      if signal not in meas.observables or not np.isfinite(meas.observables[signal]):
        continue

      if meas.corrected:
        sat_pos = meas.sat_pos_final
      else:
        sat_pos = meas.sat_pos

      '''
      if no_weight:
        weight = 1
      else:
        weight = (1 / meas.observables[signal])

      loss_vector = (sat_pos - est_pos[0:3]) / np.linalg.norm(sat_pos - est_pos[0:3])
      res = weight * ((meas.sat_vel - vel[0:3]).dot(loss_vector) -
                      (meas.observables[signal] - vel[3]))
      vals.append(res)
      '''

      f = get_prr_sympy_func(no_weight)

      val, *gradient = f(est_pos[0], est_pos[1], est_pos[2],
                         sat_pos[0], sat_pos[1], sat_pos[2],
                         meas.sat_vel[0], meas.sat_vel[1], meas.sat_vel[2],
                         meas.observables[signal],
                         vel[0], vel[1], vel[2], vel[3])
      vals.append(val)
      gradients.append(gradient)
    return np.asarray(vals), np.asarray(gradients)

  return Fx_vel

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


def get_posfix_sympy_fun(constellation):
  # Unknowns
  x, y, z = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')
  bc = sympy.Symbol('bc')
  bg = sympy.Symbol('bg')
  var = [x, y, z, bc, bg]

  # Knowns
  pr = sympy.Symbol('pr')
  sat_x, sat_y, sat_z = sympy.Symbol('sat_x'), sympy.Symbol('sat_y'), sympy.Symbol('sat_z')
  weight = sympy.Symbol('weight')

  theta = EARTH_ROTATION_RATE * (pr - bc) / SPEED_OF_LIGHT
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

  return sympy.lambdify([x, y, z, bc, bg, pr, sat_x, sat_y, sat_z, weight], res, modules=["numpy"])

# called like:
# ecef_vel, vel_res = calc_vel_fix(corrected_measurements, est_pos)
# so we can have some default values, added like v0, no_weigth and signal
def calc_vel_fix(measurements, est_pos, v0=[0, 0, 0, 0], signal='D1C'):
  '''
  Calculates gps velocity fix with WLS optimizer
  returns:
  0 -> list with velocities
  1 -> pseudorange_rate errs
  '''
  import scipy.optimize as opt  # Only use scipy here

  n = len(measurements)
  if n < 6:
    return [], []

  Fx_vel = prr_residual_h(measurements, est_pos)
  opt_vel = gauss_newton(Fx_vel, v0)
  residual, _ = Fx_vel(opt_vel)
  print(f"first velocity: {opt_vel} {residual}")
  #return opt_vel.tolist(), residual.tolist()

  # leave as comparission
  Fx_vel_orig = prr_residual(measurements, est_pos)
  opt_vel = opt.least_squares(Fx_vel_orig, v0).x
  residual = Fx_vel_orig(opt_vel, no_weight=True)
  print(f"secon velocity: {opt_vel} {residual}")

  return opt_vel, residual