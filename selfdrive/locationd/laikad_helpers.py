import numpy as np
import sympy

from laika.constants import EARTH_ROTATION_RATE, SPEED_OF_LIGHT
from laika.helpers import ConstellationId


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
