import numpy as np


def get_delta_out_limits(aEgo, aMax, aMin, jMax, jMin):

  tDelta = 0.
  if aEgo > aMax:
    tDelta = (aMax - aEgo) / jMin
  elif aEgo < aMin:
    tDelta = (aMin - aEgo) / jMax

  return tDelta


def speed_smoother(vEgo, aEgo, vT, aMax, aMin, jMax, jMin, ts):

  dV = vT - vEgo

  # recover quickly if dV is positive and aEgo is negative or viceversa
  if dV > 0. and aEgo < 0.:
    jMax *= 3.
  elif dV < 0. and aEgo > 0.:
    jMin *= 3.

  tDelta = get_delta_out_limits(aEgo, aMax, aMin, jMax, jMin)

  if (ts <= tDelta):
    if (aEgo < aMin):
      vEgo += ts * aEgo + 0.5 * ts**2 * jMax
      aEgo += ts * jMax
      return vEgo, aEgo
    elif (aEgo > aMax):
      vEgo += ts * aEgo + 0.5 * ts**2 * jMin
      aEgo += ts * jMin
      return vEgo, aEgo

  if aEgo > aMax:
    dV -= 0.5 * (aMax**2 - aEgo**2) / jMin
    vEgo += 0.5 * (aMax**2 - aEgo**2) / jMin
    aEgo += tDelta * jMin
  elif aEgo < aMin:
    dV -= 0.5 * (aMin**2 - aEgo**2) / jMax
    vEgo += 0.5 * (aMin**2 - aEgo**2) / jMax
    aEgo += tDelta * jMax

  ts -= tDelta

  jLim = jMin if aEgo >= 0 else jMax
  # if we reduce the accel to zero immediately, how much delta speed we generate?
  dv_min_shift = - 0.5 * aEgo**2 / jLim

  # flip signs so we can consider only one case
  flipped = False
  if dV < dv_min_shift:
    flipped = True
    dV *= -1
    vEgo *= -1
    aEgo *= -1
    aMax = -aMin
    jMaxcopy = -jMin
    jMin = -jMax
    jMax = jMaxcopy

  # small addition needed to avoid numerical issues with sqrt of ~zero
  aPeak = np.sqrt((0.5 * aEgo**2 / jMax + dV + 1e-9) / (0.5 / jMax - 0.5 / jMin))

  if aPeak > aMax:
    aPeak = aMax
    t1 = (aPeak - aEgo) / jMax
    if aPeak <= 0: # there is no solution, so stop after t1
      t2 = t1 + ts + 1e-9
      t3 = t2
    else:
      vChange = dV - 0.5 * (aPeak**2 - aEgo**2) / jMax + 0.5 * aPeak**2 / jMin
      if vChange < aPeak * ts:
        t2 = t1 + vChange / aPeak
      else:
        t2 = t1 + ts
      t3 = t2 - aPeak / jMin
  else:
    t1 = (aPeak - aEgo) / jMax
    t2 = t1
    t3 = t2 - aPeak / jMin

  dt1 = min(ts, t1)
  dt2 = max(min(ts, t2) - t1, 0.)
  dt3 = max(min(ts, t3) - t2, 0.)

  if ts > t3:
    vEgo += dV
    aEgo = 0.
  else:
    vEgo += aEgo * dt1 + 0.5 * dt1**2 * jMax + aPeak * dt2 + aPeak * dt3 + 0.5 * dt3**2 * jMin
    aEgo += jMax * dt1 + dt3 * jMin

  vEgo *= -1 if flipped else 1
  aEgo *= -1 if flipped else 1

  return float(vEgo), float(aEgo)
