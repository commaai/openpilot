#!/usr/bin/env python3
import numpy as np
from enum import Enum


class LongTunes(Enum):
  PEDAL = 0
  TSS2 = 1
  TSS = 2

class LatTunes(Enum):
  INDI_PRIUS = 0
  LQR_RAV4 = 1
  PID_A = 2
  PID_B = 3
  PID_C = 4
  PID_D = 5
  PID_E = 6
  PID_F = 7
  PID_G = 8
  PID_I = 9
  PID_H = 10
  PID_J = 11
  PID_K = 12
  PID_L = 13
  PID_M = 14
  PID_N = 15
  STEER_MODEL_COROLLA = 16
  STEER_MODEL_CAMRY = 17


###### LONG ######
def set_long_tune(tune, name):
  # Improved longitudinal tune
  if name == LongTunes.TSS2 or name == LongTunes.PEDAL:
    tune.deadzoneBP = [0., 8.05]
    tune.deadzoneV = [.0, .14]
    tune.kpBP = [0., 5., 20.]
    tune.kpV = [1.3, 1.0, 0.7]
    tune.kiBP = [0., 5., 12., 20., 27.]
    tune.kiV = [.35, .23, .20, .17, .1]
  # Default longitudinal tune
  elif name == LongTunes.TSS:
    tune.deadzoneBP = [0., 9.]
    tune.deadzoneV = [0., .15]
    tune.kpBP = [0., 5., 35.]
    tune.kiBP = [0., 35.]
    tune.kpV = [3.6, 2.4, 1.5]
    tune.kiV = [0.54, 0.36]
  else:
    raise NotImplementedError('This longitudinal tune does not exist')


###### LAT ######
def set_lat_tune(tune, params, name):
  # LQR takes precedence, steering model uses kf value if available, so that's at end
  if params.use_lqr:
    tune.init('lqr')

    tune.lqr.scale = 1500.0
    tune.lqr.ki = 0.05

    tune.lqr.a = [0., 1., -0.22619643, 1.21822268]
    tune.lqr.b = [-1.92006585e-04, 3.95603032e-05]
    tune.lqr.c = [1., 0.]
    tune.lqr.k = [-110.73572306, 451.22718255]
    tune.lqr.l = [0.3233671, 0.3185757]
    tune.lqr.dcGain = 0.002237852961363602

  elif name == LatTunes.INDI_PRIUS:
    if params.prius_use_pid:
      tune.pid.kpV, tune.pid.kiV = [[0.07], [0.04]]
      tune.pid.kdV = [0.]
      tune.pid.kf = 0.00009531750004645412
      tune.pid.newKfTuned = True
    else:
      tune.init('indi')
      tune.indi.innerLoopGainV = [4.0]
      tune.indi.outerLoopGainV = [3.0]
      tune.indi.timeConstantV = [0.1] if params.hasZss else [1.0]
      tune.indi.actuatorEffectivenessV = [1.0]

    tune.init('indi')
    tune.indi.innerLoopGainBP = [0.]
    tune.indi.innerLoopGainV = [4.0]
    tune.indi.outerLoopGainBP = [0.]
    tune.indi.outerLoopGainV = [3.0]
    tune.indi.timeConstantBP = [0.]
    tune.indi.timeConstantV = [1.0]
    tune.indi.actuatorEffectivenessBP = [0.]
    tune.indi.actuatorEffectivenessV = [1.0]

  elif name == LatTunes.LQR_RAV4:
    tune.init('lqr')
    tune.lqr.scale = 1500.0
    tune.lqr.ki = 0.05
    tune.lqr.a = [0., 1., -0.22619643, 1.21822268]
    tune.lqr.b = [-1.92006585e-04, 3.95603032e-05]
    tune.lqr.c = [1., 0.]
    tune.lqr.k = [-110.73572306, 451.22718255]
    tune.lqr.l = [0.3233671, 0.3185757]
    tune.lqr.dcGain = 0.002237852961363602

  elif 'STEER_MODEL' in str(name):
    tune.init('model')
    tune.model.useRates = False  # TODO: makes model sluggish, see comments in latcontrol_model.py
    tune.model.multiplier = 1.

    if name == LatTunes.STEER_MODEL_COROLLA:
      tune.model.name = "corolla_model_v5"
    elif name == LatTunes.STEER_MODEL_CAMRY:
      tune.model.name = "camryh_tss2"
      tune.model.useRates = True
    else:
      raise NotImplementedError('This steering model does not exist')

  elif 'PID' in str(name):
    tune.init('pid')
    tune.pid.kiBP = [0.0]
    tune.pid.kpBP = [0.0]
    if name == LatTunes.PID_A:
      tune.pid.kpV = [0.2]
      tune.pid.kiV = [0.05]
      tune.pid.kdV = [0.1]
      tune.pid.kf = 0.00003
    elif name == LatTunes.PID_B:
      tune.pid.kpV = [0.6]
      tune.pid.kiV = [0.05]
      tune.pid.kf = 0.00006
    elif name == LatTunes.PID_C:
      tune.pid.kpV = [0.6]
      tune.pid.kiV = [0.1]
      tune.pid.kf = 0.00006
    elif name == LatTunes.PID_D:
      tune.pid.kpV = [0.6]
      tune.pid.kiV = [0.1]
      tune.pid.kdV = [0.1]  # from RAV4_TSS2
      tune.pid.kf = 0.00007818594
    elif name == LatTunes.PID_E:
      tune.pid.kpV = [0.6]
      tune.pid.kiV = [0.15]
      tune.pid.kf = 0.00007818594
    elif name == LatTunes.PID_F:
      tune.pid.kpV = [0.723]
      tune.pid.kiV = [0.0428]
      tune.pid.kf = 0.00006
    elif name == LatTunes.PID_G:
      tune.pid.kpV = [0.18]
      tune.pid.kiV = [0.015]
      tune.pid.kf = 0.00012
    elif name == LatTunes.PID_H:
      tune.pid.kpV = [0.17]
      tune.pid.kiV = [0.03]
      tune.pid.kf = 0.00006
    elif name == LatTunes.PID_I:
      tune.pid.kpV = [0.15]
      tune.pid.kiV = [0.05]
      tune.pid.kdV = [0.1]  # from RAV4_TSS2
      tune.pid.kf = 0.00004
    elif name == LatTunes.PID_J:
      tune.pid.kpV = [0.19]
      tune.pid.kiV = [0.02]
      tune.pid.kf = 0.00007818594
    elif name == LatTunes.PID_L:
      tune.pid.kpV = [0.3]
      tune.pid.kiV = [0.05]
      tune.pid.kf = 0.00006
    elif name == LatTunes.PID_M:
      tune.pid.kpV = [0.3]
      tune.pid.kiV = [0.05]
      tune.pid.kf = 0.00007
    elif name == LatTunes.PID_N:
      tune.pid.kpV = [0.35]
      tune.pid.kiV = [0.15]
      tune.pid.kf = 0.00007818594
    else:
      raise NotImplementedError('This PID tune does not exist')
  else:
    raise NotImplementedError('This lateral tune does not exist')

  if params.use_steering_model:
    tune.init('model')
    tune.model.name = 'camryh_tss2' if params.TSS2 else 'corolla_model_v5'  # use closest available
    tune.model.useRates = bool(params.TSS2)
    tune.model.multiplier = 1.
    # use kf from PID to calculate torque multiplier
    # TODO: feed this into the model so it can extrapolate accurately
    if tune.which() == 'pid':
      # TSSP Corolla is actually 0.000069 but other cars are using stock tuning, so use stock 0.00003 for extrapolation
      MODEL_CAR_KF = 0.00006 if params.TSS2 else 0.00003
      if not np.isclose(tune.pid.kf, 0.):
        tune.model.multiplier = tune.pid.kf / MODEL_CAR_KF
