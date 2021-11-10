#!/usr/bin/env python3
from cereal import car
from selfdrive.car.toyota.values import MIN_ACC_SPEED, PEDAL_HYST_GAP



###### LONG ######
long_tunes = {}
name = 'PEDAL'
long_tunes[name] = car.CarParams.new_message().longitudinalTuning
with long_tunes[name] as tune:
  tune.deadzoneBP = [0.]
  tune.deadzoneV = [0.]
  tune.kpBP = [0., 5., MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_HYST_GAP, 35.]
  tune.kpV = [1.2, 0.8, 0.765, 2.255, 1.5]
  tune.kiBP = [0., MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_HYST_GAP, 35.]
  tune.kiV = [0.18, 0.165, 0.489, 0.36]

# Improved longitudinal tune
name = 'TSS2'
long_tunes[name] = car.CarParams.new_message().longitudinalTuning
with long_tunes[name] as tune:
  tune.deadzoneBP = [0., 8.05]
  tune.deadzoneV = [.0, .14]
  tune.kpBP = [0., 5., 20.]
  tune.kpV = [1.3, 1.0, 0.7]
  tune.kiBP = [0., 5., 12., 20., 27.]
  tune.kiV = [.35, .23, .20, .17, .1]

# Default longitudinal tune
name = 'TSSP'
long_tunes[name] = car.CarParams.new_message().longitudinalTuning
with long_tunes[name] as tune:
  tune.deadzoneBP = [0., 9.]
  tune.deadzoneV = [0., .15]
  tune.kpBP = [0., 5., 35.]
  tune.kiBP = [0., 35.]
  tune.kpV = [3.6, 2.4, 1.5]
  tune.kiV = [0.54, 0.36]


###### LAT ######
lat_tunes = {}
name = 'PRIUS_INDI'
lat_tunes[name] = car.CarParams.new_message().lateralTuning
with lat_tunes[name] as tune:
  tune.init('indi')
  tune.indi.innerLoopGainBP = [0.]
  tune.indi.innerLoopGainV = [4.0]
  tune.indi.outerLoopGainBP = [0.]
  tune.indi.outerLoopGainV = [3.0]
  tune.indi.timeConstantBP = [0.]
  tune.indi.timeConstantV = [1.0]
  tune.indi.actuatorEffectivenessBP = [0.]
  tune.indi.actuatorEffectivenessV = [1.0]

name = 'RAV4_LQR'
lat_tunes[name] = car.CarParams.new_message().lateralTuning
with lat_tunes[name] as tune:
  tune.init('lqr')
  tune.lqr.scale = 1500.0
  tune.lqr.ki = 0.05
  tune.lqr.a = [0., 1., -0.22619643, 1.21822268]
  tune.lqr.b = [-1.92006585e-04, 3.95603032e-05]
  tune.lqr.c = [1., 0.]
  tune.lqr.k = [-110.73572306, 451.22718255]
  tune.lqr.l = [0.3233671, 0.3185757]
  tune.lqr.dcGain = 0.002237852961363602


