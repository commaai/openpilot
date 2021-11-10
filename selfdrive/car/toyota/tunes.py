#!/usr/bin/env python3
from cereal import car
from selfdrive.car.toyota.values import MIN_ACC_SPEED, PEDAL_HYST_GAP

long_tunes = {}

long_tunes['pedal'] = car.CarParams.new_message().longitudinalTuning
with long_tunes['pedal'] as key:
  long_tunes[key].deadzoneBP = [0.]
  long_tunes[key].deadzoneV = [0.]
  long_tunes[key].kpBP = [0., 5., MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_HYST_GAP, 35.]
  long_tunes[key].kpV = [1.2, 0.8, 0.765, 2.255, 1.5]
  long_tunes[key].kiBP = [0., MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_HYST_GAP, 35.]
  long_tunes[key].kiV = [0.18, 0.165, 0.489, 0.36]

# Improved longitudinal tune
long_tunes['TSS2'] = car.CarParams.new_message().longitudinalTuning
with long_tunes['TSS2'] as key:
  long_tunes[key].deadzoneBP = [0., 8.05]
  long_tunes[key].deadzoneV = [.0, .14]
  long_tunes[key].kpBP = [0., 5., 20.]
  long_tunes[key].kpV = [1.3, 1.0, 0.7]
  long_tunes[key].kiBP = [0., 5., 12., 20., 27.]
  long_tunes[key].kiV = [.35, .23, .20, .17, .1]

# Default longitudinal tune
long_tunes['TSSP'] = car.CarParams.new_message().longitudinalTuning
with long_tunes['TSSP'] as key:
  long_tunes[key].deadzoneBP = [0., 9.]
  long_tunes[key].deadzoneV = [0., .15]
  long_tunes[key].kpBP = [0., 5., 35.]
  long_tunes[key].kiBP = [0., 35.]
  long_tunes[key].kpV = [3.6, 2.4, 1.5]
  long_tunes[key].kiV = [0.54, 0.36]
