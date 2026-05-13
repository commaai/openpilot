from enum import IntEnum

class Axis(IntEnum):
  TIME = 0
  EGO_POSITION = 1
  LEAD_DISTANCE= 2
  EGO_V = 3
  LEAD_V = 4
  EGO_A = 5
  D_REL = 6

axis_labels = {Axis.TIME: 'Time (s)',
               Axis.EGO_POSITION: 'Ego position (m)',
               Axis.LEAD_DISTANCE: 'Lead absolute position (m)',
               Axis.EGO_V: 'Ego Velocity (m/s)',
               Axis.LEAD_V: 'Lead Velocity (m/s)',
               Axis.EGO_A: 'Ego acceleration (m/s^2)',
               Axis.D_REL: 'Lead distance (m)'}
