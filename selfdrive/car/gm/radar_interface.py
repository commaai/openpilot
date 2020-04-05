#!/usr/bin/env python3
#copied from mock - we are hardcoding to no radar
#perhaps later we can figure out how to support both - maybe this isn't even required
from selfdrive.car.interfaces import RadarInterfaceBase

class RadarInterface(RadarInterfaceBase):
  pass
