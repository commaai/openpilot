from cereal.messaging import SubMaster


sm = SubMaster(['testJoystick'])

while 1:
  sm.update(0)
  print(sm['testJoystick'].axes)

