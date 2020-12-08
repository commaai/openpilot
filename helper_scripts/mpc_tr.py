from cereal.messaging import SubMaster
import time

sm = SubMaster(['dynamicFollowData'])

while True:
  sm.update(0)
  print('mpc_TR: {}'.format(sm['dynamicFollowData'].mpcTR))
  time.sleep(1/20)
