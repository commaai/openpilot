
import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['lateralPlan'])
while 1:
  sm.update()
  print(sm['lateralPlan'])
