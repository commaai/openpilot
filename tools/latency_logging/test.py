import cereal.messaging as messaging
import capnp

# in subscriber
sm = messaging.SubMaster(["sendcan"])
while 1:
  sm.update()
  print([can.to_dict() for can in sm["sendcan"]])
