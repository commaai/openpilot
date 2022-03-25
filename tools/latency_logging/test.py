import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['sendcan'])
while 1:
  sm.update()
  if sm.updated['sendcan']:
      print(sm.rcv_time['sendcan'], sm.logMonoTime['sendcan'])
