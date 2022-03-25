import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['procLog'])
while 1:
  sm.update()
  if sm.updated['procLog']:
      print(sm['procLog'])
