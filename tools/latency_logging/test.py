import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['lateralPlan'])
while 1:
  sm.update()
  if sm.updated['lateralPlan']:
      d ={}
      d["logMessage"] = sm['lateralPlan'].to_dict()
      d["logMonoTime"] = sm.logMonoTime['lateralPlan']
      d["recvTime"] = sm.rcv_time['lateralPlan']
      print(d)
      break
