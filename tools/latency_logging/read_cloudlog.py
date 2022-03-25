from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict


timestamps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
translationdict = {}

#r = Route("9f583b1d93915c31|2022-03-23--14-47-10")
r = Route("9f583b1d93915c31|2022-03-24--16-15-18")
lr = LogReader(r.log_paths()[0])
'''
for msg in lr:
  if msg.which() == "logMessage":
      print(msg)

exit()
'''
for msg in lr:
  if msg.which() == "logMessage" and "translation" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    logMonoTime = jmsg['msg']['logMonoTime'] 
    frame_id = jmsg['msg']['frameId']
    translationdict[logMonoTime] = frame_id


i = 0
for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = jmsg['msg']['timestamp']['time']
    frame_id = jmsg['ctx']['frame_id']
    if jmsg['msg']['timestamp']['translate']:
        try:
            frame_id = translationdict[int(frame_id)]
        except:
            i+=1
            continue
    timestamps[int(frame_id)][service][event].append(time)

print("Num failed translations:", i)

with open('timestamps.json', 'w') as outfile:
    json.dump(timestamps, outfile)
