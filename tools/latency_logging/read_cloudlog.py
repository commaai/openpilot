from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict


timestamps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
translationdict = {}

#r = Route("9f583b1d93915c31|2022-03-22--15-59-29")
r = Route("9f583b1d93915c31|2022-03-23--12-47-33")
lr = LogReader(r.log_paths()[0])

for msg in lr:
  if msg.which() == "logMessage" and "translation" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    logMonoTime = jmsg['msg']['logMonoTime'] 
    frame_id = jmsg['msg']['frameId']
    translationdict[logMonoTime] = frame_id

for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = jmsg['msg']['timestamp']['time']
    frame_id = jmsg['msg']['timestamp']['frameId']
    if jmsg['msg']['timestamp']['translate']:
        frame_id = translationdict[frame_id]
    timestamps[frame_id][service][event].append(time)

del timestamps[0]
'''

timestampsshort = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for frame_id, services in timestamps.items():
    for service, events in services.items():
        for event, times in events.items():
            timestampsshort[frame_id][service][event] = min(times)
'''

with open('timestamps.json', 'w') as outfile:
    json.dump(timestamps, outfile)
