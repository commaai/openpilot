from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict


timestamps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

r = Route("9f583b1d93915c31|2022-03-22--15-13-06")
lr = LogReader(r.log_paths()[0])

for msg in lr:
  if msg.which() == "logMessage":
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    if 'timestamp' in msg:
        try:
            jmsg = json.loads(msg)
            frame_id = jmsg['msg']['timestamp']['frameId']
            service = jmsg['ctx']['daemon']
            event = jmsg['msg']['timestamp']['event']
            time = jmsg['msg']['timestamp']['time']
            timestamps[frame_id][service][event].append(time)
        except :
            print( msg)

del timestamps[0]

timestampsshort = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for frame_id, services in timestamps.items():
    for service, events in services.items():
        for event, times in events.items():
            timestampsshort[frame_id][service][event] = min(times)


with open('timestamps.json', 'w') as outfile:
    json.dump(timestampsshort, outfile)
