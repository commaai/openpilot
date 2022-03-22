from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json

r = Route("9f583b1d93915c31|2022-03-21--18-30-38")
lr = LogReader(r.log_paths()[0])

for msg in lr:
  if msg.which() == "logMessage":
    msg_json = json.loads(msg.logMessage)
    print(msg_json['msg'])
