from selfdrive.swaglog import cloudlog

from tools.lib.route import Route
from tools.lib.logreader import LogReader

r = Route("9f583b1d93915c31|2022-03-18--08-31-46")
lr = LogReader(r.log_paths()[0])

for msg in lr:
  if msg.which() == "logMessage":
    print(msg)
