## LogReader

Route is a class for conviently accessing all the [logs](/selfdrive/loggerd/) from your routes. The LogReader class reads the non-video logs, i.e. rlog.bz2 and qlog.bz2. There's also a matching FrameReader class for reading the videos.

```python
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator

r = Route("4cf7a6ad03080c90|2021-09-29--13-46-36")

# get a list of paths for the route's rlog files
print(r.log_paths())

# and road camera (fcamera.hevc) files
print(r.camera_paths())

# setup a LogReader to read the route
lr = MultiLogIterator(r.log_paths(), wraparound=False)

# print out all the messages in the route
import codecs
codecs.register_error("strict", codecs.backslashreplace_errors)
while True:
  msg = next(lr)
  print(msg)

# print all CAN messages with timestamp
while True:
  route_time = lr.tell()
  try:
    msg = next(lr)
  except StopIteration:
    break
  except:
    raise
  try:
    typ = msg.which()
  except:
    continue
  if typ == "can":
    for CAN_msg in msg.can:
      print(f'{route_time},{CAN_msg.address},{CAN_msg.src},{CAN_msg.dat.hex()}'.format())
```
