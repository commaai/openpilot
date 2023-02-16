## LogReader

Route is a class for conveniently accessing all the [logs](/selfdrive/loggerd/) from your routes. The LogReader class reads the non-video logs, i.e. rlog.bz2 and qlog.bz2. There's also a matching FrameReader class for reading the videos.

```python
from tools.lib.route import Route
from tools.lib.logreader import LogReader

r = Route("4cf7a6ad03080c90|2021-09-29--13-46-36")

# get a list of paths for the route's rlog files
print(r.log_paths())

# and road camera (fcamera.hevc) files
print(r.camera_paths())

# setup a LogReader to read the route's first rlog
lr = LogReader(r.log_paths()[0])

# print out all the messages in the log
import codecs
codecs.register_error("strict", codecs.backslashreplace_errors)
for msg in lr:
  print(msg)

# setup a LogReader for the route's second qlog
lr = LogReader(r.log_paths()[1])

# print all the steering angles values from the log
for msg in lr:
  if msg.which() == "carState":
    print(msg.carState.steeringAngleDeg)
```

### MultiLogIterator

`MultiLogIterator` is similar to `LogReader`, but reads multiple logs. 

```python
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator

# setup a MultiLogIterator to read all the logs in the route
r = Route("4cf7a6ad03080c90|2021-09-29--13-46-36")
lr = MultiLogIterator(r.log_paths())

# print all the steering angles values from all the logs in the route
for msg in lr:
  if msg.which() == "carState":
    print(msg.carState.steeringAngleDeg)
```
