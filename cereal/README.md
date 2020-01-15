What is cereal?
----

cereal is both a messaging spec for robotics systems as well as generic high performance IPC pub sub messaging with a single publisher and multiple subscribers.

Imagine this use case:
* A sensor process reads gyro measurements directly from an IMU and publishes a sensorEvents packet
* A calibration process subscribes to the sensorEvents packet to use the IMU
* A localization process subscribes to the sensorEvents packet to use the IMU also


Messaging Spec
----

You'll find the message types in [log.capnp](log.capnp). It uses [Cap'n proto](https://capnproto.org/capnp-tool.html) and defines one struct called Event.

All Events have a logMonoTime and a valid. Then a big union defines the packet type.


Pub Sub Backends
----

cereal supports two backends, one based on [zmq](https://zeromq.org/), the other called msgq, a custom pub sub based on shared memory that doesn't require the bytes to pass through the kernel.

Example
---
```python
import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['sensorEvents'])
while 1:
  sm.update()
  print(sm['sensorEvents'])

# in publisher
pm = messaging.PubMaster(['sensorEvents'])
dat = messaging.new_message()
dat.init('sensorEvents', 1)
dat.sensorEvents[0] = {"gyro": {"v": [0.1, -0.1, 0.1]}}
pm.send('sensorEvents', dat)
```
