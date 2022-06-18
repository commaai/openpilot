What is cereal? [![cereal tests](https://github.com/commaai/cereal/workflows/Tests/badge.svg?event=push)](https://github.com/commaai/cereal/actions) [![codecov](https://codecov.io/gh/commaai/cereal/branch/master/graph/badge.svg)](https://codecov.io/gh/commaai/cereal)
----

cereal is both a messaging spec for robotics systems as well as generic high performance IPC pub sub messaging with a single publisher and multiple subscribers.

Imagine this use case:
* A sensor process reads gyro measurements directly from an IMU and publishes a `sensorEvents` packet
* A calibration process subscribes to the `sensorEvents` packet to use the IMU
* A localization process subscribes to the `sensorEvents` packet to use the IMU also


Messaging Spec
----

You'll find the message types in [log.capnp](log.capnp). It uses [Cap'n proto](https://capnproto.org/capnp-tool.html) and defines one struct called Event.

All Events have a `logMonoTime` and a `valid`. Then a big union defines the packet type.


Message definition Best Practices
----

- **All fields must describe quantities in SI units**, unless otherwise specified in the field name.

- In the context of the message they are in, field names should be completely unambiguous.

- All values should be easy to plot and be human-readable with minimal parsing.



Pub Sub Backends
----

cereal supports two backends, one based on [zmq](https://zeromq.org/) and another called msgq, a custom pub sub based on shared memory that doesn't require the bytes to pass through the kernel.

Example
---
```python
import cereal.messaging as messaging

# in subscriber
sm = messaging.SubMaster(['sensorEvents'])
while 1:
  sm.update()
  print(sm['sensorEvents'])

```

```python
# in publisher
pm = messaging.PubMaster(['sensorEvents'])
dat = messaging.new_message('sensorEvents', size=1)
dat.sensorEvents[0] = {"gyro": {"v": [0.1, -0.1, 0.1]}}
pm.send('sensorEvents', dat)
```
