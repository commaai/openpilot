# What is cereal? [![cereal tests](https://github.com/commaai/cereal/workflows/tests/badge.svg?event=push)](https://github.com/commaai/cereal/actions) [![codecov](https://codecov.io/gh/commaai/cereal/branch/master/graph/badge.svg)](https://codecov.io/gh/commaai/cereal)

cereal is both a messaging spec for robotics systems as well as generic high performance IPC pub sub messaging with a single publisher and multiple subscribers.

Imagine this use case:
* A sensor process reads gyro measurements directly from an IMU and publishes a `sensorEvents` packet
* A calibration process subscribes to the `sensorEvents` packet to use the IMU
* A localization process subscribes to the `sensorEvents` packet to use the IMU also


## Messaging Spec

You'll find the message types in [log.capnp](log.capnp). It uses [Cap'n proto](https://capnproto.org/capnp-tool.html) and defines one struct called `Event`.

All `Events` have a `logMonoTime` and a `valid`. Then a big union defines the packet type.

### Best Practices

- **All fields must describe quantities in SI units**, unless otherwise specified in the field name.
- In the context of the message they are in, field names should be completely unambiguous.
- All values should be easy to plot and be human-readable with minimal parsing.

### Maintaining backwards-compatibility

When making changes to the messaging spec you want to maintain backwards-compatibility, such that old logs can
be parsed with a new version of cereal. Adding structs and adding members to structs is generally safe, most other
things are not. Read more details [here](https://capnproto.org/language.html).

### Custom forks

Forks of [openpilot](https://github.com/commaai/openpilot) might want to add things to the messaging
spec, however this could conflict with future changes made in mainline cereal/openpilot. Rebasing against mainline openpilot
then means breaking backwards-compatibility with all old logs of your fork. So we added reserved events in
[custom.capnp](custom.capnp) that we will leave empty in mainline cereal/openpilot. **If you only modify those, you can ensure your
fork will remain backwards-compatible with all versions of mainline cereal/openpilot and your fork.**

## Pub Sub Backends

cereal supports two backends, one based on [zmq](https://zeromq.org/) and another called [msgq](messaging/msgq.cc), a custom pub sub based on shared memory that doesn't require the bytes to pass through the kernel.

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
