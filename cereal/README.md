# What is cereal?

cereal is the messaging system for openpilot. It uses [msgq](https://github.com/commaai/msgq) as a pub/sub backend, and [Cap'n proto](https://capnproto.org/capnp-tool.html) for serialization of the structs.


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
fork will remain backwards-compatible with all versions of mainline openpilot and your fork.**

An example of compatible changes:
```diff
diff --git a/cereal/custom.capnp b/cereal/custom.capnp
index 3348e859e..3365c7b98 100644
--- a/cereal/custom.capnp
+++ b/cereal/custom.capnp
@@ -10,7 +10,11 @@ $Cxx.namespace("cereal");
 # DO rename the structs
 # DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

-struct CustomReserved0 @0x81c2f05a394cf4af {
+struct SteeringInfo @0x81c2f05a394cf4af {
+  active @0 :Bool;
+  steeringAngleDeg @1 :Float32;
+  steeringRateDeg @2 :Float32;
+  steeringAccelDeg @3 :Float32;
 }

 struct CustomReserved1 @0xaedffd8f31e7b55d {
diff --git a/cereal/log.capnp b/cereal/log.capnp
index 1209f3fd9..b189f58b6 100644
--- a/cereal/log.capnp
+++ b/cereal/log.capnp
@@ -2558,14 +2558,14 @@ struct Event {

     # DO change the name of the field
     # DON'T change anything after the "@"
-    customReservedRawData0 @124 :Data;
+    rawCanData @124 :Data;
     customReservedRawData1 @125 :Data;
     customReservedRawData2 @126 :Data;

     # DO change the name of the field and struct
     # DON'T change the ID (e.g. @107)
     # DON'T change which struct it points to
-    customReserved0 @107 :Custom.CustomReserved0;
+    steeringInfo @107 :Custom.SteeringInfo;
     customReserved1 @108 :Custom.CustomReserved1;
     customReserved2 @109 :Custom.CustomReserved2;
     customReserved3 @110 :Custom.CustomReserved3;
```

---

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
