# loggerd

openpilot records routes in one minute chunks called segments. A route starts on the rising edge of ignition and ends on the falling edge.

Check out our [python library](https://github.com/commaai/openpilot/blob/master/tools/lib/logreader.py) for reading openpilot logs. Also checkout our [tools](https://github.com/commaai/openpilot/tree/master/tools) to replay and view your data. These are the same tools we use to debug and develop openpilot.

## log types

For each segment, openpilot records the following log types:

## rlog.bz2

rlogs contain all the messages passed amongst openpilot's processes. See [cereal/services.py](https://github.com/commaai/cereal/blob/master/services.py) for a list of all the logged services. They're a bzip2 archive of the serialized capnproto messages.

## {f,e,d}camera.hevc

Each camera stream is H.265 encoded and written to its respective file.
* fcamera.hevc is the road camera
* ecamera.hevc is the wide road camera
* dcamera.hevc is the driver camera

## qlog.bz2 & qcamera.ts

qlogs are a decimated subset of the rlogs. Check out [cereal/services.py](https://github.com/commaai/cereal/blob/master/services.py) for the decimation.


qcameras are H.264 encoded, lower res versions of the fcamera.hevc. The video shown in [comma connect](https://connect.comma.ai/) is from the qcameras.


qlogs and qcameras are designed to be small enough to upload instantly on slow internet and store forever, yet useful enough for most analysis and debugging.
