# Intro to openpilot

**openpilot** is a software that produces serial messages (CAN Messages) to change the acceleration and steering angle of a car given some camera streams, existing serial messages from the car, and sensor data. In addition to the real-time messages, openpilot produces logs that are used to train machine learning models at a later date.

The goal of this introduction is to introduce you to the moving pieces of openpilot, and help you understand how they work together to create these outputs.  

![conceptual_schematic](https://raw.githubusercontent.com/barbinbrad/openpilot/master/docs/assets/conceptual_schematic.png)


## CAN Messages

Modern vehicles communicate using CAN messaging, a broadcast protocol that allows many computers to talk together in a way that is tolerant to noisy environments. From the openpilot perspective, the good thing about the CAN protocol is that it is inherently trusting, allowing messages to be spoofed. The bad thing about the CAN protocol is that each manufacturer creates their own dictionary of CAN message IDs and data definitions. 

For example, one manufacturer might put the steering angle on message ID 0x30 at bytes 3 and 4, while another manufacturer describes steering angle on message ID 0xe4 at byte 2 and 3.

So even after openpilot figures out what the acceleration and steering angle *should be*, there is still much work to be done to turn it into a CAN message that will be understood by that particular make/model vehicle. 

Below is the heart of the code for turning calculations, `CC`, into manufacturer-specific CAN messages using the car interface, `CI`, which contains information about the make/model of the car:

```python
# selfdrive/controls/controlsd.py

# send car controls over can
can_sends = CI.apply(CC)
pm.send('sendcan', can_list_to_can_capnp(can_sends))
```

On each loop of the **controlsd** process, the car control message, `CC`, represents the desired setpoints of the car (acceleration, steering angle, etc). The  `CI.apply` method turns these setpoints into make/model specific CAN messages. 

On the next line, the `pm.send` function publishes the `can_send` messages on the `sendcan` topic, in Cap'n Proto format. The next snippet of code describes what happens to the message next. Don't worry if it doesn't make sense yet. 


```cpp
// selfdrive/boardd/boardd.cc

// subscribe to the sendcan topic
SubSocket * subscriber = SubSocket::create(context, "sendcan");

...

while (panda->connected) {
    // get messages as fast as possible
    Message * msg = subcriber->receive();

    // parse the cap'n proto messages
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    // send the unpacked CAN messages using the panda interface
    panda->can_send(event.getSendcan());
}
```

The **boardd** process subscribes to the `sendcan` topic and turns Cap'n Proto messages into physical CAN messages through the [panda](https://github.com/commaai/panda) hardware and firmware. The `panda->can_send` method writes to the hardware's microcontroller, which use CAN transceivers to send CAN messages directly to the vehicles CAN network.

To recap: **controlsd** is a process that takes many inputs and turns them into a list of CAN message for a particular make/model vehicle, and **boardd** is a process that talks to the vehicle through the panda interface to send (and receive) physical CAN messages. The two proccesses communicate with each other using a pub-sub messaging framework called [cereal](https://github.com/commaai/cereal) where `pm` denotes that a process is a topic publisher, and `sm` denotes that a process is a topic subscriber. 

This separation of concerns into purposeful processes that communicate through pub-sub messaging will be covered in greater detail in the next section.

## Inter-Process Communication

So far, we've mentioned two processes: **controlsd** and **boardd**. This section will give a brief overview of the major processes and how they communicate with each other. You may have noticed that processes all end with the letter d. That's a nod to linux daemons, background processes that have no user control. Similarly, openpilot's processes run without any direct user intervention. 

- **camerad**: interfaces with available camera hardware to generate images, which are encoded as videos 
- **sensorsd**: interfaces with available sensor hardware to generate data from the hardware's IMU
- **boardd**: interfaces with vehicle's CAN network and panda's ublox GPS chip to send and generate data
- **ubloxd**: converts raw GPS messages into realtime GPS data
- **locationd**: combines inputs from sensors, gps, and model into realtime estimates for movement and position
- **modeld**: transforms camera images into estimates for movement, future position, and the location of other vehicles
- **plannerd**: [TODO](#)
- **controlsd**: combines a wide range of inputs to produce a car-specific messages for a desired future state 
- **paramsd**: [TODO](#)
- **thermald**: monitors the vitals of the hardware running openpilot
- **loggerd**: logs and uploads messages history and videos
- **athenad**: opens a websocket connection so that message history and videos can be uploaded to the cloud

In the openpilot process map below, processes are represented by nodes, and the topics that they publish-to and subscribe-to are represented by the arrows between the nodes. A process is a publisher of a topic if the arrow points away from the process and is a subscriber if the arrow points toward the process. So, for example, the **thermald** proccess publishes `deviceState` and subscribes to `managerState`, `pandaState`, and `gpsLocationExternal`.

![pub_sub](https://raw.githubusercontent.com/barbinbrad/openpilot/master/docs/assets/pub_sub.png)

In the cereal pub-sub framework, any process can subscribe to any topic, but each topic can only have one publisher. Messages are stored and exchanged in [Cap'n Proto](https://capnproto.org/) format, an extremely fast and lightweight way to send objects (like JSON) as binary (like ProtoBufs). The structure of every message is contained in the [log.capnp](https://github.com/commaai/cereal/blob/master/log.capnp) file in the cereal sub-module.


## Cereal

In the log.capnp file, a single `Event` struct contains all the topic message definitions:

```capnp
# cereal/log.capnp

struct Event {
  logMonoTime @0 :UInt64;  # nanoseconds
  ...
  union {
    ...
    # ********** openpilot daemon msgs **********
    can @5 :List(CanData); # CAN received
    controlsState @7 :ControlsState;
    sensorEvents @11 :List(SensorEventData);
    pandaState @12 :PandaState;
    radarState @13 :RadarState;
    liveTracks @16 :List(LiveTracks);
    sendcan @17 :List(CanData); # CAN sent
    liveCalibration @19 :LiveCalibrationData;
    carState @22 :Car.CarState;
    ...
  }
}
```
Notice how both `can` and `sendcan` use the type `List(CanData)`. `List` is a primative type to Cap'n Proto, and `CanData` is a struct that's also defined in log.capnp. The `CanData` struct is composed exclusively of primative types, so no further definition is needed:

```capnp
# cereal/log.capnp

struct CanData {
  address @0 :UInt32; # 11 or 29-bit address
  busTime @1 :UInt16; 
  dat     @2 :Data; # up to 8 bytes of data
  src     @3 :UInt8; # 0-3 or 128-131
}
```

In C++, accessing a property `foo` from Cap'n Proto structs is as easy as calling `getFoo()`. If `foo` is a struct, `getFoo()` returns a `Foo::Reader`, a read-only class. Otherwise, if `foo` is a primative, `getFoo()` returns the primative value. To get the top-level `Event` struct, we called use a special method called `getRoot()`, which is cast as the root type `cereal::Event`:

```cpp
// selfdrive/boardd/boardd.cc

// get the root struct
cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
// pass the sendcan struct to the panda->can_send method
panda->can_send(event.getSendcan());
```

Calling `event.getSendcan()` therefore returns a `List<cereal::CanData>::Reader`, because `List` is primative but `CanData` a custom struct. Then calling `getAddress()` on the `Reader`, returns a value.

```cpp
// selfdrive/boardd/panda.cc

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
    
    const int msg_count = can_data_list.size();
    for (int i = 0; i < msg_count; i++) {
        // access structs in a list
        auto cmsg = can_data_list[i];
        // get the properties of the struct
        uint32_t address = cmsg.getAddress();
        uint8_t src = cmsg.getSrc();
    }
    ...
    usb_bulk_write(3, (unsigned char*)send.data(), send.size(), 5);
}
```

In python, reading Cap'n Proto messages is much less verbose. Here we read the deviceState messages:

```python
# selfdrive/controls/controlsd.py

import cereal.messaging as messaging
...
sm = messaging.SubMaster(['deviceState', ...)
...

while True:
    # Create events for battery, temperature, disk space, and memory
    if EON and sm['deviceState'].batteryPercent < 1 and sm['deviceState'].chargingError:
      # at zero percent battery, while discharging, OP should not allowed
      events.add(EventName.lowBattery)
    if sm['deviceState'].thermalStatus >= ThermalStatus.red:
      events.add(EventName.overheat)
    if sm['deviceState'].freeSpacePercent < 7 and not SIMULATION:
      # under 7% of space free no enable allowed
      events.add(EventName.outOfSpace)

```

Publishing a Cap'n'proto messages is just as easy. Here we see how **paramsd** uses cereal's messaging library to publish a message  on the `liveParameters` topic every time it receives a message on the `liveLocationKalman` topic:

```python
# selfdrive/locationd/paramsd.py
import cereal.messaging as messaging

# subscribe to a list of topics
sm = messaging.SubMaster(['liveLocationKalman', 'carState'], poll=['liveLocationKalman'])
# publish a list of topics
pm = messaging.PubMaster(['liveParameters'])

while True:
    # when the liveLocationKalman receives a new message
    if sm.updated['liveLocationKalman']:
      
      msg = messaging.new_message('liveParameters')
      msg.logMonoTime = sm.logMonoTime['carState']
      ...
     
      msg.liveParameters.posenetValid = True
      msg.liveParameters.sensorValid = True
      msg.liveParameters.steerRatio = float(x[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactor = float(x[States.STIFFNESS])
      msg.liveParameters.angleOffsetAverageDeg = angle_offset_average
      msg.liveParameters.angleOffsetDeg = angle_offset    
       msg.liveParameters.valid = all((
        abs(msg.liveParameters.angleOffsetAverageDeg) < 10.0,
        abs(msg.liveParameters.angleOffsetDeg) < 10.0,
        0.2 <= msg.liveParameters.stiffnessFactor <= 5.0,
        min_sr <= msg.liveParameters.steerRatio <= max_sr,
      ))  
      ....
      pm.send('liveParameters', msg)
```

Notice how the Cap'n Proto struct from cereal matches the message properties, but not all properties are required, such as `gyroBasis`. In the case that properties are not defined, they will be set to `0` or `false`:

```capnpn
# cereal/log.capnp

liveParameters @61 :LiveParametersData;

...

struct LiveParametersData {
  valid @0 :Bool;
  gyroBias @1 :Float32; #
  angleOffsetDeg @2 :Float32;
  angleOffsetAverageDeg @3 :Float32;
  stiffnessFactor @4 :Float32;
  steerRatio @5 :Float32;
  sensorValid @6 :Bool;
  yawRate @7 :Float32;
  posenetSpeed @8 :Float32;
  posenetValid @9 :Bool;
}

```

Now that we understand what the major processes are and how the talk to each other, lets take a look at how data is persisted for time travel debugging and machine learning.

## Logs

There are two types of log files in openpilot. Cap'n Proto logs, and videos. The two are used in conjuction to replay drives, diagnose problems, train machine learning models, and reverse engineer CAN message definitions. In this section, we'll focus on the Cap'n Proto logs and leave video for later in our introduction. So what is a Cap'n Proto log file?

To answer that question, we'll return our `Event` struct from cereal's [log.capnp](https://github.com/commaai/cereal/blob/master/log.capnp) file.

```capnp
struct Event {
  logMonoTime @0 :UInt64;  # nanoseconds
  valid @63 :Bool = true
  ...

  union {
    ...
    gpsNMEA @3 :GPSNMEAData;
    can @5 :List(CanData); 
    controlsState @7 :ControlsState;
    sensorEvents @11 :List(SensorEventData);
    ...
  }
}
```

The important thing to understand here is that `union` is like an `enum`, meaning that each instance of `Event` contains exactly one of the message definitions defined in the `union`. Thus, each `Event` has a `logMonoTime`, a `valid` property, and a message.

Now that we know what the `Event` struct is, let's have a look at 20ms of decompressed log data [converted](https://github.com/commaai/log_reader_js) to JSON. You may notice that the log is just a series of `Event` instances:

```json
[{
   "LogMonoTime":"31132127675929",
   "Valid":true,
   "SensorEvents":[
      {
         "Version":104,
         "Sensor":5,
         "Type":16,
         "Timestamp":"31132121969660",
         "UncalibratedDEPRECATED":false,
         "GyroUncalibrated":{
            "V":[
               -0.0488739013671875,
               0,
               0.0122222900390625,
               -0.05096435546875,
               0.00665283203125,
               0.013031005859375
            ],
            "Status":0
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":1,
         "Type":1,
         "Timestamp":"31132121969660",
         "UncalibratedDEPRECATED":false,
         "Acceleration":{
            "V":[
               9.889938354492188,
               0.772552490234375,
               0.090179443359375
            ],
            "Status":3
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":4,
         "Type":4,
         "Timestamp":"31132121969660",
         "UncalibratedDEPRECATED":false,
         "Gyro":{
            "V":[
               0.0020904541015625,
               -0.00665283203125,
               -0.0008087158203125
            ],
            "Status":3
         },
         "Source":0
      }
   ]
},
{
   "LogMonoTime":"31132130390148",
   "Valid":true,
   "Can":[
      {
         "Address":404,
         "BusTime":23453,
         "Dat":"EQAXAAKEWKM=",
         "Src":1
      },
      {
         "Address":405,
         "BusTime":23573,
         "Dat":"EQAAAAaEWJE=",
         "Src":1
      },
      {
         "Address":452,
         "BusTime":47969,
         "Dat":"BK4XCFAA/+0=",
         "Src":0
      },
      ...
   ]
},
{
   "LogMonoTime":"31132135487491",
   "Valid":true,
   "RoadCameraState":{
      "FrameId":1,
      "EncodeId":0,
      "TimestampEof":"31132109621000",
      "FrameLength":5419,
      "IntegLines":5408,
      "GlobalGain":510,
      "LensPos":635,
      "LensSag":0,
      "LensErr":0,
      "LensTruePos":400,
      "Image":"",
      "GainFrac":1,
      "FocusVal":[
         
      ],
      "FocusConf":[
         
      ],
      "SharpnessScore":[
         
      ],
      "RecoverState":0,
      "FrameType":0,
      "TimestampSof":"0",
      "Transform":[
         1,
         0,
         0,
         0,
         1,
         0,
         0,
         0,
         1
      ],
      "AndroidCaptureResult":{
         "Sensitivity":0,
         "FrameDuration":"0",
         "ExposureTime":"0",
         "RollingShutterSkew":"0",
         "ColorCorrectionTransform":[
            
         ],
         "ColorCorrectionGains":[
            
         ],
         "DisplayRotation":0
      }
   }
},
{
   "LogMonoTime":"31132140527491",
   "Valid":true,
   "Can":[
      {
         "Address":865,
         "BusTime":24666,
         "Dat":"AEcAAJYAAIM=",
         "Src":2
      },
      {
         "Address":865,
         "BusTime":53093,
         "Dat":"AEcAAJYAAIM=",
         "Src":0
      },
      {
         "Address":442,
         "BusTime":28957,
         "Dat":"ERN9+QAd0kw=",
         "Src":1
      },
      ...
   ]
},
{
   "LogMonoTime":"31132142300252",
   "Valid":true,
   "GpsLocationExternal":{
      "Flags":1,
      "Latitude":34.092524499999996,
      "Longitude":-118.32247489999999,
      "Altitude":64.429,
      "Speed":0.008999999612569809,
      "BearingDeg":0,
      "Accuracy":0.621999979019165,
      "Timestamp":"1575969208300",
      "Source":6,
      "VNED":[
         -0.00800000037997961,
         -0.004000000189989805,
         0.0010000000474974513
      ],
      "VerticalAccuracy":0.7590000033378601,
      "BearingAccuracyDeg":93.0456771850586,
      "SpeedAccuracy":0.17900000512599945
   }
},
{
   "LogMonoTime":"31132141781762",
   "Valid":true,
   "UbloxRaw":"tWIBB1wAPLZHDOMHDAoJDRz36gMAAAvl5hEDAeoMU2x5uT0bUhSt+wAAYXsBAG4CAAD3AgAA+P////z///8BAAAACQAAAAAAAACzAAAA+PmNAKUAAAASGSY7AAAAAAAAAACXnQ=="
},
{
   "LogMonoTime":"31132147796294",
   "Valid":true,
   "SensorEvents":[
      {
         "Version":104,
         "Sensor":5,
         "Type":16,
         "Timestamp":"31132131796320",
         "UncalibratedDEPRECATED":false,
         "GyroUncalibrated":{
            "V":[
               -0.0525360107421875,
               0.0158538818359375,
               0.0122222900390625,
               -0.05096435546875,
               0.00665283203125,
               0.013031005859375
            ],
            "Status":0
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":1,
         "Type":1,
         "Timestamp":"31132131796320",
         "UncalibratedDEPRECATED":false,
         "Acceleration":{
            "V":[
               9.748748779296875,
               0.8108367919921875,
               0.01361083984375
            ],
            "Status":3
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":4,
         "Type":4,
         "Timestamp":"31132131796320",
         "UncalibratedDEPRECATED":false,
         "Gyro":{
            "V":[
               -0.0015716552734375,
               0.0092010498046875,
               -0.0008087158203125
            ],
            "Status":3
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":5,
         "Type":16,
         "Timestamp":"31132141622981",
         "UncalibratedDEPRECATED":false,
         "GyroUncalibrated":{
            "V":[
               -0.0537567138671875,
               0.00244140625,
               0.013427734375,
               -0.05096435546875,
               0.00665283203125,
               0.013031005859375
            ],
            "Status":0
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":1,
         "Type":1,
         "Timestamp":"31132141622981",
         "UncalibratedDEPRECATED":false,
         "Acceleration":{
            "V":[
               9.954544067382812,
               0.87066650390625,
               0.094970703125
            ],
            "Status":3
         },
         "Source":0
      },
      {
         "Version":104,
         "Sensor":4,
         "Type":4,
         "Timestamp":"31132141622981",
         "UncalibratedDEPRECATED":false,
         "Gyro":{
            "V":[
               -0.0027923583984375,
               -0.00421142578125,
               0.000396728515625
            ],
            "Status":3
         },
         "Source":0
      }
   ]
}]

```

Phew! That's a lot data for 20ms. What you're seeing is that every message sent between processes is saved to a raw log file, called an `rlog`. This file is uploaded to comma server's (with the user's permission) when the device is able to connect to WiFi.

With so much data being saved and sent, the name of the game is compression. There are two main ways that openpilot solves the data compression problem. The first is through the data format, and the second is through sampling rates.

When it comes to data format, openpilot uses Cap'n Proto for it's small in-memory footprint, language interoperability, and lightning speed. For compression, we use the `bzip` algorithm, which favors smaller size over speed.

Let's now take a look at how the **loggerd** process saves all messages to disk in bz2 format:

```cpp
// selfdrive/loggerd/loggerd.cc

Poller * poller = Poller::create();

  // iterate through all the services (topics)
  for (const auto& it : services) {
    // subscribe to every topic
    SubSocket * sock = SubSocket::create(s.ctx, it.name);
    poller->registerSocket(sock);
    ...
  }

  // poll for new messages on all topics
  for (auto sock : poller->poll(1000)) {
      while (msg = sock->receive(true)) {
          // decide whether the message should be logged in the qlog
          const bool in_qlog = qs.freq != -1 && (qs.counter++ % qs.freq == 0);
          // save the message to disk
          logger_log(&s.logger, (uint8_t *)msg->getData(), msg->getSize(), in_qlog);
      }
  }

```

The **loggerd** process subscribes to every message, and calls `logger_log` on the message data, which eventually uses [libbz2](https://docs.oracle.com/cd/E88353_01/html/E37842/libbz2-3.html) to compress the byte array message into bz2 format, and write it to disk.  

```cpp
// selfdrive/loggerd/logger.h

inline void write(void* data, size_t size) {
    ...
    do {
        ...
        BZ2_bzWrite(&bzerror, bz_file, data, size);
    } while (bzerror == BZ_IO_ERROR && errno == EINTR);
    ...
}

```

We've just described how an `rlog.bz2` file is created. But the careful reader may have noticed something called a `qlog`. When **loggerd** calls `logger_log` it passes an argument called `in_qlog`. The `qlog.bz2` file is a subset of `rlog.bz2`. Instead of saving all messages, the `qlog` file samples every  n-th message as defined by the cereal services file:

```python
# cereal/services.py

services = {
  # service: (should_log, frequency, qlog decimation (optional))
  "sensorEvents": (True, 100., 100),
  "gpsNMEA": (True, 9.),
  "deviceState": (True, 2., 1),
  "can": (True, 100.),
  "controlsState": (True, 100., 10),
  "pandaState": (True, 2., 1),
  "radarState": (True, 20., 5),
  "roadEncodeIdx": (True, 20., 1),
  "liveTracks": (True, 20.),
  "sendcan": (True, 100.),
  "logMessage": (True, 0.),
  "liveCalibration": (True, 4., 4),
  ...
}
```

The services dictionary specifies whether the messages from that service should be logged, how many times the message is sent per second (Hz), and the sampling rate for `qlog`, called decimation.

Let's look at a few examples to see how this works. The `deviceState` message is sent 2 times per second (2Hz), and every message is saved to the `qlog`. The `radarState` message is sent 20 times per second, and one out of every 5 messages is saved to the `qlog`. The `can` message is sent 100 times per second, but no messages are saved to the `qlog`. The purpose of the `qlog` is for streaming real-time data for users with a cellular data connection. Because cellular data is more expensive, the data resource is limited.

Before moving on to the next section, let's revisit the code for deciding whether a message should be in `qlog`: 

```cpp
// selfdrive/loggerd/loggerd.cc

  // iterate through the services
  for (const auto& it : services) {
    // subscribe to all messages topics
    ...
    // store the message sampling rates for qlog
    qlog_states[sock] = {.counter = 0, .freq = it.decimation};
  }

  // poll for new messages on all topics
  for (auto sock : poller->poll(1000)) {

      QlogState &qs = qlog_states[sock];

      while (msg = sock->receive(true)) {
          // decide whether the message should be logged in the qlog
          const bool in_qlog = qs.freq != -1 && (qs.counter++ % qs.freq == 0);
          // save the message to disk
          logger_log(&s.logger, (uint8_t *)msg->getData(), msg->getSize(), in_qlog);

          // check to see if we need a new log file
          rotate_if_needed();
      }
  }

```

Now, that we understand how `rlog.bz2` and `qlog.bz2` files are created, we need to give some thought to how to break up the files into mananageble chunks. To do this, we use the function `rotate_if_needed` seen in the code above.

The purpose of the `rotate_if_needed` function is to break log files up into 60-second chunks, called segments. Because all log files are named `rlog.bz2` or `qlog.bz2`, we need some means of differentiating between them. To accomplish this, we use folders with the following naming convention `<LOG_ROOT>/<ROUTE>--<SEGMENT>/`.

`LOG_ROOT` is defined as:
```cpp
// selfdrive/hardware.h

inline std::string log_root() {
    return Hardware::PC() ? HOME + "/.comma/media/0/realdata" : "/data/media/0/realdata";
}
```

`ROUTE` is a date/timestamp that is defined once on power-up. It looks likegit  `2021-09-16--19-49-40`, which is generated by the following function:
```cpp
// selfdrive/loggerd/logger.cc

std::string logger_get_route_name() {
  char route_name[64] = {'\0'};
  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  return route_name;  
}
```

Finally, `SEGMENT` is a counter variable, converted to string. The function `rotate_if_needed` eventually increments the segment number when a new segment is needed. By default, segments are 60 seconds long, such that segment 42 represents the 42nd minute of the drive. Thus, a valid path for an `rlog` of the 42nd minute of a drive on 9/16/2021 could be: `/data/media/0/realdata/2021-09-16--19-49-40--42/rlog.bz2`.


## Persistent Parameters

While messages provide access to real-time state, and logs give us the ability to look back in time, sometimes we need to persist global data between proccesses. For example, we don't want to ask the user whether to upload logs every time we restart the device or connect to WiFi. It's better to save the user selection as a persistent parameter.

For a lot of developers, persistent key/value storage might sound like a job for a database. But remote databases are not an option because the hardware cannot always access the internet, and local databases like sqlite contain unnecessary complexity for our simple needs. In order to run many processes at near real-time speeds, everything must be optimized for speed and efficiency. 

Thus, the method employed by openpilot for persisting key/value data is to use the filesystem, using a library called **params**. Params stores key/values on the Linux filesystem here:

```cpp
// selfdrive/hardware.h

inline std::string params() {
  return Hardware::PC() ? HOME + "/.comma/params" : "/data/params";
}
```

We haven't talked about `Hardware::PC()` yet, but you can probably guess that it is a method to identify whether we're using a PC (without Android) or not. Great! Now let's take a look at some of the keys that we'll persist and the policies for when to clear the values:

```cpp
// selfdrive/common/params.cc

std::unordered_map<std::string, uint32_t> keys = {
    {"AccessToken", CLEAR_ON_MANAGER_START | DONT_LOG},
    {"ApiCache_DriveStats", PERSISTENT},
    {"ApiCache_Device", PERSISTENT},
    {"ApiCache_Owner", PERSISTENT},
    {"ApiCache_NavDestinations", PERSISTENT},
    {"AthenadPid", PERSISTENT},
    {"CalibrationParams", PERSISTENT},
    {"CarBatteryCapacity", PERSISTENT},
    {"CarParams", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"CarParamsCache", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"CarVin", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"CommunityFeaturesToggle", PERSISTENT},
    {"ControlsReady", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"CurrentRoute", CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON},
    {"DisableRadar", PERSISTENT}, // WARNING: THIS DISABLES AEB
    {"EndToEndToggle", PERSISTENT},
    {"CompletedTrainingVersion", PERSISTENT},
    {"DisablePowerDown", PERSISTENT},
    {"DisableUpdates", PERSISTENT},
    {"EnableWideCamera", CLEAR_ON_MANAGER_START},
    {"DoUninstall", CLEAR_ON_MANAGER_START},
    {"DongleId", PERSISTENT},
    ...
    {"GsmRoaming", PERSISTENT},
    {"HardwareSerial", PERSISTENT},
    {"HasAcceptedTerms", PERSISTENT},
    {"IMEI", PERSISTENT},
    {"InstallDate", PERSISTENT},
    {"IsDriverViewEnabled", CLEAR_ON_MANAGER_START},
    {"IsLdwEnabled", PERSISTENT},
    {"IsMetric", PERSISTENT},
    {"IsOffroad", CLEAR_ON_MANAGER_START},
    {"IsOnroad", PERSISTENT},
    {"IsRHD", PERSISTENT},
    {"IsTakingSnapshot", CLEAR_ON_MANAGER_START},
    {"IsUpdateAvailable", CLEAR_ON_MANAGER_START},
    {"UploadRaw", PERSISTENT},
    ...
    {"JoystickDebugMode", CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_OFF},
};
```

The **params** library has two primary methods: `Params().get(key)` and `Params().put(key, value)`. The `Params().get(key)` method is blocking, and `Params.put(key, value)` [writes atomically](https://lwn.net/Articles/457667/). Let's take a look at how these methods are used to cache the expensive operation of determining the car firmware and VIN number:

```python
# selfdrive/car/car_helpers.py

# try to load the car params from the cache
cached_params = Params().get("CarParamsCache")
if cached_params is not None:
    # use the cap'n proto struct to convert the cached bytes into a python object
    cached_params = car.CarParams.from_bytes(cached_params)

if cached_params is not None and len(cached_params.carFw) > 0 and cached_params.carVin is not VIN_UNKNOWN:
    # use the cached params
    vin = cached_params.carVin
    car_fw = list(cached_params.carFw)
else:
    # do expensive operation to get vin and firmware with CAN bus messaging
    vin = get_vin(logcan, sendcan, bus)
    car_fw = get_fw_versions(logcan, sendcan, bus)

...
# persist the car's VIN number
Params().put("CarVin", vin)
```

In the example above, we check our persistent storage for the `carParamsCache` key. If the value is present, we avoid doing the expensive `get_vin` and `get_fw_versions` function calls.

In addition to `get` and `put`, the **params** library has the methods `get_bool` and `put_bool`, which are used for true/false values. In C++, the methods are `getBool` and `putBool`. In the next example, we see how the `ControlsReady` param is used to tell the **boardd** process to wait for the **controlsd** process to finish loading the car parameters:

```python
# selfdrive/controls/controlsd.py

# after fingerprinting and loading the car interface
Params().put_bool("ControlsReady", True)
```

```cpp
// selfdrive/boardd/boardd.cc
p = Params()
...
// wait for the controls ready param to be set
while (true) {
    ...
    if (p.getBool("ControlsReady")) {
      params = p.get("CarParams");
      if (params.size() > 0) break;
    }
    util::sleep_for(100);
  }
```

This code is called when the **boardd** process starts. It waits for the `CarParams` to be set so that it can configure the panda's "safety hooks" according to the make/model of the car. Recall from list of keys that `ControlsReady` has a policy of `CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON`. That means that any time the device, car, or panda is disconnected, the parameter will be cleared. 

In the next section, we'll learn about the fingerprinting process, car interfaces, and manufacturer-specific safety hooks.


## Fingerprints & Interfaces

So far, we've looked at the mechanics of how data is transported in openpilot. In the next few sections, we'll try to understand the interfaces required to support self-driving in hundreds of different cars. To do that, let's start by defining what a fingerprint is.

In openpilot, a *fingerprint* is a dictionary of CAN message IDs and data length (in bytes). A fingerprint is used to identify a car based on the set of CAN messages sent over a few seconds, while ignoring the content of the messages. 

Now suppose there are only two cars, `XTRAIL` and `LEAF`, in the universe. And suppose that each car can have two different possible fingerprints, depending on some manufacturing variability (like the model year). Two cars with two fingerprints gives us four possible fingerprints:

```python
# selfdrive/car/nissan/values.py
fingerprints = {
  CAR.XTRAIL: [
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 520: 2, 523: 6, 548: 8, 645: 8, 658: 8, 665: 8, 666: 8, 674: 2, 682: 8, 683: 8, 689: 8, 723: 8, 758: 3, 768: 2, 783: 3, 851: 8, 855: 8, 1041: 8, 1055: 2, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1111: 4, 1227: 8, 1228: 8, 1247: 4, 1266: 8, 1273: 7, 1342: 1, 1376: 6, 1401: 8, 1474: 2, 1497: 3, 1821: 8, 1823: 8, 1837: 8, 2015: 8, 2016: 8, 2024: 8
    },
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 520: 2, 523: 6, 527: 1, 548: 8, 637: 4, 645: 8, 658: 8, 665: 8, 666: 8, 674: 2, 682: 8, 683: 8, 689: 8, 723: 8, 758: 3, 768: 6, 783: 3, 851: 8, 855: 8, 1041: 8, 1055: 2, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1111: 4, 1227: 8, 1228: 8, 1247: 4, 1266: 8, 1273: 7, 1342: 1, 1376: 6, 1401: 8, 1474: 8, 1497: 3, 1534: 6, 1792: 8, 1821: 8, 1823: 8, 1837: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 2015: 8, 2016: 8, 2024: 8
    },
  ],
  CAR.LEAF: [
    {
      2: 5, 42: 6, 264: 3, 361: 8, 372: 8, 384: 8, 389: 8, 403: 8, 459: 7, 460: 4, 470: 8, 520: 1, 569: 8, 581: 8, 634: 7, 640: 8, 644: 8, 645: 8, 646: 5, 658: 8, 682: 8, 683: 8, 689: 8, 724: 6, 758: 3, 761: 2, 783: 3, 852: 8, 853: 8, 856: 8, 861: 8, 944: 1, 976: 6, 1008: 7, 1011: 7, 1057: 3, 1227: 8, 1228: 8, 1261: 5, 1342: 1, 1354: 8, 1361: 8, 1459: 8, 1477: 8, 1497: 3, 1549: 8, 1573: 6, 1821: 8, 1837: 8, 1856: 8, 1859: 8, 1861: 8, 1864: 8, 1874: 8, 1888: 8, 1891: 8, 1893: 8, 1906: 8, 1947: 8, 1949: 8, 1979: 8, 1981: 8, 2016: 8, 2017: 8, 2021: 8, 643: 5, 1792: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2015: 8
    },
    # 2020 Leaf SV Plus
    {
      2: 5, 42: 8, 264: 3, 361: 8, 372: 8, 384: 8, 389: 8, 403: 8, 459: 7, 460: 4, 470: 8, 520: 1, 569: 8, 581: 8, 634: 7, 640: 8, 643: 5, 644: 8, 645: 8, 646: 5, 658: 8, 682: 8, 683: 8, 689: 8, 724: 6, 758: 3, 761: 2, 772: 8, 773: 6, 774: 7, 775: 8, 776: 6, 777: 7, 778: 6, 783: 3, 852: 8, 853: 8, 856: 8, 861: 8, 943: 8, 944: 1, 976: 6, 1008: 7, 1009: 8, 1010: 8, 1011: 7, 1012: 8, 1013: 8, 1019: 8, 1020: 8, 1021: 8, 1022: 8, 1057: 3, 1227: 8, 1228: 8, 1261: 5, 1342: 1, 1354: 8, 1361: 8, 1402: 8, 1459: 8, 1477: 8, 1497: 3, 1549: 8, 1573: 6, 1821: 8, 1837: 8
    },
  ],
}
```

The fingerprint values are a set of key/value pairs where the key is the message ID, and the value is the data length. Each time openpilot starts, we don't know what kind of car we have. So we start listening to CAN messages from the car.

For the sake of example, suppose that we receive a message ID 2 with a length of 5 bytes. Given the information provided above, any of the four fingerprints could be valid. Now suppose that we receive a message ID 264 with length 3. Now we can eliminate both fingerprints from `CAR.XTRAIL` because niether of the fingerprints contains message ID 264. Similarly if we receive message 42 with length 8, we can eliminate the first `CAR.LEAF` fingerprint. If, after listening to many more messages, the second `CAR.LEAF` fingerprint has not been eliminated in this way, we can conclude that the car is a Nissan Leaf.

We use this conclusion to load correct the CAN Dictionaries (DBCs), important information about the geometry and featureset of the car, and functions for reading and writing make/model-specific CAN messages. 

Reading the CAN bus is done through `selfdrive/car/<make>/carstate` and writing to the CAN bus is done through `selfdrive/car/<make>/carcontroller`. 

Every make of car contains a manufactuer-specific, `CarInterface`, `CarState`, and `CarController`, which can be found in the `selfdrive/car/<make>` directory. When needed, these classes provide differentiation between the vehicle's model.

Now that we have a basic grasp on fingerprinting, let's revisit the first lines of code we looked at. Recall that the line `CI.apply(CC)` transforms calculations for acceleration and steering angle into make/model specific CAN messages on each loop of the process:

```python
# selfdrive/controls/controlsd.py

# send car controls over can
can_sends = CI.apply(CC)
pm.send('sendcan', can_list_to_can_capnp(can_sends))
```

The car interface `CI` is determined by the `get_car` function, which follows the fingerprinting process of elimination described above.

```python
# selfdrive/controls/controlsd.py

CI, CP = get_car(can_sock, pm.sock['sendcan'])
```

The `get_car` function is called once on the intialization of the **controlsd** process. It passes the CAN Rx topic subscriber, `can_sock`, and the CAN Tx topic publisher, `sendcan`, which allows the `fingerprint` function to send and receive CAN messages. 

The goal of the `get_car` function is to dynamically `__import__` the correct `CarInterface`, `CarController`, and `CarState` from the `selfdrive/car` directories:

```python
# imports from directory selfdrive/car/<make>/
interface_names = _get_interface_names()
interfaces = load_interfaces(interface_names)

def load_interfaces(brand_names):
  ret = {}
  for brand_name in brand_names:
    path = ('selfdrive.car.%s' % brand_name)
    CarInterface = __import__(path + '.interface').CarInterface 
    CarState = __import__(path + '.carstate').CarState
    CarController = __import__(path + '.carcontroller').CarController
    
    for model_name in brand_names[brand_name]:
      ret[model_name] = (CarInterface, CarController, CarState)
  return ret

```

The `candidate` from the fingerprinting process is used to select the correct set of `CarInterface`, `CarController`, and `CarState` from the list created by `load_interface`. A valid `candidate` is not the key/value dictionary. It is the higher-level key, like `CAR.COROLLA`, for example:

```python
def get_car(logcan, sendcan):
  candidate, fingerprints, vin, car_fw, source, exact_match = fingerprint(logcan, sendcan)
  ...
  CarInterface, CarController, CarState = interfaces[candidate]
  car_params = CarInterface.get_params(candidate, fingerprints, car_fw)
  ...
  return CarInterface(car_params, CarController, CarState), car_params
```

The `CarInterface.get_params` function takes a fingerprint as an input and returns make and model-specific parameters about the car, `CP`. Here is the `get_params` function for all Toyotas. Notice how some things like the `SafetyModel` are the same across all models, while other paramaters like the `safetyParam` and `steerRatio` differ across models, depending on the `candidate`:


```python
# selfdrive/car/toyota/interface.py

def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=[]): 
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.carName = "toyota"
    ret.safetyModel = car.CarParams.SafetyModel.toyota

    ret.steerActuatorDelay = 0.12  # Default delay, Prius has larger delay
    ret.steerLimitTimer = 0.4

   ...
   elif candidate == CAR.COROLLA:
      stop_and_go = False
      ret.safetyParam = 88
      ret.wheelbase = 2.70
      ret.steerRatio = 18.27
      tire_stiffness_factor = 0.444  # not optimized yet
      ret.mass = 2860. * CV.LB_TO_KG + STD_CARGO_KG  # mean between normal and hybrid
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]
      ret.lateralTuning.pid.kf = 0.00003   # full torque for 20 deg at 80mph means 0.00007818594

    elif candidate == CAR.LEXUS_RX:
      stop_and_go = True
      ret.safetyParam = 73
      ret.wheelbase = 2.79
      ret.steerRatio = 14.8
      tire_stiffness_factor = 0.5533
      ret.mass = 4387. * CV.LB_TO_KG + STD_CARGO_KG
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.05]]
      ret.lateralTuning.pid.kf = 0.00006
   ...
```

Now that we've loaded the make's interface, and stored the parameters of the model, recall that `can_sends = CI.apply(CC)` turns calculations for acceleration and steering angle into model-specific CAN messages. But the `apply` method is just a convenient wrapper for combining the car interface, `CI`, the car controller, `CC` and car state `CS` through the car controller's `update` method.

```python
# selfdrive/car/toyota/interface

# pass in a car.CarControl
# to be called @ 100hz
class CarInterface:
  ...
  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators, ...)

    self.frame += 1
    return can_sends
```

The `frame` property is used to keep track of time, and the `actuators` argument contains the pertinent information about steering and acceleration. Here's the cereal struct definition for `actuators`:

```capnp
struct Actuators {
  # range from -1.0 - 1.0
  steer @2: Float32;
  steeringAngleDeg @3: Float32;

  accel @4: Float32; # m/s^2
  longControlState @5: LongControlState;

  enum LongControlState @0xe40f3a917d908282{
   off @0;
   pid @1;
   stopping @2;
   starting @3;
}
```

The `enabled` argument determines whether we should override the acceleration and steering, and `frame` determines how often the message should be sent. The car controllers are quite different depending on the manufacturer and underlying hardware, but here is a simple example of a car controller creating gas and brake CAN messages using the `actuators.accel` value:

```python
# selfdrive/car/honda/carcontroller.py

class CarController:
  def update(self, enabled, CS, frame, actuators, ...):
    P = self.params
    ...
    if enabled:
      gas, brake = compute_gas_brake(actuators.accel, CS.out.vEgo, CS.CP.carFingerprint)
    else:
      accel = 0.0
      gas, brake = 0.0, 0.0

    can_sends = []

    # apply brake hysteresis 
    pre_limit_brake, self.braking, self.brake_steady = actuator_hystereses(brake, self.braking, self.brake_steady, CS.out.vEgo, CS.CP.carFingerprint)

    # wind brake from air resistance decel at high speed
    wind_brake = interp(CS.out.vEgo, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15])

    # send at 50Hz instead of 100Hz
    if (frame % 2) == 0:
      idx = frame // 2
      # decide whether to brake, and remember it
      apply_brake = clip(self.brake_last - wind_brake, 0.0, 1.0)
      apply_brake = int(clip(apply_brake * P.BRAKE_MAX, 0, P.BRAKE_MAX - 1))
      can_sends.append(hondacan.create_brake_command(self.packer, apply_brake,...))
      self.apply_brake_last = apply_brake
      
      if CS.CP.enableGasInterceptor:
        # way too aggressive at low speed without this
        gas_mult = interp(CS.out.vEgo, [0., 10.], [0.4, 1.0])
        apply_gas = clip(gas_mult * gas, 0., 1.)
        can_sends.append(create_gas_command(self.packer, apply_gas, idx))

    ...

    return can_sends

```

It's important to understand that the acceleration and steering values require post-processing for different makes and models. For example, Hondas have separate messages for gas and brake, while Toyotas have a single acceleration message.

The job of `selfdrive/car/<make>/carcontroller.py` is to translate the setpoints into manufacturer specific CAN messages. It is common for the `carcontroller` to create custom CAN messages to using an additional library like `selfdrive/car/honda/hondacan.py`, which creates manufacturer specific CAN messages using the `packer` library.

```python
# selfdrive/car/honda/hondacan.py

# CAN bus layout with relay
# 0 = ACC-CAN - radar side
# 1 = F-CAN B - powertrain
# 2 = ACC-CAN - camera side
# 3 = F-CAN A - OBDII port

def get_pt_bus(car_fingerprint):
  return 1 if car_fingerprint in HONDA_BOSCH else 0

def create_brake_command(packer, apply_brake, pump_on, ...):
  brakelights = apply_brake > 0
  brake_rq = apply_brake > 0
  pcm_fault_cmd = False

  values = {
    "COMPUTER_BRAKE": apply_brake,
    "BRAKE_PUMP_REQUEST": pump_on,
    "CRUISE_OVERRIDE": pcm_override,
    "CRUISE_FAULT_CMD": pcm_fault_cmd,
    "CRUISE_CANCEL_CMD": pcm_cancel_cmd,
    "COMPUTER_BRAKE_REQUEST": brake_rq,
    "SET_ME_1": 1,
    "BRAKE_LIGHTS": brakelights,
    ...
  }

  bus = get_pt_bus(car_fingerprint)
  return packer.make_can_msg("BRAKE_COMMAND", bus, values, idx)
```

The `packer` function is attached to the car controller and passed as an argument. It is intialized by with the name of the DBC file from [opendbc](https://github.com/commaai/opendbc), a set of CAN dictionaries.

The job of the `packer` is to `make_can_msg`. Here is the DBC file excerpt from the message above.

```python
# opendbc/honda_odyssey_exl_2018_generated.dbc
...
BO_ 506 BRAKE_COMMAND: 8 ADAS
 SG_ COMPUTER_BRAKE : 7|10@0+ (1,0) [0|1] "" EBCM
 SG_ SET_ME_X00 : 13|5@0+ (1,0) [0|1] "" EBCM
 SG_ BRAKE_PUMP_REQUEST : 8|1@0+ (1,0) [0|1] "" EBCM
 SG_ SET_ME_X00_2 : 23|3@0+ (1,0) [0|1] "" EBCM
 SG_ CRUISE_OVERRIDE : 20|1@0+ (1,0) [0|1] "" EBCM
 SG_ SET_ME_X00_3 : 19|1@0+ (1,0) [0|1] "" EBCM
 SG_ CRUISE_FAULT_CMD : 18|1@0+ (1,0) [0|1] "" EBCM
 SG_ CRUISE_CANCEL_CMD : 17|1@0+ (1,0) [0|1] "" EBCM
 SG_ COMPUTER_BRAKE_REQUEST : 16|1@0+ (1,0) [0|1] "" EBCM
...
```

To learn more about the DBC file protocol, check out [@energee's excellent article](https://medium.com/@energee/what-are-dbc-files-469a3bf9b04b).

## Safety Hooks

## Panda

## Hardware

## Kalman Filters, PID Loops & MPC

## Video

## Machine Learning

## Operating System
 
## APIs
