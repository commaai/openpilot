# LatencyLogger

LatencyLogger is a tool to track the time from first pixel to actuation. Timestamps are printed in a table as well as plotted in a graph. Start openpilot with `LOG_TIMESTAMPS=1` set to enable the necessary logging.

## Usage

```
$ python latency_logger.py -h
usage: latency_logger.py [-h] [--relative] [--demo] [--plot] [route_or_segment_name]

A tool for analyzing openpilot's end-to-end latency

positional arguments:
  route_or_segment_name
                        The route to print (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --relative            Make timestamps relative to the start of each frame (default: False)
  --demo                Use the demo route instead of providing one (default: False)
  --plot                If a plot should be generated (default: False)
  --offset              Offset service to better visualize overlap (default: False)
```
To timestamp an event, use `LOGT("msg")` in c++ code or `cloudlog.timestamp("msg")` in python code. If the print is warning for frameId assignment ambiguity, use `LOGT(frameId ,"msg")`.

## Examples

Timestamps are visualized as diamonds

|  | Relative  | Absolute |
| ------------- | ------------- | ------------- |
| Inline | ![inrel](https://user-images.githubusercontent.com/42323981/170559939-465df3b1-bf87-46d5-b5ee-5cc87dc49470.png) | ![inabs](https://user-images.githubusercontent.com/42323981/170559985-a82f87e7-82c4-4e48-a348-4221568dd589.png) |
| Offset | ![offrel](https://user-images.githubusercontent.com/42323981/170559854-93fba90f-acc4-4d08-b317-d3f8fc649ea8.png) | ![offabs](https://user-images.githubusercontent.com/42323981/170559782-06ed5599-d4e3-4701-ad78-5c1eec6cb61e.png) |

Printed timestamps of a frame with internal durations.
```
Frame ID: 1202
  camerad
    wideRoadCameraState start of frame                   0.0
    roadCameraState start of frame                       0.049583
    wideRoadCameraState published                        35.01206
    WideRoadCamera: Image set                            35.020028
    roadCameraState published                            38.508261
    RoadCamera: Image set                                38.520344
    RoadCamera: Transformed                              38.616176
    wideRoadCameraState.processingTime                   3.152403049170971
    roadCameraState.processingTime                       6.453451234847307
  modeld
    Image added                                          40.909841
    Extra image added                                    42.515027
    Execution finished                                   63.002552
    modelV2 published                                    63.148747
    modelV2.modelExecutionTime                           23.62649142742157
    modelV2.gpuExecutionTime                             0.0
  plannerd
    longitudinalPlan published                           69.715999
    longitudinalPlan.solverExecutionTime                 0.5619999719783664
  controlsd
    Data sampled                                         70.217763
    Events updated                                       71.037178
    sendcan published                                    72.278775
    controlsState published                              72.825226
    Data sampled                                         80.008354
    Events updated                                       80.787666
    sendcan published                                    81.849682
    controlsState published                              82.238323
    Data sampled                                         90.521123
    Events updated                                       91.626003
    sendcan published                                    93.413218
    controlsState published                              94.143989
    Data sampled                                         100.991497
    Events updated                                       101.973774
    sendcan published                                    103.565575
    controlsState published                              104.146088
    Data sampled                                         110.284387
    Events updated                                       111.183541
    sendcan published                                    112.981692
    controlsState published                              113.731994
  boardd
    sending sendcan to panda: 250027001751393037323631   81.928119
    sendcan sent to panda: 250027001751393037323631      82.164834
    sending sendcan to panda: 250027001751393037323631   93.569986
    sendcan sent to panda: 250027001751393037323631      93.92795
    sending sendcan to panda: 250027001751393037323631   103.689167
    sendcan sent to panda: 250027001751393037323631      104.012235
    sending sendcan to panda: 250027001751393037323631   113.109555
    sendcan sent to panda: 250027001751393037323631      113.525487
    sending sendcan to panda: 250027001751393037323631   122.508434
    sendcan sent to panda: 250027001751393037323631      122.834314
```
