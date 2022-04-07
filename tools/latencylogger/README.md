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
```

## Examples
Plotting with relative starts each process at time=0 and gives a nice overview. Timestamps are visualized as diamonds. The opacity allows for visualization of overlapping services.
![relplot-1](https://user-images.githubusercontent.com/42323981/162108651-e0beee14-56e4-466d-8af1-cb37129fd94a.png)

Plotting without relative provides info about the frames relative time.
![plot-1](https://user-images.githubusercontent.com/42323981/162108694-fbfe907b-a1ee-4cc7-bc8b-162a7d9305d4.png)


Printed timestamps of a frame with internal durations.
```
Frame ID: 303
  camerad
    roadCameraState start of frame                       0.0
    wideRoadCameraState start of frame                   0.091926
    RoadCamera: Image set                                1.691696
    RoadCamera: Transformed                              1.812841
    roadCameraState published                            51.775466
    wideRoadCameraState published                        54.935164
    roadCameraState.processingTime                       1.6455530421808362
    wideRoadCameraState.processingTime                   4.790564067661762
  modeld
    Image added                                          56.628788
    Extra image added                                    57.459923
    Execution finished                                   75.091306
    modelV2 published                                    75.24797
    modelV2.modelExecutionTime                           20.00947669148445
    modelV2.gpuExecutionTime                             0.0
  plannerd
    lateralPlan published                                80.426861
    longitudinalPlan published                           85.722781
    lateralPlan.solverExecutionTime                      1.0600379901006818
    longitudinalPlan.solverExecutionTime                 1.3830000534653664
  controlsd
    Data sampled                                         89.436221
    Events updated                                       90.356522
    sendcan published                                    91.396092
    controlsState published                              91.77843
    Data sampled                                         99.885876
    Events updated                                       100.696855
    sendcan published                                    101.600489
    controlsState published                              101.941839
    Data sampled                                         110.087669
    Events updated                                       111.025365
    sendcan published                                    112.305921
    controlsState published                              112.70451
    Data sampled                                         119.692803
    Events updated                                       120.56774
    sendcan published                                    121.735016
    controlsState published                              122.142823
    Data sampled                                         129.264761
    Events updated                                       130.024282
    sendcan published                                    130.950364
    controlsState published                              131.281558
  boardd
    sending sendcan to panda: 250027001751393037323631   101.705487
    sendcan sent to panda: 250027001751393037323631      102.042462
    sending sendcan to panda: 250027001751393037323631   112.416961
    sendcan sent to panda: 250027001751393037323631      112.792269
    sending sendcan to panda: 250027001751393037323631   121.850952
    sendcan sent to panda: 250027001751393037323631      122.231103
    sending sendcan to panda: 250027001751393037323631   131.045206
    sendcan sent to panda: 250027001751393037323631      131.351296
    sending sendcan to panda: 250027001751393037323631   141.340592
    sendcan sent to panda: 250027001751393037323631      141.700744
```
