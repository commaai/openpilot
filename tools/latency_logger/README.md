# LatencyLogger

LatencyLogger is a tool to track timestamps for each frame in the pipeline. Timestamps are printed in a table as well as plotted in a graph. All times are in milliseconds. 

## Usage

```
$ python latency_logger.py -h
usage: latency_logger.py [-h] [--relative_self] [--demo] [--plot] [route_name]

A helper to run timestamp print on openpilot routes

positional arguments:
  route_name       The route to print (default: None)

optional arguments:
  -h, --help       show this help message and exit
  --relative_self  Print and plot starting a 0 each time (default: False)
  --demo           Use the demo route instead of providing one (default: False)
  --plot           If a plot should be generated (default: False)
```

## Examples
Plotting with relative_self starts each process at time=0 and gives a nice overview.
![relself](https://user-images.githubusercontent.com/42323981/161629832-c6f28874-4b0b-437a-961e-d80adbf8dd97.png)
Plotting without relative_self provides info about the frames relative time. 
![relfirst](https://user-images.githubusercontent.com/42323981/161629886-3283e7c8-1bb0-4f3c-bede-4ceac1d2e140.png)


Printed timestamps of a frame with internal durations.
```
Frame ID: 309!

  camerad
    wideRoadCameraState start                            0.0
    roadCameraState start                                0.07552
    roadCameraState published                            47.527293
    RoadCamera: Image set                                47.53547
    wideRoadCameraState published                        47.629427
    roadCameraState.processingTime                       0.041072819381952286
    wideRoadCameraState.processingTime                   0.041170213371515274
  modeld
    Image added                                          50.020177
    Extra image added                                    51.427398
    Execution finished                                   69.209873
    modelV2 published                                    69.439922
    modelV2.modelExecutionTime                           0.02127685770392418
    modelV2.gpuExecutionTime                             0.0
  plannerd
    lateralPlan published                                74.233088
    longitudinalPlan published                           79.186773
    lateralPlan.solverExecutionTime                      0.001069204998202622
    longitudinalPlan.solverExecutionTime                 0.0012989999959245324
  controlsd
    Data sampled                                         79.817234
    Events updated                                       81.043103
    sendcan published                                    82.493917
    controlsState published                              82.893756
    Data sampled                                         94.906306
    Events updated                                       95.706973
    sendcan published                                    96.692429
    controlsState published                              97.074976
    Data sampled                                         99.102137
    Events updated                                       99.867075
    sendcan published                                    100.74925
    controlsState published                              101.098985
    Data sampled                                         109.437678
    Events updated                                       110.258396
    sendcan published                                    111.288331
    controlsState published                              111.686138
    Data sampled                                         119.536504
    Events updated                                       120.330868
    sendcan published                                    121.244658
    controlsState published                              121.561217
  boardd
    sending sendcan to panda: 250027001751393037323631   96.814094
    sendcan sent to panda: 250027001751393037323631      97.153933
    sending sendcan to panda: 250027001751393037323631   100.872426
    sendcan sent to panda: 250027001751393037323631      101.230338
    sending sendcan to panda: 250027001751393037323631   111.403121
    sendcan sent to panda: 250027001751393037323631      111.794053
    sending sendcan to panda: 250027001751393037323631   121.33497
    sendcan sent to panda: 250027001751393037323631      121.665278
    sending sendcan to panda: 250027001751393037323631   131.182028
    sendcan sent to panda: 250027001751393037323631      131.509784
```
