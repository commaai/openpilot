# Rerun
Rerun is a tool to quickly visualize time series data. It supports all openpilot logs , both the `logMessages` and video logs.

[Instructions](https://rerun.io/docs/reference/viewer/overview) for navigation within the Rerun Viewer.

## Usage
```
usage: run.py [-h] [--demo] [--qcam] [--fcam] [--ecam] [--dcam] [route_or_segment_name]

A helper to run rerun on openpilot routes

positional arguments:
  route_or_segment_name
                        The route or segment name to plot (default: None)

options:
  -h, --help            show this help message and exit
  --demo                Use the demo route instead of providing one (default: False)
  --qcam                Show low-res road camera (default: False)
  --fcam                Show driving camera (default: False)
  --ecam                Show wide camera (default: False)
  --dcam                Show driver monitoring camera (default: False)
```

Examples using route name to observe accelerometer and qcamera:

`./run.sh --qcam "a2a0ccea32023010/2023-07-27--13-01-19"`

Examples using segment range (more on [SegmentRange](https://github.com/commaai/openpilot/tree/master/tools/lib)):

`./run.sh --qcam "a2a0ccea32023010/2023-07-27--13-01-19/2:4"`

## Cautions:
- Showing hevc videos (`--fcam`, `--ecam`, and `--dcam`)  are expensive, and it's recommended to use `--qcam` for optimized performance. If possible, limiting your route to a few segments using `SegmentRange` will speed up logging and reduce memory usage

## Demo
`./run.sh --qcam --demo`
