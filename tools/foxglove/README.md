# Foxglove Converter

[Foxglove Studio](https://foxglove.dev/download) is a tool to visualize robotics data. We've written a script to convert openpilot log data into mcap files for use inside foxglove studio.

## Installation

Ensure you've [set up the openpilot environment](../README.md), once the environment is setup the converter is ready to run.

## Usage

```
$ ./cereal_to_mcap.py -h
usage: cereal_to_mcap.py [-h] [--demo] [--qlog] [--ci] [--fcam] [--dcam] [--ecam] [route_or_segment_name] [segment_count]

A helper to convert openpilot routes to mcap files for foxglove studio

positional arguments:
  route_or_segment_name
                        The route or segment name to plot (cabana share URL accepted) (default: None)
  segment_count         The number of segments to plot (default: None)

options:
  -h, --help            show this help message and exit
  --demo                Use the demo route instead of providing one (default: False)
  --qlog                Use qlogs (default: False)
  --ci                  Download data from openpilot CI bucket (default: False)
  --fcam                Include fcamera data (default: False)
  --dcam                Include dcamera data (default: False)
  --ecam                Include ecamera data (default: False)

```

Examples using route name:

`./cereal_to_mcap.py "a2a0ccea32023010|2023-07-27--13-01-19"`

Examples using segment name:

`./cereal_to_mcap.py "a2a0ccea32023010|2023-07-27--13-01-19--1"`

## Demo

For a quick demo, go through the installation step and run this command:

`./cereal_to_mcap.py --demo --qlog`
