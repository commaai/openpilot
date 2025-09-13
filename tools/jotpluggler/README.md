l# JotPluggler

JotPluggler is a tool to quickly visualize openpilot logs.

## Usage

```
$ ./jotpluggler/pluggle.py -h
usage: pluggle.py [-h] [--demo] [--layout LAYOUT] [route]

A tool for visualizing openpilot logs.

positional arguments:
  route            Optional route name to load on startup.

options:
  -h, --help       show this help message and exit
  --demo           Use the demo route instead of providing one
  --layout LAYOUT  Path to YAML layout file to load on startup
```

Example using route name:

`./pluggle.py "a2a0ccea32023010/2023-07-27--13-01-19"`

Examples using segment:

`./pluggle.py "a2a0ccea32023010/2023-07-27--13-01-19/1"`

`./pluggle.py "a2a0ccea32023010/2023-07-27--13-01-19/1/q" # use qlogs`

Example using segment range:

`./pluggle.py "a2a0ccea32023010/2023-07-27--13-01-19/0:1"`

## Demo

For a quick demo, run this command:

`./pluggle.py --demo --layout=layouts/torque-controller.yaml`


## Basic Usage/Features:
- The text box to load a route is a the top left of the page, accepts standard openpilot format routes (e.g. `a2a0ccea32023010/2023-07-27--13-01-19/0:1`, `https://connect.comma.ai/a2a0ccea32023010/2023-07-27--13-01-19/`)
- The Play/Pause button is at the bottom of the screen, you can drag the bottom slider to seek. The timeline in timeseries plots are synced with the slider.
- The Timeseries List sidebar has several dropdowns, the fields each show the field name and value, synced with the timeline (will show N/A until the time of the first message in that field is reached).
- There is a search bar for the timeseries list, you can search for structs or fields, or both by separating with a "/"
- You can drag and drop any numeric/boolean field from the timeseries list into a timeseries panel.
- You can create more panels with the split buttons (buttons with two rectangles, either horizontal or vertical). You can resize the panels by dragging the grip in between any panel.
- You can load and save layouts with the corresponding buttons. Layouts will save all tabs, panels, titles, timeseries, etc.

## Layouts

If you create a layout that's useful for others, consider upstreaming it.

## Plot Interaction Controls

- **Left click and drag within the plot area** to pan X
  - Left click and drag on an axis to pan an individual axis (disabled for Y-axis)
- **Scroll in the plot area** to zoom in X axes, Y-axis is autofit
  - Scroll on an axis to zoom an individual axis
- **Right click and drag** to select data and zoom into the selected data
  - Left click while box selecting to cancel the selection
- **Double left click** to fit all visible data
  - Double left click on an axis to fit the individual axis (disabled for Y-axis, always autofit)
- **Double right click** to open the plot context menu
- **Click legend label icons** to show/hide plot items
