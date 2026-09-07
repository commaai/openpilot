# Cabana

Cabana visualizes openpilot telemetry and raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

## Usage Instructions

```bash
$ ./cabana -h
Usage: ./cabana [options] [route]

  route                     the drive to replay. find your drives at connect.comma.ai

Options:
  --help                    show this help
  --demo                    use a demo route instead of providing your own
  --auto                    Auto load the route from the best available source (no video):
                            internal, openpilotci, comma_api, car_segments, testing_closet
  --qcam                    load qcamera
  --wide-road               load wide road camera (alias: --ecam)
  --cabin                   load cabin camera (alias: --dcam)
  --msgq                    read openpilot messages from local msgq
  --stream                  alias for --msgq
  --layout <name-or-file>    open a preset, PlotJuggler XML, or Cabana JSON layout
  --panda                   read can messages from panda
  --panda-serial <serial>   read can messages from panda with given serial
  --socketcan <device>      read can messages from given SocketCAN device
  --zmq <ip-address>        read openpilot messages from zmq at the specified ip-address
  --data_dir <dir>          local directory with routes
  --no-vipc                 do not output video
  --no-cache                turn off the local route file cache
  --dbc <file>              dbc file to open
```

## Examples

### Running Cabana in Demo Mode
To run Cabana using a built-in demo route, use the following command:

```shell
cabana --demo
```

### Loading a Specific Route

To load a specific route for replay, provide the route as an argument:

```shell
cabana "5beb9b58bd12b691/0000010a--a51155e496"
```

Replace "5beb9b58bd12b691/0000010a--a51155e496" with your desired route identifier.


### Running Cabana with multiple cameras
To run Cabana with multiple cameras, use the following command:

```shell
cabana "5beb9b58bd12b691/0000010a--a51155e496" --cabin --wide-road
```

### Streaming openpilot Messages from a comma Device

[SSH into your device](https://github.com/commaai/openpilot/wiki/SSH) and start the bridge with the following command:

```shell
cd /data/openpilot
./openpilot/cereal/messaging/bridge &
```

Then Run Cabana with the device's IP address:

```shell
cabana --zmq <ipaddress>
```

Replace &lt;ipaddress&gt; with your comma device's IP address.

While streaming from the device, Cabana will log the received messages to a local directory. By default, this directory is ~/cabana_live_stream/. You can change the log directory in Cabana by navigating to menu -> tools -> settings.

After disconnecting from the device, you can replay the logged messages from the stream selector dialog -> browse local route.

### Streaming CAN Messages from Panda

To read CAN messages from a connected Panda, use the following command:

```shell
cabana --panda
```

### Using the Stream Selector Dialog

If you run Cabana without any arguments, a stream selector dialog will pop up, allowing you to choose the stream.

```shell
cabana
```

## Plotting and analysis

Cabana can use the [openpilot PlotJuggler layouts](../plotjuggler/layouts) directly, including
`tuning`, `longitudinal`, `torque`, and camera/debug presets. From this directory, try:

```shell
./cabana --demo --layout tuning
./cabana "5beb9b58bd12b691/0000010a--a51155e496" --layout ../plotjuggler/layouts/tuning.xml
./cabana --stream --layout longitudinal       # local replay or running openpilot
./cabana --zmq <ipaddress> --layout tuning    # device running the messaging bridge
```

**Layout → openpilot Presets** opens a bundled layout on the current route. Telemetry layouts
open the central **Signal Analysis** workspace, with the route's numeric cereal fields in the
left sidebar and synchronized playback/video on the right. Switch back to **CAN Editor** for
DBC work. Cereal plotting does not need a DBC or CAN messages.

Search **Route Signals** for fields such as `/carState/vEgo`, `/carControl/actuators/accel`, or
`/modelV2/position/x/0`. Double-click a field to create a plot, or drag it onto an existing plot
to compare signals. Arrays, booleans, enums, and nested numeric fields are included. Hover a
field to inspect its value. Imported equations also appear in the browser.

For decoded CAN, use **+** in the Charts toolbar to search available signals by signal name,
message name, or message ID. Select several signals to overlay them on one chart. You can also
add a signal from its message's signal view. Drag chart grips to reorder or merge charts;
**Split Chart** separates an overlay.

- **Click** a chart to seek; **drag** to zoom all charts to a time range.
- **Shift-drag** scrubs playback; **Ctrl-drag** pans; **Ctrl-wheel** zooms around the pointer.
- **View → Fit Loaded Data** fits the visible signals in the current tab.
- **View → Follow Playback** restores the rolling time window. Zoom and pan support undo/redo.
- Click a legend entry to hide/show a signal. Right-click it for **transforms and statistics**,
  also available through the chart's three-dot menu.

Transforms include scale/offset, derivative, integral, and a moving average over a configurable
number of samples. Scale and offset apply first. Derivatives omit the first sample and repeated
timestamps; integrals use trapezoids starting at zero at the first loaded sample. Moving averages
use the available samples while the window fills. Transformed signals have an asterisk in their
legend and adjusted units. Statistics show sample count, minimum, maximum, and sample mean for
the visible time range. These operations affect chart values only.

### Saved layouts and equations

**Layout → Open Layout** accepts PlotJuggler XML and Cabana JSON. XML import preserves named
tabs, chart titles, overlaid curves, colors, line styles, fixed Y limits, scale/offset transforms,
and Lua equations used by the bundled layouts. Panels are arranged in Cabana's chart grid.
The tuning layout's stateful engagement gating and multi-signal calculations run on the loaded
data using PlotJuggler's nearest-sample alignment.

Equations require a Lua 5.3 or 5.4 shared library on the system. They can use `time`, `value`,
additional inputs (`v1`, `v2`, …), persistent globals, and the Lua math library, returning a value
or `(time, value)`. Errors appear on the affected signals. Plugin panels, XY plots, time-offset
transforms, and unrestricted Lua libraries are outside this importer’s scope.

**Layout → Save Layout** saves the workspace as Cabana JSON, including equations, tabs,
chart grouping, colors, limits, signal visibility, transforms, column count, and window duration.
The workspace also restores when Cabana restarts. Layouts contain no route data and can be reused
on another route. Missing cereal fields remain visible as **No data** until their data arrives;
older layouts may reference fields no longer logged by current openpilot. PlotJuggler's optional
CAN-parser diagnostic fields are not produced by Cabana. Layouts with decoded CAN curves require
the matching DBC; invalid files or unresolved CAN signals leave the current workspace intact.

**Layout → Export Visible Data to CSV** exports visible signals in the current tab and time
range, including calculated/transformed values and transform settings. Each row contains one
sample at its original timestamp; signals with different sample rates are not resampled.
The right edge of the visible time range is excluded. Narrow panels place toolbar actions in
an overflow menu (**»**).

## Additional Information

For more information, see the [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)
