# Cabana

Cabana visualizes openpilot messages and raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

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
  --layout [LAYOUT]         open a Cabana JSON layout file
  --stream                  read openpilot messages from local msgq (alias: --msgq)
  --msgq                    read openpilot messages from local msgq
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

Cabana includes [openpilot analysis layouts](layouts), including
`tuning`, `longitudinal`, `torque`, and camera/debug presets. From this directory, try:

```shell
./cabana --demo --layout layouts/tuning.json
./cabana "5beb9b58bd12b691/0000010a--a51155e496" --layout layouts/tuning.json
./cabana --stream --layout layouts/longitudinal.json       # local replay or running openpilot
./cabana --zmq <ipaddress> --layout layouts/tuning.json    # device running the messaging bridge
```

`--layout` takes a file path relative to the directory where you run the command,
or an absolute path. Omitting its value leaves the saved session layout unchanged.

Charts, CAN inspection, and playback share one workspace. **CAN signals** and **openpilot Messages**
are independent dock panels, initially tabbed together in the sidebar. Drag either panel's title tab
to show both sources beside each other, move it elsewhere, or float it in a separate window.
Selecting a CAN message opens **CAN Details**
beside the charts, with the bit grid, signal editor, and message history. Close that pane to give
the space back to charts; selecting the message again reopens it with its inspection tabs intact.
Browsing openpilot fields leaves CAN details open, so both sources can be compared together.
This works the same for live streams and recorded routes, including dashcam-only recordings.

**Layout → openpilot Presets** opens a bundled layout on the current route and selects the
openpilot browser. The empty chart area also offers **Browse openpilot** and **Presets**.
Plotting openpilot messages does not need a DBC or CAN data. Synchronized playback and video sit
below the source panels. **CAN signals**, **openpilot Messages**, **CAN Details**, **Charts**, and
**Playback** all follow the same docking rules. Closing a panel hides it and preserves its contents;
reopen it from **View**. Closing a floating panel leaves the rest of the layout in place. Dock
positions, the selected panel, and panel visibility are remembered between sessions. **View → Reset Window Layout**
restores the default arrangement.

Inside **CAN Details**, message tabs select the CAN message being inspected; inside **Charts**, named
tabs select pages of charts. Those tabs organize a panel's contents. Move the outer panel title tab
to undock the whole inspector or chart workspace.

Browse **openpilot Messages** as a tree of messages, fields, and array indices. Search expands matching
branches and restores your previous expansion state when cleared. Search for fields such as `/carState/vEgo`, `/carControl/actuators/accel`, or
`/modelV2/position/x/0`. Double-click a field to create a plot, or drag it onto an existing plot
to compare fields. Arrays, booleans, enums, and nested numeric fields are included. Hover a
field to inspect its value. Imported equations also appear in the browser.

Use **+** in the Charts toolbar to create an empty chart, then drag openpilot fields onto it
or add decoded CAN through **Manage Signals**.
Empty charts are preserved in saved layouts. For decoded CAN, open **Manage Signals** from
the chart's menu to search by signal name, message name, or message ID.
Select several signals to overlay them on one chart. You can also
add a signal from its message's signal view. Drag chart grips to reorder or merge charts;
**Split Chart** separates an overlay.

- **Click** a chart to seek; **drag** to zoom all charts to a time range.
- **Shift-drag** scrubs playback; **Ctrl-drag** pans; **Ctrl-wheel** zooms around the pointer
  (Cmd instead of Ctrl on macOS).
- **View → Fit Loaded Data** fits the visible series in the current tab.
- **View → Follow Playback** restores the rolling time window. Zoom and pan support undo/redo.
- Click a legend entry to hide/show a series. Right-click it for **transforms and statistics**,
  also available through the chart's three-dot menu.

Transforms include scale/offset, derivative, integral, and a moving average over a configurable
number of samples. Scale and offset apply first. Derivatives omit the first sample and repeated
timestamps; integrals use trapezoids starting at zero at the first loaded sample. Moving averages
use the available samples while the window fills. Transformed series have an asterisk in their
legend and adjusted units when the source has a known unit. Statistics show sample count, minimum, maximum, and sample mean for
the visible time range. These operations affect chart values only.

### Saved layouts and equations

Use **Functions → New Function** to build a custom signal. Enter a unique name, browse for
its primary signal, and write a Python function body, for example `return value * 2.23694`
to convert `/carState/vEgo` to mph. Add inputs to use `v1`, `v2`, and so on; expand
**Global code** for numeric constants or initial state. You can also type paths for signals that have
not loaded yet. **Plot in a new chart** displays the result immediately.

Saved functions appear in the **Functions** menu for editing and in the signal browser for
plotting. Editing recalculates existing plots and dependent functions. Names stay fixed because
other signals and charts refer to them. **Delete function** removes its definition and plotted
series from all tabs. If another function uses it, update or delete that dependent function first.
Use **Layout → Save Layout** to keep the changes.


**Layout → Open Layout** accepts Cabana JSON. The bundled presets use Python equations
and preserve named tabs, chart titles, overlaid curves, colors, line styles, fixed Y limits,
and scale/offset transforms. Panels are arranged in Cabana's chart grid.
Series default to visible, untransformed values with scale 1, offset 0, and a moving-average
window of 10 samples. openpilot message fields need only a `path`; CAN signals need `message` and `signal`.

Custom signal expressions use Python.

**Layout → Save Layout** saves the workspace as Cabana JSON, including equations, tabs,
chart grouping, colors, limits, signal visibility, transforms, column count, and window duration.
The workspace also restores when Cabana restarts. Layouts contain no route data and can be reused
on another route. Missing openpilot fields remain visible as **No data** until their data arrives;
older layouts may reference fields no longer logged by current openpilot. PlotJuggler's optional
CAN-parser diagnostic fields are not produced by Cabana. Layouts with decoded CAN curves require
the matching DBC; invalid files or unresolved CAN signals leave the current workspace intact.

**Layout → Export Visible Data to CSV** exports visible series in the current tab and time
range, including calculated/transformed values and transform settings. The `source` column holds
`openpilot` for message fields or the CAN message ID. Each row contains one
sample at its original timestamp; signals with different sample rates are not resampled.
The right edge of the visible time range is excluded. Narrow panels place toolbar actions in
an overflow menu (**»**).

## Additional Information

For more information, see the [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)
