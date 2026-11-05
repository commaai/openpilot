# Cabana

Cabana is a tool developed to view raw CAN data. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

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
  --msgq                    read can messages from the msgq
  --panda                   read can messages from panda
  --panda-serial <serial>   read can messages from panda with given serial
  --socketcan <device>      read can messages from given SocketCAN device
  --remote <ip-address>     read can messages from a device at the specified ip-address
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

### Streaming CAN Messages from a comma Device

[SSH into your device](https://github.com/commaai/openpilot/wiki/SSH) and start webrtcd with the following command:

```shell
cd /data/openpilot
./openpilot/system/webrtc/webrtcd.py --host 0.0.0.0 &
```

Then Run Cabana with the device's IP address:

```shell
cabana --remote <ipaddress>
```

Replace &lt;ipaddress&gt; with your comma device's IP address.

While streaming from the device, Cabana will log the CAN messages to a local directory. By default, this directory is ~/cabana_live_stream/. You can change the log directory in Cabana by navigating to menu -> tools -> settings.

After disconnecting from the device, you can replay the logged CAN messages from the stream selector dialog -> browse local route.

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

## Additional Information

For more information, see the [openpilot wiki](https://github.com/commaai/openpilot/wiki/Cabana)

## Plotting cereal fields from recorded routes

Open a route, click **New Chart**, and select **Cereal signals** in the message dropdown.
Filter for a field such as `/carState/vEgo` or `/carState/cruiseState/speed`, then
double-click it to add it to the selection. CAN and cereal signals can share a chart.
The chart uses the same route clock, playback cursor, zoom, and seek controls as CAN plots.
**Manage Signals** can add or remove either source, and split/merge and chart restoration
also retain cereal paths.

Field discovery observes later messages, growing lists, and active union members.
Numeric, boolean, and enum values are supported; enums retain their labels in the
cursor display. Optional or nonfinite values leave gaps instead of connecting
unrelated samples. Raw values use causal sample-and-hold at the cursor.

Only requested fields are decoded, off the UI thread. Jobs own their source snapshot,
and changing the selection or route cancels stale results. Decoded fields are cached
within replay's segment window, with at most 64 retained field entries. Lists in the
browser have no fixed 64-entry truncation. Dense lines use an extrema-preserving
pixel envelope for drawing; calculations and cursor values retain original samples.

### Transforms and derived curves

A chart's menu provides **Transforms** for scale, offset, and an adjacent-sample
derivative. A positive divisor overrides elapsed time, matching JotPluggler's layout
convention; zero uses timestamp differences. Scale and offset apply after differentiation.

Use **Layouts → New Python curve** to create a NumPy expression or function body.
The linked source is available as `value` and `time` (seconds since route start);
additional sources are aligned to it with causal sample-and-hold as `v1`, `v2`, etc.
Initialization runs once per evaluation. Evaluation reuses JotPluggler's existing
Python helper in a separate process, with cancellation and a 15-second timeout.
Failures appear on the affected curve, without suppressing other charts.

### Layouts

Use **Layouts → Open layout** / **Save layout**, or start with a bundled plot layout:

```sh
./openpilot/tools/cabana/cabana --demo --layout tuning
./openpilot/tools/cabana/cabana --demo --layout CAN-bus-debug
```

Saved layouts retain chart/tab names, CAN identities, cereal paths, curve colors and
visibility, transforms, Python source, and fixed Y limits. Existing chart sessions
remain readable. JotPluggler's plot-only JSON layouts can be imported into Cabana's
existing chart grid; nested split ratios and independent X ranges are not restored.
Layouts with camera/map panes are rejected explicitly rather than partially loaded.

This support covers recorded routes and the segments in replay's cache. Live cereal
subscriptions, string-valued series, map panes, and PlotJuggler XML import are not
implemented. Python initialization restarts when the cached input window changes;
stateful results can depend on the beginning of that window. This is a migration
step, not grounds to remove PlotJuggler yet.

Build and run the native regression tests:

```sh
scons -u -j4 openpilot/tools/cabana/tests/test_cerealseries openpilot/tools/cabana/tests/test_analysis
./openpilot/tools/cabana/tests/test_cerealseries
./openpilot/tools/cabana/tests/test_analysis
```
