# Cabana

Cabana analyzes recorded and live CAN/cereal data with plots, derived curves, cameras, and maps. One use for this is creating and editing [CAN Dictionaries](http://socialledge.com/sjsu/index.php/DBC_Format) (DBC files), and the tool provides direct integration with [commaai/opendbc](https://github.com/commaai/opendbc) (a collection of DBC files), allowing you to load the DBC files direct from source, and save to your fork. In addition, you can load routes from [comma connect](https://connect.comma.ai).

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
  --msgq, --stream          read live cereal messages from msgq
  --panda                   read can messages from panda
  --panda-serial <serial>   read can messages from panda with given serial
  --socketcan <device>      read can messages from given SocketCAN device
  --remote <ip-address>     read can messages from a device at the specified ip-address
  --data_dir <dir>          local directory with routes
  --no-vipc                 do not output video
  --no-cache                turn off the local route file cache
  --dbc <file>              dbc file to open
  --layout <name-or-file>    open a bundled JSON or legacy XML layout
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

## Cereal analysis

Open a recorded route or a live source, click **New Chart**, and choose **Cereal signals**.
Paths such as `/carState/vEgo`, `/carState/cruiseState/speed`, and list entries can
share plots with CAN signals. The browser observes later messages, growing lists,
and active union members. Enums retain their labels; text fields use step traces
with readable state labels. Missing values leave gaps. Cursor values use causal
sample-and-hold.

Use a chart's **Transforms** menu for scale, value/time offsets, or derivatives.
A positive derivative divisor overrides timestamp differences; zero uses elapsed
time. Text states support time offsets but reject numerical transforms.

### Derived curves

Use **Layouts → New Python curve** for Python/NumPy expressions or function bodies.
The linked source is available as `value` and `time` (seconds since route start).
Additional sources are aligned causally as `v1`, `v2`, and so on. Initialization runs
once per evaluation. Recorded calculations read history from the beginning of the
selected route, so seeking or changing the replay cache window does not reset an
arbitrary stateful program at a different point. Named calculations can depend on
other calculations; cycles and ambiguous names report an error. Imported unplotted
Lua definitions survive layout saves and remain available to dependent curves. Reading that history can require
additional log downloads. Live calculations use a temporary disk spool from the
start of the stream; it is released when the stream and its pending readers close.

Evaluation runs in a separate process with cancellation and a 15-second timeout.
Errors appear on the affected curve. Legacy XML equations execute their original
Lua source. They require a Lua 5.3 or 5.4 shared runtime (`liblua5.4-0` on Ubuntu,
`lua` from Homebrew on macOS). Native Python layouts do not require Lua.

### Layouts, maps and cameras

Open and save layouts from the **Layouts** menu, select a bundled layout, or launch:

```sh
./openpilot/tools/cabana/cabana --demo --layout tuning
./openpilot/tools/cabana/cabana --demo --layout cameras-and-map
./openpilot/tools/cabana/cabana --demo --layout tuning.xml
```

Layouts retain nested horizontal/vertical splits, proportions, tabs, titles, curve
identities, labels, colors, visibility, Python/Lua source, transforms and axis
ranges. Drag separators to resize panes. Adding or removing panes preserves the
remaining split tree; **Use grid layout** explicitly restores Cabana's grid.
A chart's **Follow playback X** action releases an imported fixed X range.

**New camera pane** adds road, cabin, wide-road, qcamera or recorded thumbnails.
Recorded video loads on demand and follows the playback cursor. Live cameras use
visionipc; thumbnail panes decode the live thumbnail service. **New map pane** displays the route, playback position and engagement
state with cached vector roads. Wheel zoom, dragging and click-to-seek share the
same playback clock as the charts. Basemap requests use the public Overpass API;
GPS coordinates in those requests describe the displayed route.

The XML importer covers the shipped time-series layouts, derivative and
Scale/Offset transforms, and Lua equations. Unsupported XML plugin transforms,
non-time-series plots and floating-container workspaces report an
explicit error before replacing the current workspace.

### Export

**Layouts → Export plotted CSV** exports visible curves across the workspace,
including transformed and derived results and text-state labels. Columns use
causal sample-and-hold, preserve gaps, and quote text correctly.

### Live streaming

```sh
./openpilot/tools/cabana/cabana --msgq
./openpilot/tools/cabana/cabana --remote DEVICE_IP
```

For a remote source, run `openpilot/system/webrtc/webrtcd.py --host 0.0.0.0` on the
device. Cabana starts the existing bridge automatically and subscribes to non-video
cereal services. Ordinary live charts retain up to ten minutes / 256 MiB of raw
cereal messages; calculation history uses the temporary disk spool described above.

### Build checks

```sh
scons -u -j4 openpilot/tools/cabana/cabana openpilot/tools/cabana/tests/test_cerealseries openpilot/tools/cabana/tests/test_analysis
./openpilot/tools/cabana/tests/test_cerealseries
./openpilot/tools/cabana/tests/test_analysis
python3 -m pytest -q openpilot/tools/cabana/tests/test_layout_conversion.py
```

The retired PlotJuggler launcher and downloader have been removed. `op juggle` is
an alias for Cabana and supports the native route, `--demo`, `--layout`, and
`--stream` workflows. Third-party PlotJuggler plugins are not loaded by Cabana.
Long route calculations retain decoded selected history and are recomputed when
inputs change; their memory and execution time grow with recording duration.
