# Openpilot log signals in Cabana

Cabana can plot numeric cereal log fields alongside its CAN tools and video. Enable log plotting when opening a route:

```sh
./openpilot/tools/cabana/cabana --log --demo
./openpilot/tools/cabana/cabana --log "<route>"
./openpilot/tools/cabana/cabana --log-layout openpilot/tools/cabana/layouts/steering.json "<route>"
```

This is an initial integration toward commaai/openpilot#38752. PlotJuggler remains available; this change does not claim full feature parity or completion of the bounty.

## Plotting

Open **View > Log Signals**. Search for a field such as `carState/vEgo`, then choose it from **Add signal**. Select a plot to add more curves to it, or click **New plot** before adding another signal. Scalars, booleans, enums, active union members, and numeric lists are supported. List names use indices, for example `modelV2/position/x/0`. The `_valid` field under each topic retains the message validity flag.

Plots share an x range. **Follow playback** follows the same clock and selected time range as CAN charts and video. Drag or scroll in a plot to inspect a fixed range; Shift+click seeks playback. **Curves** provides scale, offset, and a first derivative in units per second. The derivative is computed after scaling/offset, and begins with a gap because the first sample has no predecessor.

**Save layout** stores curve groups and transforms as JSON. Loading a malformed layout leaves the existing plots in place. A layout can reference fields that are not present in the current route; these are reported as having no cached samples, and remain in the layout.

**Export visible CSV** exports selected, transformed samples in the displayed range using long-form rows. It includes the plot number and transform settings. Non-finite values and missing-segment boundaries appear as blank values. No interpolation is performed. Time is in seconds relative to the replay route origin.

## Data loading and scope

Log fields are extracted on replay's merge thread. Only the currently cached replay segments are retained. Seeking loads the required segments and updates the plots; exporting an interval does **not** download the entire route. Use Cabana's segment-cache setting to adjust the amount of data retained. There are no background log-field subscriptions or new message publishers.

| Capability | This change |
| --- | --- |
| Numeric cereal fields, arrays, active unions | Supported for route replay |
| Multiple curves, shared time range, video seek | Supported |
| Scale, offset, first derivative | Supported |
| Native JSON layouts and visible CSV export | Supported |
| PlotJuggler XML curve groups | Import utility; restrictions below |
| Live cereal streaming | Pending; `--log` rejects live-stream options |
| XY plots, Lua/custom math, expression dependencies | Pending |
| PlotJuggler tabs, docking, colors, axis settings | Pending |
| Full-route performance and user acceptance | Pending real-route validation |

## Importing a PlotJuggler layout

```sh
python3 openpilot/tools/cabana/import_loglayout.py \
  openpilot/tools/plotjuggler/layouts/longitudinal.xml longitudinal.json
./openpilot/tools/cabana/cabana --log-layout longitudinal.json "<route>"
```

The converter preserves raw time-series curve groups and scale/offset transforms with zero time offset. It strips the leading slash from field names. Tabs are flattened into plot groups, and styling/axis settings are not transferred. Unsupported modes, custom math curves, derivative plugins, and time shifts are rejected before creating the output file. Existing output files are never overwritten. Successful conversion does not guarantee that an older field name exists in a newer route.

## Validation

```sh
scons -u -j4 openpilot/tools/cabana/_cabana_ui \
  openpilot/tools/cabana/tests/test_logsignals \
  openpilot/tools/cabana/tests/test_log_replay \
  openpilot/tools/cabana/tests/test_logpanel
openpilot/tools/cabana/tests/test_logsignals
openpilot/tools/cabana/tests/test_log_replay
openpilot/tools/cabana/tests/test_logpanel
python3 -m unittest openpilot.tools.cabana.tests.test_loglayout -v
```

The native data tests cover typed fields, nested groups, inactive unions, arrays, non-finite values, duplicate/out-of-order samples, nanosecond precision, cache eviction, missing-segment gaps, transforms, layout validation, and CSV escaping. The replay integration test uses a generated local qlog with no CAN data to check paced playback, seeking, and exclusion of synthetic video-frame events. The headless ImGui/ImPlot test checks rendering with empty, non-finite, missing, and evicted data without requiring a display server.

Before considering removal of PlotJuggler, exercise representative long routes and streaming sessions, migrate custom layouts/formulas, verify performance and video interaction on supported desktop platforms, and complete the maintainer's acceptance period.
