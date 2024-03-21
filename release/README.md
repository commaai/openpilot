# openpilot releases


## terms

- `channel` - a named version of openpilot (git branch, casync caidx) which receives updates
- `build` - a release which is already built for the comma 3/3x and contains only required files for running openpilot and identifying the release

- `build_style` - type of build, either `debug` or `release`
  - `debug` - build with `ALLOW_DEBUG=true`, can test experimental features like longitudinal on alpha cars
  - `release` - build with `ALLOW_DEBUG=false`, experimental features disabled


## openpilot channels

| channel      | build_style | description                                                       |
| -----------  | ----------- | ----------                                                        |
| release      | `release`   | stable release of openpilot                                       |
| staging      | `release`   | release candidate of openpilot for final verification             |
| nightly      | `release`   | generated nightly from last commit passing CI tests               |
| master       | `debug`     | current master commit with experimental features enabled          |
| git branches | `debug`     | installed manually, experimental features enabled, build required |


## creating casync build

`create_casync_build.sh` - creates a casync openpilot build, ready to upload to `openpilot-releases`

```bash
# run on a tici, within the directory you want to create the build from.
# creates a prebuilt version of openpilot into BUILD_DIR and outputs the caidx
# and other casync files into CASYNC_DIR for uploading to openpilot-releases.
BUILD_DIR=/data/openpilot_build    \
CASYNC_DIR=/data/casync            \
OPENPILOT_CHANNEL=nightly          \
release/create_casync_build.sh
```

`upload_casync_release.sh` - helper for uploading a casync build to `openpilot-releases`


## release builds

to create a release build, set `RELEASE=1` environment variable when running the build script
