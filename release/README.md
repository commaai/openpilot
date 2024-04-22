# openpilot releases


## terms

- `channel` - a named version of openpilot (git branch, casync caibx) which receives updates
- `build` - a copy of openpilot ready for distribution, already built for a specific device
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


## build

`release/build_release.sh <build_dir>` - creates an openpilot build into `build_dir`, ready for distribution

## packaging a casync release

`release/package_casync_build.py <build_dir>` - packages an openpilot build into a casync tar and uploads to `openpilot-releases`

## release builds

to create a release build, set `RELEASE=1` environment variable when running the build script
