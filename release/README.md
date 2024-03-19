# openpilot releases


## terms

- `channel` - a named version of openpilot (git branch, casync caidx) which receives updates
- `prebuilt` - a release which is already built for the comma 3/3x and contains only required files for running openpilot and identifying the release
- `release` - a `prebuilt` release with `ALLOW_DEBUG=false` (`RELEASE=1` when building panda, ex: `nightly`, `release3`)


## openpilot channels

| channel      | type        | description                                                       |
| -----------  | ----------- | ----------                                                        |
| release3     | `release`   | stable release of openpilot                                       |
| staging      | `release`   | release candidate of openpilot for final verification             |
| nightly      | `release`   | generated nightly from last commit passing CI tests               |
| master       | `prebuilt`  | current master commit with experimental features enabled          |
| git branches | `git`       | installed manually, experimental features enabled, build required |


## creating casync build

`create_casync_prebuilt.sh` - creates a `prebuilt` casync openpilot build, ready to upload to `openpilot-releases`

```bash
# run on a tici, within the directory you want to create the build from.
# creates a prebuilt version of openpilot into BUILD_DIR and outputs the caidx
# and other casync files into CASYNC_DIR for uploading to openpilot-releases.
BUILD_DIR=/data/openpilot_build    \
CASYNC_DIR=/data/casync            \
OPENPILOT_CHANNEL=nightly          \
release/create_casync_prebuilt.sh
```

`upload_casync_release.sh` - helper for uploading a casync build to `openpilot-releases`


## release builds

to create a release build, set `RELEASE=1` environment variable when running the build script
