# openpilot releases


## terms

- `channel` - a named version of openpilot (git branch, casync caidx) which receives updates<br>
- `prebuilt` - a channel which is already built for the comma 3/3x and contains only required files for running openpilot and identifying the channel<br>
- `release` - a `prebuilt` channel with `ALLOW_DEBUG=false` (`RELEASE=1` when building panda, ex: `nightly`, `release3`)<br>


## openpilot channels

| channel      | type        | description                                                       |
| -----------  | ----------- | ----------                                                        |
| release3     | `release`   | channel for end users                                             |
| staging      | `release`   | staging channel for release3                                      |
| nightly      | `release`   | generated nightly from last commit passing CI tests               |
| master       | `prebuilt`  | current master commit with experimental features enabled          |
| git branches | none        | installed manually, experimental features enabled, build required |


## creating casync channel

`create_casync_prebuilt.sh` - creates a `prebuilt` casync openpilot channel, ready to upload to `openpilot-channels`

```bash
# run on a tici, within the directory you want to create the channel from.
# creates a prebuilt version of openpilot into BUILD_DIR and outputs the caidx
# and other casync files into CASYNC_DIR for uploading to openpilot-channels.
BUILD_DIR=/data/openpilot_build    \
CASYNC_DIR=/data/casync            \
OPENPILOT_CHANNEL=nightly          \
release/create_casync_prebuilt.sh
```

`upload_casync_channel.sh` - helper for uploading a casync channel to `openpilot-channels`


## release channel

to create a release channel, set `RELEASE=1` environment variable when running build script
