# openpilot releases


## terms

`channel` - a named version of openpilot with only required files for running openpilot and identifying the channel<br>
`prebuilt` - a channel prebuilt for the tici, no building required on device<br>
`release` - prebuilt with `ALLOW_DEBUG` false (RELEASE=1 when building panda). (`nightly`, `release3`)<br>

## creating casync channel

`build_casync_channel.sh` - creates a `prebuilt` openpilot channel, ready to upload to `openpilot-channels`

```bash
# run on a tici, within the directory you want to create the release from
# creates a prebuilt version of openpilot into BUILD_DIR and outputs the caidx
# and other casync files into CASYNC_DIR for uploading to openpilot-channels
BUILD_DIR=/data/openpilot_build    \
CASYNC_DIR=/data/casync            \
OPENPILOT_CHANNEL=nightly            \
release/build_casync_channel.sh
```

`upload_casync_channel.sh` - helper for uploading a casync channel to `openpilot-channels`
