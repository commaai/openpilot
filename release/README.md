# openpilot releases


## terms

`channel` - a named version of openpilot (git branches or casync caidx)<br>
`prebuilt` - a channel prebuilt for the tici, no building required on device<br>
`release` - a channel that is prebuilt and also has `ALLOW_DEBUG` false. (`nightly`, `release3`)<br>

## creating casync channel

`build_casync_channel.sh` - creates a `prebuilt` openpilot channel, ready to upload to `openpilot-channels`

```bash
# run on a tici, within the directory you want to create the release from
# creates a prebuilt version of openpilot into BUILD_DIR and outputs the caidx
# and other casync files into CASYNC_DIR for uploading to openpilot-channels
BUILD_DIR=/data/openpilot_build    \
CASYNC_DIR=/data/casync            \
RELEASE_CHANNEL=nightly            \
release/build_casync_channel.sh
```

`upload_casync_channel.sh` - helper for uploading a casync channel to `openpilot-channels`
