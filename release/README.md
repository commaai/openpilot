# openpilot releases


## terms

`channel` - a named version of openpilot (git branch, casync caidx)<br>
`prebuilt` - a channel which is built for the tici and contains only required files for running openpilot and identifying the channel<br>
`release` - a `prebuilt` channel with `ALLOW_DEBUG=false` (`RELEASE=1` when building panda, ex: `nightly`, `release3`)<br>

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
