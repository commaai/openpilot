# Camera stream

`compressed_vipc.py` connects to a remote device running openpilot, decodes the video streams, and republishes them over VisionIPC.

## Usage

### On the device
SSH into the device and run following in separate terminals:

`cd /data/openpilot && ./openpilot/system/webrtc/webrtcd.py --host 0.0.0.0`

`cd /data/openpilot/system/loggerd && ./encoderd`

`cd /data/openpilot/system/camerad && ./camerad`

Note that both the device and your PC must be on the same openpilot commit.

Alternatively paste this as a single command:
```
(
  cd /data/openpilot
  ./openpilot/system/webrtc/webrtcd.py --host 0.0.0.0 &

  cd /data/openpilot/system/camerad/
  ./camerad &

  cd /data/openpilot/system/loggerd/
  ./encoderd &

  wait
) ; trap 'kill $(jobs -p)' SIGINT
```
Ctrl+C will stop all three processes.

### On the PC
Bridge the video streams from the device, then decode them with `compressed_vipc.py`:

```~/openpilot/openpilot/system/webrtc/bridge.py <ip> narrowRoadEncodeData,wideRoadEncodeData,cabinEncodeData```

```cd ~/openpilot/openpilot/tools/camerastream && ./compressed_vipc.py```

To actually display the stream, run `watch3` in separate terminal:

```cd ~/openpilot/selfdrive/ui/ && ./watch3.py```

## compressed_vipc.py usage
```
$ python3 compressed_vipc.py -h
usage: compressed_vipc.py [-h] [--cams CAMS] [--server SERVER] [--silent]

Decode video streams and broadcast on VisionIPC

options:
  -h, --help       show this help message and exit
  --cams CAMS      Cameras to decode
  --server SERVER  choose vipc server name
  --silent         Suppress debug output
```


## Example:
```
cd ~/openpilot/tools/camerastream && ./compressed_vipc.py --cams 0
cd ~/openpilot/selfdrive/ui/ && ./watch3.py
```
