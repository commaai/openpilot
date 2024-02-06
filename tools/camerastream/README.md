# Camera stream

`compressed_vipc.py` allows to connect to Comma device and decode video streams. 

## Usage

### On the device 
SSH to the Comma device and run following lines in separate terminals:

`cd /data/openpilot/cereal/messaging && ./bridge`

`cd /data/openpilot/system/loggerd && ./encoderd`

`cd /data/openpilot/system/camerad && ./camerad`

Note: Your device need to be on `master` branch.
Make sure both the device and PC is roughly at the same commit level.
Run `./scons` on the PC and reboot the comma device if you are updating from older versions.

Alternatively paste this as a single command:
```
(
  cd /data/openpilot/cereal/messaging/
  ./bridge &

  cd /data/openpilot/system/camerad/
  ./camerad &

  cd /data/openpilot/system/loggerd/
  ./encoderd &

  wait
) ; trap 'kill $(jobs -p)' SIGINT
```
Ctrl+C will stop all three processes. 

### On the PC
Decode the stream with `compressed_vipc.py`:

```cd ~/openpilot/tools/camerastream && ./compressed_vipc.py <ip>```

Note: make sure you run python scripts in `poetry shell` virtual environment.

To actually display the stream, run `watch3` in separate terminal:

```cd ~/openpilot/selfdrive/ui/ && ./watch3```

If this step fails, try limiting active cameras by adding `--cams=0` to `compressed_vipc.py`.

## compressed_vipc.py usage
```
$ python compressed_vipc.py -h
usage: compressed_vipc.py [-h] [--nvidia] [--cams CAMS] [--silent] addr

Decode video streams and broadcast on VisionIPC

positional arguments:
  addr         Address of comma three

options:
  -h, --help   show this help message and exit
  --nvidia     Use nvidia instead of ffmpeg
  --cams CAMS  Cameras to decode
  --silent     Suppress debug output
```


## Example:
```
cd ~/openpilot/tools/camerastream && ./compressed_vipc.py comma-ffffffff --cams 0
cd ~/openpilot/selfdrive/ui/ && ./watch3
```