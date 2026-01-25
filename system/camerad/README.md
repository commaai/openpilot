# camerad - Camera Daemon

Camera capture daemon for openpilot. Distributes frames to other processes (modeld, encoderd, ui, etc.) via VisionIPC.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Camera Sensor  │────▶│    camerad      │────▶│   VisionIPC     │
│  (IMX219 etc)   │     │  (this daemon)  │     │   Consumers     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              │                        ├── modeld
                              │                        ├── encoderd
                              ▼                        ├── ui
                        ┌───────────┐                  └── loggerd
                        │  cereal   │
                        │ (pubsub)  │
                        └───────────┘
```

## Supported Platforms

| Platform | Implementation | Environment Variable |
|----------|----------------|---------------------|
| QCOM/Tici (Comma 3X) | `cameras/camera_qcom2.cc` | None (default) |
| Jetson Orin Nano | `cameras/camera_jetson.cc` | `USE_JETSON_CAMERA=1` |

## Usage on Jetson

### Required Environment Variables

```bash
# Required: Enable Jetson implementation
export USE_JETSON_CAMERA=1

# Camera settings (defaults provided)
export ROAD_CAM=0              # Sensor ID (0, 1, 2...) or GStreamer pipeline
export CAMERA_WIDTH=1280       # Frame width (default: 1280)
export CAMERA_HEIGHT=720       # Frame height (default: 720)
export CAMERA_FPS=20           # Frame rate (default: 20)

# Optional: Additional cameras
export WIDE_CAM=1              # Wide camera (sensor ID)
export DRIVER_CAM=2            # Driver camera (sensor ID)

# Disable cameras
export DISABLE_ROAD=1          # Disable road camera
```

### Custom GStreamer Pipeline

If `ROAD_CAM` contains non-numeric characters, it's interpreted as a custom pipeline:

```bash
# Custom pipeline example
export ROAD_CAM='nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=NV12 ! appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false'
```

### Running

```bash
# Standalone test
USE_JETSON_CAMERA=1 ROAD_CAM=0 ./system/camerad/camerad

# Full system (via manager)
export USE_JETSON_CAMERA=1
export ROAD_CAM=0
export DISPLAY=:0
./system/manager/manager.py
```

## Troubleshooting

### "Failed to create CaptureSession" Error

The nvargus-daemon is in an invalid state. To recover:

```bash
# Check status
./scripts/check_nvargus.sh

# Recover
sudo systemctl restart nvargus-daemon
```

### Camera Not Detected

```bash
# Check camera devices
ls -la /dev/video*

# Check sensors
v4l2-ctl --list-devices

# Test with GStreamer directly
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink
```

### Black Screen in UI

1. Check nvargus-daemon status
2. Verify `GST_ARGUS: Setup Complete` appears in logs
3. Confirm VisionIPC is working correctly

## File Structure

```
system/camerad/
├── main.cc                    # Entry point (platform branching)
├── SConscript                 # Build configuration
├── cameras/
│   ├── camera_common.h        # Common interface definitions
│   ├── camera_common.cc       # Common implementation (QCOM)
│   ├── camera_qcom2.cc        # QCOM/Spectra ISP implementation
│   ├── camera_jetson.cc       # Jetson GStreamer implementation
│   ├── spectra.cc/h           # Spectra ISP driver
│   └── ...
├── sensors/                   # Sensor drivers (QCOM)
│   ├── sensor.h
│   ├── ox03c10.cc
│   └── os04c10.cc
└── test/
    └── ...
```

## Technical Details

### Jetson Implementation (camera_jetson.cc)

- Uses **GStreamer C API** (not via OpenCV)
- **nvarguscamerasrc** plugin for CSI camera capture
- Direct **NV12 format** output (no BGR conversion)
- **appsink** callbacks for frame processing
- Proper signal handling and cleanup via **ExitHandler**

### GStreamer Pipeline

```
nvarguscamerasrc sensor-id=0
    ↓
video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=20/1
    ↓
nvvidconv
    ↓
video/x-raw, format=NV12
    ↓
appsink (max-buffers=2, drop=true, sync=false)
```

### Limitations

- **No NVENC**: Jetson Orin Nano Super lacks hardware encoder, so software encoding (libx264) is used
- **nvargus-daemon dependency**: Daemon restart required if it enters an invalid state

## Related Documentation

- [NVIDIA Argus Camera API](https://docs.nvidia.com/jetson/l4t-multimedia/group__LibargusAPI.html)
- [GStreamer nvarguscamerasrc](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html)
