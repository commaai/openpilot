## Face detection demo

Run simple face detection model on video stream from driver camera of comma three.

This example streams video frames, runs face-detection model and displays window with live detection results (bounding boxes).

```sh
# pass the ip address of comma three, if running remotely (by default localhost)
python3 face_detection.py [--host comma-ip-address]
```