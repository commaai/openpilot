**Goal**
- Rewrite `system/loggerd/encoderd.cc` as `system/loggerd/encoderd.py` with near 1:1 behavior: read VisionIPC frames, encode per stream, publish messages, rotate segments, and send periodic thumbnails.

**Design (Keep It 1:1)**
- Orchestrate in Python; keep encoder logic equivalent to C++.
- PC path (FFmpeg): use existing `av` (PyAV) to encode HEVC/H.264 with the same settings (bitrate, gop, no B-frames). Extract keyframe flags and headers from packets.
- QCOM path (V4L2 M2M): implement direct `ioctl` + `mmap` in Python (stdlib `fcntl`, `ctypes`), mirroring `v4l_encoder.cc` (formats, request/queue/dequeue, stream on/off, timestamps, flags).
- JPEG thumbnails: generate from NV12 using NumPy downscale (quarter-res) and encode with `jpeglib` (Python libjpeg bindings) to mirror C++ `jpeg_encoder.cc`; publish same schema.
- No Cython. No new dependencies beyond `av` (already available). No change to `pyproject.toml` unless requested.

**Behavior Parity**
- Startup sync across encoders using shared `start_frame_id` (same logic and margins).
- Rotation at `frames_per_seg = SEGMENT_LENGTH * MAIN_FPS` since `start_frame_id`.
- Message fields and topics identical to C++ (`EncodeIdx` data, headers on keyframes, width/height, lengths, flags, encodeId monotonic per encoder).
- CLI parity: default main encoders; `--stream` uses livestream encoders. Apply realtime priority and core affinity on device (best-effort).
- “Lagging” check (buf.get_frame_id vs extra.frame_id): not exposed in Python; acceptable to omit or replace with best-effort logging.

**Implementation Steps**
1) Skeleton `encoderd.py` + CLI (`--stream`).
2) Dataclasses for `EncoderInfo`/`LogCameraInfo` mirroring `loggerd.h`.
3) `EncoderdState` + `sync_encoders` direct port.
4) VisionIPC integration: discover streams, connect, per-stream threads.
5) FFmpeg encoder (PyAV) with matching settings and packet publish.
6) V4L2 M2M encoder via `ioctl`/`mmap` ported from `v4l_encoder.cc`.
7) JPEG cadence and publish via `jpeglib` (YUV420 input, libjpeg settings matching C++ defaults).
8) Segment rotation reopen logic.
9) Lint: run `tools/op.sh lint` and fix any issues.
10) Run existing tests (`system/loggerd/tests/test_encoder.py`).
