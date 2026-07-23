from __future__ import annotations

import os
import queue
import struct
import subprocess
import threading
from pathlib import Path

import ffmpeg

from openpilot.common.swaglog import cloudlog


def _ebml_id(value: int) -> bytes:
  return value.to_bytes((value.bit_length() + 7) // 8, "big")


def _ebml_size(value: int) -> bytes:
  for length in range(1, 9):
    if value < (1 << (7 * length)) - 1:
      return ((1 << (7 * length)) | value).to_bytes(length, "big")
  raise ValueError(f"EBML element is too large: {value}")


def _ebml_element(element_id: int, payload: bytes) -> bytes:
  return _ebml_id(element_id) + _ebml_size(len(payload)) + payload


def _ebml_uint(element_id: int, value: int) -> bytes:
  length = max(1, (value.bit_length() + 7) // 8)
  return _ebml_element(element_id, value.to_bytes(length, "big"))


def _ebml_text(element_id: int, value: str) -> bytes:
  return _ebml_element(element_id, value.encode())


def _ebml_float(element_id: int, value: float) -> bytes:
  return _ebml_element(element_id, struct.pack(">d", value))


def _matroska_header(track: bytes) -> bytes:
  ebml = b"".join(
    (
      _ebml_uint(0x4286, 1),
      _ebml_uint(0x42F7, 1),
      _ebml_uint(0x42F2, 4),
      _ebml_uint(0x42F3, 8),
      _ebml_text(0x4282, "matroska"),
      _ebml_uint(0x4287, 4),
      _ebml_uint(0x4285, 2),
    )
  )
  info = b"".join(
    (
      _ebml_uint(0x2AD7B1, 1_000_000),
      _ebml_text(0x4D80, "openpilot"),
      _ebml_text(0x5741, "openpilot"),
    )
  )
  return b"".join(
    (
      _ebml_element(0x1A45DFA3, ebml),
      _ebml_id(0x18538067),
      b"\x01\xff\xff\xff\xff\xff\xff\xff",
      _ebml_element(0x1549A966, info),
      _ebml_element(0x1654AE6B, _ebml_element(0xAE, track)),
    )
  )


class RawVideoWriter:
  def __init__(self, path: Path):
    self._file = open(path, "wb")

  def write(self, data: bytes, timestamp: int, codecconfig: bool, keyframe: bool) -> None:
    del timestamp, codecconfig, keyframe
    self._file.write(data)

  def write_audio(self, data: bytes, timestamp: int, sample_rate: int) -> None:
    del data, timestamp, sample_rate

  def close(self) -> None:
    self._file.flush()
    self._file.close()


class MatroskaVideoWriter:
  """Minimal streaming Matroska muxer for FFVHUFF packets produced by encoderd."""

  def __init__(self, path: Path, width: int, height: int, fps: int):
    self._file = open(path, "wb")
    self._frame_index = 0
    self._fps = fps
    self._write_header(width, height)

  def _write_header(self, width: int, height: int) -> None:
    bitmap_info = struct.pack(
      "<IiiHHIIiiII",
      40,
      width,
      height,
      1,
      24,
      int.from_bytes(b"FFVH", "little"),
      width * height * 3 // 2,
      0,
      0,
      0,
      0,
    )
    video = _ebml_uint(0xB0, width) + _ebml_uint(0xBA, height)
    track = b"".join(
      (
        _ebml_uint(0xD7, 1),
        _ebml_uint(0x73C5, 1),
        _ebml_uint(0x83, 1),
        _ebml_uint(0x9C, 0),
        _ebml_uint(0x23E383, round(1_000_000_000 / self._fps)),
        _ebml_text(0x86, "V_MS/VFW/FOURCC"),
        _ebml_element(0x63A2, bitmap_info),
        _ebml_element(0xE0, video),
      )
    )
    self._file.write(_matroska_header(track))

  def write(self, data: bytes, timestamp: int, codecconfig: bool, keyframe: bool) -> None:
    del timestamp, codecconfig
    timecode_ms = round(self._frame_index * 1000 / self._fps)
    flags = 0x80 if keyframe else 0
    timecode = _ebml_uint(0xE7, timecode_ms)
    block_header = b"\x81" + struct.pack(">hB", 0, flags)
    block_prefix = _ebml_id(0xA3) + _ebml_size(len(block_header) + len(data)) + block_header
    self._file.write(_ebml_id(0x1F43B675) + _ebml_size(len(timecode) + len(block_prefix) + len(data)))
    self._file.write(timecode)
    self._file.write(block_prefix)
    self._file.write(data)
    self._frame_index += 1

  def write_audio(self, data: bytes, timestamp: int, sample_rate: int) -> None:
    del data, timestamp, sample_rate

  def close(self) -> None:
    self._file.flush()
    self._file.close()


class _PipeWriter:
  def __init__(self, fd: int, name: str):
    self._fd = fd
    self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
    self._thread = threading.Thread(target=self._run, name=name, daemon=True)
    self._thread.start()

  def write(self, data: bytes) -> None:
    self._queue.put(data)

  def _run(self) -> None:
    try:
      while (data := self._queue.get()) is not None:
        view = memoryview(data)
        while view:
          written = os.write(self._fd, view)
          view = view[written:]
    except BrokenPipeError:
      cloudlog.exception("ffmpeg input pipe closed")
    finally:
      os.close(self._fd)

  def close(self) -> None:
    self._queue.put(None)
    self._thread.join()


def _mpeg_crc32(data: bytes) -> int:
  crc = 0xFFFFFFFF
  for value in data:
    crc ^= value << 24
    for _ in range(8):
      crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
  return crc


class MpegTsMuxer:
  VIDEO_PID = 0x100
  PMT_PID = 0x1000

  def __init__(self, write, fps: int):
    self._write = write
    self._fps = fps
    self._frame_index = 0
    self._continuity = {0: 0, self.PMT_PID: 0, self.VIDEO_PID: 0}
    self._codec_config = b""
    self._write_tables()

  def _packet_header(self, pid: int, payload_start: bool, adaptation_control: int) -> bytes:
    continuity = self._continuity[pid]
    self._continuity[pid] = (continuity + 1) & 0xF
    return bytes(
      (
        0x47,
        (0x40 if payload_start else 0) | (pid >> 8),
        pid & 0xFF,
        (adaptation_control << 4) | continuity,
      )
    )

  def _write_section(self, pid: int, section: bytes) -> None:
    payload = b"\0" + section
    packet = self._packet_header(pid, True, 1) + payload
    self._write(packet + b"\xff" * (188 - len(packet)))

  def _write_tables(self) -> None:
    pat = bytes(
      (
        0x00,
        0xB0,
        0x0D,
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0x00,
        0x01,
        0xE0 | (self.PMT_PID >> 8),
        self.PMT_PID & 0xFF,
      )
    )
    pat += _mpeg_crc32(pat).to_bytes(4, "big")
    self._write_section(0, pat)

    pmt = bytes(
      (
        0x02,
        0xB0,
        0x12,
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0xE0 | (self.VIDEO_PID >> 8),
        self.VIDEO_PID & 0xFF,
        0xF0,
        0x00,
        0x1B,
        0xE0 | (self.VIDEO_PID >> 8),
        self.VIDEO_PID & 0xFF,
        0xF0,
        0x00,
      )
    )
    pmt += _mpeg_crc32(pmt).to_bytes(4, "big")
    self._write_section(self.PMT_PID, pmt)

  @staticmethod
  def _pts(value: int) -> bytes:
    return bytes(
      (
        0x21 | (((value >> 30) & 0x7) << 1),
        (value >> 22) & 0xFF,
        1 | (((value >> 15) & 0x7F) << 1),
        (value >> 7) & 0xFF,
        1 | ((value & 0x7F) << 1),
      )
    )

  @staticmethod
  def _pcr(value: int) -> bytes:
    return bytes(
      (
        (value >> 25) & 0xFF,
        (value >> 17) & 0xFF,
        (value >> 9) & 0xFF,
        (value >> 1) & 0xFF,
        ((value & 1) << 7) | 0x7E,
        0,
      )
    )

  def write(self, data: bytes, codecconfig: bool, keyframe: bool) -> None:
    if codecconfig:
      self._codec_config = data
      return
    if self._frame_index % self._fps == 0:
      self._write_tables()

    pts = round(self._frame_index * 90_000 / self._fps)
    payload = self._codec_config + data
    self._codec_config = b""
    pes = b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" + self._pts(pts) + payload

    first = True
    while pes:
      if first:
        adaptation_flags = 0x50 if keyframe else 0x10
        adaptation = bytes((7, adaptation_flags)) + self._pcr(pts)
        capacity = 188 - 4 - len(adaptation)
        chunk, pes = pes[:capacity], pes[capacity:]
        packet = self._packet_header(self.VIDEO_PID, True, 3) + adaptation + chunk
      elif len(pes) < 184:
        chunk, pes = pes, b""
        adaptation_length = 183 - len(chunk)
        adaptation = bytes((adaptation_length,))
        if adaptation_length:
          adaptation += b"\0" + b"\xff" * (adaptation_length - 1)
        packet = self._packet_header(self.VIDEO_PID, False, 3) + adaptation + chunk
      else:
        chunk, pes = pes[:184], pes[184:]
        packet = self._packet_header(self.VIDEO_PID, False, 1) + chunk
      self._write(packet)
      first = False

    self._frame_index += 1


class MatroskaPcmMuxer:
  def __init__(self, write, sample_rate: int):
    self._write = write
    self._sample_rate = sample_rate
    self._sample_index = 0
    audio = b"".join(
      (
        _ebml_float(0xB5, float(sample_rate)),
        _ebml_uint(0x9F, 1),
        _ebml_uint(0x6264, 16),
      )
    )
    track = b"".join(
      (
        _ebml_uint(0xD7, 1),
        _ebml_uint(0x73C5, 1),
        _ebml_uint(0x83, 2),
        _ebml_uint(0x9C, 0),
        _ebml_text(0x86, "A_PCM/INT/LIT"),
        _ebml_element(0xE1, audio),
      )
    )
    self._write(_matroska_header(track))

  def write(self, data: bytes) -> None:
    timecode_ms = round(self._sample_index * 1000 / self._sample_rate)
    block = b"\x81" + struct.pack(">hB", 0, 0x80) + data
    cluster = _ebml_uint(0xE7, timecode_ms) + _ebml_element(0xA3, block)
    self._write(_ebml_element(0x1F43B675, cluster))
    self._sample_index += len(data) // 2


class MpegTsVideoWriter:
  def __init__(self, path: Path, fps: int, include_audio: bool):
    self._path = path
    self._fps = fps
    self._include_audio = include_audio
    self._process: subprocess.Popen | None = None
    self._video_pipe: _PipeWriter | None = None
    self._audio_pipe: _PipeWriter | None = None
    self._video_muxer: MpegTsMuxer | None = None
    self._audio_muxer: MatroskaPcmMuxer | None = None
    self._file = None
    if include_audio:
      self._path.touch()
    else:
      self._file = open(self._path, "wb")
      self._video_muxer = MpegTsMuxer(self._file.write, fps)

  def _start(self, sample_rate: int | None = None) -> None:
    if not self._include_audio:
      return
    if self._process is not None:
      return
    if sample_rate is None:
      return

    video_read, video_write = os.pipe()
    pass_fds = [video_read]
    command = [
      str(Path(ffmpeg.BIN_DIR) / "ffmpeg"),
      "-nostdin",
      "-y",
      "-loglevel",
      "warning",
      "-thread_queue_size",
      "512",
      "-f",
      "mpegts",
      "-i",
      f"pipe:{video_read}",
    ]

    audio_read, audio_write = os.pipe()
    pass_fds.append(audio_read)
    command += [
      "-thread_queue_size",
      "512",
      "-f",
      "matroska",
      "-i",
      f"pipe:{audio_read}",
      "-map",
      "0:v:0",
      "-map",
      "1:a:0",
      "-c:v",
      "copy",
      "-c:a",
      "aac",
      "-b:a",
      "32000",
    ]
    command += ["-f", "mpegts", str(self._path)]
    self._process = subprocess.Popen(command, pass_fds=pass_fds)
    os.close(video_read)
    os.close(audio_read)

    self._video_pipe = _PipeWriter(video_write, "video_writer")
    self._audio_pipe = _PipeWriter(audio_write, "audio_writer")
    self._video_muxer = MpegTsMuxer(self._video_pipe.write, self._fps)
    self._audio_muxer = MatroskaPcmMuxer(self._audio_pipe.write, sample_rate)

  def write(self, data: bytes, timestamp: int, codecconfig: bool, keyframe: bool) -> None:
    del timestamp
    self._start()
    if self._video_muxer is not None:
      self._video_muxer.write(data, codecconfig, keyframe)

  def write_audio(self, data: bytes, timestamp: int, sample_rate: int) -> None:
    del timestamp
    self._start(sample_rate)
    if self._audio_muxer is not None:
      self._audio_muxer.write(data)

  def close(self) -> None:
    if self._video_pipe is not None:
      self._video_pipe.close()
    if self._audio_pipe is not None:
      self._audio_pipe.close()

    if self._process is not None:
      try:
        return_code = self._process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        self._process.kill()
        return_code = self._process.wait()
      if return_code:
        cloudlog.error(f"ffmpeg video writer exited with code {return_code}")
    if self._file is not None:
      self._file.flush()
      self._file.close()


class VideoWriter:
  def __init__(self, path: str | Path, filename: str, width: int, height: int, fps: int, codec: str, include_audio: bool = False):
    self.path = Path(path) / filename
    self.lock_path = Path(f"{self.path}.lock")
    self.lock_path.touch(mode=0o664)
    self.closed = False

    if codec == "fullHEVC":
      self._writer = RawVideoWriter(self.path)
    elif codec == "bigBoxLossless":
      self._writer = MatroskaVideoWriter(self.path, width, height, fps)
    elif codec == "qcameraH264":
      self._writer = MpegTsVideoWriter(self.path, fps, include_audio)
    else:
      raise ValueError(f"unsupported recording codec: {codec}")

  def write(self, data: bytes, timestamp: int, codecconfig: bool, keyframe: bool) -> None:
    self._writer.write(data, timestamp, codecconfig, keyframe)

  def write_audio(self, data: bytes, timestamp: int, sample_rate: int) -> None:
    self._writer.write_audio(data, timestamp, sample_rate)

  def close(self) -> None:
    if self.closed:
      return
    self._writer.close()
    self.lock_path.unlink(missing_ok=True)
    self.closed = True
