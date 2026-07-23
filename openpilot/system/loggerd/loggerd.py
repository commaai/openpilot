#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.hardware import PC
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.logger import LoggerState
from openpilot.system.loggerd.video_writer import VideoWriter


MAIN_FPS = 20
NO_CAMERA_PATIENCE_MS = 500
PRESERVE_ATTR_NAME = "user.preserve"
PRESERVE_ATTR_VALUE = b"1"
V4L2_BUF_FLAG_KEYFRAME = 8
EVENT_UNION_FIELDS = log.Event.schema.union_fields
EVENT_WHICH_OFFSET = 24


def event_name(data: bytes) -> str:
  # Cereal events use Cap'n Proto's canonical one-segment layout. The Event
  # union discriminant is a UInt16 at byte 24.
  if len(data) >= EVENT_WHICH_OFFSET + 2 and data[:4] == b"\0\0\0\0" and data[8:12] == b"\0\0\0\0":
    which = int.from_bytes(data[EVENT_WHICH_OFFSET:EVENT_WHICH_OFFSET + 2], "little")
    if which < len(EVENT_UNION_FIELDS):
      return EVENT_UNION_FIELDS[which]

  with log.Event.from_bytes(data, traversal_limit_in_words=messaging.NO_TRAVERSAL_LIMIT) as event:
    return event.which()


@dataclass(frozen=True)
class EncoderInfo:
  publish_name: str
  idx_name: str
  filename: str | None
  record: bool = True
  include_audio: bool = False
  fps: int = MAIN_FPS


@dataclass
class ServiceState:
  name: str
  counter: int
  frequency: int | None
  encoder: bool
  preserve_segment: bool
  record_audio: bool


@dataclass
class RemoteEncoder:
  writer: VideoWriter | None = None
  encoderd_segment_offset: int = 0
  current_segment: int = -1
  queue: deque[bytes] = field(default_factory=deque)
  dropped_frames: int = 0
  recording: bool = False
  marked_ready_to_rotate: bool = False
  seen_first_packet: bool = False
  audio_initialized: bool = False

  def close_writer(self) -> None:
    if self.writer is not None:
      self.writer.close()
      self.writer = None

  def close(self) -> None:
    self.close_writer()
    self.queue.clear()


class ExitHandler:
  def __init__(self):
    self.exit = False
    self.signal = 0
    self.power_failure = False
    signal.signal(signal.SIGINT, self._handle)
    signal.signal(signal.SIGTERM, self._handle)
    if hasattr(signal, "SIGPWR"):
      signal.signal(signal.SIGPWR, self._handle)

  def _handle(self, sig: int, frame) -> None:
    del frame
    self.signal = sig
    self.power_failure = sig == getattr(signal, "SIGPWR", None)
    self.exit = True


class Loggerd:
  def __init__(self):
    self.exit_handler = ExitHandler()
    self.last_camera_seen_ms = 0.0
    self.ready_to_rotate = 0
    self.last_rotate_ms = 0.0
    self.previous_preserved_segment = -1
    self.loggerd_test = bool(os.getenv("LOGGERD_TEST"))
    self.segment_length = int(os.getenv("LOGGERD_SEGMENT_LENGTH", "60")) if self.loggerd_test else 60

    record_audio = Params().get_bool("RecordAudio")
    record_front = Params().get_bool("RecordFront")
    self.encoder_infos = {
      "roadEncodeData": EncoderInfo("roadEncodeData", "roadEncodeIdx", "fcamera.hevc"),
      "qRoadEncodeData": EncoderInfo("qRoadEncodeData", "qRoadEncodeIdx", "qcamera.ts", include_audio=record_audio),
      "wideRoadEncodeData": EncoderInfo("wideRoadEncodeData", "wideRoadEncodeIdx", "ecamera.hevc"),
      "driverEncodeData": EncoderInfo("driverEncodeData", "driverEncodeIdx", "dcamera.hevc", record=record_front),
    }
    self.max_waiting = len(self.encoder_infos)

    self.poller = messaging.Poller()
    self.service_states = {}
    self.remote_encoders = {}
    for name, service in SERVICE_LIST.items():
      encoder = name.endswith("EncodeData")
      livestream_encoder = name.startswith("livestream")
      should_record_audio = name == "rawAudioData" and record_audio
      if not (service.should_log or (encoder and not livestream_encoder) or should_record_audio):
        continue

      cloudlog.debug(f"logging {name}")
      messaging.sub_sock(name, poller=self.poller, addr="127.0.0.1", conflate=False)
      self.service_states[name] = ServiceState(
        name=name,
        counter=0,
        frequency=service.decimation,
        encoder=encoder,
        preserve_segment=name == "userBookmark",
        record_audio=should_record_audio,
      )
      if encoder:
        self.remote_encoders[name] = RemoteEncoder()

    # Subscribe before collecting initData, since hardware metadata can take long
    # enough for the camera queues to start wrapping.
    self.logger = LoggerState()
    self.encoders_with_audio = [
      self.remote_encoders[name]
      for name, service in self.service_states.items()
      if service.name in self.encoder_infos and self.encoder_infos[service.name].include_audio
    ]

  def rotate(self) -> None:
    self.logger.next()
    self.ready_to_rotate = 0
    self.last_rotate_ms = time.monotonic() * 1000
    action = "logging to" if self.logger.segment == 0 else "rotated to"
    cloudlog.warning(f"{action} {self.logger.segment_path}")

  def rotate_if_needed(self) -> None:
    all_ready = self.ready_to_rotate == self.max_waiting
    timed_out = False
    now_ms = time.monotonic() * 1000
    segment_seconds = (now_ms - self.last_rotate_ms) / 1000
    if segment_seconds > self.segment_length and not self.loggerd_test:
      if now_ms - self.last_camera_seen_ms > NO_CAMERA_PATIENCE_MS:
        timed_out = True
        cloudlog.error("no camera packets seen. auto rotating")
      elif segment_seconds > self.segment_length * 1.2:
        timed_out = True
        cloudlog.error("segment too long. auto rotating")

    if all_ready or timed_out:
      self.rotate()

  def preserve_segment(self) -> None:
    if self.logger.segment == self.previous_preserved_segment:
      return
    assert self.logger.segment_path is not None
    cloudlog.warning(f"preserving {self.logger.segment_path}")
    try:
      os.setxattr(self.logger.segment_path, PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
    except OSError:
      cloudlog.exception(f"setxattr {PRESERVE_ATTR_NAME} failed for {self.logger.segment_path}")

    params = Params()
    routes = params.get("AthenadRecentlyViewedRoutes") or ""
    params.put("AthenadRecentlyViewedRoutes", f"{routes},{self.logger.route_name}")
    self.previous_preserved_segment = self.logger.segment

  def _write_encode_data(self, event, remote: RemoteEncoder, encoder_info: EncoderInfo) -> int:
    encode_data = getattr(event, encoder_info.publish_name)
    idx = encode_data.idx
    keyframe = bool(idx.flags & V4L2_BUF_FLAG_KEYFRAME)

    if not remote.recording:
      if not keyframe:
        remote.dropped_frames += 1
        return 0
      if remote.dropped_frames:
        cloudlog.warning(f"{encoder_info.publish_name}: dropped {remote.dropped_frames} non iframe packets before init")
        remote.dropped_frames = 0
      if encoder_info.record:
        assert remote.writer is not None
        remote.writer.write(bytes(encode_data.header), idx.timestampEof // 1000, True, False)
      remote.recording = True

    if remote.writer is not None:
      remote.writer.write(bytes(encode_data.data), idx.timestampEof // 1000, False, keyframe)

    idx_msg = messaging.new_message(encoder_info.idx_name, valid=event.valid)
    idx_msg.logMonoTime = event.logMonoTime
    setattr(idx_msg, encoder_info.idx_name, idx)
    data = idx_msg.to_bytes()
    self.logger.write(data, True)
    return len(data)

  def _process_queued_encoder_messages(self, remote: RemoteEncoder, encoder_info: EncoderInfo) -> int:
    bytes_count = 0
    while remote.queue:
      queued = remote.queue.popleft()
      with log.Event.from_bytes(queued, traversal_limit_in_words=messaging.NO_TRAVERSAL_LIMIT) as event:
        bytes_count += self._write_encode_data(event, remote, encoder_info)
    return bytes_count

  def handle_encoder_message(self, data: bytes, name: str, remote: RemoteEncoder, encoder_info: EncoderInfo) -> int:
    with log.Event.from_bytes(data, traversal_limit_in_words=messaging.NO_TRAVERSAL_LIMIT) as event:
      idx = getattr(event, name).idx
      if not remote.seen_first_packet:
        remote.seen_first_packet = True
        remote.encoderd_segment_offset = idx.segmentNum
        cloudlog.debug(f"{name}: has encoderd offset {remote.encoderd_segment_offset}")
      offset_segment = idx.segmentNum - remote.encoderd_segment_offset

      if offset_segment == self.logger.segment:
        if remote.current_segment != self.logger.segment:
          remote.close_writer()
          if encoder_info.record:
            assert self.logger.segment_path is not None and encoder_info.filename is not None
            remote.writer = VideoWriter(
              self.logger.segment_path,
              encoder_info.filename,
              getattr(event, name).width,
              getattr(event, name).height,
              encoder_info.fps,
              str(idx.type),
              encoder_info.include_audio,
            )
          remote.recording = False
          remote.audio_initialized = False
          remote.current_segment = self.logger.segment
          remote.marked_ready_to_rotate = False

        if remote.audio_initialized or not encoder_info.include_audio:
          bytes_count = self._process_queued_encoder_messages(remote, encoder_info)
          bytes_count += self._write_encode_data(event, remote, encoder_info)
          return bytes_count

        if len(remote.queue) > MAIN_FPS * 10:
          cloudlog.error(f"{name}: dropping frame waiting for audio initialization, queue is too large")
        else:
          remote.queue.append(data)
        return 0

      if offset_segment > self.logger.segment:
        if not remote.marked_ready_to_rotate:
          remote.marked_ready_to_rotate = True
          self.ready_to_rotate += 1
          status = f"{self.ready_to_rotate}/{self.max_waiting}"
          cloudlog.debug(f"rotate {self.logger.segment} -> {offset_segment} ready {status} for {name}")

        if len(remote.queue) > MAIN_FPS * 10:
          cloudlog.error(f"{name}: dropping frame, queue is too large")
        else:
          remote.queue.append(data)
        return 0

      details = f"logger_segment={self.logger.segment} offset={remote.encoderd_segment_offset}"
      cloudlog.error(f"{name}: encoderd packet has an older segment: {idx.segmentNum=} {details}")
      remote.encoderd_segment_offset = -self.logger.segment
      return 0

  def handle_audio_message(self, data: bytes) -> None:
    with log.Event.from_bytes(data, traversal_limit_in_words=messaging.NO_TRAVERSAL_LIMIT) as event:
      audio = event.rawAudioData
      for encoder in self.encoders_with_audio:
        if encoder.writer is not None:
          encoder.writer.write_audio(bytes(audio.data), event.logMonoTime // 1000, audio.sampleRate)
          encoder.audio_initialized = True

  def run(self) -> None:
    self.rotate()
    Params().put("CurrentRoute", self.logger.route_name)

    message_count = 0
    bytes_count = 0
    start = time.monotonic()
    try:
      while not self.exit_handler.exit:
        ready_sockets = self.poller.poll(1000)
        if not ready_sockets:
          continue
        for sock in ready_sockets:
          service = None
          count = 0
          while not self.exit_handler.exit:
            data = sock.receive(non_blocking=True)
            if data is None:
              break

            if service is None:
              service = self.service_states[event_name(data)]

            if service.preserve_segment:
              self.preserve_segment()

            in_qlog = False
            if service.frequency is not None:
              in_qlog = service.counter % service.frequency == 0
              service.counter += 1

            if service.record_audio:
              self.handle_audio_message(data)

            if service.encoder:
              self.last_camera_seen_ms = time.monotonic() * 1000
              bytes_count += self.handle_encoder_message(
                data, service.name, self.remote_encoders[service.name], self.encoder_infos[service.name],
              )
              self.rotate_if_needed()
            else:
              self.logger.write(data, in_qlog)
              bytes_count += len(data)

            message_count += 1
            if message_count % 10_000 == 0:
              elapsed = time.monotonic() - start
              message_rate = message_count / elapsed
              byte_rate = bytes_count * 0.001 / elapsed
              cloudlog.debug(f"{message_count} messages, {message_rate:.2f} msg/sec, {byte_rate:.2f} KB/sec")

            count += 1
            if count >= 200:
              cloudlog.debug(f"large volume of '{service.name}' messages")
              break
    finally:
      cloudlog.warning("closing logger")
      self.logger.exit_signal = self.exit_handler.signal
      if self.exit_handler.power_failure:
        cloudlog.error("power failure")
        os.sync()
        cloudlog.error("sync done")
      self.logger.close()
      for remote in self.remote_encoders.values():
        remote.close()


def main() -> None:
  if not PC and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {0, 1, 2, 3})
  Loggerd().run()


if __name__ == "__main__":
  main()
