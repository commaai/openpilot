#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader


def pack_segment(rlog, road_video, wide_video, seconds, navigation):
  frames = {'narrowRoadEncodeIdx': [], 'wideRoadEncodeIdx': []}
  motions, calibrations = [], []
  for event in LogReader(rlog):
    kind = event.which()
    if kind in frames:
      frame = getattr(event, kind)
      frames[kind].append((frame.timestampEof / 1e9, frame.segmentId))
    elif kind == 'deviceMotion' and event.valid:
      motion = event.deviceMotion
      if motion.orientationNED.valid and motion.velocityDevice.valid:
        orientation = [getattr(motion.orientationNED, axis) for axis in ('x', 'y', 'z')]
        velocity = [getattr(motion.velocityDevice, axis) for axis in ('x', 'y', 'z')]
        motions.append((motion.timestamp / 1e9, orientation, velocity))
    elif kind == 'extrinsicsCalibration' and event.valid and event.extrinsicsCalibration.calStatus == 'calibrated':
      calibrations.append((event.logMonoTime / 1e9, list(event.extrinsicsCalibration.rpyCalib)))
  assert all(frames.values()) and motions and calibrations, 'Need both video indexes, deviceMotion, and calibration in the rlog'
  query = max(min(x[0] for x in values) for values in frames.values()) + seconds
  camera_times = query + np.array([-2., -1., -.5, -.1, 0.])
  images, timestamps = [], []
  for kind, video in (('wideRoadEncodeIdx', wide_video), ('narrowRoadEncodeIdx', road_video)):
    samples = np.array(sorted(frames[kind]))
    reader = FrameReader(video, hwaccel='none')
    for target in camera_times:
      index = int(np.argmin(abs(samples[:, 0] - target)))
      capture, frame = samples[index]
      assert abs(capture - target) < .06, 'Missing camera history at the requested time'
      images.append(reader.get(int(frame)).transpose(2, 0, 1))
      timestamps.append(round(capture * 1e6))
  query = timestamps[-1] / 1e6
  poses = sorted(motions, key=lambda x: x[0])
  times = np.array([p[0] for p in poses])
  query_times = query + np.arange(-19, 1) * .1
  assert times[0] <= query_times[0] and times[-1] >= query_times[-1], 'Missing ego history'
  selected = (times >= query_times[0] - .1) & (times <= query_times[-1] + .1)
  assert np.diff(times[selected]).max() < .15, 'Gap in ego history'
  orientation = rot_from_euler(np.array([p[1] for p in poses]))
  velocity_world = np.einsum('nij,nj->ni', orientation, np.array([p[2] for p in poses]))
  position_world = np.concatenate((np.zeros((1, 3)), np.cumsum(np.diff(times)[:, None] * (velocity_world[1:] + velocity_world[:-1]) / 2, axis=0)))
  history_position = np.stack([np.interp(query_times, times, position_world[:, axis]) for axis in range(3)], axis=-1)
  history_rotation = Slerp(times, Rotation.from_matrix(orientation))(query_times).as_matrix()
  calibration = [rpy for timestamp, rpy in calibrations if timestamp <= query][-1]
  world_from_vehicle = history_rotation @ rot_from_euler(calibration)
  # openpilot: forward/right/down; Alpamayo: forward/left/up.
  axes = np.diag([1., -1., -1.])
  current_from_world = axes @ world_from_vehicle[-1].T
  xyz = (history_position - history_position[-1]) @ current_from_world.T
  rot = current_from_world @ world_from_vehicle @ axes
  return {'image_frames': np.stack(images)[:, None].copy(), 'camera_indices': np.array([1]*5 + [6]*5, dtype=np.int64),
          'absolute_timestamps': np.array(timestamps, dtype=np.int64), 'ego_history_xyz': xyz[None].astype(np.float32),
          'ego_history_rot': rot[None].astype(np.float32), 'nav_text': np.array(navigation)}


def main():
  parser = argparse.ArgumentParser(description='Pack two front cameras and ego history from one openpilot segment for offline Alpamayo evaluation.')
  parser.add_argument('--rlog', required=True)
  parser.add_argument('--road-video', required=True)
  parser.add_argument('--wide-video', required=True)
  parser.add_argument('--seconds', required=True, type=float, help='Seconds after the first indexed frame; at least 2.')
  parser.add_argument('--navigation', required=True)
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  data = pack_segment(args.rlog, args.road_video, args.wide_video, args.seconds, args.navigation)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **data)
  print(args.output.resolve())


if __name__ == '__main__':
  main()
