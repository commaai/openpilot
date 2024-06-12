#!/usr/bin/env python3
import cereal.messaging as messaging

if __name__ == "__main__":
  modeld_sock = messaging.sub_sock("modelV2")

  last_frame_id = None
  start_t: int | None = None
  frame_cnt = 0
  dropped = 0

  while True:
    m = messaging.recv_one(modeld_sock)
    if m is None:
      continue

    frame_id = m.modelV2.frameId
    t = m.logMonoTime / 1e9
    frame_cnt += 1

    if start_t is None:
      start_t = t
      last_frame_id = frame_id
      continue

    d_frame = frame_id - last_frame_id
    dropped += d_frame - 1

    expected_num_frames = int((t - start_t) * 20)
    frame_drop = 100 * (1 - (expected_num_frames / frame_cnt))
    print(f"Num dropped {dropped}, Drop compared to 20Hz: {frame_drop:.2f}%")

    last_frame_id = frame_id
