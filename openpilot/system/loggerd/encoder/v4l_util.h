#pragma once

// the V4L2 encoder's logic that doesn't touch the device, so it's unit tested on PC

#include <linux/videodev2.h>

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// upstream Venus puts the codec header in front of the first slice of every IDR frame.
// returns the header's length, i.e. the offset of the first VCL NAL unit's start code, or 0 if there's no header
inline size_t header_size(const uint8_t *dat, size_t len, bool h265) {
  for (size_t i = 0; i + 3 < len; i++) {
    if (dat[i] == 0 && dat[i+1] == 0 && dat[i+2] == 1) {
      int nal_type = h265 ? (dat[i+3] >> 1) & 0x3f : dat[i+3] & 0x1f;
      bool vcl = h265 ? nal_type < 32 : (nal_type >= 1 && nal_type <= 5);
      if (vcl) return (i > 0 && dat[i-1] == 0) ? i-1 : i;
    }
  }
  return 0;
}

// the lowest H.264 level (H.264 Table A-1) whose frame size and macroblock rate fit, up to Venus' maximum of 5.1.
// live streams don't go below 3.1, like the downstream encoder's live streams
inline v4l2_mpeg_video_h264_level h264_level(int width, int height, int fps, bool live) {
  struct Level { v4l2_mpeg_video_h264_level level; int64_t max_mbps, max_fs; };
  static const Level levels[] = {
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_0, 1485, 99},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_1, 3000, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_2, 6000, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_1_3, 11880, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_0, 11880, 396},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_1, 19800, 792},
    {V4L2_MPEG_VIDEO_H264_LEVEL_2_2, 20250, 1620},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_0, 40500, 1620},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_1, 108000, 3600},
    {V4L2_MPEG_VIDEO_H264_LEVEL_3_2, 216000, 5120},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_0, 245760, 8192},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_1, 245760, 8192},
    {V4L2_MPEG_VIDEO_H264_LEVEL_4_2, 522240, 8704},
    {V4L2_MPEG_VIDEO_H264_LEVEL_5_0, 589824, 22080},
    {V4L2_MPEG_VIDEO_H264_LEVEL_5_1, 983040, 36864},
  };
  const int64_t w = (width + 15) / 16, h = (height + 15) / 16;
  const int64_t fs = w * h;
  for (const auto &l : levels) {
    if (live && l.level < V4L2_MPEG_VIDEO_H264_LEVEL_3_1) continue;
    // neither dimension may exceed sqrt(8 * MaxFS) macroblocks
    if (fs <= l.max_fs && fs * fps <= l.max_mbps && w * w <= 8 * l.max_fs && h * h <= 8 * l.max_fs) return l.level;
  }
  return V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
}

// the V4L2 OUTPUT (input frame) buffer indices, each busy from its QBUF to its DQBUF. a DMA-BUF that's queued at the
// same index each time stays attached and mapped (vb2 only remaps an index when its DMA-BUF changes), so the upstream
// kernel's VisionIPC buffers take their buffer index. msm_vidc's USERPTR buffers take any free index
class InputSlots {
public:
  explicit InputSlots(int count) : busy(count, false) {}

  // blocks until the index is free, or until any index is if it's -1
  int acquire(int index = -1) {
    assert(index < (int)busy.size());
    std::unique_lock lk(m);
    while (true) {
      int pick = index;
      if (pick < 0) {
        for (int i = 0; i < (int)busy.size() && pick < 0; i++) {
          if (!busy[i]) pick = i;
        }
      }
      if (pick >= 0 && !busy[pick]) {
        busy[pick] = true;
        return pick;
      }
      cv.wait(lk);
    }
  }

  void release(int index) {
    {
      std::lock_guard lk(m);
      busy[index] = false;
    }
    cv.notify_all();
  }

  // blocks until every index is free
  void wait_idle() {
    std::unique_lock lk(m);
    cv.wait(lk, [&] { return free_count_locked() == busy.size(); });
  }

  size_t free_count() {
    std::lock_guard lk(m);
    return free_count_locked();
  }

  size_t size() const { return busy.size(); }

private:
  size_t free_count_locked() const { return std::count(busy.begin(), busy.end(), false); }

  std::mutex m;
  std::condition_variable cv;
  std::vector<bool> busy;
};
