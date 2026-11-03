#include <chrono>
#include <thread>
#include <vector>

#include "common/tests/native_test.h"
#include "system/loggerd/encoder/v4l_util.h"

void test_header_size() {
  // HEVC: VPS, SPS, PPS, then an IDR_W_RADL slice
  const std::vector<uint8_t> hevc = {0, 0, 0, 1, 0x40, 1, 0xaa, 0, 0, 0, 1, 0x42, 1, 0xbb, 0, 0, 0, 1, 0x44, 1, 0xcc,
                                     0, 0, 0, 1, 0x26, 1, 0xdd};
  REQUIRE(header_size(hevc.data(), hevc.size(), true) == 21);
  // a slice alone has no header
  REQUIRE(header_size(hevc.data() + 21, hevc.size() - 21, true) == 0);
  // H.264: SPS, PPS with a 3 byte start code, then an IDR slice
  const std::vector<uint8_t> h264 = {0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb, 0, 0, 0, 1, 0x65, 0xcc};
  REQUIRE(header_size(h264.data(), h264.size(), false) == 11);
  // a non-IDR slice
  const std::vector<uint8_t> h264_p = {0, 0, 1, 0x41, 0xcc};
  REQUIRE(header_size(h264_p.data(), h264_p.size(), false) == 0);
  // no VCL NAL unit
  REQUIRE(header_size(h264.data(), 9, false) == 0);
}

void test_h264_level() {
  // qcamera and the livestreams, the downstream encoder's live streams use 3.1
  REQUIRE(h264_level(526, 330, 20, false) == V4L2_MPEG_VIDEO_H264_LEVEL_2_1);
  REQUIRE(h264_level(526, 330, 20, true) == V4L2_MPEG_VIDEO_H264_LEVEL_3_1);
  // the clips' full size frames are past 4.2's frame size
  REQUIRE(h264_level(1928, 1208, 20, false) == V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
  REQUIRE(h264_level(1344, 760, 20, false) == V4L2_MPEG_VIDEO_H264_LEVEL_3_2);
  REQUIRE(h264_level(1920, 1080, 30, false) == V4L2_MPEG_VIDEO_H264_LEVEL_4_0);
  REQUIRE(h264_level(1920, 1080, 60, false) == V4L2_MPEG_VIDEO_H264_LEVEL_4_2);
  REQUIRE(h264_level(176, 144, 15, false) == V4L2_MPEG_VIDEO_H264_LEVEL_1_0);
  REQUIRE(h264_level(176, 144, 30, false) == V4L2_MPEG_VIDEO_H264_LEVEL_1_1);
  // too narrow and long for a level's frame size: 8192 macroblocks, but 512 wide
  REQUIRE(h264_level(8192, 256, 1, false) == V4L2_MPEG_VIDEO_H264_LEVEL_5_1);
  // past Venus' maximum
  REQUIRE(h264_level(4096, 2304, 60, false) == V4L2_MPEG_VIDEO_H264_LEVEL_5_1);
}

void test_input_slots() {
  InputSlots slots(3);
  // an index is taken until it's released
  REQUIRE(slots.acquire(1) == 1);
  REQUIRE(slots.free_count() == 2);
  // -1 takes any free index
  REQUIRE(slots.acquire() == 0);
  REQUIRE(slots.acquire() == 2);
  REQUIRE(slots.free_count() == 0);
  for (int i : {0, 1, 2}) slots.release(i);
  REQUIRE(slots.free_count() == 3);

  // a busy index waits for its release, even with other indices free
  int busy = slots.acquire(2);
  int got = -1;
  std::thread t([&] { got = slots.acquire(2); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  REQUIRE(got == -1);
  slots.release(busy);
  t.join();
  REQUIRE(got == 2);

  // wait_idle waits for every index
  bool idle = false;
  std::thread w([&] { slots.wait_idle(); idle = true; });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  REQUIRE(!idle);
  slots.release(got);
  w.join();
  REQUIRE(idle);
}

int main() {
  return run_native_test([] {
    test_header_size();
    test_h264_level();
    test_input_slots();
  });
}
