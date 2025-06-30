#include <cassert>
#include <random>
#include <thread>
#include <chrono>
#include <cstring>
#include <vector>

#include "system/loggerd/loggerd.h"
#include "system/loggerd/encoder/v4l_encoder_burn.h"
#include "common/timing.h"
#include "msgq/visionipc/visionbuf.h"

#include "third_party/linux/include/msm_media_info.h"

#define BURN_ENCODER_COUNT 3  // Number of parallel encoders for stress testing
#define BURN_WIDTH 1920 // 4K width
#define BURN_HEIGHT 1080      // 4K height  
/*
#define BURN_WIDTH 3840 // 4K width
#define BURN_HEIGHT 2160      // 4K height  
*/
#define BURN_FPS 60           // 60 fps for maximum stress
#define BURN_BITRATE (2 * 1000000)  // 100 Mbps for maximum load
#define PRECOMPUTED_FRAMES 10

ExitHandler do_exit;

// Global precomputed VisionBuf frames - no copying needed!
static std::vector<VisionBuf> precomputed_frames;
static bool frames_initialized = false;

// Initialize precomputed random noise frames as VisionBuf objects
void init_precomputed_frames(size_t frame_size) {
  if (frames_initialized) return;  // Already initialized

  precomputed_frames.resize(PRECOMPUTED_FRAMES);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);

  LOGW("Precomputing %d VisionBuf noise frames of %zu bytes each", PRECOMPUTED_FRAMES, frame_size);

  for (int frame_idx = 0; frame_idx < PRECOMPUTED_FRAMES; frame_idx++) {
    // Allocate VisionBuf directly
    precomputed_frames[frame_idx].allocate(frame_size);

    // Fill with random noise
    uint8_t *data = (uint8_t*)precomputed_frames[frame_idx].addr;
    for (size_t i = 0; i < frame_size; i++) {
      data[i] = dis(gen);
    }
  }

  LOGW("Precomputation complete, using %zu MB total", (frame_size * PRECOMPUTED_FRAMES) / (1024 * 1024));
  frames_initialized = true;
}

// Get a pointer to a precomputed frame - no copying!
VisionBuf* get_precomputed_frame(int frame_index) {
  int idx = frame_index % PRECOMPUTED_FRAMES;
  return &precomputed_frames[idx];
}

void burn_encoder_thread(int encoder_id) {
  std::string thread_name = "burn_enc_" + std::to_string(encoder_id);
  util::set_thread_name(thread_name.c_str());

  std::string publish_name = "burnEncodeData_" + std::to_string(encoder_id);
  EncoderInfo burn_encoder_info = {
    .publish_name = publish_name.c_str(),
    .filename = nullptr,
    .record = false,
    .frame_width = BURN_WIDTH,
    .frame_height = BURN_HEIGHT,
    .fps = BURN_FPS,
    .bitrate = BURN_BITRATE,
    .encode_type = cereal::EncodeIndex::Type::FULL_H_E_V_C,
  };

  // Create the burn encoder
  V4LEncoderBurn encoder(burn_encoder_info, BURN_WIDTH, BURN_HEIGHT);
  encoder.encoder_open();

  // Calculate proper buffer size using VENUS macros for NV12 format
  size_t frame_size = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, BURN_WIDTH, BURN_HEIGHT)*2;
  //if (true) frame_size = 4202496*10;
  LOGD("Burn encoder %d: calculated buffer size %zu for %dx%d",
       encoder_id, frame_size, BURN_WIDTH, BURN_HEIGHT);

  // Initialize precomputed frames (only done once)
  init_precomputed_frames(frame_size);

  // Frame timing
  const auto frame_duration = std::chrono::microseconds(1000000 / BURN_FPS);
  auto next_frame_time = std::chrono::steady_clock::now();

  uint32_t frame_id = 0;
  LOGW("Burn encoder %d started: %dx%d@%dfps, %.1fMbps", 
       encoder_id, BURN_WIDTH, BURN_HEIGHT, BURN_FPS, BURN_BITRATE / 1000000.0);

  while (!do_exit) {
    auto current_time = std::chrono::steady_clock::now();

    // Wait until it's time for the next frame
    if (current_time < next_frame_time) {
      std::this_thread::sleep_until(next_frame_time);
    }

    // Get precomputed frame pointer - zero copy!
    VisionBuf* frame_buf = get_precomputed_frame(frame_id);

    // Create frame metadata
    VisionIpcBufExtra extra = {
      .timestamp_eof = nanos_since_boot(),
      .frame_id = frame_id++,
    };

    // Encode the frame directly from precomputed buffer
    int result = encoder.encode_frame(frame_buf, &extra);
    if (result == -1) {
      LOGE("Burn encoder %d failed to encode frame %d", encoder_id, frame_id - 1);
      // Continue anyway for stress testing
    }

    // Schedule next frame
    next_frame_time += frame_duration;

    // Log progress every 60 frames (1 second at 60fps)
    if (frame_id % 60 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto behind = std::chrono::duration_cast<std::chrono::microseconds>(now - next_frame_time);
      LOGD("Burn encoder %d: frame %d, behind by %ldμs", encoder_id, frame_id, behind.count());
    }
  }

  encoder.encoder_close();
  LOGW("Burn encoder %d stopped after %d frames", encoder_id, frame_id);
}

int main(int argc, char* argv[]) {
  LOGW("Starting encoderd_burn: %d parallel 4K@%dfps HEVC encoders",
       BURN_ENCODER_COUNT, BURN_FPS);

  if (!Hardware::PC()) {
    int ret = util::set_realtime_priority(53);
    assert(ret == 0);
  }

  std::vector<std::thread> encoder_threads;
  for (int i = 0; i < BURN_ENCODER_COUNT; i++) encoder_threads.emplace_back(burn_encoder_thread, i);
  LOGW("All burn encoders started. Press Ctrl+C to stop.");
  for (auto &thread : encoder_threads) thread.join();

  LOGW("encoderd_burn shutdown complete");
  return 0;
}
