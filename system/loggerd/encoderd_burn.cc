#include <cassert>
#include <random>
#include <thread>
#include <chrono>

#include "system/loggerd/loggerd.h"
#include "system/loggerd/encoder/v4l_encoder_burn.h"
#include "common/timing.h"
#include "msgq/visionipc/visionbuf.h"

#include "third_party/linux/include/msm_media_info.h"

#define BURN_ENCODER_COUNT 1  // Number of parallel encoders for stress testing
#define BURN_WIDTH 1920 // 4K width
#define BURN_HEIGHT 1080 // 4K height  
#define BURN_FPS 60           // 60 fps for maximum stress
#define BURN_BITRATE (100 * 1000000)  // 100 Mbps for maximum load

ExitHandler do_exit;

// Generate random noise data to simulate 4K video frames
void generate_random_noise(VisionBuf *buf, std::mt19937 &gen, std::uniform_int_distribution<uint8_t> &dis) {
  uint8_t *y_plane = (uint8_t*)buf->addr;
  uint8_t *uv_plane = y_plane + BURN_WIDTH * BURN_HEIGHT;
  
  // Fill Y plane with random values
  for (int i = 0; i < BURN_WIDTH * BURN_HEIGHT; i++) {
    y_plane[i] = dis(gen);
  }
  
  // Fill UV plane with random values (NV12 format: UV interleaved)
  for (int i = 0; i < (BURN_WIDTH * BURN_HEIGHT) / 2; i++) {
    uv_plane[i] = dis(gen);
  }
}

void burn_encoder_thread(int encoder_id) {
  std::string thread_name = "burn_enc_" + std::to_string(encoder_id);
  util::set_thread_name(thread_name.c_str());
  
  // Create burn encoder configuration - provide dummy publish name
  std::string publish_name = "burnEncodeData_" + std::to_string(encoder_id);
  EncoderInfo burn_encoder_info = {
    .publish_name = publish_name.c_str(),
    .filename = nullptr,      // No file output for burn test
    .record = false,          // Don't record for burn test
    .frame_width = BURN_WIDTH,
    .frame_height = BURN_HEIGHT,
    .fps = BURN_FPS,
    .bitrate = BURN_BITRATE,
    .encode_type = cereal::EncodeIndex::Type::FULL_H_E_V_C,  // HEVC for maximum processing
  };
  
  // Create the burn encoder
  V4LEncoderBurn encoder(burn_encoder_info, BURN_WIDTH, BURN_HEIGHT);
  encoder.encoder_open();
  
  // Calculate proper buffer size using VENUS macros for NV12 format
  size_t frame_size = VENUS_BUFFER_SIZE(COLOR_FMT_NV12, BURN_WIDTH, BURN_HEIGHT);
  if (true) frame_size = 4202496;
  LOGD("Burn encoder %d: calculated buffer size %zu for %dx%d",
       encoder_id, frame_size, BURN_WIDTH, BURN_HEIGHT);
  VisionBuf frame_buf;
  frame_buf.allocate(frame_size);
  
  // Random number generation for noise
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);
  
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
    
    // Generate random noise for this frame
    generate_random_noise(&frame_buf, gen, dis);
    
    // Create frame metadata
    VisionIpcBufExtra extra = {
      .timestamp_eof = nanos_since_boot(),
      .frame_id = frame_id++,
    };
    
    // Encode the frame
    int result = encoder.encode_frame(&frame_buf, &extra);
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
  frame_buf.free();
  LOGW("Burn encoder %d stopped after %d frames", encoder_id, frame_id);
}

int main(int argc, char* argv[]) {
  LOGW("Starting encoderd_burn: %d parallel 4K@%dfps HEVC encoders",
       BURN_ENCODER_COUNT, BURN_FPS);

  if (!Hardware::PC()) {
    // Set high priority and spread across cores for maximum CPU usage
    int ret = util::set_realtime_priority(53);  // Higher than normal encoderd
    assert(ret == 0);
    // Don't constrain to specific cores - let it use all available
  }
  
  // Start multiple parallel encoder threads
  std::vector<std::thread> encoder_threads;
  for (int i = 0; i < BURN_ENCODER_COUNT; i++) {
    encoder_threads.emplace_back(burn_encoder_thread, i);
    // Small delay between starting encoders to avoid initialization conflicts
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  LOGW("All burn encoders started. Press Ctrl+C to stop.");
  
  // Wait for all encoder threads to complete
  for (auto &thread : encoder_threads) {
    thread.join();
  }
  
  LOGW("encoderd_burn shutdown complete");
  return 0;
}
