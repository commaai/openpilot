#include "system/loggerd/clip_encoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <memory>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "common/swaglog.h"
#include "system/loggerd/encoder/v4l_decoder.h"
#include "system/loggerd/encoder/v4l_encoder.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

namespace {

constexpr double SEGMENT_DURATION = 60.0;
constexpr int CLIP_FPS = 20;
constexpr double PARALLEL_CLIP_MIN_DURATION = 2 * SEGMENT_DURATION;

const EncoderInfo clip_encoder_info = {
  .publish_name = "livestreamNarrowRoadEncodeData",
  .record = false,
  .fps = CLIP_FPS,
  .get_settings = [](int) { return EncoderSettings::StreamEncoderSettings(); },
  INIT_ENCODE_FUNCTIONS(LivestreamNarrowRoadEncode),
};

bool open_input(const std::string &path, AVFormatContext **ctx, int *stream_index) {
  if (avformat_open_input(ctx, path.c_str(), nullptr, nullptr) < 0 ||
      avformat_find_stream_info(*ctx, nullptr) < 0 ||
      (*stream_index = av_find_best_stream(*ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0)) < 0) {
    LOGE("failed to open clip input %s", path.c_str());
    avformat_close_input(ctx);
    return false;
  }
  return true;
}

void remove_file(const std::string &path) {
  std::error_code error;
  std::filesystem::remove(path, error);
}

int encode_clip_worker(const std::vector<std::string> &inputs, int width, int height,
                       double start_time, double duration, int bitrate, int speedup,
                       int64_t frame_offset, int64_t *encoded_frames,
                       V4LEncoder::PacketCallback packet_callback) try {
  EncoderInfo encoder_info = clip_encoder_info;
  encoder_info.get_settings = [bitrate](int) {
    return EncoderSettings{.encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
                           .bitrate = bitrate, .gop_size = 5};
  };
  V4LDecoder decoder;
  V4LEncoder::Options options = {
    .packet_callback = std::move(packet_callback),
    .input_format = V4L2_PIX_FMT_NV12_UBWC,
    .input_done_callback = [&decoder](VisionBuf *buf) { decoder.releaseFrame(buf); },
    .max_performance = true,
  };
  V4LEncoder encoder(encoder_info, width, height, std::move(options));
  encoder.encoder_open();
  if (!decoder.init(V4LDecoder::DEVICE, width, height, V4L2_PIX_FMT_HEVC, true, V4L2_PIX_FMT_NV12_UBWC)) return 1;

  const int64_t first_frame = std::floor(start_time * CLIP_FPS);
  const int64_t end_frame = std::ceil((start_time + duration) * CLIP_FPS);
  int64_t input_frame = 0;
  int64_t output_frame = 0;
  int64_t received_frames = 0;
  bool failed = false;
  auto pump_decoder = [&](int timeout_ms) {
    V4LDecodedFrame frame;
    if (!decoder.pump(frame, timeout_ms)) return false;
    if (!frame.buf) return true;
    ++received_frames;
    const int64_t source_frame = (int64_t)frame.token - 1;
    if (source_frame < first_frame) {
      decoder.releaseFrame(frame.buf);
      return true;
    }
    if ((frame_offset + source_frame - first_frame) % speedup != 0) {
      decoder.releaseFrame(frame.buf);
      return true;
    }

    VisionIpcBufExtra extra = {};
    extra.frame_id = output_frame;
    extra.timestamp_sof = output_frame * 1000000000ULL / CLIP_FPS;
    extra.timestamp_eof = extra.timestamp_sof;
    if (encoder.encode_frame(frame.buf, &extra) < 0) {
      decoder.releaseFrame(frame.buf);
      return false;
    }

    ++output_frame;
    return true;
  };

  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    const std::string &input = inputs[input_index];
    const int64_t segment_start_frame = input_frame;
    AVFormatContext *ctx = nullptr;
    int stream_index = -1;
    if (!open_input(input, &ctx, &stream_index)) { failed = true; break; }
    AVPacket packet = {};
    while (input_frame < end_frame && av_read_frame(ctx, &packet) >= 0) {
      if (packet.stream_index != stream_index) {
        av_packet_unref(&packet);
        continue;
      }
      if (packet.size <= 0 || (size_t)packet.size > decoder.maxPacketSize()) {
        LOGE("decoder packet too large: %d > %zu", packet.size, decoder.maxPacketSize());
        av_packet_unref(&packet);
        failed = true;
        break;
      }

      // Keep several compressed packets in flight so the firmware can sustain
      // decode/encode overlap and does not downclock due to a shallow queue.
      while (!decoder.queuePacket(&packet, input_frame + 1)) {
        if (!pump_decoder(-1)) {
          failed = true;
          break;
        }
      }
      av_packet_unref(&packet);
      if (failed) break;

      ++input_frame;
    }
    av_packet_unref(&packet);
    avformat_close_input(&ctx);
    // Only the final loggerd segment may be shorter than SEGMENT_DURATION. A
    // short intermediate segment would silently close a gap in the source.
    if (!failed && input_frame < end_frame && input_index + 1 < inputs.size() &&
        input_frame - segment_start_frame < static_cast<int64_t>(SEGMENT_DURATION * CLIP_FPS)) {
      failed = true;
    }
    if (failed || input_frame >= end_frame) break;
  }

  if (!failed) decoder.sendEOS();
  for (int empty_polls = 0; !failed && received_frames < input_frame;) {
    const int64_t before = received_frames;
    failed = !pump_decoder(1000);
    empty_polls = received_frames == before ? empty_polls + 1 : 0;
    if (empty_polls == 5) failed = true;
  }

  encoder.encoder_close();
  const int64_t source_frames = std::max<int64_t>(0, std::min(input_frame, end_frame) - first_frame);
  const int64_t first_output_frame = (speedup - frame_offset % speedup) % speedup;
  const int64_t expected_output_frames = first_output_frame < source_frames ?
    1 + (source_frames - first_output_frame - 1) / speedup : 0;
  if (failed || source_frames == 0 || output_frame != expected_output_frames) {
    LOGE("clip failed: input=%lld/%lld decoded=%lld encoded=%lld/%lld",
         (long long)input_frame, (long long)end_frame, (long long)received_frames,
         (long long)output_frame, (long long)expected_output_frames);
    return 1;
  }
  *encoded_frames = output_frame;
  return 0;
} catch (const std::exception &e) {
  LOGE("clip worker failed: %s", e.what());
  return 1;
}

struct SpoolPacket {
  uint32_t size;
  int64_t timestamp;
  bool keyframe;
};

}  // namespace

int encode_clip(const std::vector<std::string> &inputs, const std::string &output,
                double start_time, double duration, int bitrate, int speedup,
                const std::string &metadata) {
  if (inputs.empty() || !std::isfinite(start_time) || !std::isfinite(duration) ||
      start_time < 0 || duration <= 0 || bitrate <= 0 || speedup <= 0) {
    return 1;
  }

  // Inputs are consecutive loggerd segments. Skip whole files before the clip
  // so a late start does not spend hardware time decoding discarded minutes.
  const double available_duration = inputs.size() * SEGMENT_DURATION;
  if (start_time >= available_duration || duration > available_duration - start_time) return 1;
  const size_t skipped_segments = start_time / SEGMENT_DURATION;
  const std::vector<std::string> clip_inputs(inputs.begin() + skipped_segments, inputs.end());
  const double local_start = start_time - skipped_segments * SEGMENT_DURATION;

  AVFormatContext *ctx = nullptr;
  int stream = -1;
  if (!open_input(clip_inputs.front(), &ctx, &stream)) return 1;
  AVCodecParameters *codec = ctx->streams[stream]->codecpar;
  const int width = codec->width, height = codec->height;
  const bool valid_codec = codec->codec_id == AV_CODEC_ID_HEVC && width > 0 && height > 0;
  avformat_close_input(&ctx);
  if (!valid_codec) return 1;

  std::filesystem::path output_path(output);
  const std::string output_dir = output_path.has_parent_path() ? output_path.parent_path() : ".";
  auto writer = std::make_unique<VideoWriter>(output_dir.c_str(), output_path.filename().c_str(), true,
                                              width, height, CLIP_FPS, cereal::EncodeIndex::Type::QCAMERA_H264);
  if (!metadata.empty()) writer->set_metadata("ai.comma.clip.settings", metadata.c_str());
  V4LEncoder::PacketCallback write_packet = [&writer](uint8_t *data, size_t size, int64_t timestamp,
                                                      bool config, bool keyframe) {
    writer->write(data, size, timestamp, config, keyframe);
  };

  if (clip_inputs.size() < 2 || duration < PARALLEL_CLIP_MIN_DURATION) {
    int64_t encoded_frames = 0;
    const bool success = encode_clip_worker(clip_inputs, width, height, local_start, duration,
                                            bitrate, speedup, 0, &encoded_frames, write_packet) == 0;
    if (!success) {
      writer.reset();
      remove_file(output);
    }
    return success ? 0 : 1;
  }

  const size_t split = std::clamp<size_t>(std::llround((local_start + duration / 2) / SEGMENT_DURATION),
                                           1, clip_inputs.size() - 1);
  const double split_time = split * SEGMENT_DURATION;
  const std::array<std::vector<std::string>, 2> shard_inputs = {
    std::vector<std::string>(clip_inputs.begin(), clip_inputs.begin() + split),
    std::vector<std::string>(clip_inputs.begin() + split, clip_inputs.end()),
  };
  const std::array<double, 2> shard_starts = {local_start, 0};
  const std::array<double, 2> shard_durations = {
    split_time - local_start, local_start + duration - split_time,
  };
  const std::string spool_path = output + ".encoderd-" + std::to_string(getpid()) + ".tmp";
  FILE *spool = fopen(spool_path.c_str(), "w+b");
  if (!spool) {
    writer.reset();
    remove_file(output);
    return 1;
  }
  remove_file(spool_path);
  bool spool_ok = true;
  V4LEncoder::PacketCallback spool_packet = [&](uint8_t *data, size_t size, int64_t timestamp,
                                                bool config, bool keyframe) {
    if (config) return;
    const SpoolPacket packet = {(uint32_t)size, timestamp, keyframe};
    spool_ok &= fwrite(&packet, sizeof(packet), 1, spool) == 1 && fwrite(data, 1, size, spool) == size;
  };
  std::array<int, 2> results = {1, 1};
  std::array<int64_t, 2> encoded_frames = {};
  const std::array<int64_t, 2> frame_offsets = {
    0, (int64_t)std::llround(split_time * CLIP_FPS) - (int64_t)std::floor(local_start * CLIP_FPS),
  };
  std::array<std::thread, 2> workers;

  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i]() {
      results[i] = encode_clip_worker(shard_inputs[i], width, height, shard_starts[i], shard_durations[i],
                                      bitrate, speedup, frame_offsets[i], &encoded_frames[i],
                                      i == 0 ? write_packet : spool_packet);
    });
  }
  for (std::thread &worker : workers) worker.join();

  rewind(spool);
  SpoolPacket packet;
  std::vector<uint8_t> data;
  const int64_t timestamp_offset = encoded_frames[0] * 1000000 / CLIP_FPS;
  while (spool_ok && fread(&packet, sizeof(packet), 1, spool) == 1) {
    data.resize(packet.size);
    spool_ok = fread(data.data(), 1, data.size(), spool) == data.size();
    if (spool_ok) writer->write(data.data(), data.size(), packet.timestamp + timestamp_offset, false, packet.keyframe);
  }
  fclose(spool);
  bool success = results[0] == 0 && results[1] == 0 && spool_ok;
  if (!success) {
    writer.reset();
    remove_file(output);
  }
  return success ? 0 : 1;
}
