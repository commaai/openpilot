#include "system/loggerd/clip_encoder.h"

#include <cmath>
#include <filesystem>
#include <memory>

extern "C" {
#include <libavformat/avformat.h>
}

#include "common/swaglog.h"
#include "system/loggerd/encoder/v4l_encoder.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"
#include "tools/replay/qcom_decoder.h"

namespace {

constexpr int CLIP_FPS = 20;

const EncoderInfo clip_encoder_info = {
  .publish_name = "clipEncodeData",
  .record = false,
  .fps = CLIP_FPS,
  .get_settings = [](int) { return EncoderSettings::StreamEncoderSettings(); },
  INIT_ENCODE_FUNCTIONS(LivestreamRoadEncode),
};

bool open_input(const std::string &path, AVFormatContext **ctx, int *stream_index) {
  if (avformat_open_input(ctx, path.c_str(), nullptr, nullptr) < 0 || avformat_find_stream_info(*ctx, nullptr) < 0) {
    LOGE("failed to open clip input %s", path.c_str());
    return false;
  }
  *stream_index = av_find_best_stream(*ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  return *stream_index >= 0;
}

}  // namespace

int encode_clip(const std::vector<std::string> &inputs, const std::string &output,
                double start_time, double duration) {
  if (inputs.empty() || start_time < 0 || duration <= 0) return 1;

  AVFormatContext *first_ctx = nullptr;
  int first_stream = -1;
  if (!open_input(inputs.front(), &first_ctx, &first_stream)) return 1;
  AVCodecParameters *codec = first_ctx->streams[first_stream]->codecpar;
  if (codec->codec_id != AV_CODEC_ID_HEVC || codec->width <= 0 || codec->height <= 0) {
    LOGE("clip input must be HEVC with known dimensions");
    avformat_close_input(&first_ctx);
    return 1;
  }
  const int width = codec->width;
  const int height = codec->height;
  avformat_close_input(&first_ctx);

  MsmVidc decoder;
  if (!decoder.init(VIDEO_DEVICE, width, height, V4L2_PIX_FMT_HEVC, true, V4L2_PIX_FMT_NV12)) return 1;

  std::filesystem::path output_path(output);
  const std::string output_dir = output_path.has_parent_path() ? output_path.parent_path() : ".";
  VideoWriter writer(output_dir.c_str(), output_path.filename().c_str(), true,
                     width, height, CLIP_FPS, cereal::EncodeIndex::Type::QCAMERA_H264);
  V4LEncoder encoder(clip_encoder_info, width, height,
    [&writer](uint8_t *data, size_t size, int64_t timestamp, bool config, bool keyframe) {
      writer.write(data, size, timestamp, config, keyframe);
    }, V4L2_PIX_FMT_NV12, [&decoder](VisionBuf *buf) { decoder.releaseFrame(buf); }, true);
  encoder.encoder_open();

  const int64_t first_frame = std::floor(start_time * CLIP_FPS);
  const int64_t end_frame = std::ceil((start_time + duration) * CLIP_FPS);
  int64_t input_frame = 0;
  int64_t output_frame = 0;
  bool failed = false;

  for (const std::string &input : inputs) {
    AVFormatContext *ctx = nullptr;
    int stream_index = -1;
    if (!open_input(input, &ctx, &stream_index)) {
      failed = true;
      break;
    }
    AVPacket *packet = av_packet_alloc();
    while (input_frame < end_frame && av_read_frame(ctx, packet) >= 0) {
      if (packet->stream_index != stream_index) {
        av_packet_unref(packet);
        continue;
      }
      VisionBuf *frame = decoder.decodeFrameDirect(packet);
      av_packet_unref(packet);
      if (!frame) {
        failed = true;
        break;
      }
      if (input_frame++ < first_frame) {
        decoder.releaseFrame(frame);
        continue;
      }

      VisionIpcBufExtra extra = {};
      extra.frame_id = output_frame;
      extra.timestamp_sof = output_frame * 1000000000ULL / CLIP_FPS;
      extra.timestamp_eof = extra.timestamp_sof;
      if (encoder.encode_frame(frame, &extra) < 0) {
        failed = true;
        break;
      }
      ++output_frame;
      printf("out_time_us=%lld\n", (long long)(output_frame * 1000000 / CLIP_FPS));
      fflush(stdout);
    }
    av_packet_free(&packet);
    avformat_close_input(&ctx);
    if (failed || input_frame >= end_frame) break;
  }

  encoder.encoder_close();
  if (failed || input_frame < end_frame) {
    LOGE("clip ended before requested duration");
    return 1;
  }
  return 0;
}
