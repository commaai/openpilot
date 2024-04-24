#include "tools/replay/framereader.h"

#include "common/util.h"
#include "third_party/libyuv/include/libyuv.h"
#include "tools/replay/util.h"

#ifdef __APPLE__
#define HW_DEVICE_TYPE AV_HWDEVICE_TYPE_VIDEOTOOLBOX
#define HW_PIX_FMT AV_PIX_FMT_VIDEOTOOLBOX
#else
#define HW_DEVICE_TYPE AV_HWDEVICE_TYPE_CUDA
#define HW_PIX_FMT AV_PIX_FMT_CUDA
#endif


namespace {

enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  enum AVPixelFormat *hw_pix_fmt = reinterpret_cast<enum AVPixelFormat *>(ctx->opaque);
  for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
    if (*p == *hw_pix_fmt) return *p;
  }
  rWarning("Please run replay with the --no-hw-decoder flag!");
  // fallback to YUV420p
  *hw_pix_fmt = AV_PIX_FMT_NONE;
  return AV_PIX_FMT_YUV420P;
}

}  // namespace

FrameReader::FrameReader() {
  av_log_set_level(AV_LOG_QUIET);
}

FrameReader::~FrameReader() {
  if (decoder_ctx) avcodec_free_context(&decoder_ctx);
  if (input_ctx) avformat_close_input(&input_ctx);
  if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
}

bool FrameReader::load(const std::string &url, bool no_hw_decoder, std::atomic<bool> *abort, bool local_cache, int chunk_size, int retries) {
  auto local_file_path = url.find("https://") == 0 ? cacheFilePath(url) : url;
  if (!util::file_exists(local_file_path)) {
    FileReader f(local_cache, chunk_size, retries);
    if (f.read(url, abort).empty()) {
      return false;
    }
  }
  return loadFromFile(local_file_path, no_hw_decoder, abort);
}

bool FrameReader::loadFromFile(const std::string &file, bool no_hw_decoder, std::atomic<bool> *abort) {
  if (avformat_open_input(&input_ctx, file.c_str(), nullptr, nullptr) != 0 ||
      avformat_find_stream_info(input_ctx, nullptr) < 0) {
    rError("Failed to open input file or find video stream");
    return false;
  }
  input_ctx->probesize = 10 * 1024 * 1024;  // 10MB

  AVStream *video = input_ctx->streams[0];
  const AVCodec *decoder = avcodec_find_decoder(video->codecpar->codec_id);
  if (!decoder) return false;

  decoder_ctx = avcodec_alloc_context3(decoder);
  if (!decoder_ctx || avcodec_parameters_to_context(decoder_ctx, video->codecpar) != 0) {
    rError("Failed to allocate or initialize codec context");
    return false;
  }

  width = (decoder_ctx->width + 3) & ~3;
  height = decoder_ctx->height;

  if (has_hw_decoder && !no_hw_decoder && !initHardwareDecoder(HW_DEVICE_TYPE)) {
    rWarning("No device with hardware decoder found. fallback to CPU decoding.");
  }

  if (avcodec_open2(decoder_ctx, decoder, nullptr) < 0) {
    rError("Failed to open codec");
    return false;
  }

  AVPacket pkt;
  packets_info.reserve(60 * 20);  // 20fps, one minute
  while (!(abort && *abort) && av_read_frame(input_ctx, &pkt) == 0) {
    packets_info.emplace_back(PacketInfo{.flags = pkt.flags, .pos = pkt.pos});
    av_packet_unref(&pkt);
  }

  avio_seek(input_ctx->pb, 0, SEEK_SET);
  return !packets_info.empty();
}

bool FrameReader::initHardwareDecoder(AVHWDeviceType hw_device_type) {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(decoder_ctx->codec, i);
    if (!config) {
      rWarning("decoder %s does not support hw device type %s.", decoder_ctx->codec->name,
               av_hwdevice_get_type_name(hw_device_type));
      return false;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == hw_device_type) {
      hw_pix_fmt = config->pix_fmt;
      break;
    }
  }

  int ret = av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type, nullptr, nullptr, 0);
  if (ret < 0) {
    hw_pix_fmt = AV_PIX_FMT_NONE;
    has_hw_decoder = false;
    rWarning("Failed to create specified HW device %d.", ret);
    return false;
  }

  decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
  decoder_ctx->opaque = &hw_pix_fmt;
  decoder_ctx->get_format = get_hw_format;
  return true;
}

bool FrameReader::get(int idx, VisionBuf *buf) {
  if (!buf || idx < 0 || idx >= packets_info.size()) {
    return false;
  }
  return decode(idx, buf);
}

bool FrameReader::decode(int idx, VisionBuf *buf) {
  int from_idx = idx;
  if (idx != prev_idx + 1) {
    // seeking to the nearest key frame
    for (int i = idx; i >= 0; --i) {
      if (packets_info[i].flags & AV_PKT_FLAG_KEY) {
        from_idx = i;
        break;
      }
    }
    avio_seek(input_ctx->pb, packets_info[from_idx].pos, SEEK_SET);
  }
  prev_idx = idx;

  bool result = false;
  AVPacket pkt;
  for (int i = from_idx; i <= idx; ++i) {
    if (av_read_frame(input_ctx, &pkt) == 0) {
      AVFrame *f = decodeFrame(&pkt);
      if (f && i == idx) {
        result = copyBuffers(f, buf);
      }
      av_packet_unref(&pkt);
    }
  }
  return result;
}

AVFrame *FrameReader::decodeFrame(AVPacket *pkt) {
  int ret = avcodec_send_packet(decoder_ctx, pkt);
  if (ret < 0) {
    rError("Error sending a packet for decoding: %d", ret);
    return nullptr;
  }

  av_frame_.reset(av_frame_alloc());
  ret = avcodec_receive_frame(decoder_ctx, av_frame_.get());
  if (ret != 0) {
    rError("avcodec_receive_frame error: %d", ret);
    return nullptr;
  }

  if (av_frame_->format == hw_pix_fmt) {
    hw_frame.reset(av_frame_alloc());
    if ((ret = av_hwframe_transfer_data(hw_frame.get(), av_frame_.get(), 0)) < 0) {
      rError("error transferring the data from GPU to CPU");
      return nullptr;
    }
    return hw_frame.get();
  } else {
    return av_frame_.get();
  }
}

bool FrameReader::copyBuffers(AVFrame *f, VisionBuf *buf) {
  if (hw_pix_fmt == HW_PIX_FMT) {
    for (int i = 0; i < height/2; i++) {
      memcpy(buf->y + (i*2 + 0)*buf->stride, f->data[0] + (i*2 + 0)*f->linesize[0], width);
      memcpy(buf->y + (i*2 + 1)*buf->stride, f->data[0] + (i*2 + 1)*f->linesize[0], width);
      memcpy(buf->uv + i*buf->stride, f->data[1] + i*f->linesize[1], width);
    }
  } else {
    libyuv::I420ToNV12(f->data[0], f->linesize[0],
                       f->data[1], f->linesize[1],
                       f->data[2], f->linesize[2],
                       buf->y, buf->stride,
                       buf->uv, buf->stride,
                       width, height);
  }
  return true;
}
