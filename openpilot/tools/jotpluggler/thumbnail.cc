#include "tools/jotpluggler/thumbnail.h"

#include "imgui_impl_opengl3_loader.h"

#include <algorithm>
#include <cmath>
#include <limits>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
}

namespace {

bool decode_jpeg(const std::vector<uint8_t> &jpeg, int *width, int *height, std::vector<uint8_t> *rgba) {
  if (jpeg.empty()) return false;

  const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  AVCodecContext *context = codec != nullptr ? avcodec_alloc_context3(codec) : nullptr;
  AVFrame *frame = av_frame_alloc();
  AVPacket *packet = av_packet_alloc();
  if (context == nullptr || frame == nullptr || packet == nullptr) {
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&context);
    return false;
  }

  const bool packet_ready = jpeg.size() <= static_cast<size_t>(std::numeric_limits<int>::max())
                         && av_new_packet(packet, static_cast<int>(jpeg.size())) >= 0;
  if (packet_ready) {
    std::copy(jpeg.begin(), jpeg.end(), packet->data);
  }
  const bool decoded = packet_ready
                    && avcodec_open2(context, codec, nullptr) >= 0
                    && avcodec_send_packet(context, packet) >= 0
                    && avcodec_receive_frame(context, frame) >= 0;
  if (!decoded || frame->width <= 0 || frame->height <= 0) {
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&context);
    return false;
  }

  int chroma_x_shift = 0;
  int chroma_y_shift = 0;
  switch (static_cast<AVPixelFormat>(frame->format)) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
      chroma_x_shift = 1;
      chroma_y_shift = 1;
      break;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
      chroma_x_shift = 1;
      break;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
      break;
    default:
      av_packet_free(&packet);
      av_frame_free(&frame);
      avcodec_free_context(&context);
      return false;
  }

  *width = frame->width;
  *height = frame->height;
  rgba->resize(static_cast<size_t>(*width) * static_cast<size_t>(*height) * 4U);
  const bool full_range = frame->color_range == AVCOL_RANGE_JPEG
                       || frame->format == AV_PIX_FMT_YUVJ420P
                       || frame->format == AV_PIX_FMT_YUVJ422P
                       || frame->format == AV_PIX_FMT_YUVJ444P;
  for (int y = 0; y < *height; ++y) {
    const uint8_t *y_row = frame->data[0] + y * frame->linesize[0];
    const uint8_t *u_row = frame->data[1] + (y >> chroma_y_shift) * frame->linesize[1];
    const uint8_t *v_row = frame->data[2] + (y >> chroma_y_shift) * frame->linesize[2];
    uint8_t *out = rgba->data() + static_cast<size_t>(y) * static_cast<size_t>(*width) * 4U;
    for (int x = 0; x < *width; ++x) {
      const double luma = full_range ? static_cast<double>(y_row[x])
                                     : 1.164383 * (static_cast<double>(y_row[x]) - 16.0);
      const double u = static_cast<double>(u_row[x >> chroma_x_shift]) - 128.0;
      const double v = static_cast<double>(v_row[x >> chroma_x_shift]) - 128.0;
      const double red = luma + (full_range ? 1.402 : 1.596027) * v;
      const double green = luma - (full_range ? 0.344136 : 0.391762) * u
                                - (full_range ? 0.714136 : 0.812968) * v;
      const double blue = luma + (full_range ? 1.772 : 2.017232) * u;
      out[x * 4 + 0] = static_cast<uint8_t>(std::clamp(std::lround(red), 0L, 255L));
      out[x * 4 + 1] = static_cast<uint8_t>(std::clamp(std::lround(green), 0L, 255L));
      out[x * 4 + 2] = static_cast<uint8_t>(std::clamp(std::lround(blue), 0L, 255L));
      out[x * 4 + 3] = 255;
    }
  }

  av_packet_free(&packet);
  av_frame_free(&frame);
  avcodec_free_context(&context);
  return true;
}

std::string format_thumbnail_time(double seconds) {
  const int rounded = std::max(0, static_cast<int>(std::lround(seconds)));
  const int hours = rounded / 3600;
  const int minutes = (rounded % 3600) / 60;
  const int secs = rounded % 60;
  if (hours > 0) {
    return util::string_format("%d:%02d:%02d", hours, minutes, secs);
  }
  return util::string_format("%02d:%02d", minutes, secs);
}

}  // namespace

struct ThumbnailView::Impl {
  ~Impl() {
    destroy_texture();
  }

  void setThumbnails(const std::vector<ThumbnailFrame> &next_thumbnails) {
    destroy_texture();
    thumbnails = &next_thumbnails;
    displayed_index = -1;
    failed_index = -1;
  }

  void update(double tracker_time) {
    if (thumbnails == nullptr || thumbnails->empty()) return;
    auto it = std::lower_bound(thumbnails->begin(), thumbnails->end(), tracker_time,
                               [](const ThumbnailFrame &frame, double time) {
                                 return frame.timestamp < time;
                               });
    if (it == thumbnails->end()) {
      it = std::prev(thumbnails->end());
    } else if (it != thumbnails->begin()) {
      const auto previous = std::prev(it);
      if (std::abs(previous->timestamp - tracker_time) <= std::abs(it->timestamp - tracker_time)) {
        it = previous;
      }
    }
    const int index = static_cast<int>(std::distance(thumbnails->begin(), it));
    if (index == displayed_index || index == failed_index) return;

    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
    if (!decode_jpeg(it->jpeg, &width, &height, &rgba)) {
      failed_index = index;
      return;
    }

    if (texture == 0) {
      glGenTextures(1, &texture);
    }
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    texture_width = width;
    texture_height = height;
    displayed_index = index;
    failed_index = -1;
  }

  void drawSized(ImVec2 size, bool loading) const {
    size.x = std::max(1.0f, size.x);
    size.y = std::max(1.0f, size.y);
    ImGui::InvisibleButton("##thumbnail_sized", size);
    const ImVec2 pane_min = ImGui::GetItemRectMin();
    const ImVec2 pane_max = ImGui::GetItemRectMax();
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(pane_min, pane_max, IM_COL32(24, 24, 24, 255));

    if (texture != 0 && texture_width > 0 && texture_height > 0) {
      const float scale = std::min(size.x / static_cast<float>(texture_width),
                                   size.y / static_cast<float>(texture_height));
      const ImVec2 image_size(static_cast<float>(texture_width) * scale,
                              static_cast<float>(texture_height) * scale);
      const ImVec2 image_min(pane_min.x + (size.x - image_size.x) * 0.5f,
                             pane_min.y + (size.y - image_size.y) * 0.5f);
      const ImVec2 image_max(image_min.x + image_size.x, image_min.y + image_size.y);
      draw_list->AddImage(static_cast<ImTextureID>(texture), image_min, image_max);

      if (thumbnails != nullptr && displayed_index >= 0
          && displayed_index < static_cast<int>(thumbnails->size())) {
        const ThumbnailFrame &frame = (*thumbnails)[static_cast<size_t>(displayed_index)];
        const std::string label = util::string_format("%s  ·  segment %d  ·  %d/%zu",
                                                      format_thumbnail_time(frame.timestamp).c_str(),
                                                      frame.segment,
                                                      displayed_index + 1,
                                                      thumbnails->size());
        const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
        const ImVec2 label_min(image_min.x, std::max(image_min.y, image_max.y - text_size.y - 14.0f));
        draw_list->AddRectFilled(label_min, image_max, IM_COL32(0, 0, 0, 175));
        draw_list->AddText(ImVec2(label_min.x + 7.0f, label_min.y + 7.0f), IM_COL32_WHITE, label.c_str());
      }
      return;
    }

    const bool has_thumbnails = thumbnails != nullptr && !thumbnails->empty();
    const char *label = loading ? "loading" : (has_thumbnails ? "invalid thumbnail" : "no thumbnails");
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    draw_list->AddText(ImVec2(pane_min.x + (size.x - text_size.x) * 0.5f,
                              pane_min.y + (size.y - text_size.y) * 0.5f),
                       IM_COL32(187, 187, 187, 255), label);
  }

  void destroy_texture() {
    if (texture != 0) {
      glDeleteTextures(1, &texture);
    }
    texture = 0;
    texture_width = 0;
    texture_height = 0;
  }

  const std::vector<ThumbnailFrame> *thumbnails = nullptr;
  int displayed_index = -1;
  int failed_index = -1;
  GLuint texture = 0;
  int texture_width = 0;
  int texture_height = 0;
};

ThumbnailView::ThumbnailView() : impl_(std::make_unique<Impl>()) {}
ThumbnailView::~ThumbnailView() = default;

void ThumbnailView::setThumbnails(const std::vector<ThumbnailFrame> &thumbnails) {
  impl_->setThumbnails(thumbnails);
}

void ThumbnailView::update(double tracker_time) {
  impl_->update(tracker_time);
}

void ThumbnailView::drawSized(ImVec2 size, bool loading) {
  impl_->drawSized(size, loading);
}

void draw_thumbnail_pane(AppSession *session, UiState *state) {
  if (session->thumbnail_view == nullptr) {
    ImGui::TextDisabled("Thumbnails unavailable");
    return;
  }
  if (state->has_tracker_time) {
    session->thumbnail_view->update(state->tracker_time);
  }
  session->thumbnail_view->drawSized(ImGui::GetContentRegionAvail(), session->async_route_loading);
}
