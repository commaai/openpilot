// ImGui port of tools/cabana/cameraview.cc (CameraWidget) + the camera-view
// slice of tools/cabana/videowidget.cc (StreamCameraView, camera tabs).
// Timeline/thumbnails/alert overlays are P5.2's job -- see MIGRATION.md
// Phase 5. Ported 1:1 from those frozen Qt sources unless noted below.
#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "imgui_impl_opengl3_loader.h"
#include "libyuv.h"

#include "msgq/visionipc/visionipc_client.h"

#include "tools/cabana/streams/replaystream.h"
#include "tools/replay/timeline.h"

// Standalone-JPEG decode for qlog thumbnails (see ThumbnailStore below): the
// MJPEG decoder is part of libavcodec, which is already linked into
// _cabana_imgui for tools/replay's video decode path, so no new link
// dependency is needed. Wrapped in extern "C" the same way
// tools/replay/framereader.h does.
extern "C" {
#include <libavcodec/avcodec.h>
}

namespace {

// Indexed by VisionStreamType (ROAD=0, DRIVER=1, WIDE_ROAD=2) -- mirrors
// VideoWidget::vipcAvailableStreamsUpdated()'s stream_names array. VISION_STREAM_MAP
// is deliberately excluded (Qt's array would be out-of-bounds for it too; we
// just skip it defensively instead of replicating the UB).
constexpr const char *kStreamNames[3] = {"Road camera", "Driver camera", "Wide road camera"};

// Hand-rolled tab button instead of ImGui::BeginTabBar()/BeginTabItem(): see
// detail_panel.cc's draw_tab_button() comment -- native tab-item caption text
// doesn't render on the frame a tab is (re)created in this ImGui build, which
// would show blank camera tab labels in the mandatory single-shot --output
// capture. Same Button + Tab/TabSelected/TabHovered theme colors as that helper.
bool draw_camera_tab_button(const char *label, bool selected) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(selected ? ImGuiCol_TabSelected : ImGuiCol_Tab));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_TabHovered));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_TabSelected));
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
  const bool clicked = ImGui::Button(label);
  ImGui::PopStyleColor(4);
  return clicked;
}

// Exact QColor values from tools/cabana/videowidget.cc's `timeline_colors[]`,
// indexed by TimelineType (None, Engaged, AlertInfo, AlertWarning,
// AlertCritical, UserBookmark). Qt::green / Qt::magenta are (0,255,0) /
// (255,0,255).
constexpr ImVec4 kTimelineColorsF[] = {
    ImVec4(111 / 255.0f, 143 / 255.0f, 175 / 255.0f, 1.0f),
    ImVec4(0 / 255.0f, 163 / 255.0f, 108 / 255.0f, 1.0f),
    ImVec4(0 / 255.0f, 255 / 255.0f, 0 / 255.0f, 1.0f),
    ImVec4(255 / 255.0f, 195 / 255.0f, 0 / 255.0f, 1.0f),
    ImVec4(199 / 255.0f, 0 / 255.0f, 57 / 255.0f, 1.0f),
    ImVec4(255 / 255.0f, 0 / 255.0f, 255 / 255.0f, 1.0f),
};

ImU32 timeline_color(TimelineType type, float alpha = 1.0f) {
  ImVec4 c = kTimelineColorsF[static_cast<int>(type)];
  c.w = alpha;
  return ImGui::ColorConvertFloat4ToU32(c);
}

struct DecodedImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;
};

// Decodes one standalone JPEG (a qlog thumbnail) via libavcodec's native
// MJPEG decoder -- no libjpeg dependency needed. `ctx` is caller-owned and
// reused across a batch of thumbnails (one context per decode thread; see
// ThumbnailStore::handle_qlog), since allocating a fresh AVCodecContext per
// image would be wasteful and MJPEG frames have no cross-frame state to leak
// (flushed defensively before each decode regardless).
//
// system/loggerd/encoder/jpeg_encoder.cc always encodes thumbnails as 4:2:0,
// full-range (JCS_YCbCr / JFIF) JPEGs, so the YUV->RGB step below uses
// libyuv's *full range* J420 conversion (same NV12/I420-family pattern as
// CameraViewState's live-frame upload, just full- instead of studio-range) --
// this mirrors ffmpeg's own AV_PIX_FMT_YUVJ420P output for such images.
// Newer ffmpeg builds may instead report AV_PIX_FMT_YUV420P with
// color_range == AVCOL_RANGE_JPEG for the same bits, so both are handled.
bool decode_jpeg_to_rgba(const uint8_t *data, size_t size, AVCodecContext *ctx, DecodedImage *out) {
  AVPacket *pkt = av_packet_alloc();
  if (pkt == nullptr) return false;
  bool ok = false;
  if (av_new_packet(pkt, static_cast<int>(size)) == 0) {
    std::memcpy(pkt->data, data, size);
    if (AVFrame *frame = av_frame_alloc(); frame != nullptr) {
      avcodec_flush_buffers(ctx);
      if (avcodec_send_packet(ctx, pkt) == 0 && avcodec_receive_frame(ctx, frame) == 0 &&
          frame->width > 0 && frame->height > 0 &&
          (frame->format == AV_PIX_FMT_YUVJ420P || frame->format == AV_PIX_FMT_YUV420P)) {
        const bool full_range = (frame->format == AV_PIX_FMT_YUVJ420P) || (frame->color_range == AVCOL_RANGE_JPEG);
        out->width = frame->width;
        out->height = frame->height;
        out->rgba.resize(static_cast<size_t>(out->width) * static_cast<size_t>(out->height) * 4U);
        const auto convert = full_range ? libyuv::J420ToABGR : libyuv::I420ToABGR;
        convert(frame->data[0], frame->linesize[0], frame->data[1], frame->linesize[1], frame->data[2],
                frame->linesize[2], out->rgba.data(), out->width * 4, out->width, out->height);
        ok = true;
      }
      av_frame_free(&frame);
    }
  }
  av_packet_free(&pkt);
  return ok;
}

struct ThumbInfo {
  GLuint texture = 0;
  int width = 0;
  int height = 0;
};

ThumbInfo upload_rgba_texture(const DecodedImage &img) {
  ThumbInfo info;
  if (img.width <= 0 || img.height <= 0) return info;
  glGenTextures(1, &info.texture);
  glBindTexture(GL_TEXTURE_2D, info.texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.rgba.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  info.width = img.width;
  info.height = img.height;
  return info;
}

// Owns the GL textures for qlog thumbnail images -- mirrors
// StreamCameraView's `thumbnails`/`big_thumbnails` maps (tools/cabana/videowidget.h),
// merged into one native-resolution texture per timestamp here: Qt keeps two
// pre-scaled QPixmaps per thumbnail (a small badge with the alert baked in,
// and the original for the paused scrub view); this store keeps one texture
// and draws both the scaled badge and the alert text as live per-frame
// overlays instead (see draw_video_overlays), which is simpler and avoids
// double-decoding every thumbnail at two sizes. Deviation noted in report.
//
// Same function-local-static / no-GL-teardown-in-destructor rationale as
// CameraViewState below: this instance outlives the GL context (process
// atexit teardown happens after GlfwRuntime has already torn down GL), so
// leaked texture handles here are reclaimed by the OS at process exit like
// CameraViewState's texture is.
class ThumbnailStore {
 public:
  ThumbnailStore() = default;
  ThumbnailStore(const ThumbnailStore &) = delete;
  ThumbnailStore &operator=(const ThumbnailStore &) = delete;

  // (Re)connects to the current stream's qLogLoaded event exactly once per
  // stream instance -- mirrors the "wired_stream" pattern in
  // historylog_panel.cc. Clears stale thumbnails when the stream changes
  // (e.g. a new route loaded via the stream selector); Qt doesn't need this
  // since it never swaps `can` under a live VideoWidget.
  void ensure_connected(AppState &app) {
    if (wired_stream_ == app.stream.get()) return;
    wired_stream_ = app.stream.get();
    thumbs_.clear();
    if (auto *rs = dynamic_cast<ReplayStream *>(app.stream.get())) {
      rs->qLogLoaded.connect([this](std::shared_ptr<LogReader> qlog) { handle_qlog(qlog); });
    }
  }

  // Mirrors StreamCameraView::parseQLog(): the CPU-heavy JPEG decode is
  // parallelized across threads (one AVCodecContext per thread, reused for
  // that thread's whole chunk of events), matching Qt's chunked
  // std::thread split over qlog->events. GL texture upload can't be
  // parallelized (GL is single-threaded), so it happens in a final serial
  // pass after joining -- safe because ReplayStream only ever fires
  // qLogLoaded on the UI thread (replaystream.cc's onQLogLoaded goes through
  // onUiThread()/enqueue()), which is also the only thread that reads
  // thumbs_ (from draw_video_overlays), so no lock is needed around the map
  // itself.
  void handle_qlog(const std::shared_ptr<LogReader> &qlog) {
    const std::vector<Event> &events = qlog->events;
    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t chunk = (events.size() + num_threads - 1) / std::max<size_t>(1, num_threads);

    std::mutex result_mutex;
    std::vector<std::pair<uint64_t, DecodedImage>> decoded;
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads && t * chunk < events.size(); ++t) {
      const size_t start = t * chunk;
      const size_t end = std::min(start + chunk, events.size());
      threads.emplace_back([&events, start, end, &result_mutex, &decoded]() {
        const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
        if (codec == nullptr) return;
        AVCodecContext *ctx = avcodec_alloc_context3(codec);
        if (ctx == nullptr) return;
        if (avcodec_open2(ctx, codec, nullptr) < 0) {
          avcodec_free_context(&ctx);
          return;
        }
        for (size_t i = start; i < end; ++i) {
          const Event &e = events[i];
          if (e.which != cereal::Event::Which::THUMBNAIL) continue;
          capnp::FlatArrayMessageReader reader(e.data);
          auto thumb_data = reader.getRoot<cereal::Event>().getThumbnail();
          auto image_data = thumb_data.getThumbnail();
          DecodedImage img;
          if (decode_jpeg_to_rgba(image_data.begin(), image_data.size(), ctx, &img)) {
            std::lock_guard<std::mutex> lk(result_mutex);
            decoded.emplace_back(thumb_data.getTimestampEof(), std::move(img));
          }
        }
        avcodec_free_context(&ctx);
      });
    }
    for (std::thread &th : threads) th.join();

    for (auto &[ts, img] : decoded) {
      thumbs_[ts] = upload_rgba_texture(img);
    }
  }

  // mirrors the std::map::lower_bound() lookups in Qt's
  // drawThumbnail()/drawScrubThumbnail().
  const ThumbInfo *find_at_or_after(uint64_t mono_time) const {
    auto it = thumbs_.lower_bound(mono_time);
    return it != thumbs_.end() ? &it->second : nullptr;
  }

 private:
  AbstractStream *wired_stream_ = nullptr;
  std::map<uint64_t, ThumbInfo> thumbs_;
};

// mirrors videowidget.cc's MIN_VIDEO_HEIGHT / THUMBNAIL_MARGIN constants.
constexpr float kMinVideoHeight = 100.0f;
constexpr float kThumbnailMargin = 3.0f;

void draw_time_label(ImDrawList *draw_list, ImVec2 rect_min, ImVec2 rect_max, double seconds) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", seconds);
  const ImVec2 sz = ImGui::CalcTextSize(buf);
  const float x = rect_min.x + (rect_max.x - rect_min.x - sz.x) * 0.5f;
  const float y = rect_max.y - kThumbnailMargin - sz.y;
  draw_list->AddText(ImVec2(x, y), IM_COL32(255, 255, 255, 255), buf);
}

// mirrors StreamCameraView::drawThumbnail(): a small bordered badge bottom-
// aligned in the frame, positioned at the hovered time's x-fraction.
void draw_small_thumbnail(ImDrawList *draw_list, ImVec2 frame_min, ImVec2 frame_max, const ThumbInfo &thumb,
                           double thumb_time, double min_sec, double max_sec) {
  const float target_h = std::max(1.0f, kMinVideoHeight - 2.0f * kThumbnailMargin);
  const float scale = target_h / static_cast<float>(thumb.height);
  const float w = static_cast<float>(thumb.width) * scale;
  const float frame_w = frame_max.x - frame_min.x;
  const double frac = (max_sec > min_sec) ? (thumb_time - min_sec) / (max_sec - min_sec) : 0.0;
  const float pos = static_cast<float>(frac) * frame_w;
  const float x = frame_min.x + std::clamp(pos - w * 0.5f, kThumbnailMargin, std::max(kThumbnailMargin, frame_w - w - kThumbnailMargin + 1.0f));
  const float y = frame_max.y - target_h - kThumbnailMargin;

  draw_list->AddImage(static_cast<ImTextureID>(thumb.texture), ImVec2(x, y), ImVec2(x + w, y + target_h));
  draw_list->AddRect(ImVec2(x, y), ImVec2(x + w, y + target_h), IM_COL32(255, 255, 255, 255), 0.0f, 0, 2.0f);
  draw_time_label(draw_list, ImVec2(x, y), ImVec2(x + w, y + target_h), thumb_time);
}

// mirrors StreamCameraView::drawScrubThumbnail(): full-bleed aspect-fit
// preview while paused-and-hovering.
void draw_scrub_thumbnail(ImDrawList *draw_list, ImVec2 frame_min, ImVec2 frame_max, const ThumbInfo &thumb,
                           double thumb_time) {
  draw_list->AddRectFilled(frame_min, frame_max, IM_COL32(0, 0, 0, 255));
  const float avail_w = std::max(1.0f, frame_max.x - frame_min.x);
  const float avail_h = std::max(1.0f, frame_max.y - frame_min.y);
  const float ratio = static_cast<float>(thumb.width) / static_cast<float>(thumb.height);
  ImVec2 size(avail_w, avail_h);
  if (avail_w / avail_h > ratio) {
    size.x = avail_h * ratio;
  } else {
    size.y = avail_w / ratio;
  }
  const ImVec2 origin(frame_min.x + (avail_w - size.x) * 0.5f, frame_min.y + (avail_h - size.y) * 0.5f);
  draw_list->AddImage(static_cast<ImTextureID>(thumb.texture), origin, ImVec2(origin.x + size.x, origin.y + size.y));
  draw_time_label(draw_list, origin, ImVec2(origin.x + size.x, origin.y + size.y), thumb_time);
}

// mirrors StreamCameraView::drawAlert(): a semi-transparent band the height
// of the (line-broken) alert text, top-aligned in `rect`, text centered on
// top. Qt word-wraps text1/text2 as a single paragraph within the frame
// width; alert strings are short banner phrases in practice, so this draws
// text1/text2 as up to two independently centered lines instead of
// replicating full word-wrap -- visually equivalent for real alert text,
// simpler than reimplementing Qt's QFontMetrics::boundingRect wrapping.
void draw_alert_band(ImDrawList *draw_list, ImVec2 rect_min, ImVec2 rect_max, const Timeline::Entry &alert) {
  std::vector<std::string> lines;
  if (!alert.text1.empty()) lines.push_back(alert.text1);
  if (!alert.text2.empty()) lines.push_back(alert.text2);
  if (lines.empty()) return;

  const float band_w = std::max(1.0f, (rect_max.x - rect_min.x) - 2.0f);
  const float line_h = ImGui::GetTextLineHeight();
  const float band_h = line_h * static_cast<float>(lines.size());
  const ImVec2 band_min(rect_min.x + 1.0f, rect_min.y + 1.0f);

  draw_list->AddRectFilled(band_min, ImVec2(band_min.x + band_w, band_min.y + band_h), timeline_color(alert.type, 0.5f));

  const ImU32 text_color = IM_COL32(255, 255, 255, 255);  // palette().color(QPalette::BrightText)
  for (size_t i = 0; i < lines.size(); ++i) {
    const ImVec2 sz = ImGui::CalcTextSize(lines[i].c_str());
    const float x = band_min.x + std::max(0.0f, (band_w - sz.x) * 0.5f);
    draw_list->AddText(ImVec2(x, band_min.y + static_cast<float>(i) * line_h), text_color, lines[i].c_str());
  }
}

// QEasingCurve::InOutQuad, used for the "PAUSED" text fade below.
float ease_in_out_quad(float t) {
  return t < 0.5f ? 2.0f * t * t : 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
}

// mirrors StreamCameraView's fade_animation: opacity 0.2 -> 0.7 over 500ms,
// restarting every time playback transitions into paused (fade_animation's
// value holds at 0.7 for as long as paused stays true afterwards).
float paused_fade_alpha(bool paused) {
  static bool was_paused = false;
  static std::chrono::steady_clock::time_point pause_start;
  if (paused && !was_paused) pause_start = std::chrono::steady_clock::now();
  was_paused = paused;
  if (!paused) return 0.0f;
  const float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - pause_start).count();
  const float t = std::clamp(elapsed / 0.5f, 0.0f, 1.0f);
  return 0.2f + (0.7f - 0.2f) * ease_in_out_quad(t);
}

// mirrors StreamCameraView::paintGL(): thumbnail/scrub popup, then alert
// band, then the paused-fade "PAUSED" text on top of everything -- same
// z-order as the Qt reference.
void draw_video_overlays(AppState &app, ImDrawList *draw_list, ImVec2 frame_min, ImVec2 frame_max,
                          double thumbnail_display_time, ThumbnailStore &thumbs) {
  ReplayStream *rs = dynamic_cast<ReplayStream *>(app.stream.get());
  const bool paused = app.stream->isPaused();
  const bool scrubbing = paused && thumbnail_display_time >= 0.0;

  if (thumbnail_display_time >= 0.0 && rs != nullptr) {
    const uint64_t mono = app.stream->toMonoTime(thumbnail_display_time);
    if (const ThumbInfo *thumb = thumbs.find_at_or_after(mono); thumb != nullptr && thumb->texture != 0) {
      if (scrubbing) {
        draw_scrub_thumbnail(draw_list, frame_min, frame_max, *thumb, thumbnail_display_time);
      } else {
        const auto range = app.stream->timeRange().value_or(std::make_pair(app.stream->minSeconds(), app.stream->maxSeconds()));
        draw_small_thumbnail(draw_list, frame_min, frame_max, *thumb, thumbnail_display_time, range.first, range.second);
      }
    }
  }

  if (rs != nullptr) {
    const double alert_time = scrubbing ? thumbnail_display_time : app.stream->currentSec();
    if (std::optional<Timeline::Entry> alert = rs->getReplay()->findAlertAtTime(alert_time)) {
      draw_alert_band(draw_list, frame_min, frame_max, *alert);
    }
  }

  const float alpha = paused_fade_alpha(paused);
  if (alpha > 0.0f) {
    push_bold_font(16.0f);
    const char *text = "PAUSED";
    const ImVec2 sz = ImGui::CalcTextSize(text);
    const ImVec2 pos((frame_min.x + frame_max.x - sz.x) * 0.5f, (frame_min.y + frame_max.y - sz.y) * 0.5f);
    draw_list->AddText(pos, IM_COL32(200, 200, 200, static_cast<int>(255.0f * alpha)), text);
    pop_bold_font();
  }
}

constexpr float kTimelineHeight = 20.0f;

// mirrors Slider::paintEvent() + Slider::mousePressEvent() +
// VideoWidget::eventFilter()'s MouseMove/Leave -> showThumbnail() wiring:
// the colored timeline scrubber below the camera view, with click/drag-to-
// seek and a hover marker. Returns the hovered/dragged time for this frame's
// thumbnail popup (-1 when neither hovering nor dragging, i.e. Qt's Leave
// case), consumed by draw_video_overlays.
double draw_timeline_slider(AppState &app, float width) {
  static bool dragging = false;
  static double drag_value = 0.0;

  AbstractStream &stream = *app.stream;
  ReplayStream *rs = dynamic_cast<ReplayStream *>(app.stream.get());
  const auto range = stream.timeRange().value_or(std::make_pair(stream.minSeconds(), stream.maxSeconds()));
  const double min_sec = range.first;
  const double max_sec = std::max(min_sec + 0.001, range.second);

  const ImVec2 band_min = ImGui::GetCursorScreenPos();
  const ImVec2 size(width, kTimelineHeight);
  const ImVec2 band_max(band_min.x + size.x, band_min.y + size.y);
  ImGui::InvisibleButton("##timeline_slider", size);
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();

  const auto time_to_x = [&](double t) {
    const double frac = (max_sec > min_sec) ? (std::clamp(t, min_sec, max_sec) - min_sec) / (max_sec - min_sec) : 0.0;
    return band_min.x + static_cast<float>(frac) * size.x;
  };
  const auto x_to_time = [&](float x) {
    const float clamped = std::clamp(x, band_min.x, band_max.x);
    const float frac = size.x > 0.0f ? (clamped - band_min.x) / size.x : 0.0f;
    return min_sec + static_cast<double>(frac) * (max_sec - min_sec);
  };

  draw_list->AddRectFilled(band_min, band_max, timeline_color(TimelineType::None));

  if (rs != nullptr) {
    Replay *replay = rs->getReplay();
    for (const Timeline::Entry &e : *replay->getTimeline()) {
      if (e.end_time < min_sec || e.start_time > max_sec) continue;
      const float x0 = time_to_x(e.start_time);
      const float x1 = std::max(time_to_x(e.end_time), x0 + 1.0f);
      draw_list->AddRectFilled(ImVec2(x0, band_min.y), ImVec2(x1, band_max.y), timeline_color(e.type));
    }

    // Segments not yet loaded into memory, dimmed on top -- mirrors
    // Slider::paintEvent()'s "empty_color" overlay (palette().color(
    // QPalette::Window), alpha 160).
    const ImVec4 window_bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
    const ImU32 empty_color = ImGui::ColorConvertFloat4ToU32(ImVec4(window_bg.x, window_bg.y, window_bg.z, 160.0f / 255.0f));
    const auto event_data = replay->getEventData();
    for (const auto &[n, seg] : replay->route().segments()) {
      (void)seg;
      if (event_data->isSegmentLoaded(n)) continue;
      const double seg_start = n * 60.0, seg_end = (n + 1) * 60.0;
      if (seg_end < min_sec || seg_start > max_sec) continue;
      const float x0 = time_to_x(seg_start);
      const float x1 = std::max(time_to_x(seg_end), x0 + 1.0f);
      draw_list->AddRectFilled(ImVec2(x0, band_min.y), ImVec2(x1, band_max.y), empty_color);
    }
  }

  if (ImGui::IsItemActivated()) dragging = true;
  if (dragging && active) drag_value = x_to_time(ImGui::GetIO().MousePos.x);
  if (dragging && ImGui::IsItemDeactivated()) {
    dragging = false;
    app.stream->seekTo(drag_value);
  }

  const double display_pos = dragging ? drag_value : stream.currentSec();
  const float cur_x = time_to_x(display_pos);
  draw_list->AddRectFilled(ImVec2(cur_x - 1.5f, band_min.y - 2.0f), ImVec2(cur_x + 1.5f, band_max.y + 2.0f),
                            ImGui::GetColorU32(ImGuiCol_SliderGrabActive));

  double thumb_time = -1.0;
  if (hovered || active) {
    thumb_time = x_to_time(ImGui::GetIO().MousePos.x);
    const float hx = time_to_x(thumb_time);
    draw_list->AddRectFilled(ImVec2(hx - 1.0f, band_min.y), ImVec2(hx + 1.0f, band_max.y),
                              ImGui::GetColorU32(ImGuiCol_HeaderActive));
  }

  return thumb_time;
}

// Owns the background VisionIpcClient thread + the GL texture it feeds.
// Frame handoff mirrors CameraWidget's design (cameraview.cc): the vipc
// thread stores a raw VisionBuf* + meta under a mutex; the UI thread converts
// NV12 -> RGBA (libyuv, jotpluggler's proven CameraFeedView pattern) and
// uploads to GL once per new frame id, entirely off the vipc thread.
//
// Lifetime note: this is a function-local static in draw_video_panel(), so
// (unlike AppState, which is destroyed inside run() while the GL context is
// still current) its destructor runs at process atexit time, *after*
// GlfwRuntime has already torn down the GLFW/GL context -- app.cc/app.h/main.cc
// are off-limits for this workstream so there's no hook to tear this down
// earlier. See the destructor below for how that's handled.
class CameraViewState {
 public:
  CameraViewState() = default;
  // Deliberately does *not* call glDeleteTextures() here: this is a
  // function-local static, so its destructor runs at process atexit time --
  // strictly after GlfwRuntime::~GlfwRuntime() has already called
  // glfwDestroyWindow()/glfwTerminate() (run()'s locals, incl. GlfwRuntime,
  // unwind on normal return from run(), which happens before atexit-driven
  // static destructors ever start; app.cc/app.h are off-limits for this
  // workstream so there's no earlier hook to delete the texture while the
  // context is still current, the way jotpluggler's run() explicitly resets
  // its CameraFeedViews before falling out of scope). Calling any GL/GLFW
  // function against a torn-down context here would be the actual bug; one
  // leaked texture handle at process exit is reclaimed by the OS/driver like
  // every other GPU resource the process held, so it's not a real leak.
  // Only the vipc thread (which owns no GL state) needs an explicit,
  // guaranteed-safe shutdown -- see stop().
  ~CameraViewState() { stop(); }
  CameraViewState(const CameraViewState &) = delete;
  CameraViewState &operator=(const CameraViewState &) = delete;

  void ensure_running() {
    if (thread_.joinable()) return;
    stop_requested_.store(false);
    thread_ = std::thread([this] { vipc_thread(); });
  }

  // Bounded by the vipc thread's 100ms recv/retry timeout (matches
  // CameraWidget::stopVipcThread()'s QThread::wait() after
  // requestInterruption() -- no hang, ~100ms worst case).
  void stop() {
    if (!thread_.joinable()) return;
    stop_requested_.store(true);
    thread_.join();
    clear_frame();
    // Blank the display without touching GL (the destructor path has no GL
    // context): the draw gate requires texture_width_/height_ > 0.
    texture_width_ = 0;
    texture_height_ = 0;
    last_uploaded_frame_id_ = std::numeric_limits<uint32_t>::max();
  }

  // NV12 -> RGBA convert + GL upload for the latest frame, if any and if it's
  // new. Only call while the Video panel is actually visible (perf: no
  // conversion work for a hidden/collapsed dock tab).
  void sync_texture() {
    uint32_t frame_id = 0;
    int w = 0, h = 0, stride = 0;
    bool need_upload = false;
    {
      std::lock_guard<std::mutex> lk(frame_mutex_);
      if (current_frame_ == nullptr) return;
      frame_id = frame_meta_.frame_id;
      if (frame_id == last_uploaded_frame_id_) return;  // unchanged: skip conversion entirely
      w = stream_width_;
      h = stream_height_;
      stride = stream_stride_;
      if (w <= 0 || h <= 0 || stride <= 0) return;
      rgba_scratch_.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4U);
      // Proven NV12->ABGR upload pattern from tools/jotpluggler/runtime.cc
      // CameraFeedView (ABGR-named output == RGBA byte order in memory,
      // matching the GL_RGBA texture format below).
      libyuv::NV12ToABGR(current_frame_->y, stride, current_frame_->uv, stride,
                          rgba_scratch_.data(), w * 4, w, h);
      need_upload = true;
    }
    if (!need_upload) return;
    last_uploaded_frame_id_ = frame_id;
    upload_gl(w, h);
  }

  // QTabBar::autoHide equivalent: hidden entirely below 2 streams.
  void draw_tabs() {
    std::set<VisionStreamType> streams;
    {
      std::lock_guard<std::mutex> lk(frame_mutex_);
      streams = available_streams_;
    }
    if (streams.size() < 2) return;

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, ImGui::GetStyle().ItemSpacing.y));
    bool first = true;
    for (VisionStreamType type : streams) {
      if (!first) ImGui::SameLine();
      first = false;
      const char *label = (type >= VISION_STREAM_ROAD && type <= VISION_STREAM_WIDE_ROAD) ? kStreamNames[type] : "Camera";
      ImGui::PushID(static_cast<int>(type));
      if (draw_camera_tab_button(label, type == requested_type_.load())) {
        requested_type_.store(type);
      }
      ImGui::PopID();
    }
    ImGui::PopStyleVar();
    ImGui::Spacing();
  }

  // Aspect-fit centered in `avail`, black letterboxing -- mirrors
  // CameraWidget::paintGL()'s scale_x/scale_y quad transform, expressed as a
  // fit rect instead of a shader uniform since we composite via the ImGui
  // draw list rather than a raw GL viewport. Clicking the frame toggles
  // pause, mirroring CameraWidget::clicked() -> VideoWidget's
  // can->pause(!isPaused()) wiring. Returns the whole-`avail` frame rect
  // (not just the letterboxed image) in screen coordinates so the caller can
  // layer the thumbnail/alert/paused overlays (draw_video_overlays) on top
  // using the same geometry as StreamCameraView::paintGL().
  std::pair<ImVec2, ImVec2> draw_frame(AppState &app, ImVec2 avail) {
    avail.x = std::max(1.0f, avail.x);
    avail.y = std::max(1.0f, avail.y);
    const ImVec2 top_left = ImGui::GetCursorScreenPos();
    const ImVec2 bottom_right(top_left.x + avail.x, top_left.y + avail.y);

    ImGui::InvisibleButton("##camera_frame", avail);
    if (ImGui::IsItemClicked()) {
      app.stream->pause(!app.stream->isPaused());
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(top_left, bottom_right, IM_COL32(0, 0, 0, 255));

    if (texture_ != 0 && texture_width_ > 0 && texture_height_ > 0) {
      const float frame_ratio = static_cast<float>(texture_width_) / static_cast<float>(texture_height_);
      const float widget_ratio = avail.x / avail.y;
      ImVec2 size = avail;
      if (widget_ratio > frame_ratio) {
        size.x = avail.y * frame_ratio;
      } else {
        size.y = avail.x / frame_ratio;
      }
      const ImVec2 origin(top_left.x + (avail.x - size.x) * 0.5f, top_left.y + (avail.y - size.y) * 0.5f);
      draw_list->AddImage(static_cast<ImTextureID>(texture_), origin, ImVec2(origin.x + size.x, origin.y + size.y));
    }

    return {top_left, bottom_right};
  }

 private:
  void clear_frame() {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    current_frame_ = nullptr;
    available_streams_.clear();
  }

  void upload_gl(int w, int h) {
    const bool new_size = (texture_ == 0) || texture_width_ != w || texture_height_ != h;
    if (texture_ == 0) glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (new_size) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_scratch_.data());
      texture_width_ = w;
      texture_height_ = h;
    } else {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_scratch_.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);  // restore default so ImGui's own texture uploads aren't affected
  }

  // Mirrors CameraWidget::vipcThread(): reconnect loop + conflate=false, incl.
  // reconnecting when the requested stream type changes (camera tab click).
  void vipc_thread() {
    VisionStreamType cur_stream = requested_type_.load();
    std::unique_ptr<VisionIpcClient> vipc_client;

    while (!stop_requested_.load()) {
      if (!vipc_client || cur_stream != requested_type_.load()) {
        clear_frame();
        cur_stream = requested_type_.load();
        vipc_client = std::make_unique<VisionIpcClient>("camerad", cur_stream, /*conflate=*/false);
      }

      if (!vipc_client->connected) {
        clear_frame();
        std::set<VisionStreamType> streams = VisionIpcClient::getAvailableStreams("camerad", /*blocking=*/false);
        if (streams.empty()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        {
          std::lock_guard<std::mutex> lk(frame_mutex_);
          available_streams_ = streams;
        }

        if (!vipc_client->connect(/*blocking=*/false)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        std::lock_guard<std::mutex> lk(frame_mutex_);
        stream_width_ = static_cast<int>(vipc_client->buffers[0].width);
        stream_height_ = static_cast<int>(vipc_client->buffers[0].height);
        stream_stride_ = static_cast<int>(vipc_client->buffers[0].stride);
      }

      VisionIpcBufExtra meta{};
      if (VisionBuf *buf = vipc_client->recv(&meta, 100)) {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        current_frame_ = buf;
        frame_meta_ = meta;
      }
    }

    // Drop the pointer into vipc_client's buffers before it (and they) are
    // destroyed below, so a UI-thread sync_texture() racing the very end of
    // shutdown never dereferences a freed buffer.
    clear_frame();
  }

  std::thread thread_;
  std::atomic<bool> stop_requested_{false};
  std::atomic<VisionStreamType> requested_type_{VISION_STREAM_ROAD};

  // Shared with vipc_thread(); guarded by frame_mutex_.
  std::mutex frame_mutex_;
  VisionBuf *current_frame_ = nullptr;
  VisionIpcBufExtra frame_meta_{};
  int stream_width_ = 0;
  int stream_height_ = 0;
  int stream_stride_ = 0;
  std::set<VisionStreamType> available_streams_;

  // UI-thread only.
  uint32_t last_uploaded_frame_id_ = std::numeric_limits<uint32_t>::max();
  std::vector<uint8_t> rgba_scratch_;
  GLuint texture_ = 0;
  int texture_width_ = 0;
  int texture_height_ = 0;
};

}  // namespace

void draw_video_panel(AppState &app) {
  static CameraViewState state;
  static ThumbnailStore thumb_store;

  // Mirrors VideoWidget only building a camera widget when
  // !can->liveStreaming() -- i.e. a route replay. DummyStream (no stream) and
  // future live sources (panda/socketcan/msgq/zmq -- not wired up yet, see
  // main.cc) default liveStreaming()=true and show no camera pane in Qt either.
  // The colored timeline slider is part of that same camera widget in Qt
  // (VideoWidget::createCameraWidget() builds cam_widget *and* slider
  // together), so it's gated on the same condition below.
  const bool want_camera = has_stream(app) && !app.stream->liveStreaming();
  if (want_camera) {
    state.ensure_running();
    thumb_store.ensure_connected(app);
  } else {
    state.stop();
  }

  if (ImGui::Begin(VIDEO_WINDOW_TITLE)) {
    // No stream / DummyStream / no frames yet all fall out of the same empty
    // state here: available_streams_ empty (no tabs) and texture_ == 0 (plain
    // black fill in draw_frame), matching CameraWidget::paintGL's disconnected
    // look without a separate code path.
    state.sync_texture();
    state.draw_tabs();

    const float timeline_h = want_camera ? (kTimelineHeight + ImGui::GetStyle().ItemSpacing.y) : 0.0f;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y = std::max(1.0f, avail.y - timeline_h);
    const auto [frame_min, frame_max] = state.draw_frame(app, avail);

    double thumbnail_display_time = -1.0;
    if (want_camera) {
      thumbnail_display_time = draw_timeline_slider(app, ImGui::GetContentRegionAvail().x);
    }

    draw_video_overlays(app, ImGui::GetWindowDrawList(), frame_min, frame_max, thumbnail_display_time, thumb_store);
  }
  ImGui::End();
}
