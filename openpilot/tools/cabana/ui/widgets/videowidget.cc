#include "tools/cabana/ui/widgets/videowidget.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
}
#include <capnp/serialize.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

const int MIN_VIDEO_HEIGHT = 100;
const int THUMBNAIL_MARGIN = 3;
const float POINT_10_FONT_SIZE = 13.0f;  // 10 pt at 96 dpi
const float POINT_16_FONT_SIZE = 21.0f;  // 16 pt at 96 dpi
const float TOOLBAR_MARGIN_Y = 6.0f;  // between the slider and the buttons, which are as tall as the ones in the charts toolbar
const float TOOLBAR_SEPARATOR_EXTENT = 6.0f;
const float SLIDER_HEIGHT = 15.0f;     // the handle plus a 1 px margin

// Indexed by TimelineType: None, Engaged, AlertInfo, AlertWarning, AlertCritical, UserBookmark
static const ImU32 timeline_colors[] = {
  IM_COL32(111, 143, 175, 255),
  IM_COL32(0, 163, 108, 255),
  IM_COL32(0, 255, 0, 255),
  IM_COL32(255, 195, 0, 255),
  IM_COL32(199, 0, 57, 255),
  IM_COL32(255, 0, 255, 255),
};

static const float speeds[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1., 2., 3., 5.};
static const int NORMAL_SPEED_INDEX = std::find(std::begin(speeds), std::end(speeds), 1.0f) - std::begin(speeds);

static Replay *getReplay() {
  auto stream = dynamic_cast<ReplayStream *>(can);
  return stream ? stream->getReplay() : nullptr;
}

static std::string colorName(ImU32 c) {
  char buf[16];
  snprintf(buf, sizeof(buf), "#%02x%02x%02x", (c >> IM_COL32_R_SHIFT) & 0xff, (c >> IM_COL32_G_SHIFT) & 0xff, (c >> IM_COL32_B_SHIFT) & 0xff);
  return buf;
}

// the zoomed range, or the whole route
static std::pair<double, double> displayedTimeRange() {
  return can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
}

// decode with libavcodec, already linked for the replay video decoder
static bool decodeJpeg(const uint8_t *data, size_t size, RgbImage *out) {
  const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  AVCodecContext *context = codec ? avcodec_alloc_context3(codec) : nullptr;
  AVFrame *frame = av_frame_alloc();
  AVPacket *packet = av_packet_alloc();
  bool ok = false;
  if (context && frame && packet && size > 0 && size <= (size_t)INT32_MAX && av_new_packet(packet, (int)size) >= 0) {
    std::copy(data, data + size, packet->data);
    ok = avcodec_open2(context, codec, nullptr) >= 0 && avcodec_send_packet(context, packet) >= 0 &&
         avcodec_receive_frame(context, frame) >= 0 && frame->width > 0 && frame->height > 0;
  }
  int chroma_x_shift = 0, chroma_y_shift = 0;
  if (ok) {
    switch ((AVPixelFormat)frame->format) {
      case AV_PIX_FMT_YUV420P: case AV_PIX_FMT_YUVJ420P: chroma_x_shift = chroma_y_shift = 1; break;
      case AV_PIX_FMT_YUV422P: case AV_PIX_FMT_YUVJ422P: chroma_x_shift = 1; break;
      case AV_PIX_FMT_YUV444P: case AV_PIX_FMT_YUVJ444P: break;
      default: ok = false; break;
    }
  }
  if (ok) {
    out->resize(frame->width, frame->height);
    const bool full_range = frame->color_range == AVCOL_RANGE_JPEG || frame->format == AV_PIX_FMT_YUVJ420P ||
                            frame->format == AV_PIX_FMT_YUVJ422P || frame->format == AV_PIX_FMT_YUVJ444P;
    const float y_scale = full_range ? 1.0f : 1.164383f;
    const float y_offset = full_range ? 0.0f : 16.0f;
    const float kr = full_range ? 1.402f : 1.596027f;
    const float kgu = full_range ? 0.344136f : 0.391762f;
    const float kgv = full_range ? 0.714136f : 0.812968f;
    const float kb = full_range ? 1.772f : 2.017232f;
    for (int y = 0; y < frame->height; ++y) {
      const uint8_t *y_row = frame->data[0] + y * frame->linesize[0];
      const uint8_t *u_row = frame->data[1] + (y >> chroma_y_shift) * frame->linesize[1];
      const uint8_t *v_row = frame->data[2] + (y >> chroma_y_shift) * frame->linesize[2];
      uint8_t *dst = out->data.data() + (size_t)y * out->bytesPerLine();
      for (int x = 0; x < frame->width; ++x) {
        const float luma = y_scale * ((float)y_row[x] - y_offset);
        const float u = (float)u_row[x >> chroma_x_shift] - 128.0f;
        const float v = (float)v_row[x >> chroma_x_shift] - 128.0f;
        const float r = luma + kr * v;
        const float g = luma - kgu * u - kgv * v;
        const float b = luma + kb * u;
        dst[x * 4 + 0] = (uint8_t)std::clamp(std::lround(r), 0L, 255L);
        dst[x * 4 + 1] = (uint8_t)std::clamp(std::lround(g), 0L, 255L);
        dst[x * 4 + 2] = (uint8_t)std::clamp(std::lround(b), 0L, 255L);
        dst[x * 4 + 3] = 255;
      }
    }
  }
  av_packet_free(&packet);
  av_frame_free(&frame);
  avcodec_free_context(&context);
  return ok;
}

VideoWidget::VideoWidget() {
  if (!can->liveStreaming())
    createCameraWidget();

  createSpeedDropdown();

  connections_.push_back(can->timeRangeChanged.connect([this](const auto &) { timeRangeChanged(); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { msgs_received_ = true; }));
}

std::string VideoWidget::whatsThis() const {
  // one <br /> separated line per legend row, with the same entries and colors
  return "<b>Video</b><br />\n"
         "<span style=\"color:gray\">Timeline color</span><br />\n" +
         colorName(timeline_colors[(int)TimelineType::None]) + " Disengaged&nbsp;&nbsp;&nbsp;" +
         colorName(timeline_colors[(int)TimelineType::Engaged]) + " Engaged<br />\n" +
         colorName(timeline_colors[(int)TimelineType::UserBookmark]) + " User Flag&nbsp;&nbsp;&nbsp;" +
         colorName(timeline_colors[(int)TimelineType::AlertInfo]) + " Info<br />\n" +
         colorName(timeline_colors[(int)TimelineType::AlertWarning]) + " Warning&nbsp;&nbsp;&nbsp;" +
         colorName(timeline_colors[(int)TimelineType::AlertCritical]) + " Critical<br />\n"
         "<span style=\"color:gray\">Shortcuts</span><br />\n"
         "Pause/Resume: <span style=\"background-color:lightGray;color:gray\">&nbsp;space&nbsp;</span>";
}

static float toolbarHeight() { return TOOLBAR_MARGIN_Y + ImGui::GetFrameHeight(); }

void VideoWidget::drawPlaybackController() {
  beginToolbar();
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + TOOLBAR_MARGIN_Y);
  const float speed_width = menuButtonWidth("0.05x  ", true);

  const char *play_icon = can->isPaused() ? icon::PLAY : icon::PAUSE;
  const char *play_tooltip = can->isPaused() ? "Play" : "Pause";
  const char *loop_icon = getReplay() && getReplay()->loop() ? icon::REPEAT : icon::REPEAT_1;
  const std::string time_text = slider_ ? formatTime(can->currentSec(), true) + " / " + formatTime(slider_->maximum() / slider_->factor)
                                        : formatTime(can->currentSec(), true);
  const char *time_tooltip = settings.absolute_time ? "Elapsed time" : "Absolute time";

  auto seek_backward = []() { can->seekTo(can->currentSec() - 1); };
  auto toggle_play = []() { can->pause(!can->isPaused()); };
  auto seek_forward = []() { can->seekTo(can->currentSec() + 1); };

  std::vector<ToolbarItem> items = {
    {toolbarButtonWidth(icon::REWIND), [&]() { if (toolButton("rewind", icon::REWIND, "Seek backward")) seek_backward(); },
     "Seek backward", seek_backward},
    {toolbarButtonWidth(play_icon), [&]() { if (toolButton("play", play_icon, play_tooltip)) toggle_play(); },
     play_tooltip, toggle_play},
    {toolbarButtonWidth(icon::FAST_FORWARD), [&]() { if (toolButton("fast-forward", icon::FAST_FORWARD, "Seek forward")) seek_forward(); },
     "Seek forward", seek_forward},
  };
  if (can->liveStreaming()) {
    items.push_back({toolbarButtonWidth(icon::SKIP_END), [&]() {
      ImGui::BeginDisabled(!skip_to_end_enabled_);
      if (toolButton("skip-end", icon::SKIP_END, "Skip to the end")) skipToEnd();
      ImGui::EndDisabled();
    }, "Skip to the end", [this]() { skipToEnd(); }, skip_to_end_enabled_});
  }
  if (slider_ || msgs_received_) {
    // a mono font: with proportional digits the time changed width as it ticked and the items after it moved
    pushMonoFont(ImGui::GetFontSize());
    const float time_width = toolbarButtonWidth(time_text);
    popMonoFont();
    items.push_back({time_width,
                     [&]() {
                       pushMonoFont(ImGui::GetFontSize());
                       if (toolButton("time_display", time_text.c_str(), time_tooltip)) toggleTimeDisplay();
                       popMonoFont();
                     },
                     time_text, [this]() { toggleTimeDisplay(); }});
  }
  // the expanding spacer: the items after it are right aligned as long as everything fits
  const size_t spacer_index = items.size();
  if (!can->liveStreaming()) {
    items.push_back({toolbarButtonWidth(loop_icon), [&]() { if (toolButton("loop", loop_icon, "Loop playback")) loopPlaybackClicked(); },
                     "Loop playback", [this]() { loopPlaybackClicked(); }});
  }
  items.push_back({speed_width, [&]() { drawSpeedDropdown(speed_width); }});
  if (!can->liveStreaming()) {
    ToolbarItem separator{TOOLBAR_SEPARATOR_EXTENT, []() {
      // a 1 px separator line centered in TOOLBAR_SEPARATOR_EXTENT, inset from the top and bottom
      const ImVec2 min = ImGui::GetCursorScreenPos();
      ImGui::Dummy(ImVec2(TOOLBAR_SEPARATOR_EXTENT, ImGui::GetFrameHeight()));
      const float x = std::floor(min.x + TOOLBAR_SEPARATOR_EXTENT * 0.5f);
      ImGui::GetWindowDrawList()->AddLine(ImVec2(x, min.y + 4.0f), ImVec2(x, min.y + ImGui::GetFrameHeight() - 4.0f), ImGui::GetColorU32(ImGuiCol_Separator));
    }};
    separator.in_menu = false;
    items.push_back(std::move(separator));
    items.push_back({toolbarButtonWidth(icon::INFO_CIRCLE),
                     [&]() { if (toolButton("route_info", icon::INFO_CIRCLE, "View route details")) showRouteInfo(); },
                     "View route details", [this]() { showRouteInfo(); }});
  }

  drawToolbar(items, spacer_index);
  endToolbar();
}

void VideoWidget::skipToEnd() {
  // set speed to 1.0; this only checks the menu entry, the speed and the button text are unchanged
  speed_index_ = NORMAL_SPEED_INDEX;
  can->pause(false);
  can->seekTo(can->maxSeconds() + 1);
}

void VideoWidget::toggleTimeDisplay() {
  settings.absolute_time = !settings.absolute_time;
}

static std::string speedText(float speed, const char *suffix) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%gx%s", speed, suffix);
  return buf;
}

void VideoWidget::createSpeedDropdown() {
  speed_index_ = NORMAL_SPEED_INDEX;
  can->setSpeed(speeds[speed_index_]);
  speed_text_ = speedText(speeds[speed_index_], "  ");
}

void VideoWidget::drawSpeedDropdown(float width) {
  menuButton("speed_btn", speed_text_, "speed_menu", true, width);
  if (ImGui::BeginPopup("speed_menu")) {
    drawSpeedMenuItems();
    ImGui::EndPopup();
  }
}

void VideoWidget::drawSpeedMenuItems() {
  // every row declares the same width, so the popup is exactly as wide as the widest one and all the
  // highlights reach both edges; the label is padded on the right as much as the check column on the left
  const float indent = ImGui::GetFontSize();
  float label_width = 0;
  for (int i = 0; i < (int)std::size(speeds); ++i) {
    label_width = std::max(label_width, ImGui::CalcTextSize(speedText(speeds[i], "").c_str()).x);
  }
  for (int i = 0; i < (int)std::size(speeds); ++i) {
    const float speed = speeds[i];
    if (radioMenuItem(speedText(speed, "").c_str(), speed_index_ == i, indent + label_width + indent)) {
      speed_index_ = i;
      can->setSpeed(speed);
      speed_text_ = speedText(speed, "  ");
    }
  }
}

void VideoWidget::createCameraWidget() {
  camera_tab_ = std::make_unique<TabBar>();
  camera_tab_->setAutoHide(true);

  cam_widget_ = std::make_unique<StreamCameraView>("camerad", VISION_STREAM_NARROW_ROAD);

  slider_ = std::make_unique<Slider>();
  slider_->setTimeRange(can->minSeconds(), can->maxSeconds());

  connections_.push_back(slider_->sliderReleased.connect([this]() { can->seekTo(slider_->currentSecond()); }));
  connections_.push_back(cam_widget_->clicked.connect([]() { can->pause(!can->isPaused()); }));
  connections_.push_back(cam_widget_->availableStreamsUpdated.connect([this](std::set<VisionStreamType> streams) { vipcAvailableStreamsUpdated(streams); }));
  connections_.push_back(camera_tab_->currentChanged.connect([this](int index) {
    if (index != -1) cam_widget_->setStreamType((VisionStreamType)camera_tab_->tabData(index));
  }));
  connections_.push_back(static_cast<ReplayStream *>(can)->qLogLoaded.connect([this](std::shared_ptr<LogReader> qlog) { cam_widget_->parseQLog(qlog); }));
}

void VideoWidget::drawCameraWidget() {
  camera_tab_->draw();

  // cam_widget_: minimum height MIN_VIDEO_HEIGHT, takes the space left by the slider and the toolbar
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float cam_height = std::max((float)MIN_VIDEO_HEIGHT, avail.y - SLIDER_HEIGHT - toolbarHeight());
  cam_widget_->draw(ImVec2(avail.x, cam_height), thumbnail_display_time_);

  if (!slider_->isSliderDown()) slider_->setCurrentSecond(can->currentSec());
  slider_->draw(thumbnail_display_time_);
  updateSliderThumbnail();
}

void VideoWidget::vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams) {
  static const std::string stream_names[] = {"Road camera", "Driver camera", "Wide road camera"};
  for (int i = 0; i < streams.size(); ++i) {
    if (camera_tab_->count() <= i) {
      camera_tab_->addTab(std::string());
    }
    int type = *std::next(streams.begin(), i);
    camera_tab_->setTabText(i, stream_names[type]);
    camera_tab_->setTabData(i, type);
  }
  while (camera_tab_->count() > streams.size()) {
    camera_tab_->removeTab(camera_tab_->count() - 1);
  }
}

void VideoWidget::loopPlaybackClicked() {
  getReplay()->setLoop(!getReplay()->loop());
}

void VideoWidget::timeRangeChanged() {
  const auto time_range = can->timeRange();
  if (can->liveStreaming()) {
    skip_to_end_enabled_ = !time_range.has_value();
    return;
  }
  time_range ? slider_->setTimeRange(time_range->first, time_range->second)
             : slider_->setTimeRange(can->minSeconds(), can->maxSeconds());
}

std::string VideoWidget::formatTime(double sec, bool include_milliseconds) {
  if (settings.absolute_time)
    sec += std::chrono::duration<double>(can->beginDateTime().time_since_epoch()).count();
  return utils::formatSeconds(sec, include_milliseconds, settings.absolute_time);
}

void VideoWidget::setVisible(bool visible) {
  if (cam_widget_) cam_widget_->setVisible(visible);
}

void VideoWidget::showThumbnail(double seconds) {
  if (can->liveStreaming()) return;
  thumbnail_display_time_ = seconds;
}

void VideoWidget::showRouteInfo() {
  // dropped from route_info_dlgs_ once draw() returns false
  route_info_dlgs_.push_back(std::make_unique<RouteInfoDlg>());
}

void VideoWidget::updateSliderThumbnail() {
  if (slider_->underMouse()) {
    auto [min_sec, max_sec] = displayedTimeRange();
    showThumbnail(min_sec + (ImGui::GetMousePos().x - slider_->rect().Min.x) * (max_sec - min_sec) / slider_->width());
  } else if (slider_->mouseLeft()) {
    showThumbnail(-1);
  }
}

float VideoWidget::sizeHintHeight() const {
  // the camera minimum height plus the slider and the toolbar
  return MIN_VIDEO_HEIGHT + SLIDER_HEIGHT + toolbarHeight();
}

// the video pane opens with the camera at its natural aspect ratio, filling the width of the dock
float VideoWidget::defaultHeight(float width) const {
  if (!cam_widget_) return toolbarHeight();  // live streams have no camera or slider
  const float cam_height = std::max((float)MIN_VIDEO_HEIGHT, width / cam_widget_->frameAspectRatio());
  const float tab_height = camera_tab_->count() >= 2 ? ImGui::GetFrameHeight() : 0.0f;
  return cam_height + tab_height + SLIDER_HEIGHT + toolbarHeight();
}

void VideoWidget::draw() {
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0.0f));
  if (!can->liveStreaming())
    drawCameraWidget();

  drawPlaybackController();
  ImGui::PopStyleVar();

  for (auto it = route_info_dlgs_.begin(); it != route_info_dlgs_.end();) {
    it = (*it)->draw() ? it + 1 : route_info_dlgs_.erase(it);
  }
}

void Slider::draw(double thumbnail_time) {
  ImGui::InvisibleButton("##slider", ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), SLIDER_HEIGHT));
  rect_ = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  const bool hovered = ImGui::IsItemHovered();
  left_ = hovered_ && !hovered;
  hovered_ = hovered;

  if (ImGui::IsItemActivated()) handleMousePress();
  if (slider_down_) {
    if (ImGui::IsItemActive()) {
      // the handle keeps its grab offset while dragging
      setValue(pixelPosToRangeValue(ImGui::GetMousePos().x - click_offset_));
    } else {
      slider_down_ = false;
      sliderReleased();
    }
  }
  paint(thumbnail_time);
}

ImRect Slider::handleRect() const {
  const float handle_width = SLIDER_LENGTH;
  const float handle_height = std::min(SLIDER_THICKNESS, rect_.GetHeight());
  const int range = std::max(1, maximum() - minimum());
  const float x = rect_.Min.x + (float)(value() - minimum()) / range * std::max(0.0f, width() - handle_width);
  const float y = rect_.GetCenter().y - handle_height / 2;
  return ImRect(ImVec2(x, y), ImVec2(x + handle_width, y + handle_height));
}

// handle left edge (window x) -> value over the groove minus the handle width
int Slider::pixelPosToRangeValue(float x) const {
  const float handle_width = SLIDER_LENGTH;
  const float span = std::max(1.0f, width() - handle_width);
  return minimum() + (int)std::lround((maximum() - minimum()) * std::clamp((x - rect_.Min.x) / span, 0.0f, 1.0f));
}

void Slider::paint(double thumbnail_time) {
  ImDrawList *p = ImGui::GetWindowDrawList();

  ImRect handle_rect = handleRect();
  ImRect groove_rect = rect_;

  // adjust the groove height to match the handle height, rounded up to whole pixels
  float handle_height = handle_rect.GetHeight();
  const float groove_height = std::ceil(handle_height * 0.5f);
  const float center_y = rect_.GetCenter().y;
  groove_rect.Min.y = std::floor(center_y - groove_height / 2);
  groove_rect.Max.y = groove_rect.Min.y + groove_height;

  p->AddRectFilled(groove_rect.Min, groove_rect.Max, timeline_colors[(int)TimelineType::None]);

  double min = minimum() / factor;
  double max = maximum() / factor;
  const double span = std::max(max - min, 1e-9);

  auto fillRange = [&](double begin, double end, ImU32 color) {
    if (begin > max || end < min) return;

    // the edges truncate to whole pixels and the right edge is inclusive, so even an event shorter than a
    // pixel paints one full pixel in its color instead of an anti-aliased smear
    ImRect r = groove_rect;
    r.Min.x = rect_.Min.x + std::floor(((std::max(min, begin) - min) / span) * width());
    r.Max.x = rect_.Min.x + std::floor(((std::min(max, end) - min) / span) * width()) + 1.0f;
    p->AddRectFilled(r.Min, r.Max, color);
  };

  if (auto replay = getReplay()) {
    for (const auto &entry : *replay->getTimeline()) {
      fillRange(entry.start_time, entry.end_time, timeline_colors[(int)entry.type]);
    }

    ImU32 empty_color = ImGui::GetColorU32(ImGuiCol_WindowBg, 160 / 255.0f);
    const auto event_data = replay->getEventData();
    for (const auto &[n, _] : replay->route().segments()) {
      if (!event_data->isSegmentLoaded(n))
        fillRange(n * 60.0, (n + 1) * 60.0, empty_color);
    }
  }

  drawSliderHandle(p, handle_rect);

  if (thumbnail_time >= 0) {
    float left = rect_.Min.x + (float)((thumbnail_time - min) * width() / span) - 1;
    ImRect rc(ImVec2(left, rect_.Min.y + 1), ImVec2(left + 2, rect_.Max.y - 1));
    p->AddRectFilled(rc.Min, rc.Max, ImGui::GetColorU32(ImGuiCol_Header), 1.5f);  // ImGuiCol_Header is the theme highlight
  }
}

void Slider::handleMousePress() {
  // a press on the handle starts a drag and remembers the grab offset
  const ImRect handle_rect = handleRect();
  if (handle_rect.Contains(ImGui::GetMousePos())) {
    slider_down_ = true;
    click_offset_ = ImGui::GetMousePos().x - handle_rect.Min.x;
    return;
  }
  setValue(minimum() + (int)(((maximum() - minimum()) * (ImGui::GetMousePos().x - rect_.Min.x)) / width()));
  sliderReleased();
}

StreamCameraView::StreamCameraView(std::string stream_name, VisionStreamType stream_type)
    : CameraWidget(stream_name, stream_type) {
  big_thumbnail_texture_.mipmap = true;  // the hover thumbnail is drawn at a quarter of the stored size
}

StreamCameraView::~StreamCameraView() {
  for (auto &pending : pending_thumbnails_) pending.done.wait();
}

void StreamCameraView::parseQLog(std::shared_ptr<LogReader> qlog) {
  auto thumbnails = std::make_shared<std::map<uint64_t, RgbImage>>();
  auto done = ThreadPool::instance().run([qlog, thumbnails]() {
    for (const Event &e : qlog->events) {
      if (e.which != cereal::Event::Which::THUMBNAIL) continue;
      capnp::FlatArrayMessageReader reader(e.data);
      auto thumb_data = reader.getRoot<cereal::Event>().getThumbnail();
      auto image_data = thumb_data.getThumbnail();
      if (RgbImage thumb; decodeJpeg(image_data.begin(), image_data.size(), &thumb)) {
        (*thumbnails)[thumb_data.getTimestampEof()] = std::move(thumb);
      }
    }
  });
  pending_thumbnails_.push_back({std::move(done), std::move(thumbnails)});
}

void StreamCameraView::collectThumbnails() {
  for (auto it = pending_thumbnails_.begin(); it != pending_thumbnails_.end();) {
    if (it->done.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      ++it;
      continue;
    }
    for (auto &[ts, thumb] : *it->thumbnails) big_thumbnails_[ts] = std::move(thumb);
    it = pending_thumbnails_.erase(it);
  }
}

void StreamCameraView::draw(const ImVec2 &size, double thumbnail_time) {
  collectThumbnails();
  CameraWidget::draw(size);

  ImDrawList *p = ImGui::GetWindowDrawList();
  bool scrubbing = false;
  if (thumbnail_time >= 0) {
    scrubbing = can->isPaused();
    scrubbing ? drawScrubThumbnail(p, thumbnail_time) : drawThumbnail(p, thumbnail_time);
  }
  if (auto alert = getReplay()->findAlertAtTime(scrubbing ? thumbnail_time : can->currentSec())) {
    drawAlert(p, rect(), *alert, ImGui::GetFontSize());
  }

  if (can->isPaused()) {
    ImFont *font = boldFont();
    const char *text = "PAUSED";
    const ImVec2 text_size = font->CalcTextSizeA(POINT_16_FONT_SIZE, FLT_MAX, 0.0f, text);
    const ImVec2 center = rect().GetCenter();
    p->AddText(font, POINT_16_FONT_SIZE, ImVec2(center.x - text_size.x / 2, center.y - text_size.y / 2),
               IM_COL32(200, 200, 200, static_cast<int>(255 * 0.7f)), text);
  }
}

const RgbImage *StreamCameraView::thumbnailAt(double sec, uint64_t *mono_time) {
  auto it = big_thumbnails_.lower_bound(can->toMonoTime(sec));
  if (it == big_thumbnails_.end()) return nullptr;
  if (big_thumbnail_texture_.id == 0 || big_thumbnail_texture_.key != it->first) {
    big_thumbnail_texture_.upload(it->second);
    big_thumbnail_texture_.key = it->first;
  }
  if (mono_time) *mono_time = it->first;
  return &it->second;
}

void StreamCameraView::drawScrubThumbnail(ImDrawList *p, double sec) {
  p->AddRectFilled(rect().Min, rect().Max, IM_COL32(0, 0, 0, 255));
  if (const RgbImage *image = thumbnailAt(sec, nullptr)) {
    // scale to the widget size, keeping the aspect ratio
    const float scale = std::min(width() / image->width, height() / image->height);
    const ImVec2 scaled_size(std::floor(image->width * scale), std::floor(image->height * scale));
    const ImVec2 center = rect().GetCenter();
    const ImVec2 thumb_min(center.x - (int)(scaled_size.x / 2), center.y - (int)(scaled_size.y / 2));
    ImRect thumb_rect(thumb_min, ImVec2(thumb_min.x + scaled_size.x, thumb_min.y + scaled_size.y));
    p->AddImage(big_thumbnail_texture_.ref(), thumb_rect.Min, thumb_rect.Max);
    drawTime(p, thumb_rect, sec);
  }
}

void StreamCameraView::drawThumbnail(ImDrawList *p, double sec) {
  uint64_t mono_time = 0;
  if (const RgbImage *image = thumbnailAt(sec, &mono_time)) {
    // AddImage scales the stored image to the thumbnail height, keeping the aspect ratio
    const int h = MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2;
    const int w = std::max(1, (int)std::lround((double)image->width * h / image->height));
    auto [min_sec, max_sec] = displayedTimeRange();
    int pos = (sec - min_sec) * width() / (max_sec - min_sec);
    const int max_x = (int)width() - w - THUMBNAIL_MARGIN + 1;
    int x = std::clamp(pos - w / 2, THUMBNAIL_MARGIN, std::max(THUMBNAIL_MARGIN, max_x));
    int y = height() - h - THUMBNAIL_MARGIN;

    ImRect thumb_rect(ImVec2(rect().Min.x + x, rect().Min.y + y), ImVec2(rect().Min.x + x + w, rect().Min.y + y + h));
    p->AddImage(big_thumbnail_texture_.ref(), thumb_rect.Min, thumb_rect.Max);
    p->AddRect(thumb_rect.Min, thumb_rect.Max, paletteBrightText(), 0.0f, 0, 2.0f);
    if (auto alert = getReplay()->findAlertAtTime(can->toSeconds(mono_time))) {
      drawAlert(p, thumb_rect, *alert, POINT_10_FONT_SIZE);
    }
    drawTime(p, thumb_rect, sec);
  }
}

void StreamCameraView::drawTime(ImDrawList *p, const ImRect &rect, double seconds) {
  char text[32];
  snprintf(text, sizeof(text), "%.3f", seconds);
  ImFont *font = ImGui::GetFont();
  const ImVec2 text_size = font->CalcTextSizeA(POINT_10_FONT_SIZE, FLT_MAX, 0.0f, text);
  // centered horizontally, above the bottom margin
  p->AddText(font, POINT_10_FONT_SIZE, ImVec2(rect.GetCenter().x - text_size.x / 2, rect.Max.y - THUMBNAIL_MARGIN - text_size.y),
             paletteBrightText(), text);
}

void StreamCameraView::drawAlert(ImDrawList *p, const ImRect &rect, const Timeline::Entry &alert, float font_size) {
  const ImU32 pen = paletteBrightText();
  ImU32 color = withAlpha(timeline_colors[int(alert.type)], 128);
  std::string text = alert.text1;
  if (!alert.text2.empty()) text += "\n" + alert.text2;

  ImRect text_rect(ImVec2(rect.Min.x + 1, rect.Min.y + 1), ImVec2(rect.Max.x - 1, rect.Max.y - 1));
  ImFont *font = ImGui::GetFont();
  const float wrap_width = std::max(1.0f, text_rect.GetWidth());
  const ImVec2 r = font->CalcTextSizeA(font_size, FLT_MAX, wrap_width, text.c_str());
  p->AddRectFilled(ImVec2(text_rect.Min.x, text_rect.Min.y), ImVec2(text_rect.Max.x, text_rect.Min.y + r.y), color);
  // each line is centered, wrapped continuations stay left aligned
  float y = text_rect.Min.y;
  for (const auto &line : utils::split(text, '\n')) {
    const ImVec2 line_size = font->CalcTextSizeA(font_size, FLT_MAX, wrap_width, line.c_str());
    p->AddText(font, font_size, ImVec2(text_rect.Min.x + (text_rect.GetWidth() - line_size.x) / 2, y), pen, line.c_str(), nullptr, wrap_width);
    y += line_size.y;
  }
}
