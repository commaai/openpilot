#include "tools/cabana/ui/widgets/videowidget.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <mutex>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
}
#include <capnp/serialize.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

const int MIN_VIDEO_HEIGHT = 100;
const int THUMBNAIL_MARGIN = 3;
const float POINT_10_FONT_SIZE = 13.0f;  // 10 pt at 96 dpi
const float POINT_16_FONT_SIZE = 21.0f;  // 16 pt at 96 dpi
const float MENU_BUTTON_INDICATOR = 12.0f;
const float TOOLBAR_BUTTON_PADDING = 4.0f;  // auto raise button horizontal margin
const float TOOLBAR_MARGIN_Y = 6.0f;  // between the slider and the buttons, which are as tall as the ones in the charts toolbar
const float TOOLBAR_SEPARATOR_EXTENT = 6.0f;
const float MENU_ARROW_SIZE = 6.0f;            // dropdown arrow on a menu button
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

static Replay *getReplay() {
  auto stream = dynamic_cast<ReplayStream *>(can);
  return stream ? stream->getReplay() : nullptr;
}

static std::string colorName(ImU32 c) {
  char buf[16];
  snprintf(buf, sizeof(buf), "#%02x%02x%02x", (c >> IM_COL32_R_SHIFT) & 0xff, (c >> IM_COL32_G_SHIFT) & 0xff, (c >> IM_COL32_B_SHIFT) & 0xff);
  return buf;
}

static ImU32 withAlpha(ImU32 c, int alpha) {
  return (c & ~IM_COL32_A_MASK) | ((ImU32)alpha << IM_COL32_A_SHIFT);
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
  // the toolbar items sit next to each other, the buttons only carry the auto raise margin
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(TOOLBAR_ITEM_SPACING, ImGui::GetStyle().ItemSpacing.y));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(TOOLBAR_BUTTON_PADDING, ImGui::GetStyle().FramePadding.y));
  const ImGuiStyle &style = ImGui::GetStyle();
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + TOOLBAR_MARGIN_Y);
  // a button is its text plus 2 * FramePadding.x
  auto button_width = [&](const char *label) { return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2; };
  pushBoldFont();
  const float speed_width = button_width("0.05x  ") + MENU_BUTTON_INDICATOR;
  popBoldFont();

  const char *play_icon = can->isPaused() ? icon::PLAY : icon::PAUSE;
  const char *play_tooltip = can->isPaused() ? "Play" : "Pause";
  const char *loop_icon = getReplay() && getReplay()->loop() ? icon::REPEAT : icon::REPEAT_1;
  const std::string time_text = slider ? formatTime(can->currentSec(), true) + " / " + formatTime(slider->maximum() / slider->factor)
                                       : formatTime(can->currentSec(), true);
  // Qt sets the tooltip in the click handler, so there is none until the display is toggled once
  const char *time_tooltip = time_tooltip_shown_ ? (settings.absolute_time ? "Elapsed time" : "Absolute time") : nullptr;

  struct Item {
    float width;
    std::function<void()> draw;
    std::string menu_label;  // empty: dropped from the extension menu, which only carries actions
    std::function<void()> trigger;
    bool enabled = true;
  };
  auto seek_backward = []() { can->seekTo(can->currentSec() - 1); };
  auto toggle_play = []() { can->pause(!can->isPaused()); };
  auto seek_forward = []() { can->seekTo(can->currentSec() + 1); };

  std::vector<Item> items = {
    {button_width(icon::REWIND), [&]() { if (toolButton("rewind", icon::REWIND, "Seek backward")) seek_backward(); },
     "Seek backward", seek_backward},
    {button_width(play_icon), [&]() { if (toolButton("play", play_icon, play_tooltip)) toggle_play(); },
     play_tooltip, toggle_play},
    {button_width(icon::FAST_FORWARD), [&]() { if (toolButton("fast-forward", icon::FAST_FORWARD, "Seek forward")) seek_forward(); },
     "Seek forward", seek_forward},
  };
  if (can->liveStreaming()) {
    items.push_back({button_width(icon::SKIP_END), [&]() {
      ImGui::BeginDisabled(!skip_to_end_enabled_);
      if (toolButton("skip-end", icon::SKIP_END, "Skip to the end")) skipToEnd();
      ImGui::EndDisabled();
    }, "Skip to the end", [this]() { skipToEnd(); }, skip_to_end_enabled_});
  }
  // a mono font: with proportional digits the time changed width as it ticked and the items after it moved
  pushMonoFont(ImGui::GetFontSize());
  const float time_width = button_width(time_text.c_str());
  popMonoFont();
  items.push_back({time_width,
                   [&]() {
                     pushMonoFont(ImGui::GetFontSize());
                     if (toolButton("time_display", time_text.c_str(), time_tooltip)) toggleTimeDisplay();
                     popMonoFont();
                   },
                   time_text, [this]() { toggleTimeDisplay(); }});
  // the expanding spacer: the items after it are right aligned as long as everything fits
  const size_t spacer_index = items.size();
  if (!can->liveStreaming()) {
    items.push_back({button_width(loop_icon), [&]() { if (toolButton("loop", loop_icon, "Loop playback")) loopPlaybackClicked(); },
                     "Loop playback", [this]() { loopPlaybackClicked(); }});
  }
  items.push_back({speed_width, [&]() { drawSpeedDropdown(speed_width); }, "", nullptr});
  if (!can->liveStreaming()) {
    items.push_back({TOOLBAR_SEPARATOR_EXTENT, []() {
      // a 1 px separator line centered in TOOLBAR_SEPARATOR_EXTENT, inset from the top and bottom
      const ImVec2 min = ImGui::GetCursorScreenPos();
      ImGui::Dummy(ImVec2(TOOLBAR_SEPARATOR_EXTENT, ImGui::GetFrameHeight()));
      const float x = std::floor(min.x + TOOLBAR_SEPARATOR_EXTENT * 0.5f);
      ImGui::GetWindowDrawList()->AddLine(ImVec2(x, min.y + 4.0f), ImVec2(x, min.y + ImGui::GetFrameHeight() - 4.0f), ImGui::GetColorU32(ImGuiCol_Separator));
    }, "", nullptr});
    items.push_back({button_width(icon::INFO_CIRCLE),
                     [&]() { if (toolButton("route_info", icon::INFO_CIRCLE, "View route details")) showRouteInfo(); },
                     "View route details", [this]() { showRouteInfo(); }});
  }

  auto group_width = [&](size_t begin, size_t end) {
    float w = 0;
    for (size_t i = begin; i < end; ++i) w += items[i].width + (i > begin ? style.ItemSpacing.x : 0);
    return w;
  };
  const float left_width = group_width(0, spacer_index);
  const float right_width = group_width(spacer_index, items.size());

  const float start_x = ImGui::GetCursorPosX();
  const float avail = ImGui::GetContentRegionAvail().x;
  const float right_edge = start_x + avail;
  const float extension_width = button_width(icon::RAQUO);

  // when everything fits the spacer takes the slack, otherwise the extension button is reserved at the
  // right edge and the items are packed from the left until the next one does not fit
  const bool fits = left_width + style.ItemSpacing.x + right_width <= avail;
  size_t visible = items.size();
  if (!fits) {
    const float usable = avail - (extension_width + style.ItemSpacing.x);
    float used = 0;
    for (visible = 0; visible < items.size(); ++visible) {
      const float w = items[visible].width + (visible ? style.ItemSpacing.x : 0);
      if (used + w > usable) break;
      used += w;
    }
  }

  for (size_t i = 0; i < visible; ++i) {
    if (i == 0) ImGui::SetCursorPosX(start_x);
    else if (fits && i == spacer_index) ImGui::SameLine(right_edge - right_width);
    else ImGui::SameLine();
    items[i].draw();
  }

  if (visible < items.size()) {
    // the extension button sits fully inside the toolbar: its right edge is the content region right edge
    const float extension_x = std::max(start_x, right_edge - extension_width);
    visible == 0 ? ImGui::SetCursorPosX(extension_x) : ImGui::SameLine(extension_x);
    if (ImGui::Button((std::string(icon::RAQUO) + "###toolbar_extension").c_str(), ImVec2(extension_width, 0)))
      ImGui::OpenPopup("toolbar_extension_menu");
    // the popup opens inward: its right edge is aligned with the button so it stays inside the window
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y), ImGuiCond_Always, ImVec2(1, 0));
    if (ImGui::BeginPopup("toolbar_extension_menu")) {
      for (size_t i = visible; i < items.size(); ++i) {
        if (!items[i].menu_label.empty() && ImGui::MenuItem(items[i].menu_label.c_str(), nullptr, false, items[i].enabled))
          items[i].trigger();
      }
      ImGui::EndPopup();
    }
  }
  ImGui::PopStyleVar(2);
}

void VideoWidget::skipToEnd() {
  // set speed to 1.0; this only checks the menu entry, the speed and the button text are unchanged
  speed_index_ = 7;
  can->pause(false);
  can->seekTo(can->maxSeconds() + 1);
}

void VideoWidget::toggleTimeDisplay() {
  settings.absolute_time = !settings.absolute_time;
  time_tooltip_shown_ = true;
}

static const float speeds[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1., 2., 3., 5.};

static std::string speedText(float speed, const char *suffix) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%gx%s", speed, suffix);
  return buf;
}

void VideoWidget::createSpeedDropdown() {
  speed_index_ = 7;  // 1.0x
  can->setSpeed(speeds[speed_index_]);
  speed_text_ = speedText(speeds[speed_index_], "  ");
}

void VideoWidget::drawSpeedDropdown(float width) {
  const ImGuiStyle &style = ImGui::GetStyle();
  // the menu opens on press; a press while it is open toggles it closed (imgui closes the popup at the end
  // of the frame of a click outside it, so only open when it is not already open). Flat until hovered.
  pushBoldFont();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  const bool open = ImGui::ButtonEx((speed_text_ + "###speed_btn").c_str(), ImVec2(width, 0), ImGuiButtonFlags_PressedOnClick);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  popBoldFont();
  // the menu arrow at the right edge of the button
  const ImVec2 btn_min = ImGui::GetItemRectMin(), btn_max = ImGui::GetItemRectMax();
  // a small arrow centered in the indicator area, sitting on the text baseline
  const float ax = btn_max.x - MENU_BUTTON_INDICATOR * 0.5f;
  const float ay = btn_min.y + style.FramePadding.y + ImGui::GetFontSize() - 2.0f;
  ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(ax - MENU_ARROW_SIZE * 0.5f, ay - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(ax + MENU_ARROW_SIZE * 0.5f, ay - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(ax, ay), ImGui::GetColorU32(ImGuiCol_Text));
  if (open && !ImGui::IsPopupOpen("speed_menu")) ImGui::OpenPopup("speed_menu");
  ImGui::SetNextWindowPos(ImVec2(btn_min.x, btn_max.y));
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
  camera_tab = std::make_unique<TabBar>();
  camera_tab->setAutoHide(true);
  camera_tab->setExpanding(false);

  cam_widget = std::make_unique<StreamCameraView>("camerad", VISION_STREAM_NARROW_ROAD);

  slider = std::make_unique<Slider>();
  slider->setSingleStep(0);
  slider->setTimeRange(can->minSeconds(), can->maxSeconds());

  connections_.push_back(slider->sliderReleased.connect([this]() { can->seekTo(slider->currentSecond()); }));
  connections_.push_back(cam_widget->clicked.connect([]() { can->pause(!can->isPaused()); }));
  connections_.push_back(cam_widget->availableStreamsUpdated.connect([this](std::set<VisionStreamType> streams) { vipcAvailableStreamsUpdated(streams); }));
  connections_.push_back(camera_tab->currentChanged.connect([this](int index) {
    if (index != -1) cam_widget->setStreamType((VisionStreamType)camera_tab->tabData(index));
  }));
  connections_.push_back(static_cast<ReplayStream *>(can)->qLogLoaded.connect([this](std::shared_ptr<LogReader> qlog) { cam_widget->parseQLog(qlog); }));
  // eventFilter() runs right after the slider is drawn
}

void VideoWidget::drawCameraWidget() {
  camera_tab->draw();

  // cam_widget: minimum height MIN_VIDEO_HEIGHT, takes the space left by the slider and the toolbar
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float cam_height = std::max((float)MIN_VIDEO_HEIGHT, avail.y - SLIDER_HEIGHT - toolbarHeight());
  cam_widget->draw(ImVec2(avail.x, cam_height));

  if (!slider->isSliderDown()) slider->setCurrentSecond(can->currentSec());
  slider->draw();
  eventFilter();
}

void VideoWidget::vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams) {
  static const std::string stream_names[] = {"Road camera", "Driver camera", "Wide road camera"};
  for (int i = 0; i < streams.size(); ++i) {
    if (camera_tab->count() <= i) {
      camera_tab->addTab(std::string());
    }
    int type = *std::next(streams.begin(), i);
    camera_tab->setTabText(i, stream_names[type]);
    camera_tab->setTabData(i, type);
  }
  while (camera_tab->count() > streams.size()) {
    camera_tab->removeTab(camera_tab->count() - 1);
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
  time_range ? slider->setTimeRange(time_range->first, time_range->second)
             : slider->setTimeRange(can->minSeconds(), can->maxSeconds());
}

std::string VideoWidget::formatTime(double sec, bool include_milliseconds) {
  if (settings.absolute_time)
    sec += std::chrono::duration<double>(can->beginDateTime().time_since_epoch()).count();
  return utils::formatSeconds(sec, include_milliseconds, settings.absolute_time);
}

void VideoWidget::setVisible(bool visible) {
  if (cam_widget) cam_widget->setVisible(visible);
}

void VideoWidget::showThumbnail(double seconds) {
  if (can->liveStreaming()) return;

  cam_widget->thumbnail_dispaly_time = seconds;
  slider->thumbnail_dispaly_time = seconds;
}

void VideoWidget::showRouteInfo() {
  // dropped from route_info_dlgs_ once draw() returns false
  route_info_dlgs_.push_back(std::make_unique<RouteInfoDlg>());
}

void VideoWidget::eventFilter() {
  if (slider->underMouse()) {
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    showThumbnail(min_sec + (ImGui::GetMousePos().x - slider->rect().Min.x) * (max_sec - min_sec) / slider->width());
  } else if (slider->mouseLeft()) {
    showThumbnail(-1);
  }
}

float VideoWidget::sizeHintHeight() const {
  // the camera minimum height plus the slider and the toolbar
  return MIN_VIDEO_HEIGHT + SLIDER_HEIGHT + toolbarHeight();
}

// the video pane opens with the camera at its natural aspect ratio, filling the width of the dock
float VideoWidget::defaultHeight(float width) const {
  if (!cam_widget) return toolbarHeight();  // live streams have no camera or slider
  const float cam_height = std::max((float)MIN_VIDEO_HEIGHT, width / cam_widget->frameAspectRatio());
  const float tab_height = camera_tab->count() >= 2 ? ImGui::GetFrameHeight() : 0.0f;
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

Slider::Slider() {
}

void Slider::draw() {
  ImGui::InvisibleButton("##slider", ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), SLIDER_HEIGHT));
  rect_ = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  const bool hovered = ImGui::IsItemHovered();
  left_ = hovered_ && !hovered;
  hovered_ = hovered;

  if (ImGui::IsItemActivated()) mousePressEvent();
  if (slider_down_) {
    if (ImGui::IsItemActive()) {
      // the handle keeps its grab offset while dragging
      setValue(pixelPosToRangeValue(ImGui::GetMousePos().x - click_offset_));
    } else {
      slider_down_ = false;
      sliderReleased();
    }
  }
  paintEvent();
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

void Slider::paintEvent() {
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

  if (thumbnail_dispaly_time >= 0) {
    float left = rect_.Min.x + (float)((thumbnail_dispaly_time - min) * width() / span) - 1;
    ImRect rc(ImVec2(left, rect_.Min.y + 1), ImVec2(left + 2, rect_.Max.y - 1));
    p->AddRectFilled(rc.Min, rc.Max, ImGui::GetColorU32(ImGuiCol_Header), 1.5f);  // ImGuiCol_Header is the theme highlight
  }
}

void Slider::mousePressEvent() {
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
  big_thumbnail_texture.mipmap = true;  // the hover thumbnail is drawn at a quarter of the stored size
}

void StreamCameraView::parseQLog(std::shared_ptr<LogReader> qlog) {
  std::mutex mutex;
  const auto &events = qlog->events;
  unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
  size_t chunk = (events.size() + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < num_threads && t * chunk < events.size(); ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, events.size());
    threads.emplace_back([this, &mutex, &events, start, end]() {
      for (size_t i = start; i < end; ++i) {
        const Event &e = events[i];
        if (e.which == cereal::Event::Which::THUMBNAIL) {
          capnp::FlatArrayMessageReader reader(e.data);
          auto thumb_data = reader.getRoot<cereal::Event>().getThumbnail();
          auto image_data = thumb_data.getThumbnail();
          if (RgbImage thumb; decodeJpeg(image_data.begin(), image_data.size(), &thumb)) {
            std::lock_guard lock(mutex);
            big_thumbnails[thumb_data.getTimestampEof()] = std::move(thumb);
          }
        }
      }
    });
  }
  for (auto &th : threads) th.join();
}

void StreamCameraView::draw(const ImVec2 &size) {
  CameraWidget::draw(size);

  ImDrawList *p = ImGui::GetWindowDrawList();
  bool scrubbing = false;
  if (thumbnail_dispaly_time >= 0) {
    scrubbing = can->isPaused();
    scrubbing ? drawScrubThumbnail(p) : drawThumbnail(p);
  }
  if (auto alert = getReplay()->findAlertAtTime(scrubbing ? thumbnail_dispaly_time : can->currentSec())) {
    drawAlert(p, rect(), *alert, ImGui::GetFontSize());
  }

  if (can->isPaused()) {
    pushBoldFont();
    ImFont *font = ImGui::GetFont();
    popBoldFont();
    const char *text = "PAUSED";
    const ImVec2 text_size = font->CalcTextSizeA(POINT_16_FONT_SIZE, FLT_MAX, 0.0f, text);
    const ImVec2 center = rect().GetCenter();
    p->AddText(font, POINT_16_FONT_SIZE, ImVec2(center.x - text_size.x / 2, center.y - text_size.y / 2),
               IM_COL32(200, 200, 200, static_cast<int>(255 * 0.7f)), text);
  }
}

void StreamCameraView::drawScrubThumbnail(ImDrawList *p) {
  p->AddRectFilled(rect().Min, rect().Max, IM_COL32(0, 0, 0, 255));
  auto it = big_thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != big_thumbnails.end()) {
    if (big_thumbnail_texture.id == 0 || big_thumbnail_texture.key != it->first) {
      big_thumbnail_texture.upload(it->second);
      big_thumbnail_texture.key = it->first;
    }
    // scale to the widget size, keeping the aspect ratio
    const float scale = std::min(width() / it->second.width, height() / it->second.height);
    const ImVec2 scaled_size(std::floor(it->second.width * scale), std::floor(it->second.height * scale));
    const ImVec2 center = rect().GetCenter();
    ImRect thumb_rect(ImVec2(center.x - (int)(scaled_size.x / 2), center.y - (int)(scaled_size.y / 2)), ImVec2(0, 0));
    thumb_rect.Max = ImVec2(thumb_rect.Min.x + scaled_size.x, thumb_rect.Min.y + scaled_size.y);
    p->AddImage(big_thumbnail_texture.ref(), thumb_rect.Min, thumb_rect.Max);
    drawTime(p, thumb_rect, thumbnail_dispaly_time);
  }
}

void StreamCameraView::drawThumbnail(ImDrawList *p) {
  auto it = big_thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != big_thumbnails.end()) {
    const RgbImage &image = it->second;
    if (big_thumbnail_texture.id == 0 || big_thumbnail_texture.key != it->first) {
      big_thumbnail_texture.upload(image);
      big_thumbnail_texture.key = it->first;
    }
    // AddImage scales the stored image to the thumbnail height, keeping the aspect ratio
    const int h = MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2;
    const int w = std::max(1, (int)std::lround((double)image.width * h / image.height));
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    int pos = (thumbnail_dispaly_time - min_sec) * width() / (max_sec - min_sec);
    const int max_x = (int)width() - w - THUMBNAIL_MARGIN + 1;
    int x = std::clamp(pos - w / 2, THUMBNAIL_MARGIN, std::max(THUMBNAIL_MARGIN, max_x));
    int y = height() - h - THUMBNAIL_MARGIN;

    ImRect thumb_rect(ImVec2(rect().Min.x + x, rect().Min.y + y), ImVec2(rect().Min.x + x + w, rect().Min.y + y + h));
    p->AddImage(big_thumbnail_texture.ref(), thumb_rect.Min, thumb_rect.Max);
    p->AddRect(thumb_rect.Min, thumb_rect.Max, paletteBrightText(), 0.0f, 0, 2.0f);
    if (auto alert = getReplay()->findAlertAtTime(can->toSeconds(it->first))) {
      drawAlert(p, thumb_rect, *alert, POINT_10_FONT_SIZE);
    }
    drawTime(p, thumb_rect, thumbnail_dispaly_time);
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
  for (size_t pos = 0; pos <= text.size();) {
    size_t nl = text.find('\n', pos);
    const char *begin = text.c_str() + pos;
    const char *end = nl == std::string::npos ? text.c_str() + text.size() : text.c_str() + nl;
    const ImVec2 line_size = font->CalcTextSizeA(font_size, FLT_MAX, wrap_width, begin, end);
    p->AddText(font, font_size, ImVec2(text_rect.Min.x + (text_rect.GetWidth() - line_size.x) / 2, y), pen, begin, end, wrap_width);
    y += line_size.y;
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
}
