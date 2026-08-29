#include "tools/cabana/ui/widgets/videowidget.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
}
#include <capnp/serialize.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

const int MIN_VIDEO_HEIGHT = 100;
const int THUMBNAIL_MARGIN = 3;
const float POINT_10_FONT_SIZE = 13.0f;  // QFont(family, 10) at 96 dpi
const float POINT_16_FONT_SIZE = 21.0f;  // QFont(family, 16) at 96 dpi
const float MENU_BUTTON_INDICATOR = 12.0f;  // QStyle::PM_MenuButtonIndicator
const float TOOLBAR_ITEM_SPACING = 1.0f;    // QStyle::PM_ToolBarItemSpacing
const float TOOLBAR_BUTTON_PADDING = 4.0f;  // QToolButton (auto raise) horizontal margin
const float SLIDER_LENGTH = 13.0f;     // QStyle::PM_SliderLength (Fusion)
const float SLIDER_THICKNESS = 13.0f;  // QStyle::PM_SliderThickness (Fusion)

// Indexed by TimelineType: None, Engaged, AlertInfo, AlertWarning, AlertCritical, UserBookmark
static const ImU32 timeline_colors[] = {
  IM_COL32(111, 143, 175, 255),
  IM_COL32(0, 163, 108, 255),
  IM_COL32(0, 255, 0, 255),    // Qt::green
  IM_COL32(255, 195, 0, 255),
  IM_COL32(199, 0, 57, 255),
  IM_COL32(255, 0, 255, 255),  // Qt::magenta
};

static Replay *getReplay() {
  auto stream = dynamic_cast<ReplayStream *>(can);
  return stream ? stream->getReplay() : nullptr;
}

// QColor::name()
static std::string colorName(ImU32 c) {
  char buf[16];
  snprintf(buf, sizeof(buf), "#%02x%02x%02x", (c >> IM_COL32_R_SHIFT) & 0xff, (c >> IM_COL32_G_SHIFT) & 0xff, (c >> IM_COL32_B_SHIFT) & 0xff);
  return buf;
}

static ImU32 withAlpha(ImU32 c, int alpha) {
  return (c & ~IM_COL32_A_MASK) | ((ImU32)alpha << IM_COL32_A_SHIFT);
}

// palette().color(QPalette::BrightText)
static ImU32 brightText() {
  return isDarkTheme() ? IM_COL32(DarkTheme::bright_text.r, DarkTheme::bright_text.g, DarkTheme::bright_text.b, 255)
                       : IM_COL32(255, 255, 255, 255);
}

// QStyle::SC_SliderHandle (Fusion): a 13x13 handle filled with a subtle vertical gradient and a mid grey outline
static void drawSliderHandle(ImDrawList *p, const ImRect &r) {
  const bool dark = isDarkTheme();
  // buttonColor.lighter(104) / buttonColor.darker(104)
  const ImU32 top = dark ? IM_COL32(0x3e, 0x41, 0x43, 255) : IM_COL32(255, 255, 255, 255);
  const ImU32 bottom = dark ? IM_COL32(0x39, 0x3c, 0x3e, 255) : IM_COL32(0xf0, 0xf0, 0xf0, 255);
  // QFusionStylePrivate::outline: the top/left edge is one step lighter than the bottom/right edge
  const ImU32 outline_top = dark ? IM_COL32(0xa3, 0xa3, 0xa3, 255) : IM_COL32(0xab, 0xab, 0xab, 255);
  const ImU32 outline_bottom = dark ? IM_COL32(0x9c, 0x9c, 0x9c, 255) : IM_COL32(0xa4, 0xa4, 0xa4, 255);
  p->AddRectFilled(r.Min, r.Max, top, 2.0f);
  p->AddRectFilled(ImVec2(r.Min.x, r.GetCenter().y), r.Max, bottom, 2.0f, ImDrawFlags_RoundCornersBottom);
  p->AddRect(r.Min, r.Max, outline_bottom, 2.0f, 0, 1.0f);
  // the straight edges are drawn as crisp 1 px rects: an antialiased outline washes out to a much lighter grey
  const float c = 2.0f;  // corner radius
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Min.y), ImVec2(r.Max.x - c, r.Min.y + 1.0f), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x, r.Min.y + c), ImVec2(r.Min.x + 1.0f, r.Max.y - c), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Max.y - 1.0f), ImVec2(r.Max.x - c, r.Max.y), outline_bottom);
  p->AddRectFilled(ImVec2(r.Max.x - 1.0f, r.Min.y + c), ImVec2(r.Max.x, r.Max.y - c), outline_bottom);
}

// QPixmap::loadFromData(..., "jpeg") via libavcodec (already linked for the replay video decoder)
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
    for (int y = 0; y < frame->height; ++y) {
      const uint8_t *y_row = frame->data[0] + y * frame->linesize[0];
      const uint8_t *u_row = frame->data[1] + (y >> chroma_y_shift) * frame->linesize[1];
      const uint8_t *v_row = frame->data[2] + (y >> chroma_y_shift) * frame->linesize[2];
      uint8_t *dst = out->data.data() + (size_t)y * out->bytesPerLine();
      for (int x = 0; x < frame->width; ++x) {
        const double luma = full_range ? (double)y_row[x] : 1.164383 * ((double)y_row[x] - 16.0);
        const double u = (double)u_row[x >> chroma_x_shift] - 128.0;
        const double v = (double)v_row[x >> chroma_x_shift] - 128.0;
        const double r = luma + (full_range ? 1.402 : 1.596027) * v;
        const double g = luma - (full_range ? 0.344136 : 0.391762) * u - (full_range ? 0.714136 : 0.812968) * v;
        const double b = luma + (full_range ? 1.772 : 2.017232) * u;
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

// QPixmap::scaledToHeight(h, Qt::SmoothTransformation): bilinear resample
static RgbImage scaledToHeight(const RgbImage &src, int h) {
  RgbImage dst;
  if (src.isNull() || h <= 0) return dst;
  const int w = std::max(1, (int)std::lround((double)src.width * h / src.height));
  dst.resize(w, h);
  for (int y = 0; y < h; ++y) {
    const float sy = std::clamp((y + 0.5f) * src.height / h - 0.5f, 0.0f, (float)src.height - 1);
    const int y0 = (int)sy, y1 = std::min(y0 + 1, src.height - 1);
    const float fy = sy - y0;
    for (int x = 0; x < w; ++x) {
      const float sx = std::clamp((x + 0.5f) * src.width / w - 0.5f, 0.0f, (float)src.width - 1);
      const int x0 = (int)sx, x1 = std::min(x0 + 1, src.width - 1);
      const float fx = sx - x0;
      const uint8_t *p00 = &src.data[((size_t)y0 * src.width + x0) * 4], *p01 = &src.data[((size_t)y0 * src.width + x1) * 4];
      const uint8_t *p10 = &src.data[((size_t)y1 * src.width + x0) * 4], *p11 = &src.data[((size_t)y1 * src.width + x1) * 4];
      uint8_t *d = &dst.data[((size_t)y * w + x) * 4];
      for (int c = 0; c < 4; ++c) {
        const float top = p00[c] + (p01[c] - p00[c]) * fx;
        const float bottom = p10[c] + (p11[c] - p10[c]) * fx;
        d[c] = (uint8_t)std::lround(top + (bottom - top) * fy);
      }
    }
  }
  return dst;
}

VideoWidget::VideoWidget() {
  if (!can->liveStreaming())
    createCameraWidget();

  createPlaybackController();

  connections_.push_back(can->paused.connect([this]() { updatePlayBtnState(); }));
  connections_.push_back(can->resume.connect([this]() { updatePlayBtnState(); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &) { timeRangeChanged(); }));

  updatePlayBtnState();
  // setWhatsThis: the HTML table becomes one <br /> separated row per table row, with the same entries and colors
  whats_this_ = "<b>Video</b><br />\n"
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

void VideoWidget::createPlaybackController() {
  // the toolbar actions are drawn by drawPlaybackController(); only their state is created here
  if (can->liveStreaming()) {
    skip_to_end_enabled_ = true;
  }
  time_text_ = "";
  time_tooltip_ = "";

  createSpeedDropdown();
}

void VideoWidget::drawPlaybackController() {
  // QToolBar metrics: PM_ToolBarItemSpacing between the items, the buttons only carry the auto raise margin
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(TOOLBAR_ITEM_SPACING, ImGui::GetStyle().ItemSpacing.y));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(TOOLBAR_BUTTON_PADDING, ImGui::GetStyle().FramePadding.y));
  const ImGuiStyle &style = ImGui::GetStyle();
  // a button is its text plus 2 * FramePadding.x; the time display is a plain label, so it is just its text
  auto text_width = [](const char *label) { return ImGui::CalcTextSize(label).x; };
  auto button_width = [&](const char *label) { return text_width(label) + style.FramePadding.x * 2; };
  pushBoldFont();
  const float speed_width = button_width("0.05x  ") + MENU_BUTTON_INDICATOR;
  popBoldFont();

  enum ToolItem { REWIND, PLAY, FORWARD, SKIP_END, TIME_DISPLAY, LOOP, SPEED, SEPARATOR, ROUTE_INFO };
  std::vector<int> items = {REWIND, PLAY, FORWARD};
  if (can->liveStreaming()) items.push_back(SKIP_END);
  items.push_back(TIME_DISPLAY);
  // the expanding spacer: the items after it are right aligned as long as everything fits
  const size_t spacer_index = items.size();
  if (!can->liveStreaming()) {
    items.insert(items.end(), {LOOP, SPEED, SEPARATOR, ROUTE_INFO});
  } else {
    items.push_back(SPEED);
  }

  auto item_width = [&](int item) -> float {
    switch (item) {
      case REWIND: return button_width(icon::REWIND);
      case PLAY: return button_width(play_icon_);
      case FORWARD: return button_width(icon::FAST_FORWARD);
      case SKIP_END: return button_width(icon::SKIP_END);
      case TIME_DISPLAY: return text_width(time_text_.c_str());
      case LOOP: return button_width(loop_icon_);
      case SPEED: return speed_width;
      case SEPARATOR: return 1.0f;
      default: return button_width(icon::INFO_CIRCLE);
    }
  };
  auto draw_item = [&](int item) {
    switch (item) {
      case REWIND:
        if (toolButton(icon::REWIND, "Seek backward", "rewind")) can->seekTo(can->currentSec() - 1);
        break;
      case PLAY:
        if (toolButton(play_icon_, play_tooltip_.c_str(), "play")) can->pause(!can->isPaused());
        break;
      case FORWARD:
        if (toolButton(icon::FAST_FORWARD, "Seek forward", "fast-forward")) can->seekTo(can->currentSec() + 1);
        break;
      case SKIP_END:
        ImGui::BeginDisabled(!skip_to_end_enabled_);
        if (toolButton(icon::SKIP_END, "Skip to the end", "skip-end")) skipToEnd();
        ImGui::EndDisabled();
        break;
      case TIME_DISPLAY:
        // QLabel: the text with no frame around it
        if (toolButton(time_text_.c_str(), time_tooltip_.c_str(), "time_display", text_width(time_text_.c_str())))
          toggleTimeDisplay();
        break;
      case LOOP:
        if (toolButton(loop_icon_, "Loop playback", "loop")) loopPlaybackClicked();
        break;
      case SPEED: drawSpeedDropdown(); break;
      case SEPARATOR: ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical); break;
      default:
        if (toolButton(icon::INFO_CIRCLE, "View route details", "route_info")) showRouteInfo();
        break;
    }
  };

  auto group_width = [&](size_t begin, size_t end) {
    float w = 0;
    for (size_t i = begin; i < end; ++i) w += item_width(items[i]) + (i > begin ? style.ItemSpacing.x : 0);
    return w;
  };
  const float left_width = group_width(0, spacer_index);
  const float right_width = group_width(spacer_index, items.size());

  const float start_x = ImGui::GetCursorPosX();
  const float avail = ImGui::GetContentRegionAvail().x;
  const float right_edge = start_x + avail;
  const float extension_width = button_width(icon::RAQUO);

  // QToolBarLayout: when everything fits the spacer takes the slack, otherwise the extension button is
  // reserved at the right edge and the items are packed from the left until the next one does not fit
  const bool fits = left_width + style.ItemSpacing.x + right_width <= avail;
  size_t visible = items.size();
  if (!fits) {
    const float usable = avail - (extension_width + style.ItemSpacing.x);
    float used = 0;
    for (visible = 0; visible < items.size(); ++visible) {
      const float w = item_width(items[visible]) + (visible ? style.ItemSpacing.x : 0);
      if (used + w > usable) break;
      used += w;
    }
  }

  for (size_t i = 0; i < visible; ++i) {
    if (i == 0) ImGui::SetCursorPosX(start_x);
    else if (fits && i == spacer_index) ImGui::SameLine(right_edge - right_width);
    else ImGui::SameLine();
    draw_item(items[i]);
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
        switch (items[i]) {
          case REWIND:
            if (ImGui::MenuItem("Seek backward")) can->seekTo(can->currentSec() - 1);
            break;
          case PLAY:
            if (ImGui::MenuItem(play_tooltip_.c_str())) can->pause(!can->isPaused());
            break;
          case FORWARD:
            if (ImGui::MenuItem("Seek forward")) can->seekTo(can->currentSec() + 1);
            break;
          case SKIP_END:
            if (ImGui::MenuItem("Skip to the end", nullptr, false, skip_to_end_enabled_)) skipToEnd();
            break;
          case TIME_DISPLAY:
            if (ImGui::MenuItem(time_text_.c_str())) toggleTimeDisplay();
            break;
          case LOOP:
            if (ImGui::MenuItem("Loop")) loopPlaybackClicked();
            break;
          case SPEED:
            if (ImGui::BeginMenu("Speed")) {
              drawSpeedMenuItems();
              ImGui::EndMenu();
            }
            break;
          case SEPARATOR:
            ImGui::Separator();
            break;
          default:
            if (ImGui::MenuItem("Route info")) showRouteInfo();
            break;
        }
      }
      ImGui::EndPopup();
    }
  }
  ImGui::PopStyleVar(2);
}

void VideoWidget::skipToEnd() {
  // set speed to 1.0
  speed_index_ = 7;  // like the Qt code this only checks the menu entry; the speed and the button text are unchanged
  can->pause(false);
  can->seekTo(can->maxSeconds() + 1);
}

void VideoWidget::toggleTimeDisplay() {
  settings.absolute_time = !settings.absolute_time;
  time_tooltip_ = settings.absolute_time ? "Elapsed time" : "Absolute time";
  updateState();
}

static const float speeds[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1., 2., 3., 5.};

static std::string speedText(float speed, const char *suffix) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%gx%s", speed, suffix);
  return buf;
}

void VideoWidget::createSpeedDropdown() {
  for (int i = 0; i < (int)std::size(speeds); ++i) {
    const float speed = speeds[i];
    if (speed == 1.0) {
      speed_index_ = i;
      // act->trigger()
      can->setSpeed(speed);
      speed_text_ = speedText(speed, "  ");
    }
  }
}

void VideoWidget::drawSpeedDropdown() {
  const ImGuiStyle &style = ImGui::GetStyle();
  pushBoldFont();
  const float min_width = ImGui::CalcTextSize("0.05x  ").x + style.FramePadding.x * 2 + MENU_BUTTON_INDICATOR;
  // QToolButton::InstantPopup opens on press; a press while the menu is open toggles it closed (imgui closes the
  // popup at the end of the frame of a click outside it, so only open when it is not already open)
  // setAutoRaise(true): flat until hovered
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  const bool open = ImGui::ButtonEx((speed_text_ + "###speed_btn").c_str(), ImVec2(min_width, 0), ImGuiButtonFlags_PressedOnClick);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  popBoldFont();
  // QStyle::PM_MenuButtonIndicator: the menu arrow at the right edge of the button
  const ImVec2 btn_min = ImGui::GetItemRectMin(), btn_max = ImGui::GetItemRectMax();
  ImGui::RenderArrow(ImGui::GetWindowDrawList(),
                     ImVec2(btn_max.x - MENU_BUTTON_INDICATOR, (btn_min.y + btn_max.y) / 2 - ImGui::GetFontSize() * 0.5f),
                     ImGui::GetColorU32(ImGuiCol_Text), ImGuiDir_Down, 0.7f);
  if (open && !ImGui::IsPopupOpen("speed_menu")) ImGui::OpenPopup("speed_menu");
  ImGui::SetNextWindowPos(ImVec2(btn_min.x, btn_max.y));
  if (ImGui::BeginPopup("speed_menu")) {
    drawSpeedMenuItems();
    ImGui::EndPopup();
  }
}

void VideoWidget::drawSpeedMenuItems() {
  // exclusive QActionGroup: the current speed is marked with a radio bullet on the left
  const float indent = ImGui::GetFontSize();
  ImGui::Indent(indent);
  for (int i = 0; i < (int)std::size(speeds); ++i) {
    const float speed = speeds[i];
    if (ImGui::MenuItem(speedText(speed, "").c_str())) {
      speed_index_ = i;
      can->setSpeed(speed);
      speed_text_ = speedText(speed, "  ");
    }
    if (speed_index_ == i) {
      const ImVec2 item_min = ImGui::GetItemRectMin(), item_max = ImGui::GetItemRectMax();
      ImGui::RenderBullet(ImGui::GetWindowDrawList(), ImVec2(item_min.x - indent / 2, (item_min.y + item_max.y) / 2),
                          ImGui::GetColorU32(ImGuiCol_Text));
    }
  }
  ImGui::Unindent(indent);
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
  // can->paused -> cam_widget->update() and can->eventsMerged -> slider->update(): imgui redraws every frame
  connections_.push_back(cam_widget->clicked.connect([]() { can->pause(!can->isPaused()); }));
  connections_.push_back(cam_widget->availableStreamsUpdated.connect([this](std::set<VisionStreamType> streams) { vipcAvailableStreamsUpdated(streams); }));
  connections_.push_back(camera_tab->currentChanged.connect([this](int index) {
    if (index != -1) cam_widget->setStreamType((VisionStreamType)camera_tab->tabData(index));
  }));
  connections_.push_back(static_cast<ReplayStream *>(can)->qLogLoaded.connect([this](std::shared_ptr<LogReader> qlog) { cam_widget->parseQLog(qlog); }));
  // slider->installEventFilter(this): eventFilter() runs right after the slider is drawn
}

void VideoWidget::drawCameraWidget() {
  camera_tab->draw();

  // cam_widget: minimum height MIN_VIDEO_HEIGHT, takes the space left by the slider and the toolbar
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float cam_height = std::max((float)MIN_VIDEO_HEIGHT, avail.y - ImGui::GetFrameHeight() * 2 - style.ItemSpacing.y * 2);
  cam_widget->draw(ImVec2(avail.x, cam_height));

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
  bool is_looping = getReplay()->loop();
  getReplay()->setLoop(!is_looping);
  loop_icon_ = !is_looping ? icon::REPEAT : icon::REPEAT_1;
}

void VideoWidget::timeRangeChanged() {
  const auto time_range = can->timeRange();
  if (can->liveStreaming()) {
    skip_to_end_enabled_ = !time_range.has_value();
    return;
  }
  time_range ? slider->setTimeRange(time_range->first, time_range->second)
             : slider->setTimeRange(can->minSeconds(), can->maxSeconds());
  updateState();
}

std::string VideoWidget::formatTime(double sec, bool include_milliseconds) {
  if (settings.absolute_time)
    sec += std::chrono::duration<double>(can->beginDateTime().time_since_epoch()).count();
  return utils::formatSeconds(sec, include_milliseconds, settings.absolute_time);
}

void VideoWidget::updateState() {
  if (slider) {
    if (!slider->isSliderDown()) {
      slider->setCurrentSecond(can->currentSec());
    }
    if (camera_tab->count() == 0) {  //  No streams available
      // cam_widget->update(): imgui redraws every frame, the alert events are drawn regardless
    }
    time_text_ = formatTime(can->currentSec(), true) + " / " + formatTime(slider->maximum() / slider->factor);
  } else {
    time_text_ = formatTime(can->currentSec(), true);
  }
}

void VideoWidget::updatePlayBtnState() {
  play_icon_ = can->isPaused() ? icon::PLAY : icon::PAUSE;
  play_tooltip_ = can->isPaused() ? "Play" : "Pause";
}

void VideoWidget::setVisible(bool visible) {
  if (cam_widget) cam_widget->setVisible(visible);
}

void VideoWidget::showThumbnail(double seconds) {
  if (can->liveStreaming()) return;

  cam_widget->thumbnail_dispaly_time = seconds;
  slider->thumbnail_dispaly_time = seconds;
  // cam_widget->update(), slider->update(): imgui redraws every frame
}

void VideoWidget::showRouteInfo() {
  // WA_DeleteOnClose: dropped from route_info_dlgs_ once draw() returns false
  route_info_dlgs_.push_back(std::make_unique<RouteInfoDlg>());
}

void VideoWidget::eventFilter() {
  if (slider->underMouse()) {  // QEvent::MouseMove (setMouseTracking(true))
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    showThumbnail(min_sec + (ImGui::GetMousePos().x - slider->rect().Min.x) * (max_sec - min_sec) / slider->width());
  } else if (slider->mouseLeft()) {  // QEvent::Leave
    showThumbnail(-1);
  }
}

void VideoWidget::draw() {
  if (!can->liveStreaming())
    drawCameraWidget();

  drawPlaybackController();

  for (auto it = route_info_dlgs_.begin(); it != route_info_dlgs_.end();) {
    it = (*it)->draw() ? it + 1 : route_info_dlgs_.erase(it);
  }
}

// ToolButton

bool toolButton(const char *icon, const char *tooltip, const char *id, float width) {
  const std::string label = std::string(icon) + "###" + id;
  // setAutoRaise(true)
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  const bool pressed = ImGui::Button(label.c_str(), ImVec2(width, 0));
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  if (tooltip && tooltip[0] && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", tooltip);
  return pressed;
}

// TabBar

int TabBar::addTab(const std::string &text) {
  tabs_.push_back({text, 0, next_id_++});
  int index = count() - 1;
  // the "x" close button is drawn by BeginTabItem(p_open) and reports through closeTabClicked()
  if (current_index_ == -1) {  // QTabBar makes the first tab current
    current_index_ = index;
    select_current_ = true;
    currentChanged(index);
  }
  return index;
}

void TabBar::removeTab(int index) {
  tabs_.erase(tabs_.begin() + index);
  if (index == current_index_) {
    // QTabBar::SelectRightTab: the tab that moved into this index, else the one to the left
    current_index_ = count() ? std::min(index, count() - 1) : -1;
    select_current_ = true;
    currentChanged(current_index_);
  } else if (index < current_index_) {
    --current_index_;
  }
}

void TabBar::draw() {
  if (auto_hide_ && count() < 2) return;  // setAutoHide(true)
  if (!ImGui::BeginTabBar("##tabbar")) return;
  for (int i = 0; i < count(); ++i) {
    bool open = true;
    const std::string label = tabs_[i].text + "###tab" + std::to_string(tabs_[i].id);
    const ImGuiTabItemFlags flags = (select_current_ && i == current_index_) ? ImGuiTabItemFlags_SetSelected : 0;
    if (ImGui::BeginTabItem(label.c_str(), tabs_closable_ ? &open : nullptr, flags)) {
      if (i != current_index_) {
        current_index_ = i;
        currentChanged(i);
      }
      ImGui::EndTabItem();
    }
    if (!open) closeTabClicked(i);
  }
  select_current_ = false;
  ImGui::EndTabBar();
}

// Slider
Slider::Slider() {
  // setMouseTracking(true): imgui always reports hover
}

void Slider::draw() {
  ImGui::InvisibleButton("##slider", ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), ImGui::GetFrameHeight()));
  rect_ = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  const bool hovered = ImGui::IsItemHovered();
  left_ = hovered_ && !hovered;
  hovered_ = hovered;

  if (ImGui::IsItemActivated()) mousePressEvent();
  if (slider_down_) {
    if (ImGui::IsItemActive()) {
      // QSlider::mouseMoveEvent: pixelPosToRangeValue(pos - clickOffset), the handle keeps its grab offset
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

// QSliderPrivate::pixelPosToRangeValue: handle left edge (window x) -> value over the groove minus the handle width
int Slider::pixelPosToRangeValue(float x) const {
  const float handle_width = SLIDER_LENGTH;
  const float span = std::max(1.0f, width() - handle_width);
  return minimum() + (int)std::lround((maximum() - minimum()) * std::clamp((x - rect_.Min.x) / span, 0.0f, 1.0f));
}

void Slider::paintEvent() {
  ImDrawList *p = ImGui::GetWindowDrawList();

  ImRect handle_rect = handleRect();
  ImRect groove_rect = rect_;

  // Adjust groove height to match handle height
  float handle_height = handle_rect.GetHeight();
  const float groove_height = handle_height * 0.5f;
  const float center_y = rect_.GetCenter().y;
  groove_rect.Min.y = center_y - groove_height / 2;
  groove_rect.Max.y = center_y + groove_height / 2;

  p->AddRectFilled(groove_rect.Min, groove_rect.Max, timeline_colors[(int)TimelineType::None]);

  double min = minimum() / factor;
  double max = maximum() / factor;
  const double span = std::max(max - min, 1e-9);

  auto fillRange = [&](double begin, double end, ImU32 color) {
    if (begin > max || end < min) return;

    ImRect r = groove_rect;
    r.Min.x = rect_.Min.x + ((std::max(min, begin) - min) / span) * width();
    r.Max.x = rect_.Min.x + ((std::min(max, end) - min) / span) * width();
    p->AddRectFilled(r.Min, r.Max, color);
  };

  if (auto replay = getReplay()) {
    for (const auto &entry : *replay->getTimeline()) {
      fillRange(entry.start_time, entry.end_time, timeline_colors[(int)entry.type]);
    }

    ImU32 empty_color = ImGui::GetColorU32(ImGuiCol_WindowBg, 160 / 255.0f);  // palette().color(QPalette::Window) with alpha 160
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
    p->AddRectFilled(rc.Min, rc.Max, ImGui::GetColorU32(ImGuiCol_Header), 1.5f);  // palette().highlight(): ImGuiCol_Header is the theme highlight (style.cc), as in chart.cc
  }
}

void Slider::mousePressEvent() {
  // QSlider::mousePressEvent: a press on the handle starts a drag (isSliderDown) and remembers the grab offset
  const ImRect handle_rect = handleRect();
  if (handle_rect.Contains(ImGui::GetMousePos())) {
    slider_down_ = true;
    click_offset_ = ImGui::GetMousePos().x - handle_rect.Min.x;
    return;
  }
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !isSliderDown()) {
    setValue(minimum() + (int)(((maximum() - minimum()) * (ImGui::GetMousePos().x - rect_.Min.x)) / width()));
    sliderReleased();
  }
}

// StreamCameraView
StreamCameraView::StreamCameraView(std::string stream_name, VisionStreamType stream_type)
    : CameraWidget(stream_name, stream_type) {
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
            Thumbnail generated_thumb = generateThumbnail(thumb, can->toSeconds(thumb_data.getTimestampEof()));
            std::lock_guard lock(mutex);
            thumbnails[thumb_data.getTimestampEof()] = std::move(generated_thumb);
            big_thumbnails[thumb_data.getTimestampEof()] = std::move(thumb);
          }
        }
      }
    });
  }
  for (auto &th : threads) th.join();
  // update(): imgui redraws every frame
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

StreamCameraView::Thumbnail StreamCameraView::generateThumbnail(const RgbImage &thumb, double seconds) {
  Thumbnail scaled;
  scaled.image = scaledToHeight(thumb, MIN_VIDEO_HEIGHT - THUMBNAIL_MARGIN * 2);
  // the 2px BrightText border and the alert are painted over the image in drawThumbnail()
  scaled.alert = getReplay()->findAlertAtTime(seconds);
  return scaled;
}

void StreamCameraView::drawScrubThumbnail(ImDrawList *p) {
  p->AddRectFilled(rect().Min, rect().Max, IM_COL32(0, 0, 0, 255));
  auto it = big_thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != big_thumbnails.end()) {
    if (big_thumbnail_texture.id == 0 || big_thumbnail_texture.key != it->first) {
      big_thumbnail_texture.upload(it->second);
      big_thumbnail_texture.key = it->first;
    }
    // scaled(rect().size(), Qt::KeepAspectRatio)
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
  auto it = thumbnails.lower_bound(can->toMonoTime(thumbnail_dispaly_time));
  if (it != thumbnails.end()) {
    const Thumbnail &thumb = it->second;
    if (thumbnail_texture.id == 0 || thumbnail_texture.key != it->first) {
      thumbnail_texture.upload(thumb.image);
      thumbnail_texture.key = it->first;
    }
    auto [min_sec, max_sec] = can->timeRange().value_or(std::make_pair(can->minSeconds(), can->maxSeconds()));
    int pos = (thumbnail_dispaly_time - min_sec) * width() / (max_sec - min_sec);
    const int max_x = (int)width() - thumb.image.width - THUMBNAIL_MARGIN + 1;
    int x = std::clamp(pos - thumb.image.width / 2, THUMBNAIL_MARGIN, std::max(THUMBNAIL_MARGIN, max_x));
    int y = height() - thumb.image.height - THUMBNAIL_MARGIN;

    ImRect thumb_rect(ImVec2(rect().Min.x + x, rect().Min.y + y), ImVec2(rect().Min.x + x + thumb.image.width, rect().Min.y + y + thumb.image.height));
    p->AddImage(thumbnail_texture.ref(), thumb_rect.Min, thumb_rect.Max);
    // generateThumbnail: QPen(BrightText, 2) rect and the alert at that time
    p->AddRect(thumb_rect.Min, thumb_rect.Max, brightText(), 0.0f, 0, 2.0f);
    if (thumb.alert) {
      drawAlert(p, thumb_rect, *thumb.alert, POINT_10_FONT_SIZE);
    }
    drawTime(p, thumb_rect, thumbnail_dispaly_time);
  }
}

void StreamCameraView::drawTime(ImDrawList *p, const ImRect &rect, double seconds) {
  char text[32];
  snprintf(text, sizeof(text), "%.3f", seconds);
  ImFont *font = ImGui::GetFont();
  const ImVec2 text_size = font->CalcTextSizeA(POINT_10_FONT_SIZE, FLT_MAX, 0.0f, text);
  // rect.adjusted(0, 0, 0, -THUMBNAIL_MARGIN), Qt::AlignHCenter | Qt::AlignBottom
  p->AddText(font, POINT_10_FONT_SIZE, ImVec2(rect.GetCenter().x - text_size.x / 2, rect.Max.y - THUMBNAIL_MARGIN - text_size.y),
             brightText(), text);
}

void StreamCameraView::drawAlert(ImDrawList *p, const ImRect &rect, const Timeline::Entry &alert, float font_size) {
  const ImU32 pen = brightText();
  ImU32 color = withAlpha(timeline_colors[int(alert.type)], 128);  // setAlphaF(0.5)
  std::string text = alert.text1;
  if (!alert.text2.empty()) text += "\n" + alert.text2;

  ImRect text_rect(ImVec2(rect.Min.x + 1, rect.Min.y + 1), ImVec2(rect.Max.x - 1, rect.Max.y - 1));
  ImFont *font = ImGui::GetFont();
  const float wrap_width = std::max(1.0f, text_rect.GetWidth());
  const ImVec2 r = font->CalcTextSizeA(font_size, FLT_MAX, wrap_width, text.c_str());
  p->AddRectFilled(ImVec2(text_rect.Min.x, text_rect.Min.y), ImVec2(text_rect.Max.x, text_rect.Min.y + r.y), color);
  // Qt::AlignTop | Qt::AlignHCenter | Qt::TextWordWrap: each line is centered, wrapped continuations stay left aligned
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
