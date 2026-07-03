#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>

#include "tools/cabana/settings.h"

namespace {

// exact speed list from tools/cabana/videowidget.cc createSpeedDropdown()
constexpr double SPEEDS[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1., 2., 3., 5.};

bool g_seek_dragging = false;
float g_seek_drag_value = 0.0f;

// mirrors tools/cabana/utils/util.cc formatSeconds() for elapsed playback time
std::string format_duration(double sec, bool include_ms) {
  sec = std::max(sec, 0.0);
  const int total = static_cast<int>(sec);
  const int hh = total / 3600;
  const int mm = (total % 3600) / 60;
  const int ss = total % 60;
  char buf[64];
  if (include_ms) {
    const int ms = static_cast<int>(std::lround((sec - total) * 1000.0));
    if (sec > 3600.0) {
      snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", hh, mm, ss, ms);
    } else {
      snprintf(buf, sizeof(buf), "%02d:%02d.%03d", mm, ss, ms);
    }
  } else if (sec > 3600.0) {
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hh, mm, ss);
  } else {
    snprintf(buf, sizeof(buf), "%02d:%02d", mm, ss);
  }
  return buf;
}

// mirrors VideoWidget::formatTime() absolute_time branch (local wall-clock time)
std::string format_absolute(double epoch_sec, bool include_ms) {
  const time_t whole = static_cast<time_t>(epoch_sec);
  std::tm tm_buf{};
  localtime_r(&whole, &tm_buf);
  char buf[64];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
           tm_buf.tm_mday, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  std::string s = buf;
  if (include_ms) {
    const int ms = static_cast<int>(std::lround((epoch_sec - whole) * 1000.0));
    char ms_buf[16];
    snprintf(ms_buf, sizeof(ms_buf), ".%03d", ms);
    s += ms_buf;
  }
  return s;
}

std::string format_time(const AppState &app, double sec, bool include_ms) {
  if (settings.absolute_time) {
    const double begin = std::chrono::duration<double>(app.stream->beginDateTime().time_since_epoch()).count();
    return format_absolute(begin + sec, include_ms);
  }
  return format_duration(sec, include_ms);
}

std::string format_speed(double speed) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%gx", speed);
  return buf;
}

}  // namespace

void draw_transport_bar(AppState &app) {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - TRANSPORT_BAR_HEIGHT));
  ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, TRANSPORT_BAR_HEIGHT));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav |
                                 ImGuiWindowFlags_NoSavedSettings;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 0.0f));
  if (!ImGui::Begin("##transport_bar", nullptr, flags)) {
    ImGui::End();
    ImGui::PopStyleVar();
    return;
  }

  const AbstractStream &stream = *app.stream;
  const bool live = has_stream(app);
  const float min_sec = static_cast<float>(stream.minSeconds());
  const float max_sec = std::max(min_sec + 0.001f, static_cast<float>(stream.maxSeconds()));
  const std::string total_label = format_time(app, max_sec, false);
  const std::string speed_label = format_speed(stream.getSpeed());
  const std::string route_label = stream.routeName();
  const float speed_w = ImGui::CalcTextSize("00.00x").x + ImGui::GetFrameHeight();
  const float total_w = ImGui::CalcTextSize(total_label.c_str()).x;
  const float route_w = ImGui::CalcTextSize(route_label.c_str()).x;
  const float spacing = ImGui::GetStyle().ItemSpacing.x;

  ImGui::SetCursorPosY((TRANSPORT_BAR_HEIGHT - ImGui::GetFrameHeight()) * 0.5f);
  ImGui::BeginDisabled(!live);

  const bool paused = stream.isPaused();
  const float button_w = ImGui::GetFrameHeight();
  if (ImGui::Button(paused ? icon::PLAY_FILL : icon::PAUSE_FILL, ImVec2(button_w, 0.0f))) {
    app.stream->pause(!paused);
  }

  ImGui::SameLine();
  const std::string current_label = format_time(app, stream.currentSec(), true);
  if (ImGui::Button(current_label.c_str())) {
    settings.absolute_time = !settings.absolute_time;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s", settings.absolute_time ? "Elapsed time" : "Absolute time");
  }

  ImGui::SameLine();
  const float trailing_w = 3.0f * spacing + total_w + speed_w + route_w;
  const float slider_w = std::max(20.0f, ImGui::GetContentRegionAvail().x - trailing_w);
  ImGui::SetNextItemWidth(slider_w);
  float slider_value = g_seek_dragging ? g_seek_drag_value : static_cast<float>(stream.currentSec());
  slider_value = std::clamp(slider_value, min_sec, max_sec);
  ImGui::SliderFloat("##seek", &slider_value, min_sec, max_sec, "");
  if (ImGui::IsItemActive()) {
    g_seek_dragging = true;
    g_seek_drag_value = slider_value;
    ImGui::SetTooltip("%s", format_time(app, slider_value, true).c_str());
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    app.stream->seekTo(slider_value);
  }
  if (ImGui::IsItemDeactivated()) {
    g_seek_dragging = false;
  }

  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(total_label.c_str());

  ImGui::SameLine();
  ImGui::SetNextItemWidth(speed_w);
  if (ImGui::BeginCombo("##speed", speed_label.c_str())) {
    for (double speed : SPEEDS) {
      const std::string label = format_speed(speed);
      if (ImGui::Selectable(label.c_str(), speed == stream.getSpeed())) {
        app.stream->setSpeed(static_cast<float>(speed));
      }
    }
    ImGui::EndCombo();
  }

  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("%s", route_label.c_str());

  ImGui::End();
  ImGui::PopStyleVar();
}
