#include "tools/loggy/panes/camera.h"

#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "imgui_impl_opengl3_loader.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <cstdio>
#include <string>
#include <utility>

namespace loggy {
namespace {

struct CameraPaneState {
  CameraViewKind view = CameraViewKind::Road;
  bool fit_to_pane = true;
};

struct CameraPaneSnapshot {
  CameraViewKind view = CameraViewKind::Road;
  size_t segment_file_count = 0;
  size_t frame_count = 0;
  TimelineSpanKind overlay_kind = TimelineSpanKind::None;
  bool has_frame = false;
  bool ingest_failed = false;
};

struct CameraRouteInfoLines {
  std::string line1;
  std::string line2;
  std::string line3;
};

struct CameraTextureState {
  GLuint texture = 0;
  int width = 0;
  int height = 0;
  bool live = false;
  std::optional<CameraDecodeKey> key;
  // Frame actually on the GPU, not the seek target (REVIEW defect B).
  uint32_t frame_id = 0;
  double timestamp = 0.0;
  std::string error;
};

CameraPaneState parse_camera_pane_state(std::string_view state_json);
std::string camera_pane_state_json(const CameraPaneState &state);
CameraPaneSnapshot prepare_camera_pane_snapshot(const Session &session, const CameraPaneState &state);
CameraRouteInfoLines build_camera_route_info_lines(const CameraPaneSnapshot &snapshot,
                                                   const CameraDecodeStatus &decode_status,
                                                   const LiveCameraFrameStatus *live_status,
                                                   const CameraTextureState &texture,
                                                   bool has_texture,
                                                   float text_area_width);

std::string trim_overlay_text(const std::string &text, float max_width) {
  if (text.empty() || max_width <= 0.0f) return text;

  const size_t max_chars = std::max<size_t>(3, static_cast<size_t>(max_width / 7.0f));
  if (text.size() <= max_chars) return text;

  if (max_chars <= 3) return text.substr(0, max_chars);
  return text.substr(0, max_chars - 3).append("...");
}

int camera_combo_index(CameraViewKind view) {
  return static_cast<int>(camera_view_index(view));
}

CameraViewKind camera_from_combo_index(int index) {
  const auto &specs = camera_view_specs();
  if (index < 0 || index >= static_cast<int>(specs.size())) return CameraViewKind::Road;
  return specs[static_cast<size_t>(index)].view;
}

struct CameraPaneTransientState {
  CameraPaneState state;
  std::string state_json;
  bool valid = false;
  std::array<CameraTextureState, 4> textures;
};

CameraPaneTransientState &camera_pane_transient_state(PaneInstance &pane) {
  if (CameraPaneTransientState *state = std::any_cast<CameraPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = CameraPaneTransientState{};
  return std::any_cast<CameraPaneTransientState &>(pane.transient_state);
}

CameraPaneState &camera_pane_state(PaneInstance &pane) {
  CameraPaneTransientState &transient = camera_pane_transient_state(pane);
  if (!transient.valid || transient.state_json != pane.state_json) {
    transient.state = parse_camera_pane_state(pane.state_json);
    transient.state_json = pane.state_json;
    transient.valid = true;
  }
  return transient.state;
}

ImU32 camera_timeline_color(TimelineSpanKind kind, uint8_t alpha) {
  const TimelineColor color = timeline_span_color(kind, alpha);
  return IM_COL32(color.r, color.g, color.b, color.a);
}

void upload_camera_frame(CameraTextureState *texture, DecodedCameraFrame frame, bool live_frame) {
  if (texture == nullptr) return;
  texture->live = live_frame;
  if (!frame.ok) {
    texture->error = frame.error;
    return;
  }
  if (frame.width <= 0 || frame.height <= 0 || frame.rgba.empty()) {
    texture->error = "empty decoded frame";
    return;
  }

  if (texture->texture == 0) glGenTextures(1, &texture->texture);
  glBindTexture(GL_TEXTURE_2D, texture->texture);
  if (texture->width != frame.width || texture->height != frame.height) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame.width, frame.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, frame.rgba.data());
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height,
                    GL_RGBA, GL_UNSIGNED_BYTE, frame.rgba.data());
  }
  glBindTexture(GL_TEXTURE_2D, 0);

  texture->width = frame.width;
  texture->height = frame.height;
  texture->key = frame.key;
  texture->frame_id = frame.frame_id;
  texture->timestamp = frame.timestamp;
  texture->error.clear();
}

bool draw_camera_controls(CameraPaneState *state) {
  if (state == nullptr) return false;
  bool changed = false;
  int index = camera_combo_index(state->view);
  ImGui::SetNextItemWidth(150.0f);
  if (ImGui::BeginCombo("View", camera_view_spec(state->view).label)) {
    const auto &specs = camera_view_specs();
    for (int i = 0; i < static_cast<int>(specs.size()); ++i) {
      const bool selected = i == index;
      if (ImGui::Selectable(specs[static_cast<size_t>(i)].label, selected)) {
        index = i;
        state->view = camera_from_combo_index(index);
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  if (ImGui::GetContentRegionAvail().x > 86.0f) ImGui::SameLine();
  changed = ImGui::Checkbox("Fit", &state->fit_to_pane) || changed;
  return changed;
}

void draw_camera_canvas(const CameraPaneSnapshot &snapshot,
                        bool fit_to_pane,
                        const CameraTextureState &texture,
                        const CameraDecodeStatus &decode_status,
                        const LiveCameraFrameStatus *live_status) {
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float width = std::max(240.0f, avail.x);
  const float height = std::max(180.0f, avail.y);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##camera_canvas", ImVec2(width, height));

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + width, pos.y + height);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(color_rgb(28, 30, 32)), 0.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(color_rgb(92, 96, 98)), 0.0f, 0, 1.0f);

  const bool live_mode = live_status != nullptr;
  const bool has_texture = texture.texture != 0 && texture.width > 0 && texture.height > 0 &&
                           texture.live == live_mode;
  const float aspect = has_texture ? static_cast<float>(texture.width) / static_cast<float>(texture.height)
                                   : 16.0f / 9.0f;
  ImVec2 video_min = pos;
  ImVec2 video_max = max;
  if (fit_to_pane) {
    float video_w = width;
    float video_h = video_w / aspect;
    if (video_h > height) {
      video_h = height;
      video_w = video_h * aspect;
    }
    video_min = ImVec2(pos.x + (width - video_w) * 0.5f, pos.y + (height - video_h) * 0.5f);
    video_max = ImVec2(video_min.x + video_w, video_min.y + video_h);
  }
  if (has_texture) {
    draw_list->AddImage(static_cast<ImTextureID>(texture.texture), video_min, video_max);
  } else {
    draw_list->AddRectFilled(video_min, video_max, ImGui::GetColorU32(color_rgb(38, 41, 43)), 0.0f);
  }
  draw_list->AddRect(video_min, video_max, ImGui::GetColorU32(color_rgb(116, 178, 255, 0.75f)), 0.0f, 0, 1.5f);

  const float info_width = std::max(0.0f, std::min(video_max.x - video_min.x - 22.0f,
                                                  (video_max.x - video_min.x) * 0.62f));
  const CameraRouteInfoLines info = build_camera_route_info_lines(
    snapshot, decode_status, live_status, texture, has_texture, info_width);

  const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
  const ImU32 muted = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  const ImVec2 info_min(video_min.x + 8.0f, video_min.y + 8.0f);
  const float line_height = ImGui::GetTextLineHeight();
  const float info_height = line_height * 2.8f;
  draw_list->AddRectFilled(info_min, ImVec2(info_min.x + info_width, info_min.y + info_height),
                          ImGui::GetColorU32(color_rgb(20, 22, 24, 0.60f)), 4.0f);
  draw_list->AddRect(info_min, ImVec2(info_min.x + info_width, info_min.y + info_height),
                     ImGui::GetColorU32(color_rgb(84, 87, 89, 0.85f)), 4.0f, 0, 1.0f);
  if (!info.line1.empty()) {
    draw_list->AddText(ImVec2(info_min.x + 7.0f, info_min.y + 4.0f), text, info.line1.c_str());
  }
  if (!info.line2.empty()) {
    draw_list->AddText(ImVec2(info_min.x + 7.0f, info_min.y + 22.0f), muted, info.line2.c_str());
  }
  if (!info.line3.empty()) {
    draw_list->AddText(ImVec2(info_min.x + 7.0f, info_min.y + 40.0f), muted, info.line3.c_str());
  }

  if (snapshot.overlay_kind != TimelineSpanKind::None) {
    const char *label = timeline_span_label(snapshot.overlay_kind);
    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const float pad_x = 10.0f;
    const float pad_y = 5.0f;
    const ImVec2 badge_max(video_max.x - 12.0f, video_max.y - 12.0f);
    const ImVec2 badge_min(std::max(video_min.x + 12.0f, badge_max.x - label_size.x - pad_x * 2.0f),
                           std::max(video_min.y + 12.0f, badge_max.y - label_size.y - pad_y * 2.0f));
    draw_list->AddRectFilled(badge_min, badge_max, camera_timeline_color(snapshot.overlay_kind, 210), 4.0f);
    draw_list->AddRect(badge_min, badge_max, camera_timeline_color(snapshot.overlay_kind, 255), 4.0f, 0, 1.0f);
    draw_list->AddText(ImVec2(badge_min.x + pad_x, badge_min.y + pad_y), IM_COL32(255, 255, 255, 255), label);
  }
}

CameraRouteInfoLines build_camera_route_info_lines(const CameraPaneSnapshot &snapshot,
                                                  const CameraDecodeStatus &decode_status,
                                                  const LiveCameraFrameStatus *live_status,
                                                  const CameraTextureState &texture,
                                                  bool has_texture,
                                                  float text_area_width) {
  const CameraViewSpec &spec = camera_view_spec(snapshot.view);
  CameraRouteInfoLines lines;
  const float info_width = std::max(0.0f, text_area_width);
  char buffer[256] = {};

  if (live_status != nullptr) {
    if (live_status->has_frame) {
      std::snprintf(buffer, sizeof(buffer), "%s  f%d  %s", spec.label, live_status->frame_id,
                    live_camera_stream_label(snapshot.view).c_str());
    } else {
      std::snprintf(buffer, sizeof(buffer), "%s live", spec.label);
    }
    lines.line1 = trim_overlay_text(buffer, info_width);

    if (!live_status->supported || !live_status->error.empty()) {
      lines.line2 = "VisionIPC unsupported";
    } else if (live_status->connected) {
      std::snprintf(buffer, sizeof(buffer), "camerad / %s  frame %zu",
                    live_camera_stream_label(snapshot.view).c_str(), live_status->received_frames);
      lines.line2 = trim_overlay_text(buffer, info_width);
    } else {
      lines.line2 = "waiting for camerad VisionIPC";
    }

    if (!live_status->supported || !live_status->error.empty()) {
      lines.line3 = trim_overlay_text(live_status->error, info_width);
    } else if (has_texture && texture.width > 0 && texture.height > 0) {
      std::snprintf(buffer, sizeof(buffer), "live %dx%d", texture.width, texture.height);
      lines.line3 = trim_overlay_text(buffer, info_width);
    } else if (!live_status->connected) {
      lines.line3 = "route video unavailable";
    } else {
      lines.line3 = "connected, waiting for frames";
    }

    if (lines.line2.empty()) lines.line2 = "camerad / live";
    return lines;
  }

  // seg/frame/t always come from the displayed texture, never the seek target (REVIEW defect B).
  const bool have_frame_info = has_texture && texture.key.has_value();
  if (have_frame_info || snapshot.has_frame) {
    if (have_frame_info) {
      std::snprintf(buffer, sizeof(buffer), "%s  seg %d  frame %u  t %.2f",
                    spec.label, texture.key->segment, texture.frame_id, texture.timestamp);
    } else {
      std::snprintf(buffer, sizeof(buffer), "%s  loading...", spec.label);
    }
    lines.line1 = trim_overlay_text(buffer, info_width);
    std::snprintf(buffer, sizeof(buffer), "%zu files  %zu frames",
                  snapshot.segment_file_count, snapshot.frame_count);
    lines.line2 = trim_overlay_text(buffer, info_width);
  } else if (snapshot.segment_file_count > 0) {
    std::snprintf(buffer, sizeof(buffer), "%s  no encode index", spec.label);
    lines.line1 = trim_overlay_text(buffer, info_width);
    std::snprintf(buffer, sizeof(buffer), "%zu files, %zu frames", snapshot.segment_file_count,
                  snapshot.frame_count);
    lines.line2 = trim_overlay_text(buffer, info_width);
  } else {
    std::snprintf(buffer, sizeof(buffer), "%s  no route video", spec.label);
    lines.line1 = trim_overlay_text(buffer, info_width);
    lines.line2 = "no route video for this view";
  }

  if (decode_status.loading) {
    std::snprintf(buffer, sizeof(buffer), "decoding...  cache %zu  queued %zu",
                  decode_status.cached_frames, decode_status.queued_frames);
  } else if (!decode_status.error.empty()) {
    std::snprintf(buffer, sizeof(buffer), "%s", decode_status.error.c_str());
  } else if (have_frame_info) {
    std::snprintf(buffer, sizeof(buffer), "decoded %dx%d", texture.width, texture.height);
  } else if (has_texture) {
    std::snprintf(buffer, sizeof(buffer), "cache %zu", decode_status.cached_frames);
  } else if (snapshot.ingest_failed) {
    std::snprintf(buffer, sizeof(buffer), "no camera data");  // terminal (REVIEW defect C)
  } else {
    std::snprintf(buffer, sizeof(buffer), "waiting for frame index");
  }
  lines.line3 = trim_overlay_text(buffer, info_width);
  return lines;
}

CameraPaneState parse_camera_pane_state(std::string_view state_json) {
  CameraPaneState state;
  if (state_json.empty()) return state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["camera_view"].is_string()) {
    state.view = camera_view_from_layout_name(json["camera_view"].string_value());
  }
  if (json["fit"].is_bool()) state.fit_to_pane = json["fit"].bool_value();
  return state;
}

std::string camera_pane_state_json(const CameraPaneState &state) {
  return json11::Json(json11::Json::object{
    {"camera_view", camera_view_layout_name(state.view)},
    {"fit", state.fit_to_pane},
  }).dump();
}

CameraPaneSnapshot prepare_camera_pane_snapshot(const Session &session, const CameraPaneState &state) {
  CameraPaneSnapshot snapshot;
  snapshot.view = state.view;
  snapshot.overlay_kind = session.timeline.kind_at_time(session.playback.tracker_time());
  const CameraFeedIndex &index = session.camera_index(state.view);
  snapshot.segment_file_count = index.segment_files.size();
  snapshot.frame_count = index.entries.size();
  snapshot.has_frame = camera_frame_at_time(index, session.playback.tracker_time()).has_value();
  snapshot.ingest_failed = session.ingest_status().state == RouteIngestState::Failed;
  return snapshot;
}

}  // namespace

void draw_camera_pane(Session &session, PaneInstance &pane) {
  CameraPaneTransientState &transient = camera_pane_transient_state(pane);
  CameraPaneState &state = camera_pane_state(pane);
  if (draw_camera_controls(&state)) pane.state_json = camera_pane_state_json(state);
  transient.state_json = pane.state_json;

  const CameraPaneSnapshot snapshot = prepare_camera_pane_snapshot(session, state);
  std::array<CameraTextureState, 4> &textures = transient.textures;
  CameraTextureState &texture = textures[camera_view_index(state.view)];

  if (session.config.stream) {
    LiveCameraFrameSource &source = session.live_camera_source;
    source.request_frame(state.view);
    if (std::optional<DecodedCameraFrame> frame = source.take_frame(state.view)) {
      upload_camera_frame(&texture, std::move(*frame), true);
    }
    const LiveCameraFrameStatus live_status = source.status(state.view);
    ImGui::TextDisabled("Live VisionIPC | %s", live_status.connected ? "connected" : "waiting");
    draw_camera_canvas(snapshot, state.fit_to_pane, texture, CameraDecodeStatus{}, &live_status);
    return;
  }

  CameraFrameDecoder &decoder = session.camera_decoder(state.view);
  decoder.request_frame(session.playback.tracker_time());
  if (std::optional<DecodedCameraFrame> frame = decoder.take_frame()) {
    upload_camera_frame(&texture, std::move(*frame), false);
  }
  const CameraDecodeStatus decode_status = decoder.status();

  ImGui::TextDisabled("%zu files | %zu frames", snapshot.segment_file_count, snapshot.frame_count);
  draw_camera_canvas(snapshot, state.fit_to_pane, texture, decode_status, nullptr);
}

}  // namespace loggy
