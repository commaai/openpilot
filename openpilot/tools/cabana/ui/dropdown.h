#pragma once

#include <cstring>
#include <vector>

#include "tools/cabana/ui/theme.h"

namespace dropdown {
constexpr float PADDING_X = 9.0f;
constexpr float PADDING_Y = 6.0f;
constexpr float SPACING_X = 10.0f;
constexpr float SPACING_Y = 8.0f;
constexpr float ROUNDING = 6.0f;
constexpr float BORDER = 1.0f;
constexpr int WINDOW_STYLE_VARS = 3;

inline void pushStyle() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(PADDING_X, PADDING_Y));
  ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, ROUNDING);
  ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, BORDER);
}

inline bool finishBegin(bool open) {
  if (open) ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(SPACING_X, SPACING_Y));
  else ImGui::PopStyleVar(WINDOW_STYLE_VARS);
  return open;
}
inline bool BeginPopup(const char *id, ImGuiWindowFlags flags = 0) {
  pushStyle();
  return finishBegin(ImGui::BeginPopup(id, flags));
}
inline bool BeginPopupContextItem(const char *id = nullptr, ImGuiPopupFlags flags = ImGuiPopupFlags_MouseButtonRight) {
  pushStyle();
  return finishBegin(ImGui::BeginPopupContextItem(id, flags));
}
inline void EndPopup() {
  ImGui::PopStyleVar();
  ImGui::EndPopup();
  ImGui::PopStyleVar(WINDOW_STYLE_VARS);
}
inline bool BeginMenu(const char *label, bool enabled = true) {
  pushStyle();
  return finishBegin(ImGui::BeginMenu(label, enabled));
}
inline void EndMenu() {
  ImGui::PopStyleVar();
  ImGui::EndMenu();
  ImGui::PopStyleVar(WINDOW_STYLE_VARS);
}
inline bool BeginCombo(const char *label, const char *preview, ImGuiComboFlags flags = 0) {
  pushStyle();
  return finishBegin(ImGui::BeginCombo(label, preview, flags));
}
inline void EndCombo() {
  ImGui::PopStyleVar();
  ImGui::EndCombo();
  ImGui::PopStyleVar(WINDOW_STYLE_VARS);
}

inline bool Item(const char *label, const char *shortcut = nullptr, bool selected = false, bool enabled = true) {
  return ImGui::MenuItem(label, shortcut, selected, enabled);
}
inline bool Item(const char *label, const char *shortcut, bool *selected, bool enabled = true) {
  if (!Item(label, shortcut, selected && *selected, enabled)) return false;
  if (selected) *selected = !*selected;
  return true;
}

inline bool Combo(const char *label, int *index, const char *const items[], int count) {
  bool changed = false;
  if (BeginCombo(label, *index >= 0 && *index < count ? items[*index] : "")) {
    for (int i = 0; i < count; ++i) {
      ImGui::PushID(i);
      if (Item(items[i], nullptr, i == *index) && i != *index) {
        *index = i;
        changed = true;
      }
      if (i == *index) ImGui::SetItemDefaultFocus();
      ImGui::PopID();
    }
    EndCombo();
  }
  return changed;
}
inline bool Combo(const char *label, int *index, const char *items) {
  std::vector<const char *> labels;
  for (const char *item = items; *item; item += std::strlen(item) + 1) labels.push_back(item);
  return Combo(label, index, labels.data(), labels.size());
}
}  // namespace dropdown
