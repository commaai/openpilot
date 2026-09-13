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

class PopupViewportScope {
public:
  PopupViewportScope() : main_(ImGui::GetMainViewport()), flags_(main_->Flags) {
    // Keep detached popups above their owner; reset ownership when docked.
    ImGuiWindowClass window_class;
    const ImGuiViewport *owner = ImGui::GetWindowViewport();
    if (owner != main_) {
      // Popups ignore NoAutoMerge, so exclude the main viewport during Begin.
      main_->Flags &= ~ImGuiViewportFlags_CanHostOtherWindows;
      window_class.ParentViewportId = owner->ID;
    }
    ImGui::SetNextWindowClass(&window_class);
  }
  ~PopupViewportScope() { main_->Flags = flags_; }

private:
  ImGuiViewport *main_;
  ImGuiViewportFlags flags_;
};

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

inline void PositionBelowItem(const char *id, bool align_right = false) {
  const ImRect anchor(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  char name[32];
  ImFormatString(name, IM_ARRAYSIZE(name), "##Popup_%08x", ImGui::GetID(id));
  if (ImGuiWindow *popup = ImGui::FindWindowByName(name); popup && popup->WasActive) {
    // Place like a combo using the owner's current monitor bounds.
    const auto *viewport = static_cast<ImGuiViewportP *>(ImGui::GetWindowViewport());
    ImRect bounds = viewport->GetMainRect();
    if ((ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) && viewport->PlatformMonitor >= 0) {
      const auto &monitor = ImGui::GetPlatformIO().Monitors[viewport->PlatformMonitor];
      bounds = ImRect(monitor.WorkPos, ImVec2(monitor.WorkPos.x + monitor.WorkSize.x, monitor.WorkPos.y + monitor.WorkSize.y));
    }
    bounds.Expand(ImVec2(-ImGui::GetStyle().DisplaySafeAreaPadding.x, -ImGui::GetStyle().DisplaySafeAreaPadding.y));
    ImGuiDir direction = align_right ? ImGuiDir_Left : ImGuiDir_Down;
    const ImVec2 pos = ImGui::FindBestWindowPosForPopupEx(anchor.GetBL(), ImGui::CalcWindowNextAutoFitSize(popup),
                                                       &direction, bounds, anchor, ImGuiPopupPositionPolicy_ComboBox);
    ImGui::SetNextWindowPos(pos);
  } else {
    ImGui::SetNextWindowPos(align_right ? anchor.GetBR() : anchor.GetBL(), ImGuiCond_Always, ImVec2(align_right ? 1.0f : 0.0f, 0));
  }
}

inline bool BeginPopup(const char *id, ImGuiWindowFlags flags = 0) {
  PopupViewportScope viewport_scope;
  pushStyle();
  return finishBegin(ImGui::BeginPopup(id, flags));
}
inline bool BeginPopupContextItem(const char *id = nullptr, ImGuiPopupFlags flags = ImGuiPopupFlags_MouseButtonRight) {
  PopupViewportScope viewport_scope;
  pushStyle();
  return finishBegin(ImGui::BeginPopupContextItem(id, flags));
}
inline void EndPopup() {
  ImGui::PopStyleVar();
  ImGui::EndPopup();
  ImGui::PopStyleVar(WINDOW_STYLE_VARS);
}
inline bool BeginMenu(const char *label, bool enabled = true) {
  PopupViewportScope viewport_scope;
  pushStyle();
  return finishBegin(ImGui::BeginMenu(label, enabled));
}
inline void EndMenu() {
  ImGui::PopStyleVar();
  ImGui::EndMenu();
  ImGui::PopStyleVar(WINDOW_STYLE_VARS);
}
inline bool BeginCombo(const char *label, const char *preview, ImGuiComboFlags flags = 0) {
  PopupViewportScope viewport_scope;
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
