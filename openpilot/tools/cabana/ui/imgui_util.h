#pragma once

#include <string>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/color.h"

inline ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

inline ImU32 toImU32(const CabanaColor &c) { return IM_COL32(c.r, c.g, c.b, c.a); }
inline ImVec4 toImVec4(const CabanaColor &c) { return ImVec4(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f); }

inline int imguiResizeCallback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    auto *s = static_cast<std::string *>(data->UserData);
    s->resize(data->BufTextLen);
    data->Buf = s->data();
  }
  return 0;
}

inline bool inputText(const char *label, std::string *s, const char *hint = "", ImGuiInputTextFlags flags = 0) {
  return ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1,
                                  flags | ImGuiInputTextFlags_CallbackResize, imguiResizeCallback, s);
}

// auto-raise icon button with a tooltip
inline bool toolButton(const char *id, const char *icon, const char *tooltip = nullptr, const char *text = nullptr,
                       float width = 0.0f) {
  std::string label = text && *text ? std::string(icon) + " " + text + "###" + id : std::string(icon) + "###" + id;
  // no frame, transparent until hovered
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  bool clicked = ImGui::Button(label.c_str(), ImVec2(width, 0));
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  if (tooltip && *tooltip) ImGui::SetItemTooltip("%s", tooltip);
  return clicked;
}

// Escape closes a dialog only when nothing is open above it: a combo drops its list first.
inline bool dialogEscapePressed() {
  if (!ImGui::IsKeyPressed(ImGuiKey_Escape, false)) return false;
  ImGuiContext &g = *GImGui;
  return g.OpenPopupStack.Size > 0 && g.OpenPopupStack.back().Window == ImGui::GetCurrentWindow();
}

// [Cancel] [Accept], right aligned. reject_label = nullptr for an accept-only box.
inline bool dialogButtons(const char *accept_label, bool *accepted, bool *rejected, bool accept_enabled = true,
                          const char *reject_label = "Cancel") {
  const float button_width = 80.0f;
  const int count = reject_label ? 2 : 1;
  const float total = button_width * count + ImGui::GetStyle().ItemSpacing.x * (count - 1);
  const float avail = ImGui::GetContentRegionAvail().x;
  if (avail > total) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - total);
  bool pressed = false;
  if (reject_label) {
    if (ImGui::Button(reject_label, ImVec2(button_width, 0.0f))) {
      if (rejected) *rejected = true;
      pressed = true;
    }
    ImGui::SameLine();
  }
  ImGui::BeginDisabled(!accept_enabled);
  if (ImGui::Button(accept_label, ImVec2(button_width, 0.0f))) {
    if (accepted) *accepted = true;
    pressed = true;
  }
  ImGui::EndDisabled();
  return pressed;
}

// horizontal header labels are centered
inline void tableHeadersRow() {
  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  for (int c = 0, count = ImGui::TableGetColumnCount(); c < count; ++c) {
    if (!ImGui::TableSetColumnIndex(c)) continue;
    const char *name = ImGui::TableGetColumnName(c);
    if (!name) name = "";
    const float offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(name).x) * 0.5f;
    if (offset > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::PushID(c);
    ImGui::TableHeader(name);
    ImGui::PopID();
  }
}

// a 16px box vertically centered in the frame height so rows keep their layout; ImGui::Checkbox draws a
// frame height (22 px) square.
const float CHECKBOX_SIZE = 16.0f;
inline bool checkBox(const char *label, bool *v) {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImGuiID id = window->GetID(label);
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  const float frame_h = ImGui::GetFrameHeight();
  const ImVec2 pos = window->DC.CursorPos;
  const ImRect total_bb(pos, ImVec2(pos.x + CHECKBOX_SIZE + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), pos.y + frame_h));
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if (!ImGui::ItemAdd(total_bb, id)) return false;
  bool hovered, held;
  const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
  if (pressed) {
    *v = !*v;
    ImGui::MarkItemEdited(id);
  }
  const float y = pos.y + IM_TRUNC((frame_h - CHECKBOX_SIZE) * 0.5f);
  const ImRect check_bb(ImVec2(pos.x, y), ImVec2(pos.x + CHECKBOX_SIZE, y + CHECKBOX_SIZE));
  ImGui::RenderNavCursor(total_bb, id);
  const ImU32 bg = ImGui::GetColorU32((held && hovered) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
  ImGui::RenderFrame(check_bb.Min, check_bb.Max, bg, true, style.FrameRounding);
  if (*v) {
    const float pad = ImMax(1.0f, IM_TRUNC(CHECKBOX_SIZE / 6.0f));
    ImGui::RenderCheckMark(window->DrawList, ImVec2(check_bb.Min.x + pad, check_bb.Min.y + pad), ImGui::GetColorU32(ImGuiCol_CheckMark), CHECKBOX_SIZE - pad * 2.0f);
  }
  if (label_size.x > 0.0f) ImGui::RenderText(ImVec2(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + style.FramePadding.y), label);
  return pressed;
}

inline bool comboBox(const char *label, int *index, const char *const items[], int count) {
  return ImGui::Combo(label, index, items, count);
}

ImU32 highlightedTextColor();
ImU32 paletteBrightText();

// the next window is a real OS window instead of being drawn inside the main one
inline void setNextWindowFloatsOut() {
  ImGuiWindowClass window_class;
  window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
  ImGui::SetNextWindowClass(&window_class);
}

// full width groove, filled left of the handle, 13x13 handle (style.cc)
bool fusionSliderInt(const char *label, int *v, int min, int max, float width);

void pushMonoFont();
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
