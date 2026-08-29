#pragma once

#include <string>

#include "imgui.h"
#include "imgui_internal.h"

inline ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

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

// Escape closes a dialog only when nothing is open above it. Qt gives Escape to the top-most popup
// (a combo drops its list without closing the dialog); imgui does not, so check the popup stack.
inline bool dialogEscapePressed() {
  if (!ImGui::IsKeyPressed(ImGuiKey_Escape, false)) return false;
  ImGuiContext &g = *GImGui;
  return g.OpenPopupStack.Size > 0 && g.OpenPopupStack.back().Window == ImGui::GetCurrentWindow();
}

// QDialogButtonBox: [Cancel] [Accept], right aligned. reject_label = nullptr for an accept-only box.
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

// QHeaderView::defaultAlignment: horizontal header labels are centered
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

inline bool comboBox(const char *label, int *index, const char *const items[], int count) {
  return ImGui::Combo(label, index, items, count);
}

ImU32 highlightedTextColor();  // QPalette::HighlightedText
void pushMonoFont();
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
