#pragma once

#include <string>

#include "imgui.h"

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

inline bool comboBox(const char *label, int *index, const char *const items[], int count) {
  return ImGui::Combo(label, index, items, count);
}

void pushMonoFont();
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
