#pragma once

#include "common/util.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

inline ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(static_cast<float>(r) / 255.0f,
                static_cast<float>(g) / 255.0f,
                static_cast<float>(b) / 255.0f,
                alpha);
}

inline ImVec4 color_rgb(const std::array<uint8_t, 3> &color, float alpha = 1.0f) {
  return color_rgb(color[0], color[1], color[2], alpha);
}

inline std::string lowercase_copy(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline int imgui_resize_callback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag != ImGuiInputTextFlags_CallbackResize || data->UserData == nullptr) return 0;
  auto *text = static_cast<std::string *>(data->UserData);
  text->resize(static_cast<size_t>(data->BufTextLen));
  data->Buf = text->data();
  return 0;
}

inline bool input_text_string(const char *label,
                              std::string *text,
                              ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputText(label, text->data(), text->capacity() + 1,
                          flags, imgui_resize_callback, text);
}

inline bool input_text_with_hint_string(const char *label,
                                        const char *hint,
                                        std::string *text,
                                        ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputTextWithHint(label, hint, text->data(), text->capacity() + 1,
                                  flags, imgui_resize_callback, text);
}

inline bool input_text_multiline_string(const char *label,
                                        std::string *text,
                                        const ImVec2 &size = ImVec2(0.0f, 0.0f),
                                        ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputTextMultiline(label, text->data(), text->capacity() + 1,
                                   size, flags, imgui_resize_callback, text);
}

inline bool is_local_stream_address(std::string_view address) {
  return address.empty() || address == "127.0.0.1" || address == "localhost";
}

inline void ensure_parent_dir(const std::filesystem::path &path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

inline std::string shell_quote(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

struct CommandResult {
  int exit_code = 0;
  std::string output;
};

std::string read_file_or_throw(const std::filesystem::path &path);
void write_file_or_throw(const std::filesystem::path &path, std::string_view contents);
void write_file_or_throw(const std::filesystem::path &path, const void *data, size_t size);
void run_system_or_throw(const std::string &command, std::string_view action);
CommandResult run_process_capture_output(const std::vector<std::string> &args);
