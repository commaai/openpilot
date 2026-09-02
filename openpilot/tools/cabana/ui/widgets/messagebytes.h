#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/core/color.h"

class MessageBytesDelegate {
public:
  MessageBytesDelegate(bool multiple_lines = false) : multiple_lines_(multiple_lines) {}
  // text cells elide, the bytes cell paints colored hex
  void paint(ImDrawList *painter, const ImRect &rect, bool selected, bool inactive, const std::string &text,
             const std::vector<uint8_t> *bytes = nullptr, const std::vector<CabanaColor> *colors = nullptr) const;
  ImVec2 sizeHint(const std::vector<uint8_t> *bytes) const;
  bool multipleLines() const { return multiple_lines_; }
  void setMultipleLines(bool v) { multiple_lines_ = v; }
  ImVec2 sizeForBytes(int n) const;
  void updateFontMetrics();  // the fonts are only valid inside a frame

private:
  ImVec2 byte_size_ = {};
  bool multiple_lines_ = false;
};
