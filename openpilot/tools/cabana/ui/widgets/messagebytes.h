#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/core/color.h"

// the cells of the messages and history log tables. Only valid inside a frame: the sizes come from the
// mono font.

ImVec2 byteCellSize();                              // one "00 " cell
ImVec2 bytesCellSize(int n, bool multiple_lines);   // a cell of n bytes, the table's cell padding included
ImU32 cellTextColor(bool selected, bool inactive);  // inactive rows gray the text and fade the highlighted text

void drawTextCell(ImDrawList *dl, const ImRect &rect, const std::string &text, bool selected, bool inactive);
void drawBytesCell(ImDrawList *dl, const ImRect &rect, const std::vector<uint8_t> &bytes, const std::vector<CabanaColor> *colors,
                   bool selected, bool inactive, bool multiple_lines);
