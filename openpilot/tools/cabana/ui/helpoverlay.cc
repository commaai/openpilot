#include "tools/cabana/ui/helpoverlay.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdio>
#include <cstring>

#include "tools/cabana/ui/util.h"

namespace {
struct HelpRun {
  std::string text;
  bool bold = false;
  bool chip = false;    // background-color:lightGray
  bool swatch = false;  // a colored square
  ImU32 color = 0;      // 0 = default text color
};

ImU32 helpColor(const std::string &name) {
  if (name == "gray") return IM_COL32(128, 128, 128, 255);
  if (name == "blue") return IM_COL32(0, 0, 255, 255);
  if (name == "red") return IM_COL32(255, 0, 0, 255);
  unsigned rgb = 0;
  if (name.size() == 7 && name[0] == '#' && sscanf(name.c_str() + 1, "%6x", &rgb) == 1) {
    return IM_COL32((rgb >> 16) & 0xff, (rgb >> 8) & 0xff, rgb & 0xff, 255);
  }
  return 0;
}

std::vector<std::vector<HelpRun>> parseHelpHtml(const std::string &raw) {
  std::vector<std::vector<HelpRun>> lines(1);
  HelpRun style;
  std::vector<HelpRun> span_stack;
  bool prev_space = true;  // html collapses whitespace; leading whitespace is dropped
  std::string pending;
  auto flush = [&]() {
    if (!pending.empty()) {
      HelpRun run = style;
      run.text = pending;
      lines.back().push_back(run);
      pending.clear();
    }
  };
  auto push_swatch = [&](ImU32 color) {
    flush();
    HelpRun run = style;
    run.swatch = true;
    if (color) run.color = color;
    lines.back().push_back(run);
    prev_space = false;
  };
  for (size_t i = 0; i < raw.size(); ++i) {
    const char c = raw[i];
    if (c == '<') {
      const size_t close = raw.find('>', i);
      if (close == std::string::npos) break;
      const std::string tag = raw.substr(i + 1, close - i - 1);
      i = close;
      if (tag.compare(0, 3, "!--") == 0) continue;
      flush();
      if (tag == "b") {
        style.bold = true;
      } else if (tag == "/b") {
        style.bold = false;
      } else if (tag.compare(0, 2, "br") == 0) {
        lines.emplace_back();
        prev_space = true;
      } else if (tag.compare(0, 4, "span") == 0) {
        span_stack.push_back(style);
        const size_t st = tag.find("style=\"");
        if (st != std::string::npos) {
          const std::string css = tag.substr(st + 7, tag.find('"', st + 7) - st - 7);
          size_t pos = 0;
          while (pos < css.size()) {
            const size_t semi = css.find(';', pos);
            const std::string decl = css.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
            const size_t colon = decl.find(':');
            if (colon != std::string::npos) {
              const std::string key = decl.substr(0, colon), value = decl.substr(colon + 1);
              if (key == "color") style.color = helpColor(value);
              if (key == "background-color") style.chip = true;
            }
            if (semi == std::string::npos) break;
            pos = semi + 1;
          }
        }
      } else if (tag == "/span") {
        if (!span_stack.empty()) {
          style = span_stack.back();
          span_stack.pop_back();
        }
      }
    } else if (c == '&') {
      static const std::pair<const char *, const char *> entities[] = {{"&nbsp;", " "}};
      bool matched = false;
      for (const auto &[name, text] : entities) {
        if (raw.compare(i, strlen(name), name) == 0) {
          pending += text;
          prev_space = false;
          i += strlen(name) - 1;
          matched = true;
          break;
        }
      }
      if (!matched && raw.compare(i, 7, "&#9632;") == 0) {  // the filled square of the byte color legend
        push_swatch(0);
        i += 6;
        matched = true;
      }
      if (!matched) {
        pending += c;
        prev_space = false;
      }
    } else if (isspace(static_cast<unsigned char>(c))) {
      if (!prev_space) pending += ' ';
      prev_space = true;
    } else if (c == '#' && i + 6 < raw.size() && helpColor(raw.substr(i, 7)) != 0) {  // #rrggbb legend token
      push_swatch(helpColor(raw.substr(i, 7)));
      i += 6;
    } else {
      pending += c;
      prev_space = false;
    }
  }
  flush();
  for (auto &line : lines) {  // trim the collapsed whitespace at the line ends
    if (!line.empty() && !line.back().text.empty() && line.back().text.back() == ' ') line.back().text.pop_back();
    if (!line.empty() && !line.front().text.empty() && line.front().text.front() == ' ') line.front().text.erase(0, 1);
  }
  while (!lines.empty() && lines.back().empty()) lines.pop_back();
  return lines;
}
}  // namespace

void HelpOverlay::toggle() {
  visible_ = !visible_;
  opened_frame_ = ImGui::GetFrameCount();
}

void HelpOverlay::add(const std::string &text, const ImRect &rect) {
  if (visible_) texts_.push_back({text, rect, ImGui::GetWindowViewport()});
}

void HelpOverlay::draw() {
  if (!visible_) return;
  // Each panel belongs to a viewport; detached panels need their own foreground layer.
  std::vector<ImGuiViewport *> viewports{ImGui::GetMainViewport()};
  for (const auto &entry : texts_) {
    if (std::find(viewports.begin(), viewports.end(), entry.viewport) == viewports.end()) viewports.push_back(entry.viewport);
  }
  for (auto *viewport : viewports) {
    ImGui::GetForegroundDrawList(viewport)->AddRectFilled(viewport->Pos, ImVec2(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y), IM_COL32(0, 0, 0, 50));
  }
  ImFont *font = ImGui::GetFont();
  ImFont *bold_font = boldFont() ? boldFont() : font;
  const float font_size = ImGui::GetFontSize();
  const float line_h = ImGui::GetTextLineHeightWithSpacing();
  auto run_width = [&](const HelpRun &r) {
    if (r.swatch) return font_size;
    return (r.bold ? bold_font : font)->CalcTextSizeA(font_size, FLT_MAX, 0.0f, r.text.c_str()).x;
  };
  for (const auto &[raw, rect, viewport] : texts_) {
    ImDrawList *dl = ImGui::GetForegroundDrawList(viewport);
    if (raw.empty()) continue;
    const auto lines = parseHelpHtml(raw);
    float width = 0;
    for (const auto &line : lines) {
      float w = 0;
      for (const auto &r : line) w += run_width(r);
      width = std::max(width, w);
    }
    const ImVec2 size(width, lines.size() * line_h);
    const ImVec2 center((rect.Min.x + rect.Max.x) * 0.5f, (rect.Min.y + rect.Max.y) * 0.5f);
    const ImVec2 min(center.x - size.x * 0.5f - 8.0f, center.y - size.y * 0.5f - 8.0f);
    const ImVec2 max(center.x + size.x * 0.5f + 8.0f, center.y + size.y * 0.5f + 8.0f);
    dl->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_PopupBg), ImGui::GetStyle().PopupRounding);
    dl->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Border), ImGui::GetStyle().PopupRounding);
    float y = min.y + 8.0f;
    for (const auto &line : lines) {
      float x = min.x + 8.0f;
      for (const auto &r : line) {
        const float w = run_width(r);
        const ImU32 color = r.color ? r.color : ImGui::GetColorU32(ImGuiCol_Text);
        if (r.swatch) {
          dl->AddRectFilled(ImVec2(x + 2, y + 3), ImVec2(x + font_size - 2, y + font_size - 1), color);
        } else {
          if (r.chip) dl->AddRectFilled(ImVec2(x, y), ImVec2(x + w, y + font_size), ImGui::GetColorU32(ImGuiCol_Button), 3.0f);
          dl->AddText(r.bold ? bold_font : font, font_size, ImVec2(x, y), color, r.text.c_str());
        }
        x += w;
      }
      y += line_h;
    }
  }
  texts_.clear();
  // ignore the release of the click that opened the overlay
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ImGui::GetFrameCount() != opened_frame_) visible_ = false;
}
