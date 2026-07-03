// F1 help overlay -- ImGui port of tools/cabana/mainwin.cc's HelpOverlay.
//
// Qt's HelpOverlay is a translucent QWidget that installs an event filter on
// MainWindow, fills itself with QColor(0,0,0,50), then for each of
// {MessagesWidget, BinaryView, SignalView, ChartsWidget, VideoWidget} looks
// up `w->whatsThis()` and draws it in a small tooltip-colored box centered
// over that widget; a mouse release anywhere closes it.
//
// This port keeps the same idea -- dim the whole viewport, then draw a
// labeled callout centered on each dock window's *live* rect (queried via
// ImGui::FindWindowByName using the exact titles app.h declares) -- with one
// structural adjustment: in this ImGui port BinaryView and SignalView (the
// "SignalEditor") aren't separate top-level widgets, they're panes stacked
// inside the single "Detail" dock window (see detail_panel.cc draw_msg_tab),
// and the playback toolbar that lived inside Qt's VideoWidget is now its own
// fixed bar at the bottom of the whole app (draw_transport_bar in
// transport.cc), not part of the Video dock. So:
//   - "Detail" callout folds BinaryView's + SignalView's whatsThis together
//     (both really do live in that one window here).
//   - "Video" keeps just the timeline-color legend from VideoWidget's text.
//   - "Transport" is a new callout (Qt had no separate widget for it) that
//     carries the "Pause/Resume: space" shortcut Qt's VideoWidget whatsThis
//     used to include, since the play/pause button now lives there instead.
// Help text is plain (no HTML like Qt's whatsThis) since this UI has no rich
// text renderer; every line below is a faithful, concise transcription of
// the corresponding setWhatsThis(...) string. Click anywhere or Esc closes.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>

#include "imgui_internal.h"  // ImGui::FindWindowByName, ImGuiWindow, ImRect

namespace {

struct HelpOverlayState {
  bool active = false;
  bool just_opened = false;  // suppress the "click closes" check on the opening frame
};
HelpOverlayState g_help;

struct Callout {
  const char *window_title;  // exact ImGui window title to look up via FindWindowByName
  const char *heading;
  std::vector<std::string> lines;
};

// tools/cabana/messageswidget.cc setWhatsThis()
const Callout &messages_callout() {
  static const Callout c{
      MESSAGES_WINDOW_TITLE,
      "Messages",
      {
          "Byte color: gray = constant, blue = increasing, red = decreasing",
          "Shortcuts",
          "  Horizontal Scrolling:  Shift+Wheel",
      },
  };
  return c;
}

// tools/cabana/binaryview.cc + tools/cabana/signalview.cc setWhatsThis()
// (folded together: both panes live inside this single dock window here)
const Callout &detail_callout() {
  static const Callout c{
      CENTER_WINDOW_TITLE,
      "Detail",
      {
          "Binary View",
          "  Delete Signal:  x / Backspace / Delete",
          "  Change endianness:  e",
          "  Change signedness:  s",
          "  Open chart:  c / p / g",
          "Signal Editor",
          "  Edit name, size, endianness, factor/offset and bit range;",
          "  drag on the bits above to create or resize a signal.",
      },
  };
  return c;
}

// tools/cabana/videowidget.cc setWhatsThis() (timeline legend only --
// Pause/Resume moved to the Transport callout below)
const Callout &video_callout() {
  static const Callout c{
      VIDEO_WINDOW_TITLE,
      "Video",
      {
          "Timeline color",
          "  Disengaged / Engaged",
          "  User Flag / Info",
          "  Warning / Critical",
      },
  };
  return c;
}

// tools/cabana/chart/chartswidget.cc setWhatsThis()
const Callout &charts_callout() {
  static const Callout c{
      CHARTS_WINDOW_TITLE,
      "Charts",
      {
          "Click:  seek to a corresponding time",
          "Drag:  zoom into the chart",
          "Shift + Drag:  scrub through the chart to view values",
          "Right Mouse:  open the context menu",
      },
  };
  return c;
}

// New in this port -- transport.cc is a standalone bar (Qt's playback
// toolbar lived inside VideoWidget, whose whatsThis carried this shortcut).
const Callout &transport_callout() {
  static const Callout c{
      "##transport_bar",
      "Transport",
      {
          "Pause/Resume:  space",
          "Drag the seek bar to scrub; click the time to toggle",
          "elapsed/absolute display; the dropdown sets playback speed.",
      },
  };
  return c;
}

ImRect viewport_rect() {
  const ImGuiViewport *vp = ImGui::GetMainViewport();
  return ImRect(vp->Pos, ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y));
}

// Centers a small tooltip-colored box (mirrors palette().toolTipBase() in
// Qt's drawHelpForWidget) over `target`'s live rect, clamped to stay fully
// on-screen, and draws `heading` + `lines` inside it.
void draw_callout(ImDrawList *dl, const ImRect &target, const ImRect &clamp_rect, const Callout &c) {
  push_bold_font();
  ImFont *heading_font = ImGui::GetFont();
  const float heading_size = ImGui::GetFontSize();
  const ImVec2 heading_text_size = heading_font->CalcTextSizeA(heading_size, FLT_MAX, 0.0f, c.heading);
  pop_bold_font();

  ImFont *body_font = ImGui::GetFont();
  const float body_size = ImGui::GetFontSize();
  const float line_h = ImGui::GetTextLineHeightWithSpacing();

  const ImVec2 pad(12.0f, 10.0f);
  float box_w = heading_text_size.x;
  for (const std::string &l : c.lines) box_w = std::max(box_w, ImGui::CalcTextSize(l.c_str()).x);
  box_w += pad.x * 2.0f;
  const float box_h = heading_text_size.y + pad.y * 0.5f + static_cast<float>(c.lines.size()) * line_h + pad.y * 2.0f;

  const ImVec2 center((target.Min.x + target.Max.x) * 0.5f, (target.Min.y + target.Max.y) * 0.5f);
  ImVec2 box_min(center.x - box_w * 0.5f, center.y - box_h * 0.5f);
  box_min.x = std::clamp(box_min.x, clamp_rect.Min.x + 8.0f, std::max(clamp_rect.Min.x + 8.0f, clamp_rect.Max.x - box_w - 8.0f));
  box_min.y = std::clamp(box_min.y, clamp_rect.Min.y + 8.0f, std::max(clamp_rect.Min.y + 8.0f, clamp_rect.Max.y - box_h - 8.0f));
  const ImVec2 box_max(box_min.x + box_w, box_min.y + box_h);

  dl->AddRectFilled(box_min, box_max, ImGui::GetColorU32(ImGuiCol_PopupBg), 4.0f);
  dl->AddRect(box_min, box_max, ImGui::GetColorU32(ImGuiCol_Border), 4.0f);

  ImVec2 cursor(box_min.x + pad.x, box_min.y + pad.y);
  dl->AddText(heading_font, heading_size, cursor, ImGui::GetColorU32(ImGuiCol_Text), c.heading);
  cursor.y += heading_text_size.y + pad.y * 0.5f;
  const ImU32 body_col = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  for (const std::string &l : c.lines) {
    dl->AddText(body_font, body_size, cursor, body_col, l.c_str());
    cursor.y += line_h;
  }

  // Faint highlight border around the panel this callout describes -- mirrors
  // the callout being visually anchored to `w` in Qt (there via position
  // only; the extra outline makes the association clearer in ImGui, where
  // panels sit edge-to-edge with no gap for the eye to use instead).
  dl->AddRect(target.Min, target.Max, ImGui::GetColorU32(ImGuiCol_TabSelectedOverline), 0.0f, 0, 2.0f);
}

void draw_one(ImDrawList *dl, const ImRect &clamp_rect, const Callout &c) {
  ImGuiWindow *win = ImGui::FindWindowByName(c.window_title);
  if (win == nullptr || !win->WasActive || win->Hidden) return;
  const ImRect rect(win->Pos, ImVec2(win->Pos.x + win->Size.x, win->Pos.y + win->Size.y));
  draw_callout(dl, rect, clamp_rect, c);
}

}  // namespace

void toggle_help_overlay() {
  g_help.active = !g_help.active;
  g_help.just_opened = g_help.active;
}

void draw_help_overlay(AppState & /*app*/) {
  if (!g_help.active) return;

  const ImRect vp_rect = viewport_rect();
  ImDrawList *dl = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());

  // Qt: painter.fillRect(rect(), QColor(0, 0, 0, 50));
  dl->AddRectFilled(vp_rect.Min, vp_rect.Max, IM_COL32(0, 0, 0, 50));

  draw_one(dl, vp_rect, messages_callout());
  draw_one(dl, vp_rect, detail_callout());
  draw_one(dl, vp_rect, video_callout());
  draw_one(dl, vp_rect, charts_callout());
  draw_one(dl, vp_rect, transport_callout());

  const bool was_just_opened = g_help.just_opened;
  g_help.just_opened = false;
  const bool close_click = !was_just_opened && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
  const bool close_esc = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
  if (close_click || close_esc) {
    g_help.active = false;
  }
}
