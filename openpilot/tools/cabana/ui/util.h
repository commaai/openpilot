#pragma once

#include <functional>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/color.h"
#include "tools/cabana/utils/util.h"

struct GLFWwindow;

inline ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}

inline ImU32 toImU32(const CabanaColor &c) { return IM_COL32(c.r, c.g, c.b, c.a); }
inline ImVec4 toImVec4(const CabanaColor &c) { return ImVec4(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f); }
inline ImU32 withAlpha(ImU32 c, int alpha) { return (c & ~IM_COL32_A_MASK) | ((ImU32)alpha << IM_COL32_A_SHIFT); }

// the dock window identity of the messages panel (the visible title changes, the part after ### is the id)
constexpr const char *MESSAGES_PANEL_ID = "###MessagesPanel";

struct InputContext {
  std::string *str;
  ImGuiInputTextCallback validator;
  ValidState (*validate)(const std::string &) = nullptr;
  const std::string *last_valid = nullptr;
};

int inputCallback(ImGuiInputTextCallbackData *data);

// text input with an optional validator; `s` grows through the resize callback
bool validatedInput(const char *label, std::string *s, ImGuiInputTextCallback validator, const char *hint = "",
                    ImGuiInputTextFlags flags = 0);

inline bool inputText(const char *label, std::string *s, const char *hint = "", ImGuiInputTextFlags flags = 0) {
  return validatedInput(label, s, nullptr, hint, flags);
}

bool inputTextMultiline(const char *label, std::string *s, const ImVec2 &size, ImGuiInputTextFlags flags = 0);

// an input with a trailing clear button once it holds text; true when the text changed
bool clearableInput(const char *label, std::string *s, const char *hint = "", ImGuiInputTextCallback validator = nullptr);

bool comboBox(const char *label, int *index, const std::vector<std::string> &items);

// numeric items (bus ids, bus speeds) are formatted as they are drawn
template <typename T>
inline bool comboBox(const char *label, int *index, const T *values, int count) {
  bool changed = false;
  const std::string preview = *index >= 0 && *index < count ? std::to_string(values[*index]) : "";
  if (ImGui::BeginCombo(label, preview.c_str())) {
    for (int i = 0; i < count; ++i) {
      ImGui::PushID(i);
      if (ImGui::Selectable(std::to_string(values[i]).c_str(), i == *index) && *index != i) {
        *index = i;
        changed = true;
      }
      if (i == *index) ImGui::SetItemDefaultFocus();
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  return changed;
}

// an edit that makes the text Invalid is refused inside the imgui buffer, like a validated line edit
bool validatedText(const char *label, std::string *s, ValidState (*validate)(const std::string &),
                   const char *hint = "", ImGuiInputTextCallback filter = nullptr);

// InputText char filters; the std::string validators in utils/util.h are run again when the edit is committed
int nameValidator(ImGuiInputTextCallbackData *data);
int nodeValidator(ImGuiInputTextCallbackData *data);
int doubleValidator(ImGuiInputTextCallbackData *data);
int ipValidator(ImGuiInputTextCallbackData *data);
int nonWhitespaceValidator(ImGuiInputTextCallbackData *data);

// auto-raise icon button with a tooltip
bool toolButton(const char *id, const char *icon, const char *tooltip = nullptr, const char *text = nullptr);

// tooltip for the last item that also shows while the item is disabled
void disabledItemTooltip(const char *text);

// exclusive menu action: the bullet sits in the check column and the whole row highlights. `width` is the
// minimum row width, so a narrow popup stays wide enough for every row while the highlight spans the popup.
bool radioMenuItem(const char *label, bool checked, float width = 0.0f);

// A queued modal popup submitted from whichever call site is nested in the top-most modal. draw() is called
// both nested in a modal dialog and at the root level; only the level that opened the popup may submit it
// (opening at level 0 would make imgui close the parent modal).
struct PopupOwner {
  ImGuiID popup_id = 0, owner_id = 0;

  // false: this call site must skip the popup this frame
  bool begin(const char *id);

  void reset() { popup_id = owner_id = 0; }
};

// Escape closes a dialog only when nothing is open above it: a combo drops its list first.
bool dialogEscapePressed();

// the window of the top-most open popup, nullptr when none is open
ImGuiWindow *topPopupWindow();

// [Cancel] [Accept], right aligned. reject_label = nullptr for an accept-only box. Escape rejects.
bool dialogButtons(const char *accept_label, bool *accepted, bool *rejected, bool accept_enabled = true,
                   const char *reject_label = "Cancel");

// horizontal header labels are centered. Returns the column a right click was released on, or -1.
int tableHeadersRow();

// no hover highlight, only the selection background. Selectable() prefers HeaderHovered over Header
// whenever the row is hovered, even when it is selected, so a selected row has to keep the selection color
// as its hover color or it looks unselected.
bool viewSelectable(const char *label, bool selected, ImGuiSelectableFlags flags, const ImVec2 &size);

// a 16px box vertically centered in the frame height so rows keep their layout; ImGui::Checkbox draws a
// frame height (22 px) square.
bool checkBox(const char *label, bool *v);

// the next items on the line are right aligned as a block `width` wide
void alignRight(float width);

// text drawn inside a rect; align 0 = left/top, 0.5 = center, 1 = right/bottom. font_size 0: the current size
void drawText(ImDrawList *dl, const ImRect &rect, const char *text, ImU32 col, ImFont *font = nullptr,
              float font_size = 0.0f, const ImVec2 &align = ImVec2(0.5f, 0.5f));
// elided to the rect width, vertically centered
void drawElidedText(ImDrawList *dl, const ImRect &rect, const std::string &text, ImU32 col, bool align_right = false);
// the colored square in front of a signal name: a text line minus 4 px, drawn 2 px below `pos`
float markerSize();
void drawColorMarker(ImDrawList *dl, const ImVec2 &pos, ImU32 col);

void loadFonts();
void applyTheme(int theme);  // safe to call at runtime
bool isDarkTheme();  // the theme applyTheme() resolved

ImU32 highlightedTextColor();
ImU32 paletteBrightText();

// the next window is a real OS window instead of being drawn inside the main one
void setNextWindowFloatsOut();
#ifdef __APPLE__
// the app menu takes its name from the main bundle, and a bare binary gets an info dictionary with its
// file name in it. That dictionary is mutable, so the name is set before glfw brings up cocoa
void setMacAppName(const char *name);
// the native Cocoa full screen toggle (glfw's monitor switch is not full screen on macOS)
bool isNativeFullScreen(GLFWwindow *window);
void toggleNativeFullScreen(GLFWwindow *window);
#endif

// centered, floating out, sized on first appearance
void setNextDialogWindow(const ImVec2 &size);
// centered modal dialog. false when the popup is not submitted this frame.
bool beginDialog(const char *id, PopupOwner *owner, const ImVec2 &size, ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize);

const float TOOLBAR_ITEM_SPACING = 1.0f;
const float TOOLBAR_BUTTON_PADDING = 4.0f;  // auto raise button horizontal margin
const float SLIDER_LENGTH = 13.0f;
const float SLIDER_THICKNESS = 13.0f;

// a tool bar item: `draw` submits it. Items that do not fit go into a ">>" menu, where an item with a
// menu_label becomes a MenuItem that runs `trigger`, and an item without one draws itself.
struct ToolbarItem {
  float width;
  std::function<void()> draw;
  std::string menu_label;
  std::function<void()> trigger;
  bool enabled = true;
  bool in_menu = true;  // false: left out of the ">>" menu (a separator)
};
void beginToolbar();  // item spacing and button padding of a tool bar, until endToolbar()
void endToolbar();
float toolbarButtonWidth(const std::string &label);
// the width of every item plus the spacing between neighbors and the two groups
float toolbarWidth(const std::vector<ToolbarItem> &items, size_t spacer_index);
// items before spacer_index sit at the left, the rest are right aligned; the overflow goes into the ">>" menu
void drawToolbar(const std::vector<ToolbarItem> &items, size_t spacer_index);

// an auto-raise button that opens `popup_id` below itself, with a dropdown arrow after the text. width 0:
// sized to the text, otherwise the arrow sits at the right edge
float menuButtonWidth(const std::string &text, bool bold = false);
bool menuButton(const char *id, const std::string &text, const char *popup_id, bool bold = false, float width = 0.0f);

// a 13x13 handle filled with a subtle vertical gradient and a mid grey outline
void drawSliderHandle(ImDrawList *p, const ImRect &r);

// full width groove, filled left of the handle, 13x13 handle (style.cc)
bool fusionSliderInt(const char *label, int *v, int min, int max, float width);

ImFont *boldFont();
ImFont *monoFont();
void pushMonoFont(float size = 0.0f);  // 0: the size the font was loaded at
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
