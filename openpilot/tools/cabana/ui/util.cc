#include "tools/cabana/ui/util.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <GLFW/glfw3.h>
extern "C" {
struct objc_object;
struct objc_selector;
objc_object *glfwGetCocoaWindow(GLFWwindow *window);
objc_selector *sel_registerName(const char *name);
void objc_msgSend(void);
}
#endif

#include "tools/cabana/ui/icons.h"

namespace {
ImU32 u32(const ImVec4 &c) { return ImGui::ColorConvertFloat4ToU32(c); }
}  // namespace

int inputCallback(ImGuiInputTextCallbackData *data) {
  auto *ctx = static_cast<InputContext *>(data->UserData);
  if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
    return ctx->validator ? ctx->validator(data) : 0;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackEdit) {
    if (ctx->validate(std::string(data->Buf, data->BufTextLen)) == ValidState::Invalid) {
      data->DeleteChars(0, data->BufTextLen);
      data->InsertChars(0, ctx->last_valid->c_str());
    }
    return 0;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    ctx->str->resize(data->BufTextLen);
    data->Buf = ctx->str->data();
  }
  return 0;
}

bool validatedInput(const char *label, std::string *s, ImGuiInputTextCallback validator, const char *hint,
                    ImGuiInputTextFlags flags) {
  InputContext ctx{s, validator};
  flags |= ImGuiInputTextFlags_CallbackResize;
  if (validator) flags |= ImGuiInputTextFlags_CallbackCharFilter;
  return ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1, flags, inputCallback, &ctx);
}

bool inputTextMultiline(const char *label, std::string *s, const ImVec2 &size, ImGuiInputTextFlags flags) {
  InputContext ctx{s, nullptr};
  return ImGui::InputTextMultiline(label, s->data(), s->capacity() + 1, size, flags | ImGuiInputTextFlags_CallbackResize,
                                   inputCallback, &ctx);
}

bool clearableInput(const char *label, std::string *s, const char *hint, ImGuiInputTextCallback validator) {
  bool changed = validatedInput(label, s, validator, hint);
  if (!s->empty()) {
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::PushID(label);
    if (iconButton("clear", icon::X_LG)) {
      s->clear();
      changed = true;
    }
    ImGui::PopID();
  }
  return changed;
}

bool comboBox(const char *label, int *index, const std::vector<std::string> &items) {
  bool changed = false;
  const int count = (int)items.size();
  if (ImGui::BeginCombo(label, *index >= 0 && *index < count ? items[*index].c_str() : "")) {
    for (int i = 0; i < count; ++i) {
      ImGui::PushID(i);
      if (ImGui::Selectable(items[i].c_str(), i == *index) && *index != i) {
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

bool validatedText(const char *label, std::string *s, ValidState (*validate)(const std::string &),
                   const char *hint, ImGuiInputTextCallback filter) {
  const std::string last_valid = *s;  // a refused edit never reaches *s
  InputContext ctx{s, filter, validate, &last_valid};
  ImGuiInputTextFlags flags = ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_CallbackEdit;
  if (filter) flags |= ImGuiInputTextFlags_CallbackCharFilter;
  ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1, flags, inputCallback, &ctx);
  return *s != last_valid;
}

int nameValidator(ImGuiInputTextCallbackData *data) {
  // [A-Za-z0-9_], spaces rewritten to '_'
  if (data->EventChar == ' ') {
    data->EventChar = '_';
    return 0;
  }
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_')) ? 0 : 1;
}

int nodeValidator(ImGuiInputTextCallbackData *data) {
  // \w+(,\w+)*
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_' || data->EventChar == ',')) ? 0 : 1;
}

int doubleValidator(ImGuiInputTextCallbackData *data) {
  // C-locale floating-point
  const ImWchar c = data->EventChar;
  return (c < 128 && (std::isdigit((int)c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E')) ? 0 : 1;
}

int ipValidator(ImGuiInputTextCallbackData *data) {
  // [0-9.]
  const ImWchar c = data->EventChar;
  return ((c >= '0' && c <= '9') || c == '.') ? 0 : 1;
}

int nonWhitespaceValidator(ImGuiInputTextCallbackData *data) {
  // \S+
  return (data->EventChar < 128 && std::isspace((int)data->EventChar)) ? 1 : 0;
}

float iconButtonWidth() { return ImGui::GetFrameHeight(); }

namespace {
constexpr float ICON_BUTTON_GLYPH_SCALE = 0.8f;

// Exclude rasterization padding when centering icons; it varies between glyphs.
struct GlyphInk { float x0, y0, x1, y1; };
GlyphInk glyphInk(const ImFontGlyph *g) {
  ImTextureData *tex = ImGui::GetIO().Fonts->TexData;
  const int px0 = (int)std::lround(g->U0 * tex->Width), px1 = (int)std::lround(g->U1 * tex->Width);
  const int py0 = (int)std::lround(g->V0 * tex->Height), py1 = (int)std::lround(g->V1 * tex->Height);
  int ix0 = px1, iy0 = py1, ix1 = px0, iy1 = py0;
  for (int y = py0; y < py1; ++y) {
    for (int x = px0; x < px1; ++x) {
      const unsigned char *p = (const unsigned char *)tex->GetPixelsAt(x, y);
      const unsigned char alpha = tex->Format == ImTextureFormat_Alpha8 ? p[0] : p[3];
      if (alpha < 64) continue;
      ix0 = std::min(ix0, x); ix1 = std::max(ix1, x + 1);
      iy0 = std::min(iy0, y); iy1 = std::max(iy1, y + 1);
    }
  }
  if (ix0 >= ix1 || py1 <= py0 || px1 <= px0) return {g->X0, g->Y0, g->X1, g->Y1};
  const float sx = (g->X1 - g->X0) / (px1 - px0), sy = (g->Y1 - g->Y0) / (py1 - py0);
  return {g->X0 + (ix0 - px0) * sx, g->Y0 + (iy0 - py0) * sy, g->X0 + (ix1 - px0) * sx, g->Y0 + (iy1 - py0) * sy};
}

// The scan reads the atlas pixels: do it once per icon, and again when the atlas repacked the glyph.
const GlyphInk &cachedGlyphInk(const ImFontGlyph *g, float size, unsigned int codepoint) {
  struct Entry { ImVec4 uv; GlyphInk ink; };
  static std::unordered_map<uint64_t, Entry> cache;
  const uint64_t key = ((uint64_t)(uint32_t)size << 32) | codepoint;
  const ImVec4 uv(g->U0, g->V0, g->U1, g->V1);
  auto it = cache.find(key);
  if (it == cache.end() || memcmp(&it->second.uv, &uv, sizeof(uv)) != 0) it = cache.insert_or_assign(key, Entry{uv, glyphInk(g)}).first;
  return it->second.ink;
}

bool squareIconButton(const char *id, const char *icon) {
  const bool clicked = ImGui::Button((std::string("###") + id).c_str(), ImVec2(iconButtonWidth(), 0.0f));
  const ImRect r(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  unsigned int codepoint = 0;
  ImTextCharFromUtf8(&codepoint, icon, nullptr);
  // Leave a margin even for icons that fill the glyph bounds.
  const float size = std::round(ImGui::GetFontSize() * ICON_BUTTON_GLYPH_SCALE);
  ImFontBaked *baked = ImGui::GetFont()->GetFontBaked(size);
  if (const ImFontGlyph *g = baked->FindGlyph((ImWchar)codepoint)) {
    const GlyphInk ink = cachedGlyphInk(g, size, codepoint);
    // Preserve half-logical-pixel positions on HiDPI displays.
    const float snap = std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x);
    auto snapped = [snap](float v) { return std::round(v * snap) / snap; };
    const ImVec2 pos(snapped(r.GetCenter().x - (ink.x0 + ink.x1) * 0.5f), snapped(r.GetCenter().y - (ink.y0 + ink.y1) * 0.5f));
    // AddText truncates to whole logical pixels, undoing the framebuffer snapping above.
    ImGui::GetWindowDrawList()->AddImage(ImGui::GetIO().Fonts->TexRef, ImVec2(pos.x + g->X0, pos.y + g->Y0),
                                         ImVec2(pos.x + g->X1, pos.y + g->Y1), ImVec2(g->U0, g->V0), ImVec2(g->U1, g->V1),
                                         ImGui::GetColorU32(ImGuiCol_Text));
  }
  return clicked;
}
}  // namespace

bool iconButton(const char *id, const char *icon, const char *tooltip) {
  const bool clicked = squareIconButton(id, icon);
  if (tooltip && *tooltip) ImGui::SetItemTooltip("%s", tooltip);
  return clicked;
}

void disabledItemTooltip(const char *text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", text);
}

bool radioMenuItem(const char *label, bool checked, float width) {
  const float indent = ImGui::GetFontSize();
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const bool clicked = ImGui::Selectable((std::string("##") + label).c_str(), false, ImGuiSelectableFlags_None,
                                         ImVec2(ImMax(width, ImGui::GetContentRegionAvail().x), 0.0f));
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  ImDrawList *painter = ImGui::GetWindowDrawList();
  if (checked) ImGui::RenderBullet(painter, ImVec2(pos.x + indent / 2, pos.y + ImGui::GetTextLineHeight() / 2), color);
  painter->AddText(ImVec2(pos.x + indent, pos.y), color, label);
  return clicked;
}

bool PopupOwner::begin(const char *id) {
  ImGuiWindow *window = ImGui::GetCurrentWindowRead();  // GetCurrentWindow() would mark the fallback window as used
  if (popup_id == 0) {
    // a pending popup may only be opened from the call nested in the top-most modal, or from any call
    // when there is no modal at all
    ImGuiWindow *modal = ImGui::GetTopMostPopupModal();
    if (modal != nullptr && modal != window) return false;
    ImGui::OpenPopup(id);
    popup_id = window->GetID(id);
    owner_id = window->ID;
  } else if (owner_id != window->ID) {
    return false;
  } else if (!ImGui::IsPopupOpen(popup_id, ImGuiPopupFlags_AnyPopupLevel)) {
    // reopen if imgui closed the popup underneath us (host window change)
    ImGui::OpenPopup(id);
  }
  return true;
}

ImGuiWindow *topPopupWindow() {
  ImGuiContext &g = *GImGui;
  return g.OpenPopupStack.Size > 0 ? g.OpenPopupStack.back().Window : nullptr;
}

bool dialogEscapePressed() {
  return ImGui::IsKeyPressed(ImGuiKey_Escape, false) && topPopupWindow() == ImGui::GetCurrentWindow();
}

bool dialogButtons(const char *accept_label, bool *accepted, bool *rejected, bool accept_enabled,
                   const char *reject_label) {
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
  if (rejected && dialogEscapePressed()) {
    *rejected = true;
    pressed = true;
  }
  return pressed;
}

int tableHeadersRow() {
  int clicked = -1;
  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  for (int c = 0, count = ImGui::TableGetColumnCount(); c < count; ++c) {
    if (!ImGui::TableSetColumnIndex(c)) continue;
    const char *name = ImGui::TableGetColumnName(c);
    if (!name) name = "";
    const float offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(name).x) * 0.5f;
    if (offset > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::PushID(c);
    ImGui::TableHeader(name);
    // same timing as TableHeader's own TableOpenContextMenu, so a menu opened by the caller is opened last
    // and replaces the (disabled) table context menu in the popup stack
    if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) clicked = c;
    ImGui::PopID();
  }
  return clicked;
}

bool viewSelectable(const char *label, bool selected, ImGuiSelectableFlags flags, const ImVec2 &size) {
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
  const bool clicked = ImGui::Selectable(label, selected, flags, size);
  ImGui::PopStyleColor(2);
  return clicked;
}

bool checkBox(const char *label, bool *v) {
  const float box = CHECKBOX_SIZE;
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImGuiID id = window->GetID(label);
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  const float frame_h = ImGui::GetFrameHeight();
  const ImVec2 pos = window->DC.CursorPos;
  const ImRect total_bb(pos, ImVec2(pos.x + box + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), pos.y + frame_h));
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if (!ImGui::ItemAdd(total_bb, id)) return false;
  bool hovered, held;
  const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
  if (pressed) {
    *v = !*v;
    ImGui::MarkItemEdited(id);
  }
  const float y = pos.y + IM_TRUNC((frame_h - box) * 0.5f);
  const ImRect check_bb(ImVec2(pos.x, y), ImVec2(pos.x + box, y + box));
  ImGui::RenderNavCursor(total_bb, id);
  const ImU32 bg = ImGui::GetColorU32((held && hovered) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
  ImGui::RenderFrame(check_bb.Min, check_bb.Max, bg, true, style.FrameRounding);
  if (*v) {
    const float pad = ImMax(1.0f, IM_TRUNC(box / 6.0f));
    ImGui::RenderCheckMark(window->DrawList, ImVec2(check_bb.Min.x + pad, check_bb.Min.y + pad), ImGui::GetColorU32(ImGuiCol_CheckMark), box - pad * 2.0f);
  }
  if (label_size.x > 0.0f) ImGui::RenderText(ImVec2(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + style.FramePadding.y), label);
  return pressed;
}

void alignRight(float width) {
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - width));
}

void drawText(ImDrawList *dl, const ImRect &rect, const char *text, ImU32 col, ImFont *font, float font_size, const ImVec2 &align) {
  if (font == nullptr) font = ImGui::GetFont();
  if (font_size <= 0.0f) font_size = ImGui::GetFontSize();
  const ImVec2 size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, text);
  const ImVec2 pos(rect.Min.x + (rect.GetWidth() - size.x) * align.x, rect.Min.y + (rect.GetHeight() - size.y) * align.y);
  dl->AddText(font, font_size, pos, col, text);
}

void drawElidedText(ImDrawList *dl, const ImRect &rect, const std::string &text, ImU32 col, bool align_right) {
  const ImVec2 size = ImGui::CalcTextSize(text.c_str());
  const float y = rect.Min.y + std::max(0.0f, (rect.GetHeight() - size.y) * 0.5f);
  if (size.x <= rect.GetWidth()) {
    dl->AddText(ImVec2(align_right ? rect.Max.x - size.x : rect.Min.x, y), col, text.c_str());
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::RenderTextEllipsis(dl, ImVec2(rect.Min.x, y), ImVec2(rect.Max.x, y + size.y), rect.Max.x, text.c_str(), nullptr, &size);
    ImGui::PopStyleColor();
  }
}

float markerSize() { return ImGui::GetTextLineHeight() - 4; }

void drawColorMarker(ImDrawList *dl, const ImVec2 &pos, ImU32 col) {
  const float size = markerSize();
  dl->AddRectFilled(ImVec2(pos.x, pos.y + 2), ImVec2(pos.x + size, pos.y + 2 + size), col);
}

#ifdef __APPLE__
void setMacAppName(const char *name) {
  auto info = (CFMutableDictionaryRef)CFBundleGetInfoDictionary(CFBundleGetMainBundle());
  if (info == nullptr) return;
  CFStringRef value = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
  CFDictionarySetValue(info, CFSTR("CFBundleName"), value);
  CFRelease(value);
}

bool isNativeFullScreen(GLFWwindow *window) {
  constexpr unsigned long NS_WINDOW_STYLE_MASK_FULL_SCREEN = 1ul << 14;
  objc_object *ns_window = glfwGetCocoaWindow(window);
  if (ns_window == nullptr) return false;
  auto styleMask = (unsigned long (*)(objc_object *, objc_selector *))objc_msgSend;
  return (styleMask(ns_window, sel_registerName("styleMask")) & NS_WINDOW_STYLE_MASK_FULL_SCREEN) != 0;
}

void toggleNativeFullScreen(GLFWwindow *window) {
  auto toggle = (void (*)(objc_object *, objc_selector *, objc_object *))objc_msgSend;
  toggle(glfwGetCocoaWindow(window), sel_registerName("toggleFullScreen:"), nullptr);
}
#endif

void setNextWindowFloatsOut() {
  ImGuiWindowClass window_class;
  window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
  ImGui::SetNextWindowClass(&window_class);
}

void setNextDialogWindow(const ImVec2 &size) {
  if (size.x > 0.0f || size.y > 0.0f) ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  setNextWindowFloatsOut();
}

bool beginDialog(const char *id, PopupOwner *owner, const ImVec2 &size, ImGuiWindowFlags flags) {
  if (!owner->begin(id)) return false;
  setNextDialogWindow(size);
  return ImGui::BeginPopupModal(id, nullptr, flags | ImGuiWindowFlags_NoSavedSettings);
}

// tool bar

ToolbarItem toolbarAction(const char *id, const char *icon, const char *label, std::function<void()> trigger, bool enabled, bool tight) {
  return {iconButtonWidth(), [=]() {
    ImGui::BeginDisabled(!enabled);
    if (iconButton(id, icon)) trigger();
    ImGui::EndDisabled();
    disabledItemTooltip(label);
  }, label, trigger, enabled, true, tight};
}

ToolbarItem toolbarMenu(const char *id, const std::string &text, const char *label, std::function<void()> items, bool bold, bool tight, float width) {
  if (width <= 0.0f) width = menuButtonWidth(text, bold);
  ToolbarItem item{width, [id, text, items, bold, width]() {
    const std::string popup_id = std::string(id) + "_menu";
    menuButton(id, text, popup_id.c_str(), bold, width);
    if (ImGui::BeginPopup(popup_id.c_str())) {
      items();
      ImGui::EndPopup();
    }
  }, label};
  item.tight = tight;
  item.submenu = std::move(items);
  return item;
}

float toolbarButtonWidth(const std::string &label) {
  return ImGui::CalcTextSize(label.c_str(), nullptr, true).x + ImGui::GetStyle().FramePadding.x * 2;
}

static float toolbarSpacing(const ToolbarItem &item) {
  return item.tight ? ImGui::GetStyle().ItemInnerSpacing.x : ImGui::GetStyle().ItemSpacing.x;
}

static float toolbarGroupWidth(const std::vector<ToolbarItem> &items, size_t begin, size_t end) {
  float w = 0;
  for (size_t i = begin; i < end; ++i) w += items[i].width + (i > begin ? toolbarSpacing(items[i]) : 0);
  return w;
}

float toolbarWidth(const std::vector<ToolbarItem> &items, size_t spacer_index) {
  spacer_index = std::min(spacer_index, items.size());
  float w = toolbarGroupWidth(items, 0, spacer_index) + toolbarGroupWidth(items, spacer_index, items.size());
  if (spacer_index > 0 && spacer_index < items.size()) w += ImGui::GetStyle().ItemSpacing.x;
  return w;
}

void drawToolbar(const std::vector<ToolbarItem> &items, size_t spacer_index, float width) {
  const ImGuiStyle &style = ImGui::GetStyle();
  spacer_index = std::min(spacer_index, items.size());
  const float right_width = toolbarGroupWidth(items, spacer_index, items.size());
  const float start_x = ImGui::GetCursorPosX();
  const float avail = width < 0.0f ? ImGui::GetContentRegionAvail().x : width;
  const float right_edge = start_x + avail;
  const float extension_width = iconButtonWidth();

  // when everything fits the spacer takes the slack, otherwise the extension button is reserved at the
  // right edge and the items are packed from the left until the next one does not fit
  // a caller may size a flexible item from the same available width: allow for the float error of the round trip
  const bool fits = toolbarWidth(items, spacer_index) <= avail + 0.5f;
  size_t visible = items.size();
  if (!fits) {
    const float usable = avail - (extension_width + style.ItemSpacing.x);
    float used = 0;
    for (visible = 0; visible < items.size(); ++visible) {
      const float w = items[visible].width + (visible ? toolbarSpacing(items[visible]) : 0);
      if (used + w > usable) break;
      used += w;
    }
  }

  for (size_t i = 0; i < visible; ++i) {
    if (i == 0) ImGui::SetCursorPosX(start_x);
    else if (fits && i == spacer_index) ImGui::SameLine(right_edge - right_width);
    else ImGui::SameLine(0.0f, toolbarSpacing(items[i]));
    items[i].draw();
  }

  if (visible < items.size()) {
    // the extension button sits fully inside the toolbar: its right edge is the content region right edge
    const float extension_x = std::max(start_x, right_edge - extension_width);
    visible == 0 ? ImGui::SetCursorPosX(extension_x) : ImGui::SameLine(extension_x);
    if (iconButton("toolbar_extension", icon::CHEVRON_DOUBLE_RIGHT, "More")) ImGui::OpenPopup("toolbar_extension_menu");
    // the popup opens inward: its right edge is aligned with the button so it stays inside the window
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y), ImGuiCond_Always, ImVec2(1, 0));
    if (ImGui::BeginPopup("toolbar_extension_menu")) {
      for (size_t i = visible; i < items.size(); ++i) {
        if (!items[i].in_menu) continue;
        if (items[i].menu_label.empty()) {
          items[i].draw();
        } else if (items[i].submenu) {
          if (ImGui::BeginMenu(items[i].menu_label.c_str(), items[i].enabled)) {
            items[i].submenu();
            ImGui::EndMenu();
          }
        } else if (ImGui::MenuItem(items[i].menu_label.c_str(), nullptr, false, items[i].enabled)) {
          items[i].trigger();
        }
      }
      ImGui::EndPopup();
    }
  }
}

const float MENU_ARROW_SIZE = 6.0f;     // dropdown arrow on a menu button
const float MENU_ARROW_SPACING = 5.0f;  // gap between the label and the dropdown arrow

float menuButtonWidth(const std::string &text, bool bold) {
  if (bold) pushBoldFont();
  const float w = ImGui::CalcTextSize(text.c_str(), nullptr, true).x + MENU_ARROW_SPACING + MENU_ARROW_SIZE +
                  ImGui::GetStyle().FramePadding.x * 2;
  if (bold) popBoldFont();
  return w;
}

bool menuButton(const char *id, const std::string &text, const char *popup_id, bool bold, float width) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const bool popup_open = ImGui::IsPopupOpen(popup_id);
  if (width <= 0.0f) width = menuButtonWidth(text, bold);
  // ImGui closes popups at frame end on outside clicks. Only open a closed popup so a second press toggles it off.
  if (bold) pushBoldFont();
  const float text_width = ImGui::CalcTextSize(text.c_str(), nullptr, true).x;
  const float ascent = ImGui::GetFontBaked()->Ascent;
  // the text and the arrow are centered as a group in the button
  const float padding_x = std::max(style.FramePadding.x, (width - (text_width + MENU_ARROW_SPACING + MENU_ARROW_SIZE)) * 0.5f);
  ImGui::PushStyleColor(ImGuiCol_Button, popup_open ? style.Colors[ImGuiCol_ButtonActive] : style.Colors[ImGuiCol_Button]);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(padding_x, style.FramePadding.y));
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  const bool clicked = ImGui::ButtonEx((text + "###" + id).c_str(), ImVec2(width, 0.0f), ImGuiButtonFlags_PressedOnClick);
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor();
  if (bold) popBoldFont();
  // a 6 px arrow right after the text, sitting on the text baseline
  const ImVec2 min = ImGui::GetItemRectMin();
  const float x = min.x + padding_x + text_width + MENU_ARROW_SPACING;
  const float baseline = min.y + style.FramePadding.y + ascent;
  ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(x, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE * 0.5f, baseline),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled));
  if (clicked && !popup_open) ImGui::OpenPopup(popup_id);
  // the menu drops down from below the button, not at the mouse cursor
  ImGui::SetNextWindowPos(ImVec2(min.x, ImGui::GetItemRectMax().y), ImGuiCond_Always);
  return clicked;
}

void drawSliderHandle(ImDrawList *p, const ImRect &r) {
  const Palette &pal = palette();
  p->AddRectFilled(r.Min, r.Max, u32(pal.button_hovered), 2.0f);
  p->AddRectFilled(ImVec2(r.Min.x, r.GetCenter().y), r.Max, u32(pal.button), 2.0f, ImDrawFlags_RoundCornersBottom);
  p->AddRect(r.Min, r.Max, u32(pal.border), 2.0f, 0, 1.0f);
}

bool fusionSliderInt(const char *label, int *v, int min, int max, float width) {
  // Keep ImGui slider input handling, but replace its frame and grab with custom drawing.
  ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::SetNextItemWidth(width);
  bool changed = ImGui::SliderInt(label, v, min, max, "", ImGuiSliderFlags_NoInput);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(5);

  const ImVec2 bb_min = ImGui::GetItemRectMin(), bb_max = ImGui::GetItemRectMax();
  const float cy = (bb_min.y + bb_max.y) * 0.5f;
  const float groove_h = SLIDER_THICKNESS * 0.5f;
  const float handle_h = std::min(SLIDER_THICKNESS, bb_max.y - bb_min.y);
  const float x0 = bb_min.x + SLIDER_LENGTH * 0.5f, x1 = bb_max.x - SLIDER_LENGTH * 0.5f;
  const float t = max > min ? (float)(*v - min) / (float)(max - min) : 0.0f;
  const float hx = x0 + (x1 - x0) * t;
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const float groove_y0 = cy - groove_h * 0.5f, groove_y1 = cy + groove_h * 0.5f;
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(bb_max.x, groove_y1), u32(palette().separator), groove_h * 0.5f);
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(hx, groove_y1), u32(palette().accent), groove_h * 0.5f);
  drawSliderHandle(dl, ImRect(ImVec2(hx - SLIDER_LENGTH * 0.5f, cy - handle_h * 0.5f),
                              ImVec2(hx + SLIDER_LENGTH * 0.5f, cy + handle_h * 0.5f)));
  return changed;
}
