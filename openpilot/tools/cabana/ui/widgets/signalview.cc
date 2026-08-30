#include "tools/cabana/ui/widgets/signalview.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <future>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/widgets/videowidget.h"

// bootstrap glyphs merged into the fonts (utils::icon("name") in Qt); file local, other widgets define their own

namespace {
// QStyle pixel metrics (Fusion): PM_TreeViewIndentation, PM_FocusFrameHMargin + 1, PM_FocusFrameVMargin, PM_ToolBarItemSpacing
constexpr float INDENTATION = 20.0f;
constexpr float H_MARGIN = 3.0f;
constexpr float V_MARGIN = 2.0f;
// signal rows are taller than a frame so the sparklines have room to read
constexpr float SIGNAL_ROW_EXTRA = 5.0f;  // the QToolButton in the row makes it 27 px tall at the 16 px font
constexpr float TOOLBAR_ITEM_SPACING = 4.0f;

std::string trimmed(const std::string &s) {
  size_t b = s.find_first_not_of(" \t\n\r\f\v");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\n\r\f\v");
  return s.substr(b, e - b + 1);
}

bool containsCaseInsensitive(const std::string &text, const std::string &sub) {
  auto it = std::search(text.begin(), text.end(), sub.begin(), sub.end(),
                        [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); });
  return it != text.end();
}

std::string toString(double v) {  // QString::number(double)
  char buf[32];
  snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

// Qt rendered the tooltip as rich text; drop the tags for the plain text imgui tooltip
std::string stripHtml(const std::string &s) {
  std::string out;
  bool in_tag = false;
  for (char c : s) {
    if (c == '<') in_tag = true;
    else if (c == '>') in_tag = false;
    else if (!in_tag) out += c;
  }
  return trimmed(out);
}

// QFontMetrics::elidedText + drawText
void drawElidedText(ImDrawList *painter, const ImRect &rect, const std::string &text, ImU32 color, bool align_right) {
  const ImVec2 size = ImGui::CalcTextSize(text.c_str());
  const float y = rect.Min.y + (rect.GetHeight() - size.y) * 0.5f;
  if (size.x <= rect.GetWidth()) {
    painter->AddText(ImVec2(align_right ? rect.Max.x - size.x : rect.Min.x, y), color, text.c_str());
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::RenderTextEllipsis(painter, ImVec2(rect.Min.x, y), ImVec2(rect.Max.x, y + size.y), rect.Max.x, text.c_str(), nullptr, &size);
    ImGui::PopStyleColor();
  }
}

// drawText with a specific font size, vertically centered, optionally aligned to the top/bottom (Qt::AlignTop/AlignBottom)
void drawSmallText(ImDrawList *painter, const ImRect &rect, float font_size, const std::string &text, ImU32 color, int valign = 0) {
  ImFont *font = ImGui::GetFont();
  const ImVec2 size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, text.c_str());
  float y = rect.Min.y + (rect.GetHeight() - size.y) * 0.5f;
  if (valign < 0) y = rect.Min.y;
  else if (valign > 0) y = rect.Max.y - size.y;
  painter->AddText(font, font_size, ImVec2(rect.Min.x, y), color, text.c_str());
}

struct InputContext {
  std::string *str;
  ImGuiInputTextCallback validator;
};

int inputCallback(ImGuiInputTextCallbackData *data) {
  auto *ctx = static_cast<InputContext *>(data->UserData);
  if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
    return ctx->validator ? ctx->validator(data) : 0;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    ctx->str->resize(data->BufTextLen);
    data->Buf = ctx->str->data();
  }
  return 0;
}
}  // namespace

// validators

int nameValidator(ImGuiInputTextCallbackData *data) {
  // NameValidator: [A-Za-z0-9_], spaces rewritten to '_'
  if (data->EventChar == ' ') {
    data->EventChar = '_';
    return 0;
  }
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_')) ? 0 : 1;
}

int nodeValidator(ImGuiInputTextCallbackData *data) {
  // NodeValidator: \w+(,\w+)*
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_' || data->EventChar == ',')) ? 0 : 1;
}

int doubleValidator(ImGuiInputTextCallbackData *data) {
  // DoubleValidator: C-locale floating-point
  const ImWchar c = data->EventChar;
  return (c < 128 && (std::isdigit((int)c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E')) ? 0 : 1;
}

int nonWhitespaceValidator(ImGuiInputTextCallbackData *data) {
  // NonWhitespaceValidator: \S+
  return (data->EventChar < 128 && std::isspace((int)data->EventChar)) ? 1 : 0;
}

bool validatedInput(const char *label, std::string *s, ImGuiInputTextCallback validator, const char *hint, ImGuiInputTextFlags flags) {
  InputContext ctx{s, validator};
  flags |= ImGuiInputTextFlags_CallbackResize;
  if (validator) flags |= ImGuiInputTextFlags_CallbackCharFilter;
  return ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1, flags, inputCallback, &ctx);
}

// SignalModel

static std::string signalTypeToString(cabana::Signal::Type type) {
  if (type == cabana::Signal::Type::Multiplexor) return "Multiplexor Signal";
  else if (type == cabana::Signal::Type::Multiplexed) return "Multiplexed Signal";
  else return "Normal Signal";
}

SignalModel::SignalModel() : root(new Item) {
  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(dbc()->msgUpdated.connect([this](MessageId id) { handleMsgChanged(id); }));
  connections_.push_back(dbc()->msgRemoved.connect([this](MessageId id) { handleMsgChanged(id); }));
  connections_.push_back(dbc()->signalAdded.connect([this](MessageId id, const cabana::Signal *sig) { handleSignalAdded(id, sig); }));
  connections_.push_back(dbc()->signalUpdated.connect([this](const cabana::Signal *sig) { handleSignalUpdated(sig); }));
  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) { handleSignalRemoved(sig); }));
}

void SignalModel::insertItem(SignalModel::Item *root_item, int pos, const cabana::Signal *sig) {
  Item *parent_item = new Item{.type = Item::Sig, .parent = root_item, .sig = sig, .title = sig->name};
  root_item->children.insert(root_item->children.begin() + pos, parent_item);
  std::string titles[]{"Name", "Size", "Receiver Nodes", "Little Endian", "Signed", "Offset", "Factor", "Type",
                       "Multiplex Value", "Extra Info", "Unit", "Comment", "Minimum Value", "Maximum Value", "Value Table"};
  for (int i = 0; i < std::size(titles); ++i) {
    auto item = new Item{.type = (Item::Type)(i + Item::Name), .parent = parent_item, .sig = sig, .title = titles[i]};
    parent_item->children.push_back(item);
    if (item->type == Item::ExtraInfo) {
      parent_item = item;
    }
  }
}

void SignalModel::setMessage(const MessageId &id) {
  msg_id = id;
  filter_str = "";
  refresh();
}

void SignalModel::setFilter(const std::string &txt) {
  filter_str = txt;
  refresh();
}

void SignalModel::refresh() {
  root.reset(new SignalModel::Item);
  if (auto msg = dbc()->msg(msg_id)) {
    for (auto s : msg->getSignals()) {
      if (filter_str.empty() || containsCaseInsensitive(s->name, filter_str)) {
        insertItem(root.get(), root->children.size(), s);
      }
    }
  }
  modelReset();
  rowsChanged();  // modelReset
}

int SignalModel::flags(const Item *item, int column) const {
  if (!item || item == root.get()) return NoItemFlags;

  int flags = ItemIsSelectable | ItemIsEnabled;
  if (column == 1  && item->children.empty()) {
    flags |= (item->type == Item::Endian || item->type == Item::Signed) ? ItemIsUserCheckable : ItemIsEditable;
  }
  if (item->type == Item::MultiplexValue && item->sig->type != cabana::Signal::Type::Multiplexed) {
    flags &= ~ItemIsEnabled;
  }
  return flags;
}

int SignalModel::signalRow(const cabana::Signal *sig) const {
  for (int i = 0; i < root->children.size(); ++i) {
    if (root->children[i]->sig == sig) return i;
  }
  return -1;
}

std::string SignalModel::data(const Item *item, int column) const {
  if (item && item != root.get()) {
    if (column == 0) {
      return item->type == Item::Sig ? item->sig->name : item->title;
    } else {
      switch (item->type) {
        case Item::Sig: return item->sig_val;
        case Item::Name: return item->sig->name;
        case Item::Size: return std::to_string(item->sig->size);
        case Item::Node: return item->sig->receiver_name;
        case Item::SignalType: return signalTypeToString(item->sig->type);
        case Item::MultiplexValue: return std::to_string(item->sig->multiplex_value);
        case Item::Offset: return doubleToString(item->sig->offset);
        case Item::Factor: return doubleToString(item->sig->factor);
        case Item::Unit: return item->sig->unit;
        case Item::Comment: return item->sig->comment;
        case Item::Min: return doubleToString(item->sig->min);
        case Item::Max: return doubleToString(item->sig->max);
        case Item::Desc: {
          std::string val_desc;
          for (auto &[val, desc] : item->sig->val_desc) {
            if (!val_desc.empty()) val_desc += " ";
            val_desc += toString(val) + " \"" + desc + "\"";
          }
          return val_desc;
        }
        default: break;
      }
    }
  }
  return {};
}

bool SignalModel::checkState(const Item *item) const {
  if (item->type == Item::Endian) return item->sig->is_little_endian;
  if (item->type == Item::Signed) return item->sig->is_signed;
  return false;
}

std::string SignalModel::toolTip(const Item *item, int column) const {
  if (item && item->type == Item::Sig) {
    return (column == 0) ? utils::signalToolTip(item->sig) : std::string();
  }
  return {};
}

bool SignalModel::setData(Item *item, const ItemValue &value) {
  cabana::Signal s = *item->sig;
  switch (item->type) {
    case Item::Name: s.name = value.toString(); break;
    case Item::Size: s.size = value.toInt(); break;
    case Item::Node: s.receiver_name = trimmed(value.toString()); break;
    case Item::SignalType: s.type = (cabana::Signal::Type)value.toInt(); break;
    case Item::MultiplexValue: s.multiplex_value = value.toInt(); break;
    case Item::Endian: s.is_little_endian = value.toBool(); break;
    case Item::Signed: s.is_signed = value.toBool(); break;
    case Item::Offset: s.offset = value.toDouble(); break;
    case Item::Factor: s.factor = value.toDouble(); break;
    case Item::Unit: s.unit = value.toString(); break;
    case Item::Comment: s.comment = value.toString(); break;
    case Item::Min: s.min = value.toDouble(); break;
    case Item::Max: s.max = value.toDouble(); break;
    case Item::Desc: s.val_desc = value.toValueDescription(); break;
    default: return false;
  }
  bool ret = saveSignal(item->sig, s);
  return ret;
}

bool SignalModel::saveSignal(const cabana::Signal *origin_s, cabana::Signal &s) {
  auto msg = dbc()->msg(msg_id);
  if (s.name != origin_s->name && msg->sig(s.name) != nullptr) {
    std::string text = "There is already a signal with the same name '" + s.name + "'";
    MessageBox::warning("Failed to save signal", text);
    return false;
  }

  if (s.is_little_endian != origin_s->is_little_endian) {
    s.start_bit = flipBitPos(s.start_bit);
  }
  UndoStack::instance()->push(new EditSignalCommand(msg_id, origin_s, s));
  return true;
}

void SignalModel::handleMsgChanged(MessageId id) {
  if (id.address == msg_id.address) {
    refresh();
  }
}

void SignalModel::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id == msg_id) {
    if (filter_str.empty()) {
      int i = dbc()->msg(msg_id)->indexOf(sig);
      insertItem(root.get(), i, sig);
      rowsChanged();  // rowsInserted
    } else if (containsCaseInsensitive(sig->name, filter_str)) {
      refresh();
    }
  }
}

void SignalModel::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    if (filter_str.empty()) {
      // move row when the order changes.
      int to = dbc()->msg(msg_id)->indexOf(sig);
      if (to != row) {
        auto item = root->children[row];
        root->children.erase(root->children.begin() + row);
        root->children.insert(root->children.begin() + to, item);
      }
    }
  }
}

void SignalModel::handleSignalRemoved(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    delete root->children[row];
    root->children.erase(root->children.begin() + row);
    rowsChanged();  // rowsRemoved
  }
}

// SignalItemDelegate

SignalItemDelegate::SignalItemDelegate() {
  name_validator = nameValidator;
  node_validator = nodeValidator;
  double_validator = doubleValidator;
  // updateEditorGeometry runs on the first paint; seed the size of the [plot][remove] widget (two 22px
  // ToolButtons + PM_ToolBarItemSpacing) so the first updateState() calls already leave room for the sparklines
  button_size = ImVec2(22 * 2 + TOOLBAR_ITEM_SPACING, 22);

  // the sig pointers die with the signal
  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) {
    if (desc_sig_ == sig) desc_sig_ = nullptr;
    if ((editing_item_ && editing_item_->sig == sig) || (open_item_ && open_item_->sig == sig) ||
        (focus_item_ && focus_item_->sig == sig)) closeEditor();
  }));
  connections_.push_back(dbc()->fileChanged.connect([this]() {
    desc_sig_ = nullptr;
    closeEditor();
  }));
}

float SignalItemDelegate::textWidth(const std::string &text, float font_size) {
  ImFont *font = ImGui::GetFont();
  if (!font || ImGui::GetFontSize() <= 0) return 0;  // no frame rendered yet
  return font->CalcTextSizeA(font_size > 0 ? font_size : ImGui::GetFontSize(), FLT_MAX, 0.0f, text.c_str()).x;
}

float SignalItemDelegate::rowHeight() const {
  return ImGui::GetFrameHeight();
}

// only the top level signal rows are taller; the expanded sub-rows keep the Qt row height
float SignalItemDelegate::signalRowHeight() const {
  return ImGui::GetFrameHeight() + SIGNAL_ROW_EXTRA;
}

float SignalItemDelegate::sizeHint(const SignalModel::Item *item, int column, float widget_width, const std::string &text) const {
  float width = widget_width / 2;
  if (column == 0) {
    float spacing = INDENTATION + color_label_width + 8;
    std::string txt = text;
    if (item->type == SignalModel::Item::Sig && item->sig->type != cabana::Signal::Type::Normal) {
      txt += item->sig->type == cabana::Signal::Type::Multiplexor ? std::string(" M ") : " m" + std::to_string(item->sig->multiplex_value) + " ";
      spacing += H_MARGIN * 2;
    }
    width = std::min<float>(widget_width / 3.0, textWidth(txt) + spacing);
  }
  return width;
}

void SignalItemDelegate::paint(ImDrawList *painter, const ImRect &option_rect, const SignalModel::Item *item, int column,
                               bool selected, const std::string &text, float viewport_x) const {
  const float h_margin = H_MARGIN;
  const float v_margin = V_MARGIN;

  ImRect rect(option_rect.Min.x + h_margin, option_rect.Min.y + v_margin, option_rect.Max.x - h_margin, option_rect.Max.y - v_margin);
  // selection background is painted by the row's Selectable
  const ImU32 text_color = selected ? highlightedTextColor() : ImGui::GetColorU32(ImGuiCol_Text);

  if (column == 0) {
    if (item->type == SignalModel::Item::Sig) {
      // color label
      ImRect icon_rect(rect.Min.x, rect.Min.y, rect.Min.x + color_label_width, rect.Max.y);
      const CabanaColor c = item->sig->color.darker(item->highlight ? 125 : 0);
      painter->AddRectFilled(icon_rect.Min, icon_rect.Max, IM_COL32(c.r, c.g, c.b, c.a), 3.0f);
      const std::string number = std::to_string(const_cast<SignalModel::Item *>(item)->row() + 1);
      ImFont *font = ImGui::GetFont();
      const ImVec2 size = font->CalcTextSizeA(label_font, FLT_MAX, 0.0f, number.c_str());
      painter->AddText(font, label_font, ImVec2(icon_rect.Min.x + (icon_rect.GetWidth() - size.x) * 0.5f, icon_rect.Min.y + (icon_rect.GetHeight() - size.y) * 0.5f),
                       item->highlight ? IM_COL32_WHITE : IM_COL32_BLACK, number.c_str());

      rect.Min.x = icon_rect.Max.x + h_margin * 2;
      // multiplexer indicator
      if (item->sig->type != cabana::Signal::Type::Normal) {
        std::string indicator = item->sig->type == cabana::Signal::Type::Multiplexor ? std::string(" M ") : " m" + std::to_string(item->sig->multiplex_value) + " ";
        ImRect indicator_rect(rect.Min.x, rect.Min.y, rect.Min.x + ImGui::CalcTextSize(indicator.c_str()).x, rect.Max.y);
        painter->AddRectFilled(indicator_rect.Min, indicator_rect.Max, IM_COL32(160, 160, 164, 255), 3.0f);  // Qt::gray
        drawElidedText(painter, indicator_rect, indicator, IM_COL32_WHITE, false);
        rect.Min.x = indicator_rect.Max.x + h_margin * 2;
      }
    } else {
      rect.Min.x = viewport_x + INDENTATION + color_label_width + h_margin * 3;
    }

    // name
    if (rect.GetWidth() > 0) drawElidedText(painter, rect, text, text_color, false);
  } else if (column == 1) {
    if (!item->sparkline.isEmpty()) {
      const ImVec2 sparkline_size = item->sparkline.size;
      item->sparkline.draw(painter, rect.Min);
      // min-max value
      rect.Min.x += sparkline_size.x + 1;
      float value_adjust = 10;
      if (!item->sparkline.isEmpty() && (item->highlight || selected)) {
        painter->AddLine(rect.Min, ImVec2(rect.Min.x, rect.Max.y), text_color);
        rect.Min.x += 5;
        rect.Min.y -= v_margin;
        rect.Max.y += v_margin;
        std::string min = toString(item->sparkline.min_val);
        std::string max = toString(item->sparkline.max_val);
        drawSmallText(painter, rect, minmax_font, max, text_color, -1);
        drawSmallText(painter, rect, minmax_font, min, text_color, 1);
        value_adjust = std::max(textWidth(min, minmax_font), textWidth(max, minmax_font)) + 5;
      } else if (!item->sparkline.isEmpty() && item->sig->type == cabana::Signal::Type::Multiplexed) {
        // display freq of multiplexed signal
        char freq[64];
        snprintf(freq, sizeof(freq), "%.2g hz", item->sparkline.freq());
        ImRect freq_rect(rect.Min.x + 5, rect.Min.y, rect.Max.x, rect.Max.y);
        drawSmallText(painter, freq_rect, label_font, freq, text_color);
        value_adjust = textWidth(freq, label_font) + 10;
      }
      // signal value
      rect.Min.x += value_adjust;
      rect.Max.x -= button_size.x;
      if (rect.GetWidth() > 0) drawElidedText(painter, rect, text, text_color, true);
    } else {
      // QStyledItemDelegate::paint
      if (rect.GetWidth() > 0) drawElidedText(painter, rect, text, text_color, false);
    }
  }
}

void SignalItemDelegate::createEditor(SignalModel::Item *item, SignalModel *model) {
  const bool take_focus = focus_item_ == item;
  if (take_focus) focus_item_ = nullptr;
  if (item->type == SignalModel::Item::Name || item->type == SignalModel::Item::Node || item->type == SignalModel::Item::Offset ||
      item->type == SignalModel::Item::Factor || item->type == SignalModel::Item::MultiplexValue ||
      item->type == SignalModel::Item::Min || item->type == SignalModel::Item::Max) {
    ImGuiInputTextCallback validator = nullptr;
    if (item->type == SignalModel::Item::Name) validator = name_validator;
    else if (item->type == SignalModel::Item::Node) validator = node_validator;
    else validator = double_validator;

    take_focus_ = take_focus;
    lineEditor(item, model, validator);
  } else if (item->type == SignalModel::Item::Size) {
    int v = item->sig->size;
    if (take_focus) ImGui::SetKeyboardFocusHere();
    bool changed = ImGui::InputInt("##editor", &v, 1, 100, ImGuiInputTextFlags_AutoSelectAll);
    if (ImGui::IsItemDeactivatedAfterEdit() || (changed && !ImGui::IsItemActive())) {
      setModelData(item, model, std::clamp(v, 1, CAN_MAX_DATA_BYTES));
    }
    // Enter, Escape and a click outside close the editor; the step buttons keep it open like QSpinBox
    if (ImGui::IsItemDeactivated() && (!ImGui::IsItemHovered() || ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                                       ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false))) {
      open_item_ = nullptr;
    }
  } else if (item->type == SignalModel::Item::SignalType) {
    // the QComboBox editor is closed by Enter and Escape; the cell is painted as text again next frame
    if (combo_focused_ && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                           ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
      open_item_ = nullptr;
      combo_focused_ = false;
      return;
    }
    std::vector<std::pair<std::string, int>> items;
    items.emplace_back(signalTypeToString(cabana::Signal::Type::Normal), (int)cabana::Signal::Type::Normal);
    if (!dbc()->msg(model->msg_id)->multiplexor) {
      items.emplace_back(signalTypeToString(cabana::Signal::Type::Multiplexor), (int)cabana::Signal::Type::Multiplexor);
    } else if (item->sig->type != cabana::Signal::Type::Multiplexor) {
      items.emplace_back(signalTypeToString(cabana::Signal::Type::Multiplexed), (int)cabana::Signal::Type::Multiplexed);
    }
    std::vector<const char *> names;
    int current = -1;  // QComboBox::currentIndex() is -1 when the current type is not an item (Multiplexor)
    for (int i = 0; i < items.size(); ++i) {
      names.push_back(items[i].first.c_str());
      if (items[i].second == (int)item->sig->type) current = i;
    }
    const ImGuiID popup_id = ImHashStr("##ComboPopup", 0, ImGui::GetID("##editor"));
    if (take_focus) ImGui::SetKeyboardFocusHere();
    if (ImGui::Combo("##editor", &current, names.data(), names.size())) {
      setModelData(item, model, items[current].second);
      open_item_ = nullptr;  // activated: commitData + closeEditor
    }
    combo_focused_ = ImGui::IsItemFocused() || ImGui::IsPopupOpen(popup_id, ImGuiPopupFlags_None);
    if (!take_focus && !combo_focused_) open_item_ = nullptr;  // the editor is closed when it loses the focus
  } else if (item->type == SignalModel::Item::Desc) {
    ImGui::PushStyleColor(ImGuiCol_Header, (ImU32)0);
    const bool clicked = ImGui::Selectable("##editor", false, 0, ImVec2(0, rowHeight()));
    ImGui::PopStyleColor();
    drawElidedText(ImGui::GetWindowDrawList(), ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), model->data(item, 1),
                   highlightedTextColor(), false);
    if (clicked || take_focus) {
      desc_dlg_ = std::make_unique<ValueDescriptionDlg>(item->sig->val_desc);
      desc_dlg_->title = item->sig->name;
      desc_sig_ = item->sig;
    }
  } else {
    // QStyledItemDelegate::createEditor: plain QLineEdit, no validator
    take_focus_ = take_focus;
    lineEditor(item, model, nullptr);
  }
}

void SignalItemDelegate::closeEditor() {
  editing_item_ = open_item_ = focus_item_ = nullptr;
  editor_active_ = refocus_editor_ = enter_pressed_ = combo_focused_ = false;
}

// QValidator::validate for the editor of `item`; mutates `text` like NameValidator does (spaces -> '_')
ValidState SignalItemDelegate::validateEditor(const SignalModel::Item *item, std::string &text) {
  if (item->type == SignalModel::Item::Name) return validateName(text);
  if (item->type == SignalModel::Item::Node) return validateNodes(text);
  if (item->type == SignalModel::Item::Offset || item->type == SignalModel::Item::Factor ||
      item->type == SignalModel::Item::MultiplexValue || item->type == SignalModel::Item::Min ||
      item->type == SignalModel::Item::Max) {
    return validateDouble(text);
  }
  return ValidState::Acceptable;  // no validator
}

// QLineEdit editor: Enter and focus out only commit when the validator reports Acceptable (an Intermediate
// or Invalid value keeps the editor open with the typed text and commits nothing), Escape reverts to the
// value the editor was opened with.
void SignalItemDelegate::lineEditor(SignalModel::Item *item, SignalModel *model, ImGuiInputTextCallback validator) {
  const bool editing = editing_item_ == item;
  const bool was_active = editing && editor_active_;  // the editor had the focus at the end of the last frame

  std::string text = editing ? edit_text_ : model->data(item, 1);
  if (std::exchange(take_focus_, false)) ImGui::SetKeyboardFocusHere();
  if (editing && refocus_editor_) {
    ImGui::SetKeyboardFocusHere();  // QLineEdit keeps the focus when the input is not acceptable
    refocus_editor_ = false;
  }
  validatedInput("##editor", &text, validator, "", ImGuiInputTextFlags_AutoSelectAll);
  if (ImGui::IsItemActivated()) {
    editing_item_ = item;
    edit_original_ = model->data(item, 1);
  }
  if (editing_item_ != item) return;

  edit_text_ = text;
  editor_active_ = ImGui::IsItemActive();
  if (was_active) {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      // closeEditor(RevertModelCache): no commit, the editor is closed and the original value comes back
      edit_text_ = edit_original_;
      editing_item_ = open_item_ = nullptr;
      enter_pressed_ = false;
      return;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) {
      enter_pressed_ = true;
    }
  }
  if (ImGui::IsItemDeactivated()) {
    const bool by_enter = std::exchange(enter_pressed_, false);
    if (!ImGui::IsItemDeactivatedAfterEdit()) {
      editing_item_ = open_item_ = nullptr;  // nothing was typed, nothing to commit
    } else if (validateEditor(item, edit_text_) == ValidState::Acceptable) {
      setModelData(item, model, edit_text_);
      editing_item_ = open_item_ = nullptr;
    } else if (by_enter) {
      refocus_editor_ = true;  // the editor stays open with the text the user typed
    } else {
      editing_item_ = open_item_ = nullptr;
    }
  }
}

void SignalItemDelegate::setModelData(SignalModel::Item *item, SignalModel *model, const ItemValue &value) const {
  pending_commit = [item, model, value]() { model->setData(item, value); };
}

void SignalItemDelegate::drawValueDescriptionDlg(SignalModel *model) {
  if (!desc_dlg_) return;
  if (desc_dlg_->draw()) return;

  if (desc_dlg_->accepted) {
    // dlg.exec() returned: apply to the Desc item of the edited signal
    for (auto sig_item : model->root->children) {
      if (sig_item->sig != desc_sig_) continue;
      for (auto child : sig_item->children) {
        if (child->type != SignalModel::Item::ExtraInfo) continue;
        for (auto extra : child->children) {
          if (extra->type == SignalModel::Item::Desc) setModelData(extra, model, desc_dlg_->val_desc);
        }
      }
    }
  }
  desc_dlg_.reset();
  desc_sig_ = nullptr;
}

// SignalView

SignalView::SignalView(ChartsWidget *charts) : charts(charts) {
  // title bar: signal_count_lb, filter_edit (NonWhitespaceValidator, clear button, "Filter Signal") drawn in draw()

  // WARNING: increasing the maximum range can result in severe performance degradation.
  // 30s is a reasonable value at present.
  const int max_range = sparkline_range_max; // 30s
  settings.sparkline_range = std::clamp(settings.sparkline_range, 1, max_range);
  // sparkline_label, sparkline_range_slider (1..max_range, "Sparkline time range") and the collapse button are drawn in draw()

  // tree view
  model = std::make_unique<SignalModel>();
  delegate = std::make_unique<SignalItemDelegate>();
  updateToolBar();

  connections_.push_back(model->rowsChanged.connect([this]() { rowsChanged(); }));
  // QAbstractItemView::reset() closes the open editors; the items they point at are deleted by refresh()
  connections_.push_back(model->modelReset.connect([this]() {
    delegate->closeEditor();
    // QTreeView lays the new rows out before rowsChanged() runs, so its visible range is current. Ours is
    // computed while drawing; reset it to the top so the sparklines are ready in the frame that paints them.
    if (first_visible_row_ != -1) {
      last_visible_row_ = std::min(model->rowCount() - 1, last_visible_row_ - first_visible_row_);
      first_visible_row_ = 0;
    }
  }));
  connections_.push_back(dbc()->signalAdded.connect([this](MessageId id, const cabana::Signal *sig) { handleSignalAdded(id, sig); }));
  connections_.push_back(dbc()->signalUpdated.connect([this](const cabana::Signal *sig) { handleSignalUpdated(sig); }));
  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) { handleSignalRemoved(sig); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { handleSignalRemoved(nullptr); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool) { updateState(msgs); }));
}

std::string SignalView::whatsThis() const {
  return R"(
    <b>Signal view</b><br />
    <!-- TODO: add description here -->
  )";
}

void SignalView::setMessage(const MessageId &id) {
  max_value_width = 0;
  filter_edit.clear();
  model->setMessage(id);
}

void SignalView::rowsChanged() {
  // the [plot][remove] index widgets are drawn per row in drawIndexWidget()
  updateToolBar();
  updateChartState();
  updateState();
}

void SignalView::rowClicked(SignalModel::Item *item) {
  if (item->type == SignalModel::Item::Sig || item->type == SignalModel::Item::ExtraInfo) {
    item->expanded = !item->expanded;
  }
}

void SignalView::selectSignal(const cabana::Signal *sig, bool expand) {
  if (int row = model->signalRow(sig); row != -1) {
    auto item = model->root->children[row];
    if (expand) {
      item->expanded = !item->expanded;
    }
    scroll_to_sig_ = sig;  // scrollTo(idx, PositionAtTop)
    current_sig_ = sig;
    current_type_ = SignalModel::Item::Sig;
  }
}

void SignalView::updateChartState() {
  for (auto item : model->root->children) {
    item->chart_opened = charts->hasSignal(model->msg_id, item->sig);
  }
}

void SignalView::signalHovered(const cabana::Signal *sig) {
  auto &children = model->root->children;
  for (int i = 0; i < children.size(); ++i) {
    children[i]->highlight = children[i]->sig == sig;
  }
}

void SignalView::updateToolBar() {
  signal_count_lb = "Signals: " + std::to_string(model->rowCount());
  sparkline_label = utils::formatSeconds(settings.sparkline_range);
}

void SignalView::setSparklineRange(int value) {
  settings.sparkline_range = value;
  updateToolBar();
  updateState();
}

void SignalView::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id.address == model->msg_id.address) {
    selectSignal(sig);
  }
}

void SignalView::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = model->signalRow(sig); row != -1)
    updateState();
}

void SignalView::handleSignalRemoved(const cabana::Signal *sig) {
  if (!sig || current_sig_ == sig) {
    // QItemSelectionModel moves the current index to the row that took the removed one, or to the last row
    current_sig_ = nullptr;
    auto &children = model->root->children;
    if (sig && !children.empty() && current_row_ >= 0) {
      current_sig_ = children[std::min<int>(current_row_, children.size() - 1)]->sig;
      current_type_ = SignalModel::Item::Sig;
    }
  }
  if (!sig || scroll_to_sig_ == sig) scroll_to_sig_ = nullptr;
  if (!sig || hovered_sig_ == sig) hovered_sig_ = nullptr;
}

std::pair<int, int> SignalView::visibleSignalRange() {
  // computed while drawing the tree: the first top-level row whose own row is visible (a signal whose
  // header is scrolled out but whose children are visible is skipped, like Qt), and the last top-level
  // row with any visible row
  return {first_visible_row_, last_visible_row_};
}

void SignalView::updateState(const std::set<MessageId> *msgs) {
  const auto &last_msg = can->lastMessage(model->msg_id);
  if (model->rowCount() == 0 || (msgs && !msgs->count(model->msg_id)) || last_msg.dat.size() == 0) return;

  for (auto item : model->root->children) {
    double value = 0;
    if (item->sig->getValue(last_msg.dat.data(), last_msg.dat.size(), &value)) {
      item->sig_val = item->sig->formatValue(value);
      max_value_width = std::max(max_value_width, SignalItemDelegate::textWidth(item->sig_val));
    }
  }

  auto [first_visible, last_visible] = visibleSignalRange();
  if (first_visible != -1 && last_visible != -1 && last_visible < model->rowCount()) {
    const float min_max_width = SignalItemDelegate::textWidth("-000.00", delegate->minmax_font) + 5;
    float available_width = value_column_width - delegate->button_size.x;
    float value_width = std::min<float>(max_value_width + min_max_width, available_width / 2);
    ImVec2 size(std::floor(available_width - value_width),
                std::floor(delegate->signalRowHeight() - V_MARGIN * 2));

    // plain locals: capturing structured bindings in a lambda is C++20
    const auto range = can->eventsInRange(model->msg_id, std::make_pair(last_msg.ts -settings.sparkline_range, last_msg.ts));
    const CanEventIter first = range.first, last = range.second;
    std::vector<std::future<void>> futures;
    for (int i = first_visible; i <= last_visible; ++i) {
      auto item = model->root->children[i];
      futures.push_back(ThreadPool::instance().run([item, first, last, size]() {
        item->sparkline.update(item->sig, first, last, settings.sparkline_range, size);
      }));
    }
    for (auto &f : futures) f.get();
  }
}

void SignalView::draw() {
  // QFrame::StyledPanel
  if (!ImGui::BeginChild("SignalView", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
    ImGui::EndChild();
    return;
  }

  // title bar
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(signal_count_lb.c_str());
  ImGui::SameLine();
  ImGui::SetNextItemWidth(160.0f);
  if (validatedInput("##filter_edit", &filter_edit, nonWhitespaceValidator, "Filter Signal")) {
    model->setFilter(filter_edit);  // textEdited
  }
  if (!filter_edit.empty()) {
    // clear button
    ImGui::SameLine(0.0f, 0.0f);
    if (ImGui::SmallButton((std::string(icon::X) + "##clear").c_str())) {
      filter_edit.clear();
      model->setFilter(filter_edit);
    }
  }

  // stretch: the sparkline controls sit at the right edge
  const float slider_width = 120.0f;
  const ImGuiStyle &style = ImGui::GetStyle();
  const float right_width = ImGui::CalcTextSize(sparkline_label.c_str()).x + style.ItemSpacing.x + slider_width + style.ItemSpacing.x +
                            ImGui::GetFont()->CalcTextSizeA(12.0f, FLT_MAX, 0.0f, icon::DASH_SQUARE).x + style.FramePadding.x * 2;
  ImGui::SameLine();
  const float right_x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - right_width;
  if (right_x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right_x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(sparkline_label.c_str());
  ImGui::SameLine();
  int range = settings.sparkline_range;
  if (fusionSliderInt("##sparkline_range_slider", &range, 1, sparkline_range_max, slider_width)) {
    setSparklineRange(range);
  }
  ImGui::SetItemTooltip("Sparkline time range");
  ImGui::SameLine();
  // auto-raise ToolButton with setIconSize({12, 12})
  ImGui::PushFont(ImGui::GetFont(), 12.0f);
  const bool collapse = toolButton(icon::DASH_SQUARE, "Collapse All", "collapse_all");
  ImGui::PopFont();
  if (collapse) collapseAll();

  // tree view
  drawTree();
  delegate->drawValueDescriptionDlg(model.get());
  // model changes run after the tree is drawn: dbc()->signalUpdated/signalRemoved reorder or delete the rows
  if (delegate->pending_commit) std::exchange(delegate->pending_commit, nullptr)();
  if (pending_action_) std::exchange(pending_action_, nullptr)();
  current_row_ = model->signalRow(current_sig_);  // currentIndex row, used when the row is removed

  ImGui::EndChild();
}

void SignalView::collapseAll() {
  for (auto item : model->root->children) {
    item->expanded = false;
    for (auto child : item->children) child->expanded = false;
  }
}

void SignalView::drawTree() {
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));  // QFrame::NoFrame
  const float min_height = std::max(ImGui::GetContentRegionAvail().y, 300.0f);  // setMinimumHeight(300)
  const bool visible = ImGui::BeginChild("tree", ImVec2(0, min_height), ImGuiChildFlags_None);
  ImGui::PopStyleVar();
  if (visible) {
    DrawContext ctx{ImGui::GetWindowDrawList(), ImGui::GetCursorScreenPos().x, ImGui::GetContentRegionAvail().x, delegate->rowHeight()};
    const float scroll_value = ImGui::GetScrollY();
    const float scroll_range = ImGui::GetScrollMaxY();
    // the press that closes an open editor is consumed by the focus change, the index widgets never see it
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) editor_open_on_press_ = delegate->open_item_ != nullptr;

    int first_visible = -1, last_visible = -1;
    auto &children = model->root->children;
    for (int i = 0; i < children.size(); ++i) {
      ctx.any_visible = false;
      const bool header_visible = drawItem(children[i], 0, ctx);
      if (header_visible && first_visible == -1) first_visible = i;
      if (ctx.any_visible) last_visible = i;
    }
    if (first_visible == -1 && last_visible != -1) last_visible = -1;
    first_visible_row_ = first_visible;
    last_visible_row_ = last_visible;
    scroll_to_sig_ = nullptr;

    // header()->sectionResized / resizeEvent / scrollbar valueChanged, rangeChanged
    bool changed = false;
    if (ctx.name_width > 0) name_column_width = ctx.name_width;
    if (ctx.value_column_width > 0 && ctx.value_column_width != value_column_width) {
      value_column_width = ctx.value_column_width;
      changed = true;
    }
    if (scroll_value != scroll_value_ || scroll_range != scroll_range_) {
      scroll_value_ = scroll_value;
      scroll_range_ = scroll_range;
      changed = true;
    }
    if (changed) updateState();

    // QAbstractItemView::mousePressEvent: a press on the viewport that hits no row clears the
    // selection and the current index. clicked() is not emitted, so rowClicked() does not run.
    if (!ctx.mouse_on_row && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      current_sig_ = nullptr;
      current_type_ = SignalModel::Item::Root;
    }

    // entered / viewportEntered / leaveEvent
    if (ctx.hovered_sig != hovered_sig_) {
      hovered_sig_ = ctx.hovered_sig;
      highlight(hovered_sig_);
    }
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
}

bool SignalView::drawItem(SignalModel::Item *item, int depth, DrawContext &ctx) {
  const int flags = model->flags(item, 0);
  const bool selected = item->sig == current_sig_ && item->type == current_type_;
  const float row_height = item->type == SignalModel::Item::Sig ? delegate->signalRowHeight() : ctx.row_height;
  const ImVec2 row_min = ImGui::GetCursorScreenPos();
  const ImVec2 row_max(row_min.x + ctx.width, row_min.y + row_height);
  const bool row_visible = ImGui::IsRectVisible(row_min, row_max);
  ctx.any_visible |= row_visible;

  ImGui::PushID(item);
  ImGui::BeginDisabled(!(flags & SignalModel::ItemIsEnabled));
  // QTreeView has no hover highlight, only the selection background
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
  const bool row_clicked = ImGui::Selectable("##row", selected, ImGuiSelectableFlags_AllowOverlap, ImVec2(0, row_height));
  ImGui::PopStyleColor(2);
  // QTreeView::mousePressEvent: a press on the branch indicator only toggles the expansion, the current
  // index does not change and clicked() is not emitted
  const float branch_x = row_min.x + depth * INDENTATION;
  const bool on_branch = !item->children.empty() && ImGui::GetMousePos().x >= branch_x &&
                         ImGui::GetMousePos().x < branch_x + INDENTATION;
  if (row_clicked && on_branch) {
    item->expanded = !item->expanded;
  } else if (row_clicked) {
    // setCurrentIndex + clicked
    current_sig_ = item->sig;
    current_type_ = item->type;
    // AllEditTriggers: currentChanged opens the editor of the new current item. The name column and the
    // non-editable cells (signal rows, check boxes) have no editor, so a click there only makes the cell current.
    delegate->closeEditor();
    if ((model->flags(item, 1) & SignalModel::ItemIsEditable) && ImGui::GetMousePos().x >= row_min.x + name_column_width) {
      delegate->focus_item_ = delegate->open_item_ = item;
    }
    rowClicked(item);
  }
  if (item->type == SignalModel::Item::Sig && item->sig == scroll_to_sig_) {
    ImGui::SetScrollHereY(0.0f);
    scroll_to_sig_ = nullptr;
  }
  if (ImGui::IsMouseHoveringRect(row_min, row_max)) {
    ctx.mouse_on_row = true;  // indexAt(pos) is valid
    if (ImGui::IsWindowHovered()) ctx.hovered_sig = item->sig;
  }

  // drawBranches
  if (!item->children.empty()) {
    const float arrow_size = ImGui::GetFontSize() * 0.7f;
    ImGui::RenderArrow(ctx.draw_list, ImVec2(row_min.x + depth * INDENTATION + 4.0f, row_min.y + (row_height - arrow_size) * 0.5f),
                       ImGui::GetColorU32(ImGuiCol_Text), item->expanded ? ImGuiDir_Down : ImGuiDir_Right, 0.7f);
  }

  // column 0
  const std::string text0 = model->data(item, 0);
  ctx.name_width = std::max(ctx.name_width, delegate->sizeHint(item, 0, ctx.width, text0));
  const ImRect rect0(ImVec2(row_min.x + (depth + 1) * INDENTATION, row_min.y), ImVec2(row_min.x + name_column_width, row_max.y));
  delegate->paint(ctx.draw_list, rect0, item, 0, selected, text0, ctx.viewport_x);
  if (item->type == SignalModel::Item::Sig && ImGui::IsMouseHoveringRect(ImVec2(row_min.x, row_min.y), rect0.Max) &&
      ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && ImGui::BeginTooltip()) {
    ImGui::TextUnformatted(stripHtml(model->toolTip(item, 0)).c_str());
    ImGui::EndTooltip();
  }

  // column 1
  const ImRect rect1(ImVec2(row_min.x + name_column_width, row_min.y), row_max);
  ctx.value_column_width = rect1.GetWidth();
  const int flags1 = model->flags(item, 1);
  if (item->type == SignalModel::Item::Sig) {
    delegate->paint(ctx.draw_list, rect1, item, 1, selected, item->sig_val, ctx.viewport_x);
    drawIndexWidget(item, rect1);
  } else if (flags1 & SignalModel::ItemIsUserCheckable) {
    bool checked = model->checkState(item);
    ImGui::SetCursorScreenPos(ImVec2(rect1.Min.x + H_MARGIN, rect1.Min.y));
    if (checkBox("##check", &checked)) delegate->setModelData(item, model.get(), checked);
  } else if (flags1 & SignalModel::ItemIsEditable) {
    // QAbstractItemView only creates the editor for the current item; the others paint through the delegate
    if (selected && delegate->open_item_ == item) {
      ImGui::SetCursorScreenPos(rect1.Min);
      ImGui::SetNextItemWidth(rect1.GetWidth());
      delegate->createEditor(item, model.get());
    } else {
      delegate->paint(ctx.draw_list, rect1, item, 1, selected, model->data(item, 1), ctx.viewport_x);
    }
  } else {
    delegate->paint(ctx.draw_list, rect1, item, 1, selected, model->data(item, 1), ctx.viewport_x);
  }
  ImGui::EndDisabled();
  ImGui::PopID();
  ImGui::SetCursorScreenPos(ImVec2(row_min.x, row_max.y));

  if (item->expanded) {
    for (auto child : item->children) drawItem(child, depth + 1, ctx);
  }
  return row_visible;
}

void SignalView::drawIndexWidget(SignalModel::Item *item, const ImRect &rect) {
  // rowsChanged(): plot_btn + remove_btn, right aligned in the value column
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 2.0f));
  const ImVec2 btn_size(ImGui::CalcTextSize(icon::GRAPH_UP).x + 6.0f, ImGui::GetFrameHeight());
  const ImVec2 size(btn_size.x * 2 + TOOLBAR_ITEM_SPACING, btn_size.y);
  ImGui::SetCursorScreenPos(ImVec2(rect.Max.x - size.x, rect.Min.y + (rect.GetHeight() - size.y) * 0.5f));

  const auto sig = item->sig;
  const bool checked = item->chart_opened;
  if (checked) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  if (ImGui::Button((std::string(icon::GRAPH_UP) + "##plot").c_str(), btn_size) && !editor_open_on_press_) {
    item->chart_opened = !checked;
    showChart(model->msg_id, sig, item->chart_opened, ImGui::GetIO().KeyShift);
  }
  if (checked) ImGui::PopStyleColor();
  ImGui::SetItemTooltip("%s", checked ? "Close Plot" : "Show Plot\nSHIFT click to add to previous opened plot");
  ImGui::SameLine(0.0f, TOOLBAR_ITEM_SPACING);
  if (ImGui::Button((std::string(icon::X) + "##remove").c_str(), btn_size) && !editor_open_on_press_) {
    pending_action_ = [this, sig]() { UndoStack::instance()->push(new RemoveSigCommand(model->msg_id, sig)); };
  }
  ImGui::SetItemTooltip("Remove signal");
  ImGui::PopStyleVar();
  delegate->button_size = size;  // updateEditorGeometry
}

// ValueDescriptionDlg

ValueDescriptionDlg::ValueDescriptionDlg(const ValueDescription &descriptions) {
  for (auto &[val, desc] : descriptions) {
    table.emplace_back(toString(val), desc);
  }
}

bool ValueDescriptionDlg::draw() {
  const std::string popup_id = title + "###ValueDescriptionDlg";
  if (!opened_) {
    ImGui::OpenPopup(popup_id.c_str());
    opened_ = true;
  }
  ImGui::SetNextWindowSize(ImVec2(500.0f, 0.0f), ImGuiCond_Appearing);  // setMinimumWidth(500)
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  bool open = true;
  // not drawn while the dock is collapsed or another modal is on top; only closed once the popup is gone
  if (!ImGui::BeginPopupModal(popup_id.c_str(), &open)) return ImGui::IsPopupOpen(popup_id.c_str());

  bool closing = false;
  // toolbar
  if (ImGui::Button(icon::PLUS)) {
    table.emplace_back("", "");
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(current_row == -1);
  if (ImGui::Button(icon::DASH) && current_row < table.size()) {
    table.erase(table.begin() + current_row);
    current_row = -1;
  }
  ImGui::EndDisabled();

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("table", 3, flags, ImVec2(0.0f, 300.0f))) {
    ImGui::TableSetupScrollFreeze(1, 1);
    // vertical header: the 1 based row number
    ImGui::TableSetupColumn("##row_number", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoHeaderLabel,
                            ImGui::CalcTextSize("000").x + ImGui::GetStyle().CellPadding.x * 2);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    for (int row = 0; row < table.size(); ++row) {
      ImGui::PushID(row);
      ImGui::TableNextRow();
      if (row == current_row) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, ImGui::GetColorU32(ImGuiCol_Header));
      ImGui::TableSetColumnIndex(0);
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(std::to_string(row + 1).c_str());
      ImGui::TableSetColumnIndex(1);
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (Delegate::createEditor(0, &table[row].first)) current_row = row;
      ImGui::TableSetColumnIndex(2);
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (Delegate::createEditor(1, &table[row].second)) current_row = row;
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  // btn_box
  bool accept = false, reject = false;
  if (dialogButtons("OK", &accept, &reject)) {
    if (accept) save();
    closing = true;
  }
  if (!open) closing = true;
  if (closing) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  return !closing;
}

void ValueDescriptionDlg::save() {
  for (int i = 0; i < table.size(); ++i) {
    std::string val = trimmed(table[i].first);
    std::string desc = trimmed(table[i].second);
    if (!val.empty() && !desc.empty()) {
      val_desc.push_back({std::atof(val.c_str()), desc});
    }
  }
  accepted = true;  // QDialog::accept()
}

bool ValueDescriptionDlg::Delegate::createEditor(int column, std::string *text) {
  // QLineEdit editor; column 0 has a DoubleValidator. Returns true when the cell was clicked (row selection).
  // the two cells of a row share the row's id scope, so each editor needs its own column id
  ImGui::PushID(column);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  validatedInput("##edit", text, column == 0 ? doubleValidator : nullptr);
  ImGui::PopStyleVar();
  const bool clicked = ImGui::IsItemActivated() || ImGui::IsItemClicked();
  ImGui::PopID();
  return clicked;
}
