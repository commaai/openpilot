#include "tools/cabana/ui/widgets/signalview.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <future>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"
#include "tools/cabana/ui/icons.h"

namespace {
constexpr float H_MARGIN = 3.0f;
constexpr float V_MARGIN = 2.0f;
// signal rows are taller than a frame so the sparklines have room to read
constexpr float SIGNAL_ROW_EXTRA = 5.0f;  // the tool button in the row makes it 27 px tall at the 16 px font
constexpr float SIGNAL_ROW_SCALE = 1.25f;
constexpr float FILTER_WIDTH = 160.0f;
constexpr float SPARKLINE_SLIDER_WIDTH = 120.0f;
// WARNING: increasing the maximum range can result in severe performance degradation.
// 30s is a reasonable value at present.
constexpr int SPARKLINE_RANGE_MAX = 30;
constexpr float LABEL_FONT = 12.0f;   // Inter needs 12 px for 8 px tall digits
constexpr float MINMAX_FONT = 10.0f;
constexpr int COLOR_LABEL_WIDTH = 18;

std::string signalTypeToString(cabana::Signal::Type type) {
  if (type == cabana::Signal::Type::Multiplexor) return "Multiplexor Signal";
  else if (type == cabana::Signal::Type::Multiplexed) return "Multiplexed Signal";
  else return "Normal Signal";
}

std::string multiplexIndicator(const cabana::Signal *sig) {
  return sig->type == cabana::Signal::Type::Multiplexor ? std::string(" M ") : " m" + std::to_string(sig->multiplex_value) + " ";
}

bool propertyButton(const char *label) {
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  const bool clicked = ImGui::Button(label, ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFrameHeight()));
  ImGui::PopStyleColor();
  ImGui::PopStyleVar(2);
  return clicked;
}

// Signal rows leave room for the original sparklines and their min/max labels.
float signalRowHeight() {
  return std::floor((ImGui::GetFrameHeight() + SIGNAL_ROW_EXTRA) * SIGNAL_ROW_SCALE);
}

// column 0 has a double validator. Returns true when the cell was clicked (row selection); the two cells
// of a row share the row's id scope, so each editor needs its own column id
bool valueDescriptionEditor(int column, std::string *text) {
  ImGui::PushID(column);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  validatedInput("##edit", text, column == 0 ? doubleValidator : nullptr);
  ImGui::PopStyleVar();
  const bool clicked = ImGui::IsItemActivated() || ImGui::IsItemClicked();
  ImGui::PopID();
  return clicked;
}

}  // namespace

SignalModel::SignalModel() : root_(new Item) {
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
  msg_id_ = id;
  filter_str_ = "";
  refresh();
}

void SignalModel::setFilter(const std::string &txt) {
  filter_str_ = txt;
  refresh();
}

void SignalModel::refresh() {
  root_.reset(new SignalModel::Item);
  if (auto msg = dbc()->msg(msg_id_)) {
    for (auto s : msg->getSignals()) {
      if (filter_str_.empty() || utils::containsCI(s->name, filter_str_)) {
        insertItem(root_.get(), root_->children.size(), s);
      }
    }
  }
  modelReset();
  rowsChanged();
}

bool SignalModel::isEnabled(const Item *item) {
  return !(item->type == Item::MultiplexValue && item->sig->type != cabana::Signal::Type::Multiplexed);
}

bool SignalModel::isCheckable(const Item *item) {
  return item->type == Item::Endian || item->type == Item::Signed;
}

bool SignalModel::isEditable(const Item *item) {
  return item->children.empty() && !isCheckable(item);
}

int SignalModel::signalRow(const cabana::Signal *sig) const {
  for (int i = 0; i < root_->children.size(); ++i) {
    if (root_->children[i]->sig == sig) return i;
  }
  return -1;
}

std::string SignalModel::valueText(const Item *item) const {
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
        val_desc += utils::toString(val) + " \"" + desc + "\"";
      }
      return val_desc;
    }
    default: return {};
  }
}

bool SignalModel::setData(Item *item, const ItemValue &value) {
  cabana::Signal s = *item->sig;
  switch (item->type) {
    case Item::Name: s.name = value.toString(); break;
    case Item::Size: s.size = value.toInt(); break;
    case Item::Node: s.receiver_name = utils::trimmed(value.toString()); break;
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
  return saveSignal(item->sig, s);
}

bool SignalModel::saveSignal(const cabana::Signal *origin_s, cabana::Signal &s) {
  auto msg = dbc()->msg(msg_id_);
  if (s.name != origin_s->name && msg->sig(s.name) != nullptr) {
    std::string text = "There is already a signal with the same name '" + s.name + "'";
    MessageBox::warning("Failed to save signal", text);
    return false;
  }

  if (s.is_little_endian != origin_s->is_little_endian) {
    s.start_bit = flipBitPos(s.start_bit);
  }
  UndoStack::instance()->push(new EditSignalCommand(msg_id_, origin_s, s));
  return true;
}

void SignalModel::handleMsgChanged(MessageId id) {
  if (id.address == msg_id_.address) {
    refresh();
  }
}

void SignalModel::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id == msg_id_) {
    if (filter_str_.empty()) {
      int i = dbc()->msg(msg_id_)->indexOf(sig);
      insertItem(root_.get(), i, sig);
      rowsChanged();
    } else if (utils::containsCI(sig->name, filter_str_)) {
      refresh();
    }
  }
}

void SignalModel::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    if (filter_str_.empty()) {
      // move row when the order changes.
      int to = dbc()->msg(msg_id_)->indexOf(sig);
      if (to != row) {
        auto item = root_->children[row];
        root_->children.erase(root_->children.begin() + row);
        root_->children.insert(root_->children.begin() + to, item);
      }
    }
  }
}

void SignalModel::handleSignalRemoved(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    delete root_->children[row];
    root_->children.erase(root_->children.begin() + row);
    rowsChanged();
  }
}

float SignalView::textWidth(const std::string &text, float font_size) {
  ImFont *font = ImGui::GetFont();
  if (!font || ImGui::GetFontSize() <= 0) return 0;  // no frame rendered yet
  return font->CalcTextSizeA(font_size > 0 ? font_size : ImGui::GetFontSize(), FLT_MAX, 0.0f, text.c_str()).x;
}

void SignalView::paintCell(ImDrawList *painter, const ImRect &option_rect, const SignalModel::Item *item, int column,
                           bool selected, const std::string &text) const {
  const float h_margin = H_MARGIN;
  const float v_margin = V_MARGIN;

  ImRect rect(option_rect.Min.x + h_margin, option_rect.Min.y + v_margin, option_rect.Max.x - h_margin, option_rect.Max.y - v_margin);
  // selection background is painted by the row's Selectable
  const ImU32 text_color = ImGui::GetColorU32(selected ? palette().text_selected : palette().text);

  if (column == 0) {
    if (item->type == SignalModel::Item::Sig) {
      // color label
      ImRect icon_rect(rect.Min.x, rect.Min.y, rect.Min.x + COLOR_LABEL_WIDTH, rect.Max.y);
      painter->AddRectFilled(icon_rect.Min, icon_rect.Max, toImU32(item->sig->color.darker(item->highlight ? 125 : 0)), ImGui::GetStyle().FrameRounding);
      drawText(painter, icon_rect, std::to_string(item->row() + 1).c_str(), item->highlight ? IM_COL32_WHITE : IM_COL32_BLACK,
               nullptr, LABEL_FONT);

      rect.Min.x = icon_rect.Max.x + h_margin * 2;
      // multiplexer indicator
      if (item->sig->type != cabana::Signal::Type::Normal) {
        const std::string indicator = multiplexIndicator(item->sig);
        ImRect indicator_rect(rect.Min.x, rect.Min.y, rect.Min.x + ImGui::CalcTextSize(indicator.c_str()).x, rect.Max.y);
        painter->AddRectFilled(indicator_rect.Min, indicator_rect.Max, IM_COL32(160, 160, 164, 255), ImGui::GetStyle().FrameRounding);
        drawElidedText(painter, indicator_rect, indicator, IM_COL32_WHITE, false);
        rect.Min.x = indicator_rect.Max.x + h_margin * 2;
      }
    }

    // name
    if (rect.GetWidth() > 0) drawElidedText(painter, rect, text, text_color, false);
  } else if (column == 1) {
    if (!item->sparkline.isEmpty()) {
      const ImVec2 sparkline_size = item->sparkline.size;
      item->sparkline.draw(painter, rect.Min, selected ? text_color : 0);
      // min-max value
      rect.Min.x += sparkline_size.x + 1;
      float value_adjust = 10;
      if (item->highlight || selected) {
        painter->AddLine(rect.Min, ImVec2(rect.Min.x, rect.Max.y), text_color);
        rect.Min.x += 5;
        rect.Min.y -= v_margin;
        rect.Max.y += v_margin;
        std::string min = utils::toString(item->sparkline.min_val);
        std::string max = utils::toString(item->sparkline.max_val);
        drawText(painter, rect, max.c_str(), text_color, nullptr, MINMAX_FONT, ImVec2(0.0f, 0.0f));
        drawText(painter, rect, min.c_str(), text_color, nullptr, MINMAX_FONT, ImVec2(0.0f, 1.0f));
        value_adjust = std::max(textWidth(min, MINMAX_FONT), textWidth(max, MINMAX_FONT)) + 5;
      } else if (item->sig->type == cabana::Signal::Type::Multiplexed) {
        // display freq of multiplexed signal
        char freq[64];
        snprintf(freq, sizeof(freq), "%.2g hz", item->sparkline.freq());
        ImRect freq_rect(rect.Min.x + 5, rect.Min.y, rect.Max.x, rect.Max.y);
        drawText(painter, freq_rect, freq, text_color, nullptr, LABEL_FONT, ImVec2(0.0f, 0.5f));
        value_adjust = textWidth(freq, LABEL_FONT) + 10;
      }
      // signal value
      rect.Min.x += value_adjust;
    }
    // Monospaced digits prevent the value width from changing during playback.
    rect.Max.x -= button_size_.x;
    pushMonoFont(ImGui::GetFontSize());
    if (rect.GetWidth() > 0) drawElidedText(painter, rect, text, text_color, true);
    popMonoFont();
  }
}

void SignalView::drawEditor(SignalModel::Item *item) {
  const bool take_focus = focus_item_ == item;
  if (take_focus) focus_item_ = nullptr;
  if (item->type == SignalModel::Item::Name || item->type == SignalModel::Item::Node || item->type == SignalModel::Item::Offset ||
      item->type == SignalModel::Item::Factor || item->type == SignalModel::Item::MultiplexValue ||
      item->type == SignalModel::Item::Min || item->type == SignalModel::Item::Max) {
    ImGuiInputTextCallback validator = nullptr;
    if (item->type == SignalModel::Item::Name) validator = nameValidator;
    else if (item->type == SignalModel::Item::Node) validator = nodeValidator;
    else validator = doubleValidator;

    drawLineEditor(item, validator, take_focus);
  } else if (item->type == SignalModel::Item::Size) {
    if (take_focus) {
      edit_int_ = item->sig->size;
      ImGui::SetKeyboardFocusHere();
    }
    bool changed = inputInt("##editor", &edit_int_, 1, 100, ImGuiInputTextFlags_AutoSelectAll);
    if (ImGui::IsItemDeactivated() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      open_item_ = nullptr;  // InputInt already reverted the value; only the commit has to be skipped
      return;
    }
    if (ImGui::IsItemDeactivatedAfterEdit() || (changed && !ImGui::IsItemActive())) {
      edit_int_ = std::clamp(edit_int_, 1, CAN_MAX_DATA_BYTES);
      queueCommit(item, edit_int_);
    }
    // Enter, Escape and a click outside close the editor; the step buttons keep it open
    if (ImGui::IsItemDeactivated() && (!ImGui::IsItemHovered() || ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                                       ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false))) {
      open_item_ = nullptr;
    }
  } else if (item->type == SignalModel::Item::SignalType) {
    // the combo editor is closed by Enter and Escape; the cell is painted as text again next frame
    if (combo_focused_ && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                           ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
      open_item_ = nullptr;
      combo_focused_ = false;
      return;
    }
    std::vector<std::pair<std::string, int>> items;
    items.emplace_back(signalTypeToString(cabana::Signal::Type::Normal), (int)cabana::Signal::Type::Normal);
    if (!dbc()->msg(model_.msgId())->multiplexor) {
      items.emplace_back(signalTypeToString(cabana::Signal::Type::Multiplexor), (int)cabana::Signal::Type::Multiplexor);
    } else if (item->sig->type != cabana::Signal::Type::Multiplexor) {
      items.emplace_back(signalTypeToString(cabana::Signal::Type::Multiplexed), (int)cabana::Signal::Type::Multiplexed);
    }
    std::vector<const char *> names;
    int current = -1;  // -1 when the current type is not an item (Multiplexor)
    for (int i = 0; i < items.size(); ++i) {
      names.push_back(items[i].first.c_str());
      if (items[i].second == (int)item->sig->type) current = i;
    }
    const ImGuiID popup_id = ImHashStr("##ComboPopup", 0, ImGui::GetID("##editor"));
    if (take_focus) ImGui::SetKeyboardFocusHere();
    if (dropdown::Combo("##editor", &current, names.data(), names.size())) {
      queueCommit(item, items[current].second);
      open_item_ = nullptr;  // commit and close the editor
    }
    combo_focused_ = ImGui::IsItemFocused() || ImGui::IsPopupOpen(popup_id, ImGuiPopupFlags_None);
    if (!take_focus && !combo_focused_) open_item_ = nullptr;  // the editor is closed when it loses the focus
  } else if (item->type == SignalModel::Item::Desc) {
    const bool clicked = propertyButton((model_.valueText(item) + "###editor").c_str());
    if (clicked || take_focus) {
      desc_dlg_ = std::make_unique<ValueDescriptionDlg>(item->sig->val_desc);
      desc_dlg_->title = item->sig->name;
      desc_sig_ = item->sig;
    }
  } else {
    // plain text input, no validator
    drawLineEditor(item, nullptr, take_focus);
  }
}

void SignalView::commitEditor() {
  SignalModel::Item *item = editing_item_;
  std::string text = edit_text_;
  auto commits = std::move(pending_commits_);
  closeEditor();
  pending_commits_ = std::move(commits);
  if (item && validateEditor(item, text) == ValidState::Acceptable) {
    queueCommit(item, text);
  }
}

void SignalView::closeEditor() {
  editing_item_ = open_item_ = focus_item_ = nullptr;
  editor_active_ = refocus_editor_ = enter_pressed_ = combo_focused_ = false;
  pending_commits_.clear();  // the items they captured may have been deleted by the caller
}

// validate the editor of `item`; mutates `text` like the name validator does (spaces -> '_')
ValidState SignalView::validateEditor(const SignalModel::Item *item, std::string &text) {
  if (item->type == SignalModel::Item::Name) return validateName(text);
  if (item->type == SignalModel::Item::Node) return validateNodes(text);
  if (item->type == SignalModel::Item::Offset || item->type == SignalModel::Item::Factor ||
      item->type == SignalModel::Item::MultiplexValue || item->type == SignalModel::Item::Min ||
      item->type == SignalModel::Item::Max) {
    return validateDouble(text);
  }
  return ValidState::Acceptable;  // no validator
}

// Enter and focus out only commit when the validator reports Acceptable (an Intermediate or Invalid value
// keeps the editor open with the typed text and commits nothing), Escape reverts to the value the editor
// was opened with.
void SignalView::drawLineEditor(SignalModel::Item *item, ImGuiInputTextCallback validator, bool take_focus) {
  const bool editing = editing_item_ == item;
  const bool was_active = editing && editor_active_;  // the editor had the focus at the end of the last frame

  std::string text = editing ? edit_text_ : model_.valueText(item);
  if (take_focus) ImGui::SetKeyboardFocusHere();
  if (editing && refocus_editor_) {
    ImGui::SetKeyboardFocusHere();  // keep the focus when the input is not acceptable
    refocus_editor_ = false;
  }
  validatedInput("##editor", &text, validator, "", ImGuiInputTextFlags_AutoSelectAll);
  if (ImGui::IsItemActivated()) editing_item_ = item;
  if (editing_item_ != item) return;

  edit_text_ = text;
  editor_active_ = ImGui::IsItemActive();
  if (was_active) {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      // InputText already reverted the text; only the commit has to be skipped
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
      queueCommit(item, edit_text_);
      editing_item_ = open_item_ = nullptr;
    } else if (by_enter) {
      refocus_editor_ = true;  // the editor stays open with the text the user typed
    } else {
      editing_item_ = open_item_ = nullptr;
    }
  }
}

void SignalView::queueCommit(SignalModel::Item *item, const ItemValue &value) {
  pending_commits_.push_back([this, item, value]() { model_.setData(item, value); });
}

void SignalView::drawValueDescriptionDlg() {
  if (!desc_dlg_) return;
  if (desc_dlg_->draw()) return;

  if (desc_dlg_->accepted) {
    // the dialog closed: apply to the Desc item of the edited signal
    for (auto sig_item : model_.root()->children) {
      if (sig_item->sig != desc_sig_) continue;
      for (auto child : sig_item->children) {
        if (child->type != SignalModel::Item::ExtraInfo) continue;
        for (auto extra : child->children) {
          if (extra->type == SignalModel::Item::Desc) queueCommit(extra, desc_dlg_->val_desc);
        }
      }
    }
  }
  desc_dlg_.reset();
  desc_sig_ = nullptr;
}

static ImVec2 indexButtonsSize(float button) {
  return ImVec2(button * 2 + ImGui::GetStyle().ItemSpacing.x, button);
}

SignalView::SignalView(ChartsWidget *charts) : charts_(charts) {
  settings.sparkline_range = std::clamp(settings.sparkline_range, 1, SPARKLINE_RANGE_MAX);

  // Reserve button space for updateState() calls before the first draw (no frame yet: derive the frame height).
  button_size_ = indexButtonsSize(UI_FONT_SIZE + ImGui::GetStyle().FramePadding.y * 2.0f);
  updateToolBar();

  connections_.push_back(model_.rowsChanged.connect([this]() { rowsChanged(); }));
  // a reset closes the open editors; the items they point at are deleted by refresh()
  connections_.push_back(model_.modelReset.connect([this]() {
    closeEditor();
    properties_sig_ = nullptr;
    // the visible range is computed while drawing; reset it to the top so the sparklines are ready in the
    // frame that paints them
    if (first_visible_row_ != -1) {
      last_visible_row_ = std::min(model_.rowCount() - 1, last_visible_row_ - first_visible_row_);
      first_visible_row_ = 0;
    }
  }));
  connections_.push_back(dbc()->signalAdded.connect([this](MessageId id, const cabana::Signal *sig) { handleSignalAdded(id, sig); }));
  connections_.push_back(dbc()->signalUpdated.connect([this](const cabana::Signal *sig) { handleSignalUpdated(sig); }));
  // the sig pointers die with the signal
  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) {
    if (desc_sig_ == sig) desc_sig_ = nullptr;
    if (properties_sig_ == sig) properties_sig_ = nullptr;
    if ((editing_item_ && editing_item_->sig == sig) || (open_item_ && open_item_->sig == sig) ||
        (focus_item_ && focus_item_->sig == sig)) closeEditor();
    handleSignalRemoved(sig);
  }));
  connections_.push_back(dbc()->fileChanged.connect([this]() {
    desc_sig_ = nullptr;
    closeEditor();
    handleSignalRemoved(nullptr);
  }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool) { updateState(msgs); }));
}

std::string SignalView::whatsThis() const {
  return R"(
    <b>Signal view</b><br />
    Select a signal to highlight its bits. Double-click a signal or choose Edit to open its properties.<br />
    Drag column dividers to resize. The time-range slider controls the existing sparklines.<br />
    Shift-click the plot button to add a signal to the previously opened plot.
  )";
}

void SignalView::setMessage(const MessageId &id) {
  filter_edit_.clear();
  model_.setMessage(id);
}

void SignalView::rowsChanged() {
  updateToolBar();
  updateChartState();
  updateState();
}

void SignalView::selectSignal(const cabana::Signal *sig) {
  if (model_.signalRow(sig) != -1) {
    scroll_to_sig_ = sig;
    current_sig_ = sig;
    if (properties_sig_ != sig) commitEditor();
    properties_sig_ = sig;
  }
}

void SignalView::updateChartState() {
  for (auto item : model_.root()->children) {
    item->chart_opened = charts_->hasSignal(model_.msgId(), item->sig);
  }
}

void SignalView::signalHovered(const cabana::Signal *sig) {
  auto &children = model_.root()->children;
  for (int i = 0; i < children.size(); ++i) {
    children[i]->highlight = children[i]->sig == sig;
  }
}

void SignalView::updateToolBar() {
  signal_count_lb_ = "Signals: " + std::to_string(model_.rowCount());
  sparkline_label_ = utils::formatSeconds(settings.sparkline_range);
}

void SignalView::setSparklineRange(int value) {
  settings.sparkline_range = value;
  updateToolBar();
  updateState();
}

void SignalView::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id.address == model_.msgId().address) {
    selectSignal(sig);
  }
}

void SignalView::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = model_.signalRow(sig); row != -1)
    updateState();
}

void SignalView::handleSignalRemoved(const cabana::Signal *sig) {
  if (!sig || current_sig_ == sig) {
    // the current index moves to the row that took the removed one, or to the last row
    current_sig_ = nullptr;
    auto &children = model_.root()->children;
    if (sig && !children.empty() && current_row_ >= 0) {
      current_sig_ = children[std::min<int>(current_row_, children.size() - 1)]->sig;
    }
  }
  if (!sig || scroll_to_sig_ == sig) scroll_to_sig_ = nullptr;
  if (!sig || hovered_sig_ == sig) hovered_sig_ = nullptr;
}

float SignalView::widestValueWidth(const cabana::Signal *sig) {
  const double raw_max = sig->is_signed ? std::ldexp(1.0, sig->size - 1) - 1 : std::ldexp(1.0, sig->size) - 1;
  const double raw_min = sig->is_signed ? -std::ldexp(1.0, sig->size - 1) : 0.0;
  float width = 0;
  pushMonoFont(ImGui::GetFontSize());
  for (double raw : {raw_min, raw_max}) {
    // An endpoint may map to a short enum (e.g. "Fault"); still reserve the full numeric value and unit.
    char numeric[128];
    snprintf(numeric, sizeof(numeric), "%.*f", sig->precision, raw * sig->factor + sig->offset);
    const std::string value = std::string(numeric) + (sig->unit.empty() ? "" : " " + sig->unit);
    width = std::max(width, textWidth(value));
  }
  for (const auto &[_, desc] : sig->val_desc) {
    width = std::max(width, textWidth(desc));
  }
  popMonoFont();
  return width;
}

void SignalView::setVisible(bool visible) {
  if (std::exchange(visible_, visible) != visible && visible) updateState();
}

void SignalView::updateState(const std::set<MessageId> *msgs) {
  if (!visible_) return;
  const auto &last_msg = can->lastMessage(model_.msgId());
  if (model_.rowCount() == 0 || (msgs && !msgs->count(model_.msgId())) || last_msg.dat.size() == 0) return;

  // sized for the widest value the signals can produce, not the widest one in the last message: sizing
  // to the current values moved the sparklines every time a value changed length
  float max_value_width = 0;
  for (auto item : model_.root()->children) {
    double value = 0;
    if (item->sig->getValue(last_msg.dat.data(), last_msg.dat.size(), &value)) {
      item->sig_val = item->sig->formatValue(value);
    }
    max_value_width = std::max(max_value_width, widestValueWidth(item->sig));
  }

  if (first_visible_row_ != -1 && last_visible_row_ != -1 && last_visible_row_ < model_.rowCount()) {
    const float min_max_width = textWidth("-000.00", MINMAX_FONT) + 11;
    float available_width = std::max(1.0f, value_column_width_ - button_size_.x - H_MARGIN * 2);
    float value_width = std::min<float>(max_value_width + min_max_width, available_width / 2);
    ImVec2 size(std::floor(available_width - value_width),
                std::floor(signalRowHeight() - ImGui::GetStyle().CellPadding.y * 2 - V_MARGIN * 2));

    // the window ends at the playback clock, not at the last message: its timestamp only moves when a
    // message of this id arrives, so a slow message held the sparkline still for several updates and
    // then jumped it, which reads as a hitch at the message rate
    const double window_end = can->currentSec();
    // a little data from before the window: the sparkline clips it, so the oldest samples and their
    // points slide off the left edge instead of disappearing the moment they age out
    const double lead_in = settings.sparkline_range * 0.05;
    // plain locals: capturing structured bindings in a lambda is C++20
    const auto range = can->eventsInRange(model_.msgId(), std::make_pair(window_end - settings.sparkline_range - lead_in, window_end));
    const CanEventIter first = range.first, last = range.second;
    std::vector<std::future<void>> futures;
    for (int i = first_visible_row_; i <= last_visible_row_; ++i) {
      auto item = model_.root()->children[i];
      futures.push_back(ThreadPool::instance().run([item, first, last, size, window_end]() {
        item->sparkline.update(item->sig, first, last, settings.sparkline_range, size, window_end);
      }));
    }
    for (auto &f : futures) f.get();
  }
}

// the sparkline label and range slider
float SignalView::toolBarRightWidth(const std::string &range_label) {
  const ImGuiStyle &style = ImGui::GetStyle();
  return ImGui::CalcTextSize(range_label.c_str()).x + style.ItemSpacing.x + SPARKLINE_SLIDER_WIDTH;
}

// the width at which the tool bar stops squishing: the signal count and the filter box on the left, the
// sparkline controls on the right, plus the borders and padding of the view's own child window
float SignalView::minimumWidth() {
  const ImGuiStyle &style = ImGui::GetStyle();
  const float left_width = ImGui::CalcTextSize("Signals: 000").x + style.ItemSpacing.x + FILTER_WIDTH;
  // formatSeconds is mm:ss for every value the range slider allows
  return left_width + style.ItemSpacing.x + toolBarRightWidth("00:00") + (style.WindowPadding.x + style.ChildBorderSize) * 2;
}

void SignalView::draw() {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, palette().surface);
  if (!ImGui::BeginChild("SignalView", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
    ImGui::EndChild();
    ImGui::PopStyleColor();
    return;
  }

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(signal_count_lb_.c_str());
  ImGui::SameLine();
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x - toolBarRightWidth(sparkline_label_) - ImGui::GetStyle().ItemSpacing.x,
                                       80.0f, FILTER_WIDTH);
  ImGui::SetNextItemWidth(filter_width);
  if (clearableInput("##filter_edit", &filter_edit_, "Filter Signal", nonWhitespaceValidator)) {
    model_.setFilter(filter_edit_);
  }

  // stretch: the sparkline controls sit at the right edge
  const float controls_width = toolBarRightWidth(sparkline_label_);
  if (ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x - ImGui::GetItemRectMax().x > controls_width + ImGui::GetStyle().ItemSpacing.x) {
    alignRight(controls_width);
  }
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(sparkline_label_.c_str());
  ImGui::SameLine();
  int range = settings.sparkline_range;
  if (fusionSliderInt("##sparkline_range_slider", &range, 1, SPARKLINE_RANGE_MAX, SPARKLINE_SLIDER_WIDTH)) {
    setSparklineRange(range);
  }
  ImGui::SetItemTooltip("Sparkline time range");

  if (model_.signalRow(properties_sig_) >= 0) {
    const float available = std::max(1.0f, ImGui::GetContentRegionAvail().y);
    const float minimum = std::min(available * 0.4f, ImGui::GetFrameHeightWithSpacing() * 4);
    const float maximum = std::max(minimum, available - ImGui::GetFrameHeightWithSpacing() * 5);
    ImGui::SetNextWindowSizeConstraints(ImVec2(0, minimum), ImVec2(FLT_MAX, maximum));
    if (ImGui::BeginChild("signal_rows", ImVec2(0, std::clamp(available * 0.5f, minimum, maximum)),
                          ImGuiChildFlags_ResizeY, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
      drawSignals();
    }
    ImGui::EndChild();
    drawProperties();
  } else {
    ImGui::TextDisabled("Select a signal to edit its properties below.");
    drawSignals();
  }
  // Model changes run after both tables are drawn: dbc()->signalUpdated/signalRemoved reorder or delete the rows
  for (auto &commit : std::exchange(pending_commits_, {})) commit();
  if (pending_action_) std::exchange(pending_action_, nullptr)();
  current_row_ = model_.signalRow(current_sig_);  // used when the row is removed

  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void SignalView::drawSignals() {
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) editor_open_on_press_ = open_item_ != nullptr;
  // Let the table subtract its own vertical scrollbar from the available width.
  // Forcing inner_width to the outer width clips the last button and adds a horizontal scrollbar.
  const bool narrow = ImGui::GetContentRegionAvail().x < 420.0f + ImGui::GetStyle().ScrollbarSize;
  const auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                     ImGuiTableFlags_PadOuterX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp |
                     (narrow ? ImGuiTableFlags_ScrollX : 0);
  const cabana::Signal *hovered = nullptr;
  // Scroll very narrow docks instead of reducing names, sparklines and values to a few pixels.
  const float inner_width = narrow ? 420.0f : 0.0f;
  if (ImGui::BeginTable("signals", 2, flags, ImVec2(0, std::max(1.0f, ImGui::GetContentRegionAvail().y)), inner_width)) {
    ImGui::TableSetupColumn("Signal", ImGuiTableColumnFlags_WidthStretch, 1);
    ImGui::TableSetupColumn("History / Value", ImGuiTableColumnFlags_WidthStretch, 2);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();
    int first_visible = -1, last_visible = -1;
    float value_width = value_column_width_;
    for (int i = 0; i < model_.rowCount(); ++i) {
      auto item = model_.root()->children[i];
      ImGui::PushID(item->sig);
      ImGui::TableNextRow(0, signalRowHeight());
      ImGui::TableSetColumnIndex(0);
      const ImVec2 pos = ImGui::GetCursorScreenPos();
      const float cell_height = signalRowHeight() - ImGui::GetStyle().CellPadding.y * 2;
      const ImRect name_rect(pos, ImVec2(pos.x + ImGui::GetContentRegionAvail().x, pos.y + cell_height));
      const bool selected = current_sig_ == item->sig;
      if (selected) {
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGui::GetStyleColorVec4(ImGuiCol_Header));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetStyleColorVec4(ImGuiCol_Header));
      }
      if (ImGui::Selectable("##signal", selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap, ImVec2(0, cell_height))) {
        selectSignal(item->sig);
      }
      if (selected) ImGui::PopStyleColor(2);
      const bool visible = ImGui::IsItemVisible();
      if (visible) {
        if (first_visible == -1) first_visible = i;
        last_visible = i;
      }
      if (ImGui::IsItemHovered()) {
        hovered = item->sig;
        ImGui::SetItemTooltip("%s", utils::stripHtml(utils::signalToolTip(item->sig)).c_str());
      }
      if (scroll_to_sig_ == item->sig) {
        ImGui::SetScrollHereY();
        scroll_to_sig_ = nullptr;
      }
      if (visible) paintCell(ImGui::GetWindowDrawList(), name_rect, item, 0, selected, item->sig->name);
      ImGui::TableSetColumnIndex(1);
      const ImVec2 value_pos = ImGui::GetCursorScreenPos();
      const ImRect value_rect(value_pos, ImVec2(value_pos.x + ImGui::GetContentRegionAvail().x, value_pos.y + cell_height));
      value_width = value_rect.GetWidth();
      if (visible) {
        paintCell(ImGui::GetWindowDrawList(), value_rect, item, 1, selected, item->sig_val);
        drawIndexWidget(item, value_rect);
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
    const bool changed = first_visible != first_visible_row_ || last_visible != last_visible_row_ || value_width != value_column_width_;
    first_visible_row_ = first_visible;
    last_visible_row_ = last_visible;
    value_column_width_ = value_width;
    if (changed) updateState();
  }
  const auto highlighted = hovered ? hovered : (model_.signalRow(current_sig_) != -1 ? current_sig_ : nullptr);
  if (highlighted != hovered_sig_) {
    hovered_sig_ = highlighted;
    highlight(highlighted);
  }
}

void SignalView::drawProperty(SignalModel::Item *item) {
  ImGui::PushID(item->type);
  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(item->title.c_str());
  ImGui::TableSetColumnIndex(1);
  ImGui::BeginDisabled(!SignalModel::isEnabled(item));
  if (SignalModel::isCheckable(item)) {
    bool checked = item->type == SignalModel::Item::Endian ? item->sig->is_little_endian : item->sig->is_signed;
    if (checkBox("##check", &checked)) queueCommit(item, checked);
  } else if (open_item_ == item) {
    ImGui::SetNextItemWidth(-FLT_MIN);
    drawEditor(item);
  } else {
    const std::string text = model_.valueText(item);
    // Match the editor's frame padding and baseline so opening it never shifts the table.
    if (propertyButton((text + "###value").c_str())) {
      commitEditor();
      open_item_ = focus_item_ = item;
    }
    ImGui::SetItemTooltip("Click to edit %s", item->title.c_str());
  }
  ImGui::EndDisabled();
  ImGui::PopID();
}

void SignalView::commitProperties() {
  commitEditor();
  for (auto &commit : std::exchange(pending_commits_, {})) commit();
}

void SignalView::drawProperties() {
  ImGui::Text("Properties: %s", properties_sig_ ? properties_sig_->name.c_str() : "No signal selected");
  ImGui::SetItemTooltip("Select another signal above to edit it. Drag the divider to resize.");
  const int row = model_.signalRow(properties_sig_);
  const bool narrow = ImGui::GetContentRegionAvail().x < 360.0f + ImGui::GetStyle().ScrollbarSize;
  const auto flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX |
                     ImGuiTableFlags_ScrollY | (narrow ? ImGuiTableFlags_ScrollX : 0);
  if (row >= 0 && ImGui::BeginTable("properties", 2, flags, ImVec2(0, std::max(1.0f, ImGui::GetContentRegionAvail().y)), narrow ? 360.0f : 0.0f)) {
    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 150);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
    for (auto child : model_.root()->children[row]->children) {
      if (child->type == SignalModel::Item::ExtraInfo) {
        for (auto extra : child->children) drawProperty(extra);
      } else {
        drawProperty(child);
      }
    }
    ImGui::EndTable();
  }
  drawValueDescriptionDlg();
}

void SignalView::drawIndexWidget(SignalModel::Item *item, const ImRect &rect) {
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const ImVec2 size = indexButtonsSize(iconButtonWidth());
  ImGui::SetCursorScreenPos(ImVec2(rect.Max.x - size.x, rect.Min.y + (rect.GetHeight() - size.y) * 0.5f));

  const auto sig = item->sig;
  const bool checked = item->chart_opened;
  if (checked) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  if (iconButton("plot", icon::GRAPH_UP) && !editor_open_on_press_) {
    item->chart_opened = !checked;
    showChart(model_.msgId(), sig, item->chart_opened, ImGui::GetIO().KeyShift);
  }
  if (checked) ImGui::PopStyleColor();
  ImGui::SetItemTooltip("%s", checked ? "Close Plot" : "Show Plot\nShift-click to add to the previously opened plot");
  ImGui::SameLine(0.0f, spacing);
  if (iconButton("remove", icon::X_LG) && !editor_open_on_press_) {
    pending_action_ = [this, sig]() { UndoStack::instance()->push(new RemoveSigCommand(model_.msgId(), sig)); };
  }
  ImGui::SetItemTooltip("Remove signal");
  button_size_ = size;
}

ValueDescriptionDlg::ValueDescriptionDlg(const ValueDescription &descriptions) {
  for (auto &[val, desc] : descriptions) {
    table_.emplace_back(utils::toString(val), desc);
  }
}

bool ValueDescriptionDlg::draw() {
  const std::string popup_id = title + "###ValueDescriptionDlg";
  if (!opened_) {
    ImGui::OpenPopup(popup_id.c_str());
    opened_ = true;
  }
  setNextDialogWindow(ImVec2(500.0f, 0.0f));
  bool open = true;
  // not drawn while the dock is collapsed or another modal is on top; only closed once the popup is gone
  if (!ImGui::BeginPopupModal(popup_id.c_str(), &open, ImGuiWindowFlags_NoSavedSettings)) return ImGui::IsPopupOpen(popup_id.c_str());

  bool closing = false;
  if (stepButton("add", true, "Add")) {
    table_.emplace_back("", "");
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(current_row_ == -1);
  if (stepButton("remove", false, "Remove") && current_row_ < table_.size()) {
    table_.erase(table_.begin() + current_row_);
    current_row_ = -1;
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
    for (int row = 0; row < table_.size(); ++row) {
      ImGui::PushID(row);
      ImGui::TableNextRow();
      if (row == current_row_) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, ImGui::GetColorU32(ImGuiCol_Header));
      ImGui::TableSetColumnIndex(0);
      ImGui::AlignTextToFramePadding();
      ImGui::TextColored(row == current_row_ ? palette().text_selected : palette().text, "%d", row + 1);
      ImGui::TableSetColumnIndex(1);
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (valueDescriptionEditor(0, &table_[row].first)) current_row_ = row;
      ImGui::TableSetColumnIndex(2);
      ImGui::SetNextItemWidth(-FLT_MIN);
      if (valueDescriptionEditor(1, &table_[row].second)) current_row_ = row;
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

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
  for (int i = 0; i < table_.size(); ++i) {
    std::string val = utils::trimmed(table_[i].first);
    std::string desc = utils::trimmed(table_[i].second);
    if (!val.empty() && !desc.empty()) {
      val_desc.push_back({utils::toDouble(val), desc});
    }
  }
  accepted = true;
}
