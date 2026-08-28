#include "tools/cabana/ui/widgets/detailwidget.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdio>
#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

namespace {

// QString::trimmed
std::string trimmed(const std::string &s) {
  const char *ws = " \t\n\r\f\v";
  auto b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

// QString::compare(..., Qt::CaseInsensitive) == 0
bool iequals(const std::string &a, const std::string &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return std::tolower((unsigned char)x) == std::tolower((unsigned char)y); });
}

// NameValidator: reject the edit when the new text is invalid, otherwise keep the fixed up text
void applyNameValidator(std::string &input, const std::string &before) {
  std::string fixed = input;
  input = ::validateName(fixed) == ValidState::Invalid ? before : fixed;
}

// QFontMetrics::elidedText(text, Qt::ElideRight, width)
std::string elidedText(const std::string &text, float width) {
  if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
  const float dots = ImGui::CalcTextSize("...").x;
  size_t n = text.size();
  while (n > 0) {
    --n;
    while (n > 0 && (text[n] & 0xC0) == 0x80) --n;  // keep UTF-8 sequences whole
    if (ImGui::CalcTextSize(text.c_str(), text.c_str() + n).x + dots <= width) break;
  }
  return text.substr(0, n) + "...";
}

}  // namespace

// ElidedLabel

ElidedLabel::ElidedLabel(const std::string &text) : text_(trimmed(text)) {}

void ElidedLabel::draw(float width) {
  if (width != lastWidth_) {  // resizeEvent
    lastWidth_ = width;
    lastText_ = elidedText_ = "";
  }

  const std::string &curText = text_;
  if (curText != lastText_) {
    elidedText_ = elidedText(curText, width);
    lastText_ = curText;
  }

  ImGui::TextUnformatted(elidedText_.c_str());
  if (!tooltip_.empty()) ImGui::SetItemTooltip("%s", tooltip_.c_str());
  if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    clicked();
  }
}

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts) : charts(charts) {
  // tabbar: drawn by drawTabBar(), auto hidden with fewer than two tabs

  createToolBar();

  // warning: drawn by draw() when warning_widget_visible

  // msg widget
  binary_view = std::make_unique<BinaryView>();
  signal_view = std::make_unique<SignalView>(charts);

  history_log = std::make_unique<LogsWidget>();

  connections_.push_back(binary_view->signalHovered.connect([this](const cabana::Signal *s) { signal_view->signalHovered(s); }));
  connections_.push_back(binary_view->signalClicked.connect([this](const cabana::Signal *s) { signal_view->selectSignal(s, true); }));
  connections_.push_back(binary_view->editSignal.connect([this](const cabana::Signal *origin_s, cabana::Signal &s) { signal_view->model->saveSignal(origin_s, s); }));
  connections_.push_back(binary_view->showChart.connect([this](const MessageId &id, const cabana::Signal *sig, bool show, bool merge) { this->charts->showChart(id, sig, show, merge); }));
  connections_.push_back(signal_view->showChart.connect([this](const MessageId &id, const cabana::Signal *sig, bool show, bool merge) { this->charts->showChart(id, sig, show, merge); }));
  connections_.push_back(signal_view->highlight.connect([this](const cabana::Signal *sig) { binary_view->highlight(sig); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool) { updateState(msgs); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { refresh(); }));
  connections_.push_back(charts->seriesChanged.connect([this]() { signal_view->updateChartState(); }));
}

void DetailWidget::createToolBar() {
  // the toolbar widgets are drawn by drawToolBar(); heatmap_live starts checked
  connections_.push_back(can->timeRangeChanged.connect([=](const std::optional<std::pair<double, double>> &range) {
    char text[64];
    if (range) snprintf(text, sizeof(text), "%.3f - %.3f", range->first, range->second);
    heatmap_all_text = range ? text : "All";
    // (range ? heatmap_all : heatmap_live)->setChecked(true), toggled -> setHeatmapLiveMode
    const bool live = !range;
    if (std::exchange(heatmap_live, live) != live) binary_view->setHeatmapLiveMode(live);
  }));
}

void DetailWidget::drawToolBar() {
  const ImGuiStyle &style = ImGui::GetStyle();
  auto radio_width = [&](const char *label) { return ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize(label).x; };
  auto button_width = [&](const char *label) { return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2; };
  const float right_width = ImGui::CalcTextSize("Heatmap:").x + style.ItemSpacing.x + radio_width("Live") + style.ItemSpacing.x +
                            radio_width(heatmap_all_text.c_str()) + style.ItemSpacing.x * 3 + 1.0f +
                            button_width(icon::PENCIL) + style.ItemSpacing.x + button_width(icon::X_LG);
  const float avail = ImGui::GetContentRegionAvail().x;
  const float start_x = ImGui::GetCursorPosX();

  ImGui::AlignTextToFramePadding();
  pushBoldFont();  // QLabel{font-weight:bold;}
  name_label.draw(std::max(1.0f, avail - right_width - style.ItemSpacing.x));
  popBoldFont();

  // spacer
  ImGui::SameLine(start_x + std::max(avail - right_width, 0.0f));

  // Heatmap label and radio buttons
  ImGui::TextUnformatted("Heatmap:");
  ImGui::SameLine();
  if (ImGui::RadioButton("Live##heatmap_live", heatmap_live) && !heatmap_live) {
    heatmap_live = true;
    binary_view->setHeatmapLiveMode(true);
  }
  ImGui::SameLine();
  if (ImGui::RadioButton((heatmap_all_text + "##heatmap_all").c_str(), !heatmap_live) && heatmap_live) {
    heatmap_live = false;
    binary_view->setHeatmapLiveMode(false);
  }

  // Edit and remove buttons
  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  if (ImGui::Button(icon::PENCIL)) editMsg();
  ImGui::SetItemTooltip("Edit Message");
  ImGui::SameLine();
  ImGui::BeginDisabled(!action_remove_msg_enabled);
  if (ImGui::Button(icon::X_LG)) removeMsg();
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Remove Message");
}

void DetailWidget::showTabBarContextMenu(int index) {
  if (index >= 0) {
    if (ImGui::BeginPopupContextItem()) {
      if (ImGui::MenuItem("Close Other Tabs")) {
        std::rotate(tabbar.begin(), tabbar.begin() + index, tabbar.begin() + index + 1);  // moveTab(index, 0)
        setMessage(tabbar[0].id);                                                         // setCurrentIndex(0)
        while (tabbar.size() > 1) {
          tabbar.erase(tabbar.begin() + 1);
        }
      }
      ImGui::EndPopup();
    }
  }
}

void DetailWidget::drawTabBar() {
  if (tabbar.size() < 2) return;  // setAutoHide(true)
  if (!ImGui::BeginTabBar("tabbar", ImGuiTabBarFlags_FittingPolicyScroll | ImGuiTabBarFlags_NoTooltip)) return;
  // setCurrentIndex requests made during this loop are applied on the next frame
  const bool select_current = std::exchange(tabbar_select_current, false);
  for (int i = 0; i < (int)tabbar.size(); ++i) {
    const MessageId id = tabbar[i].id;
    const std::string label = id.toString();
    bool open = true;
    const ImGuiTabItemFlags flags = (select_current && id == msg_id) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
    const bool current = ImGui::BeginTabItem(label.c_str(), &open, flags);
    if (current) ImGui::EndTabItem();
    ImGui::SetItemTooltip("%s", tabbar[i].tooltip.c_str());
    showTabBarContextMenu(i);
    // currentChanged; a programmatic selection takes effect on the next frame, ignore the old tab until then
    if (current && !select_current && id != msg_id) setMessage(id);
    if (!open) {  // tabCloseRequested
      removeTab(i);
      break;
    }
  }
  ImGui::EndTabBar();
}

// QTabBar::removeTab: closing the current tab selects the one to its right
void DetailWidget::removeTab(int index) {
  const bool was_current = tabbar[index].id == msg_id;
  tabbar.erase(tabbar.begin() + index);
  if (was_current && !tabbar.empty()) {
    setMessage(tabbar[std::clamp<int>(index, 0, tabbar.size() - 1)].id);
  }
}

int DetailWidget::findOrAddTab(const MessageId& message_id) {
  int index = tabbar.size() - 1;
  for (/**/; index >= 0; --index) {
    if (tabbar[index].id == message_id) break;
  }
  if (index == -1) {
    tabbar.push_back({message_id, msgName(message_id)});
    index = tabbar.size() - 1;
  }
  return index;
}

void DetailWidget::setMessage(const MessageId &message_id) {
  if (std::exchange(msg_id, message_id) == message_id) return;

  findOrAddTab(message_id);
  tabbar_select_current = true;

  signal_view->setMessage(msg_id);
  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);
  refresh();
}

std::pair<std::string, std::vector<std::string>> DetailWidget::serializeMessageIds() const {
  std::vector<std::string> msgs;
  for (int i = 0; i < (int)tabbar.size(); ++i) {
    MessageId id = tabbar[i].id;
    msgs.push_back(id.toString());
  }
  return std::make_pair(msg_id.toString(), msgs);
}

void DetailWidget::restoreTabs(const std::string &active_msg_id, const std::vector<std::string>& msg_ids) {
  for (const auto& str_id : msg_ids) {
    MessageId id = MessageId::fromString(str_id);
    if (dbc()->msg(id) != nullptr)
      findOrAddTab(id);
  }

  auto active_id = MessageId::fromString(active_msg_id);
  if (dbc()->msg(active_id) != nullptr)
    setMessage(active_id);
}

void DetailWidget::refresh() {
  std::vector<std::string> warnings;
  auto msg = dbc()->msg(msg_id);
  if (msg) {
    if (msg_id.source == INVALID_SOURCE) {
      warnings.push_back("No messages received.");
    } else if (msg->size != can->lastMessage(msg_id).dat.size()) {
      warnings.push_back("Message size (" + std::to_string(msg->size) + ") is incorrect.");
    }
    for (auto s : binary_view->getOverlappingSignals()) {
      warnings.push_back(s->name + " has overlapping bits.");
    }
  }
  std::string msg_name = msg ? msg->name + " (" + msg->transmitter + ")" : msgName(msg_id);
  name_label.setText(msg_name);
  name_label.setToolTip(msg_name);
  action_remove_msg_enabled = msg != nullptr;

  if (!warnings.empty()) {
    warning_label.clear();
    for (size_t i = 0; i < warnings.size(); ++i) {
      if (i) warning_label += '\n';
      warning_label += warnings[i];
    }
    warning_icon = msg ? icon::EXCLAMATION_TRIANGLE : icon::INFO_CIRCLE;
  }
  warning_widget_visible = !warnings.empty();
}

void DetailWidget::updateState(const std::set<MessageId> *msgs) {
  if ((msgs && !msgs->count(msg_id)))
    return;

  if (tab_widget_index == 0)
    binary_view->updateState();
  else
    history_log->updateState();
}

void DetailWidget::editMsg() {
  auto msg = dbc()->msg(msg_id);
  int size = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  edit_dlg_ = std::make_unique<EditMessageDialog>(msg_id, msgName(msg_id), size, ImGui::GetWindowWidth());
}

void DetailWidget::removeMsg() {
  UndoStack::instance()->push(new RemoveMsgCommand(msg_id));
}

void DetailWidget::drawTabWidget() {
  // QTabWidget::South: the pages first, the tab bar below them
  const float tab_height = ImGui::GetFrameHeight();
  const float content_height = ImGui::GetContentRegionAvail().y - tab_height - ImGui::GetStyle().ItemSpacing.y;
  ImGui::BeginChild("tab_widget", ImVec2(0, std::max(content_height, 1.0f)), ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  if (tab_widget_index == 0) {
    // splitter: binary_view keeps its size hint, signal_view takes the rest
    const float min_height = binary_view->minimumSizeHint().y;
    const float avail = ImGui::GetContentRegionAvail().y;
    const float max_height = std::max(avail - 6.0f - ImGui::GetStyle().ItemSpacing.y * 2 - 1.0f, 1.0f);
    const float height = std::clamp(std::max(splitter_pos, min_height), 1.0f, max_height);
    ImGui::BeginChild("binary_view", ImVec2(0, height));
    binary_view_rect_ = ImGui::GetCurrentWindow()->Rect();
    binary_view->draw();
    ImGui::EndChild();
    ImGui::InvisibleButton("##splitter", ImVec2(-1.0f, 6.0f));
    if (ImGui::IsItemActive()) splitter_pos = std::clamp(height + ImGui::GetIO().MouseDelta.y, min_height, max_height);
    if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    ImGui::BeginChild("signal_view", ImVec2(0, 0));
    signal_view_rect_ = ImGui::GetCurrentWindow()->Rect();
    signal_view->draw();
    ImGui::EndChild();
  } else {
    history_log->draw();
  }
  ImGui::EndChild();

  if (ImGui::BeginTabBar("tab_widget_tabs")) {
    // the Qt tabs were "&Msg" and "&Logs"; the Alt+M/Alt+L mnemonics are not ported
    const std::string labels[] = {std::string(icon::FILE_EARMARK_RULED) + " Msg", std::string(icon::STOPWATCH) + " Logs"};
    for (int i = 0; i < 2; ++i) {
      if (ImGui::BeginTabItem(labels[i].c_str())) {
        if (tab_widget_index != i) {  // currentChanged
          tab_widget_index = i;
          if (i == 1) history_log->showEvent();
          updateState();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

void DetailWidget::draw() {
  drawTabBar();
  drawToolBar();

  // warning
  if (warning_widget_visible) {
    ImGui::TextUnformatted(warning_icon);
    ImGui::SameLine();
    ImGui::TextUnformatted(warning_label.c_str());
  }

  drawTabWidget();

  if (edit_dlg_ && !edit_dlg_->draw()) {
    if (edit_dlg_->accepted()) {
      UndoStack::instance()->push(new EditMsgCommand(edit_dlg_->msg_id, trimmed(edit_dlg_->name_edit), edit_dlg_->size_spin,
                                                     trimmed(edit_dlg_->node), trimmed(edit_dlg_->comment_edit)));
    }
    edit_dlg_.reset();
  }
}

std::string DetailWidget::whatsThis() const {
  return binary_view->whatsThis();
}

// HelpOverlay: the whatsThis text and last drawn rect of the binary view and the signal view
std::vector<std::pair<std::string, ImRect>> DetailWidget::helpRects() const {
  std::vector<std::pair<std::string, ImRect>> rects;
  if (tab_widget_index == 0) {
    rects.emplace_back(binary_view->whatsThis(), binary_view_rect_);
    rects.emplace_back(signal_view->whatsThis(), signal_view_rect_);
  }
  return rects;
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const MessageId &msg_id, const std::string &title, int size, float parent_width)
    : msg_id(msg_id), original_name(title), name_edit(title), size_spin(size), width_(parent_width * 0.9f) {
  window_title_ = "Edit message: " + msg_id.toString();

  if (auto msg = dbc()->msg(msg_id)) {
    node = msg->transmitter;
    comment_edit = msg->comment;
  }
  validateName(name_edit);
}

bool EditMessageDialog::draw() {
  if (closed_) return false;
  if (!opened_) {
    ImGui::OpenPopup(window_title_.c_str());
    opened_ = true;
  }
  ImGui::SetNextWindowSize(ImVec2(width_, 0.0f), ImGuiCond_Always);  // setFixedWidth, height fits the form
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  bool open = true;
  if (ImGui::BeginPopupModal(window_title_.c_str(), &open)) {
    const float label_width = ImGui::CalcTextSize("Comment").x + ImGui::GetStyle().ItemSpacing.x * 2;
    auto row = [&](const char *label) {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(label);
      ImGui::SameLine(label_width);
      ImGui::SetNextItemWidth(-FLT_MIN);
    };

    if (error_label_visible) {
      row("");
      ImGui::TextUnformatted(error_label.c_str());
    }
    row("Name");
    const std::string name_before = name_edit;
    if (inputText("##name", &name_edit)) {
      applyNameValidator(name_edit, name_before);
      validateName(name_edit);  // textEdited
    }

    row("Size");
    if (ImGui::InputInt("##size", &size_spin)) size_spin = std::clamp(size_spin, 1, CAN_MAX_DATA_BYTES);

    row("Node");
    const std::string node_before = node;
    if (inputText("##node", &node)) applyNameValidator(node, node_before);
    row("Comment");
    ImGui::InputTextMultiline("##comment", comment_edit.data(), comment_edit.capacity() + 1, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 4),
                              ImGuiInputTextFlags_CallbackResize, imguiResizeCallback, &comment_edit);
    const bool comment_active = ImGui::IsItemActive();

    // btn_box
    ImGui::BeginDisabled(!btn_box_ok_enabled);
    if (ImGui::Button("OK", ImVec2(80.0f, 0.0f))) {
      accepted_ = true;
      closed_ = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f))) closed_ = true;
    // QDialog: Enter triggers the default (OK) button, Escape rejects
    if (!closed_ && btn_box_ok_enabled && !comment_active && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
      accepted_ = true;
      closed_ = true;
    }
    if (!closed_ && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) closed_ = true;
    if (!open) closed_ = true;
    if (closed_) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  } else {
    closed_ = true;  // closed from outside
  }
  return !closed_;
}

void EditMessageDialog::validateName(const std::string &text) {
  bool valid = !iequals(text, UNTITLED);
  error_label_visible = false;
  if (!text.empty() && valid && text != original_name) {
    valid = dbc()->msg(msg_id.source, text) == nullptr;
    if (!valid) {
      error_label = "Name already exists";
      error_label_visible = true;
    }
  }
  btn_box_ok_enabled = valid;
}

// CenterWidget

CenterWidget::CenterWidget() {}

DetailWidget* CenterWidget::ensureDetailWidget() {
  if (!detail_widget) {
    welcome_widget = false;
    detail_widget = std::make_unique<DetailWidget>(charts_);
  }
  return detail_widget.get();
}

void CenterWidget::clear() {
  detail_widget.reset();
  charts_ = nullptr;  // MainWindow recreates the ChartsWidget after startStream
  if (!welcome_widget) {
    welcome_widget = true;
  }
}

void CenterWidget::draw() {
  if (detail_widget) {
    detail_widget->draw();
  } else if (welcome_widget) {
    drawWelcomeWidget();
  }
}

void CenterWidget::drawWelcomeWidget() {
  // setBackgroundRole(QPalette::Base)
  const ImVec2 win_pos = ImGui::GetWindowPos(), win_size = ImGui::GetWindowSize();
  ImGui::GetWindowDrawList()->AddRectFilled(win_pos, ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y), ImGui::GetColorU32(ImGuiCol_ChildBg));

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 origin = ImGui::GetCursorPos();
  auto centered = [&](const char *text, float y) {
    const ImVec2 size = ImGui::CalcTextSize(text);
    ImGui::SetCursorPos(ImVec2(origin.x + (avail.x - size.x) * 0.5f, y));
    ImGui::TextUnformatted(text);
  };
  ImGui::PushStyleColor(ImGuiCol_Text, colorRgb(169, 169, 169));  // QLabel{color:darkGray;}
  float y = origin.y + avail.y * 0.5f - 90.0f;
  pushLargeFont();  // font-size:50px;font-weight:bold;
  centered("CABANA", y);
  y += ImGui::GetTextLineHeightWithSpacing();
  popLargeFont();

  auto newShortcutRow = [&](const char *title, const char *key) {
    const float w = ImGui::CalcTextSize(title).x + ImGui::CalcTextSize(key).x + 40.0f;
    ImGui::SetCursorPos(ImVec2(origin.x + (avail.x - w) * 0.5f, y));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title);
    ImGui::SameLine();
    ImGui::BeginDisabled();
    ImGui::SmallButton(key);
    ImGui::EndDisabled();
    y += ImGui::GetFrameHeightWithSpacing();
  };

  centered("<-Select a message to view details", y);
  y += ImGui::GetTextLineHeightWithSpacing();
  newShortcutRow("Pause", "Space");
  newShortcutRow("Help", "F1");
  newShortcutRow("WhatsThis", "Shift+F1");
  ImGui::PopStyleColor();
}
