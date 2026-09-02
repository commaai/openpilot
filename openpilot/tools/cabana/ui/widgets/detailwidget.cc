#include "tools/cabana/ui/widgets/detailwidget.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdio>
#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

namespace {

bool iequals(const std::string &a, const std::string &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return std::tolower((unsigned char)x) == std::tolower((unsigned char)y); });
}

}  // namespace

ElidedLabel::ElidedLabel(const std::string &text) : text_(utils::trimmed(text)) {}

void ElidedLabel::draw(float width) {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  const ImVec2 pos(window->DC.CursorPos.x, window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset);
  const ImRect bb(pos, ImVec2(pos.x + width, pos.y + ImGui::GetTextLineHeight()));
  ImGui::ItemSize(bb.GetSize(), 0.0f);
  if (ImGui::ItemAdd(bb, 0)) {
    ImGui::RenderTextEllipsis(window->DrawList, bb.Min, bb.Max, bb.Max.x, text_.c_str(), nullptr, nullptr);
  }
  if (!tooltip_.empty()) ImGui::SetItemTooltip("%s", tooltip_.c_str());
  if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    clicked();
  }
}

DetailWidget::DetailWidget(ChartsWidget *charts) : charts_(charts) {
  tabbar_.setUsesScrollButtons(true);
  tabbar_.setAutoHide(true);
  tabbar_.setTabsClosable(true);
  connections_.push_back(tabbar_.currentChanged.connect([this](int index) {
    if (index >= 0) setMessage(MessageId::fromString(tabbar_.tabText(index)));
  }));
  connections_.push_back(tabbar_.tabCloseRequested.connect([this](int index) { tabbar_.removeTab(index); }));
  connections_.push_back(tabbar_.tabContextMenu.connect([this](int index) { showTabBarContextMenu(index); }));
  binary_view_ = std::make_unique<BinaryView>();
  signal_view_ = std::make_unique<SignalView>(charts);

  history_log_ = std::make_unique<LogsWidget>();

  connections_.push_back(binary_view_->signalHovered.connect([this](const cabana::Signal *s) { signal_view_->signalHovered(s); }));
  connections_.push_back(binary_view_->signalClicked.connect([this](const cabana::Signal *s) { signal_view_->selectSignal(s, true); }));
  connections_.push_back(binary_view_->editSignal.connect([this](const cabana::Signal *origin_s, cabana::Signal &s) { signal_view_->saveSignal(origin_s, s); }));
  connections_.push_back(binary_view_->showChart.connect([this](const MessageId &id, const cabana::Signal *sig, bool show, bool merge) { charts_->showChart(id, sig, show, merge); }));
  connections_.push_back(signal_view_->showChart.connect([this](const MessageId &id, const cabana::Signal *sig, bool show, bool merge) { charts_->showChart(id, sig, show, merge); }));
  connections_.push_back(signal_view_->highlight.connect([this](const cabana::Signal *sig) { binary_view_->highlight(sig); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool) { updateState(msgs); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { refresh(); }));
  connections_.push_back(charts->seriesChanged.connect([this]() { signal_view_->updateChartState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const std::optional<std::pair<double, double>> &range) {
    char text[64];
    if (range) snprintf(text, sizeof(text), "%.3f - %.3f", range->first, range->second);
    heatmap_all_text_ = range ? text : "All";
    const bool live = !range;
    if (std::exchange(heatmap_live_, live) != live) binary_view_->setHeatmapLiveMode(live);
  }));
}

void DetailWidget::drawToolBar() {
  const ImGuiStyle &style = ImGui::GetStyle();
  auto radio_width = [&](const char *label) { return ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize(label).x; };
  auto button_width = [&](const char *label) { return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2; };
  const float right_width = ImGui::CalcTextSize("Heatmap:").x + style.ItemSpacing.x + radio_width("Live") + style.ItemSpacing.x +
                            radio_width(heatmap_all_text_.c_str()) + style.ItemSpacing.x * 3 + 1.0f +
                            button_width(icon::PENCIL) + style.ItemSpacing.x + button_width(icon::X_LG);
  const float avail = ImGui::GetContentRegionAvail().x;

  ImGui::AlignTextToFramePadding();
  pushBoldFont();
  name_label_.draw(std::max(1.0f, avail - right_width - style.ItemSpacing.x));
  popBoldFont();

  alignRight(right_width);
  ImGui::TextUnformatted("Heatmap:");
  ImGui::SameLine();
  if (ImGui::RadioButton("Live##heatmap_live_", heatmap_live_) && !heatmap_live_) {
    heatmap_live_ = true;
    binary_view_->setHeatmapLiveMode(true);
  }
  ImGui::SameLine();
  if (ImGui::RadioButton((heatmap_all_text_ + "##heatmap_all").c_str(), !heatmap_live_) && heatmap_live_) {
    heatmap_live_ = false;
    binary_view_->setHeatmapLiveMode(false);
  }

  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  if (ImGui::Button(icon::PENCIL)) editMsg();
  ImGui::SetItemTooltip("Edit Message");
  ImGui::SameLine();
  ImGui::BeginDisabled(!action_remove_msg_enabled_);
  if (ImGui::Button(icon::X_LG)) UndoStack::instance()->push(new RemoveMsgCommand(msg_id_));
  ImGui::EndDisabled();
  disabledItemTooltip("Remove Message");
}

void DetailWidget::showTabBarContextMenu(int index) {
  if (ImGui::BeginPopupContextItem()) {
    if (ImGui::MenuItem("Close Other Tabs")) {
      tabbar_.moveTab(index, 0);
      tabbar_.setCurrentIndex(0);
      while (tabbar_.count() > 1) tabbar_.removeTab(1);
    }
    ImGui::EndPopup();
  }
}

int DetailWidget::findOrAddTab(const MessageId &message_id) {
  const std::string text = message_id.toString();
  int index = tabbar_.count() - 1;
  for (/**/; index >= 0; --index) {
    if (tabbar_.tabText(index) == text) break;
  }
  if (index == -1) {
    index = tabbar_.addTab(text);
    tabbar_.setTabToolTip(index, msgName(message_id));
  }
  return index;
}

void DetailWidget::setMessage(const MessageId &message_id) {
  if (std::exchange(msg_id_, message_id) == message_id) return;

  tabbar_.setCurrentIndex(findOrAddTab(message_id));

  signal_view_->setMessage(msg_id_);
  binary_view_->setMessage(msg_id_);
  history_log_->setMessage(msg_id_);
  refresh();
}

std::pair<std::string, std::vector<std::string>> DetailWidget::serializeMessageIds() const {
  std::vector<std::string> msgs;
  for (int i = 0; i < tabbar_.count(); ++i) msgs.push_back(tabbar_.tabText(i));
  return std::make_pair(msg_id_.toString(), msgs);
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
  auto msg = dbc()->msg(msg_id_);
  if (msg) {
    if (msg_id_.source == INVALID_SOURCE) {
      warnings.push_back("No messages received.");
    } else if (msg->size != can->lastMessage(msg_id_).dat.size()) {
      warnings.push_back("Message size (" + std::to_string(msg->size) + ") is incorrect.");
    }
    for (auto s : binary_view_->getOverlappingSignals()) {
      warnings.push_back(s->name + " has overlapping bits.");
    }
  }
  std::string msg_name = msg ? msg->name + " (" + msg->transmitter + ")" : msgName(msg_id_);
  name_label_.setText(msg_name);
  name_label_.setToolTip(msg_name);
  action_remove_msg_enabled_ = msg != nullptr;

  if (!warnings.empty()) {
    warning_label_.clear();
    for (size_t i = 0; i < warnings.size(); ++i) {
      if (i) warning_label_ += '\n';
      warning_label_ += warnings[i];
    }
    warning_icon_ = msg ? icon::EXCLAMATION_TRIANGLE : icon::INFO_CIRCLE;
  }
  warning_widget_visible_ = !warnings.empty();
}

void DetailWidget::updateState(const std::set<MessageId> *msgs) {
  if ((msgs && !msgs->count(msg_id_)))
    return;

  if (tab_widget_index_ == 0)
    binary_view_->updateState();
  else
    history_log_->updateState();
}

void DetailWidget::editMsg() {
  auto msg = dbc()->msg(msg_id_);
  int size = msg ? msg->size : can->lastMessage(msg_id_).dat.size();
  edit_dlg_ = std::make_unique<EditMessageDialog>(msg_id_, msgName(msg_id_), size, ImGui::GetWindowWidth());
}

void DetailWidget::drawTabWidget() {
  // the pages first, the tab bar below them
  const float tab_height = ImGui::GetFrameHeight();
  const float content_height = ImGui::GetContentRegionAvail().y - tab_height - ImGui::GetStyle().ItemSpacing.y;
  ImGui::BeginChild("tab_widget", ImVec2(0, std::max(content_height, 1.0f)), ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  if (tab_widget_index_ == 0) {
    // binary_view_ keeps its size hint, signal_view_ takes the rest
    const float min_height = binary_view_->minimumSizeHint().y;
    const float avail = ImGui::GetContentRegionAvail().y;
    const float max_height = std::max(avail - 6.0f - ImGui::GetStyle().ItemSpacing.y * 2 - 1.0f, 1.0f);
    const float height = std::clamp(min_height, 1.0f, max_height);
    ImGui::BeginChild("binary_view", ImVec2(0, height));
    binary_view_rect_ = ImGui::GetCurrentWindow()->Rect();
    binary_view_->draw();
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
    const float spacing = ImGui::GetStyle().ItemSpacing.y;
    const ImRect child_rect = ImGui::GetCurrentWindow()->Rect();
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(child_rect.Min.x, ImGui::GetItemRectMin().y - spacing),
                                              ImVec2(child_rect.Max.x, ImGui::GetItemRectMax().y + spacing),
                                              ImGui::GetColorU32(ImGuiCol_WindowBg));
    ImGui::BeginChild("signal_view", ImVec2(0, 0));
    signal_view_rect_ = ImGui::GetCurrentWindow()->Rect();
    signal_view_->draw();
    ImGui::EndChild();
  } else {
    history_log_->draw();
  }
  ImGui::EndChild();

  const std::string labels[] = {std::string(icon::FILE_EARMARK_RULED) + " Messages", std::string(icon::STOPWATCH) + " Logs"};
  // the tabs are centered in the bar: the bar itself starts at the first tab, so its separator only spans the
  // tabs and the full width one is drawn underneath it
  const ImGuiStyle &style = ImGui::GetStyle();
  float tabs_width = 0.0f;
  for (int i = 0; i < 2; ++i) {
    tabs_width += ImGui::TabItemCalcSize(labels[i].c_str(), false).x + (i ? style.ItemInnerSpacing.x : 0.0f);
  }
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  const float separator_y = ImGui::GetCursorScreenPos().y + ImGui::GetFrameHeight() - 1.0f;
  window->DrawList->AddLine(ImVec2(window->WorkRect.Min.x, separator_y), ImVec2(window->WorkRect.Max.x, separator_y),
                            ImGui::GetColorU32(ImGuiCol_TabSelected), style.TabBarBorderSize);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - tabs_width) * 0.5f));

  if (ImGui::BeginTabBar("tab_widget_tabs")) {
    for (int i = 0; i < 2; ++i) {
      if (ImGui::BeginTabItem(labels[i].c_str())) {
        if (tab_widget_index_ != i) {
          tab_widget_index_ = i;
          if (i == 1) history_log_->onShown();
          updateState();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

void DetailWidget::draw() {
  tabbar_.draw();
  drawToolBar();

  if (warning_widget_visible_) {
    ImGui::TextUnformatted(warning_icon_);
    ImGui::SameLine();
    ImGui::TextUnformatted(warning_label_.c_str());
  }

  drawTabWidget();

  if (edit_dlg_ && !edit_dlg_->draw()) {
    if (edit_dlg_->accepted()) {
      const auto r = edit_dlg_->result();
      UndoStack::instance()->push(new EditMsgCommand(r.msg_id, r.name, r.size, r.node, r.comment));
    }
    edit_dlg_.reset();
  }
}

// HelpOverlay: the whatsThis text and last drawn rect of the binary view and the signal view
std::vector<std::pair<std::string, ImRect>> DetailWidget::helpRects() const {
  std::vector<std::pair<std::string, ImRect>> rects;
  if (tab_widget_index_ == 0) {
    rects.emplace_back(binary_view_->whatsThis(), binary_view_rect_);
    rects.emplace_back(signal_view_->whatsThis(), signal_view_rect_);
  }
  return rects;
}

EditMessageDialog::EditMessageDialog(const MessageId &msg_id, const std::string &title, int size, float parent_width)
    : msg_id_(msg_id), original_name_(title), name_edit_(title), size_spin_(size), width_(parent_width * 0.9f) {
  window_title_ = "Edit message: " + msg_id.toString();

  if (auto msg = dbc()->msg(msg_id)) {
    node_ = msg->transmitter;
    comment_edit_ = msg->comment;
  }
  validateName(name_edit_);
}

EditMessageDialog::Result EditMessageDialog::result() const {
  return {msg_id_, utils::trimmed(name_edit_), utils::trimmed(node_), utils::trimmed(comment_edit_), size_spin_};
}

bool EditMessageDialog::draw() {
  if (closed_) return false;
  if (!opened_) {
    ImGui::OpenPopup(window_title_.c_str());
    opened_ = true;
  }
  setNextDialogWindow(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(width_, 0.0f), ImGuiCond_Always);  // fixed width, the height fits the form
  bool open = true;
  if (ImGui::BeginPopupModal(window_title_.c_str(), &open)) {
    const float label_width = ImGui::CalcTextSize("Comment").x + ImGui::GetStyle().ItemSpacing.x * 2;
    auto row = [&](const char *label) {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(label);
      ImGui::SameLine(label_width);
      ImGui::SetNextItemWidth(-FLT_MIN);
    };

    if (!error_label_.empty()) {
      row("");
      ImGui::TextUnformatted(error_label_.c_str());
    }
    row("Name");
    if (validatedInput("##name", &name_edit_, nameValidator)) {
      validateName(name_edit_);
    }

    row("Size");
    if (ImGui::InputInt("##size", &size_spin_)) size_spin_ = std::clamp(size_spin_, 1, CAN_MAX_DATA_BYTES);

    row("Node");
    validatedInput("##node", &node_, nameValidator);
    row("Comment");
    inputTextMultiline("##comment", &comment_edit_, ImVec2(-FLT_MIN, 192.0f));
    const bool comment_active = ImGui::IsItemActive();

    bool accept = false, reject = false;
    if (dialogButtons("OK", &accept, &reject, ok_enabled_)) {
      accepted_ = accept;
      closed_ = true;
    }
    // Enter triggers the default (OK) button
    if (!closed_ && ok_enabled_ && !comment_active && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
      accepted_ = true;
      closed_ = true;
    }
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
  error_label_.clear();
  if (!text.empty() && valid && text != original_name_) {
    valid = dbc()->msg(msg_id_.source, text) == nullptr;
    if (!valid) error_label_ = "Name already exists";
  }
  ok_enabled_ = valid;
}

DetailWidget* CenterWidget::ensureDetailWidget() {
  if (!detail_widget) {
    detail_widget = std::make_unique<DetailWidget>(charts_);
  }
  return detail_widget.get();
}

void CenterWidget::clear() {
  detail_widget.reset();
  charts_ = nullptr;  // MainWindow recreates the ChartsWidget after startStream
}

void CenterWidget::draw() {
  if (detail_widget) {
    detail_widget->draw();
  } else {
    drawWelcomeWidget();
  }
}

void CenterWidget::drawWelcomeWidget() {
  const ImVec2 win_pos = ImGui::GetWindowPos(), win_size = ImGui::GetWindowSize();
  ImGui::GetWindowDrawList()->AddRectFilled(win_pos, ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y), ImGui::GetColorU32(ImGuiCol_ChildBg));

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 origin = ImGui::GetCursorPos();
  auto centered = [&](const char *text, float y) {
    const ImVec2 size = ImGui::CalcTextSize(text);
    ImGui::SetCursorPos(ImVec2(origin.x + (avail.x - size.x) * 0.5f, y));
    ImGui::TextUnformatted(text);
  };
  ImGui::PushStyleColor(ImGuiCol_Text, colorRgb(169, 169, 169));
  float y = origin.y + avail.y * 0.5f - 90.0f;
  pushLargeFont();
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
