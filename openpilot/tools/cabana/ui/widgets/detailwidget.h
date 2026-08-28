#pragma once

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/ui/widgets/binaryview.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/widgets/historylog.h"
#include "tools/cabana/ui/widgets/signalview.h"
#include "tools/cabana/ui/icons.h"


// utils/elidedlabel: a label that elides its text to the available width
class ElidedLabel {
public:
  explicit ElidedLabel(const std::string &text = {});
  void setText(const std::string &text) { text_ = text; }
  void setToolTip(const std::string &tip) { tooltip_ = tip; }
  void draw(float width);  // paintEvent; a width change is the resizeEvent

  Observable<> clicked;

private:
  std::string text_, tooltip_;
  std::string lastText_, elidedText_;
  float lastWidth_ = -1;
};

// modal, non-blocking: DetailWidget::draw() polls draw() and applies the result once it returns false
class EditMessageDialog {
public:
  EditMessageDialog(const MessageId &msg_id, const std::string &title, int size, float parent_width);
  void validateName(const std::string &text);
  bool draw();  // false once closed
  bool accepted() const { return accepted_; }

  MessageId msg_id;
  std::string original_name;
  bool btn_box_ok_enabled = true;
  std::string name_edit;
  std::string node;
  std::string comment_edit;
  std::string error_label;
  bool error_label_visible = false;
  int size_spin;

private:
  std::string window_title_;
  float width_;
  bool opened_ = false;
  bool accepted_ = false;
  bool closed_ = false;
};

class DetailWidget {
public:
  DetailWidget(ChartsWidget *charts);
  void setMessage(const MessageId &message_id);
  void refresh();
  void draw();  // tab bar of message ids, toolbar, warning, Msg/Logs tabs
  std::pair<std::string, std::vector<std::string>> serializeMessageIds() const;
  void restoreTabs(const std::string &active_msg_id, const std::vector<std::string> &msg_ids);
  std::string whatsThis() const;

private:
  void createToolBar();
  void drawToolBar();
  void drawTabBar();
  void drawTabWidget();
  int findOrAddTab(const MessageId& message_id);
  void showTabBarContextMenu(int index);
  void removeTab(int index);
  void editMsg();
  void removeMsg();
  void updateState(const std::set<MessageId> *msgs = nullptr);

  struct Tab {
    MessageId id;         // tabData
    std::string tooltip;  // setTabToolTip
  };

  MessageId msg_id;
  const char *warning_icon = nullptr;
  std::string warning_label;
  ElidedLabel name_label;
  bool warning_widget_visible = false;
  std::vector<Tab> tabbar;
  bool tabbar_select_current = false;  // tabbar->setCurrentIndex, applied on the next draw
  int tab_widget_index = 0;            // tab_widget->currentIndex()
  bool action_remove_msg_enabled = false;
  bool heatmap_live = true;
  std::string heatmap_all_text = "All";
  float splitter_pos = 192.0f;  // binary_view height; QAbstractScrollArea::sizeHint().height()
  std::unique_ptr<LogsWidget> history_log;
  std::unique_ptr<BinaryView> binary_view;
  std::unique_ptr<SignalView> signal_view;
  ChartsWidget *charts;
  std::unique_ptr<EditMessageDialog> edit_dlg_;
  Connections connections_;
};

class CenterWidget {
public:
  CenterWidget();
  void setChartsWidget(ChartsWidget *charts) { charts_ = charts; }  // Qt's DetailWidget got it from its MainWindow parent
  void setMessage(const MessageId &message_id) { ensureDetailWidget()->setMessage(message_id); }
  DetailWidget* getDetailWidget() { return detail_widget.get(); }
  DetailWidget* ensureDetailWidget();
  void clear();
  void draw();  // the welcome widget until a message is selected, then the DetailWidget

private:
  void drawWelcomeWidget();
  std::unique_ptr<DetailWidget> detail_widget;
  bool welcome_widget = true;
  ChartsWidget *charts_ = nullptr;
};
