#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/core/observable.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/chart/sparkline.h"

// QValidator ports (utils/qtutil.h): InputText char filters, rejecting the characters the Qt validators reject.
// The std::string validators in utils/util.h are run again when the edit is committed.
int nameValidator(ImGuiInputTextCallbackData *data);
int nodeValidator(ImGuiInputTextCallbackData *data);
int doubleValidator(ImGuiInputTextCallbackData *data);
int nonWhitespaceValidator(ImGuiInputTextCallbackData *data);
// QLineEdit with an optional validator; `s` grows through the resize callback like inputText() in imgui_util.h
bool validatedInput(const char *label, std::string *s, ImGuiInputTextCallback validator, const char *hint = "",
                    ImGuiInputTextFlags flags = 0);

// QVariant stand-in for SignalModel::setData: the editors hand over text, numbers, check states or a value table
class ItemValue {
public:
  ItemValue(const std::string &s) : str_(s) {}
  ItemValue(const char *s) : str_(s) {}
  ItemValue(int v) : str_(std::to_string(v)) {}
  ItemValue(bool v) : str_(v ? "1" : "0") {}
  ItemValue(double v) : str_(doubleToString(v)) {}
  ItemValue(const ValueDescription &v) : val_desc_(v) {}
  std::string toString() const { return str_; }
  int toInt() const { return std::atoi(str_.c_str()); }
  bool toBool() const { return str_ == "1" || str_ == "true"; }
  double toDouble() const { return std::atof(str_.c_str()); }
  const ValueDescription &toValueDescription() const { return val_desc_; }

private:
  std::string str_;
  ValueDescription val_desc_;
};

class SignalModel {
public:
  struct Item {
    enum Type {Root, Sig, Name, Size, Node, Endian, Signed, Offset, Factor, SignalType, MultiplexValue, ExtraInfo, Unit, Comment, Min, Max, Desc };
    ~Item() { for (auto c : children) delete c; }
    inline int row() {
      auto it = std::find(parent->children.begin(), parent->children.end(), this);
      return it != parent->children.end() ? std::distance(parent->children.begin(), it) : -1;
    }

    Type type = Type::Root;
    Item *parent = nullptr;
    std::vector<Item *> children;

    const cabana::Signal *sig = nullptr;
    std::string title;
    bool highlight = false;
    std::string sig_val = "-";
    Sparkline sparkline;
    bool expanded = false;      // QTreeView::isExpanded
    bool chart_opened = false;  // plot_btn checked state
  };
  // Qt::ItemFlags
  enum ItemFlag { NoItemFlags = 0, ItemIsSelectable = 1, ItemIsEnabled = 2, ItemIsEditable = 4, ItemIsUserCheckable = 8 };

  SignalModel();
  int rowCount() const { return root->children.size(); }
  int columnCount() const { return 2; }
  std::string data(const Item *item, int column) const;  // Qt::DisplayRole / Qt::EditRole
  bool checkState(const Item *item) const;                // Qt::CheckStateRole, column 1
  std::string toolTip(const Item *item, int column) const;  // Qt::ToolTipRole
  int flags(const Item *item, int column) const;
  bool setData(Item *item, const ItemValue &value);
  void setMessage(const MessageId &id);
  void setFilter(const std::string &txt);
  bool saveSignal(const cabana::Signal *origin_s, cabana::Signal &s);
  int signalRow(const cabana::Signal *sig) const;

  Observable<> rowsChanged;  // modelReset, rowsInserted and rowsRemoved

private:
  void insertItem(SignalModel::Item *root_item, int pos, const cabana::Signal *sig);
  void handleSignalAdded(MessageId id, const cabana::Signal *sig);
  void handleSignalUpdated(const cabana::Signal *sig);
  void handleSignalRemoved(const cabana::Signal *sig);
  void handleMsgChanged(MessageId id);
  void refresh();

  MessageId msg_id;
  std::string filter_str;
  std::unique_ptr<Item> root;
  Connections connections_;
  friend class SignalView;
  friend class SignalItemDelegate;
};

// exec() is non-blocking: draw() returns false once closed, `accepted` tells whether Ok was pressed
class ValueDescriptionDlg {
public:
  ValueDescriptionDlg(const ValueDescription &descriptions);
  bool draw();
  ValueDescription val_desc;
  std::string title;  // setWindowTitle
  bool accepted = false;

private:
  struct Delegate {
    static bool createEditor(int column, std::string *text);
  };

  void save();
  std::vector<std::pair<std::string, std::string>> table;  // QTableWidget rows: {Value, Description}
  int current_row = -1;
  bool opened_ = false;
};

class SignalItemDelegate {
public:
  SignalItemDelegate();
  // viewport_x: left edge of the tree viewport (Qt paints child rows at absolute viewport coordinates)
  void paint(ImDrawList *painter, const ImRect &rect, const SignalModel::Item *item, int column, bool selected,
             const std::string &text, float viewport_x) const;
  float sizeHint(const SignalModel::Item *item, int column, float widget_width, const std::string &text) const;  // column width
  float rowHeight() const;
  // draws the editor for `item` at the cursor; commits through setModelData (QStyledItemDelegate commit on focus out)
  void createEditor(SignalModel::Item *item, SignalModel *model);
  void setModelData(SignalModel::Item *item, SignalModel *model, const ItemValue &value) const;
  void drawValueDescriptionDlg(SignalModel *model);  // continuation of the ValueDescriptionDlg exec() in createEditor
  static float textWidth(const std::string &text, float font_size = 0);

  ImGuiInputTextCallback name_validator, double_validator, node_validator;
  const float label_font = 11.0f;   // QFont pointSize 8
  const float minmax_font = 10.0f;  // QFont pixelSize 10
  const int color_label_width = 18;
  mutable ImVec2 button_size = {};

private:
  SignalModel::Item *editing_item_ = nullptr;  // the open QLineEdit editor
  std::string edit_text_;
  std::unique_ptr<ValueDescriptionDlg> desc_dlg_;
  const cabana::Signal *desc_sig_ = nullptr;
};

class SignalView {
public:
  SignalView(ChartsWidget *charts);
  void setMessage(const MessageId &id);
  void draw();
  void signalHovered(const cabana::Signal *sig);  // slot for BinaryView::signalHovered
  void updateChartState();
  void selectSignal(const cabana::Signal *sig, bool expand = false);
  void rowClicked(SignalModel::Item *item);
  std::string whatsThis() const;
  std::unique_ptr<SignalModel> model;

  // signals
  Observable<const cabana::Signal *> highlight;
  Observable<const MessageId &, const cabana::Signal *, bool, bool> showChart;

private:
  void rowsChanged();
  void updateToolBar();
  void setSparklineRange(int value);
  void handleSignalAdded(MessageId id, const cabana::Signal *sig);
  void handleSignalUpdated(const cabana::Signal *sig);
  void updateState(const std::set<MessageId> *msgs = nullptr);
  std::pair<int, int> visibleSignalRange();  // top-level rows, -1 when invalid

  // TreeView
  struct DrawContext {
    ImDrawList *draw_list;
    float viewport_x;
    float width;
    float row_height;
    float name_width = 0;
    float value_column_width = 0;
    bool any_visible = false;
    const cabana::Signal *hovered_sig = nullptr;
  };
  void drawTree();
  bool drawItem(SignalModel::Item *item, int depth, DrawContext &ctx);  // returns whether the row is visible
  void drawIndexWidget(SignalModel::Item *item, const ImRect &rect);    // the [plot][remove] widget from rowsChanged
  void collapseAll();
  float max_value_width = 0;
  float value_column_width = 0;
  float name_column_width = 150;
  int first_visible_row_ = -1;
  int last_visible_row_ = -1;
  float scroll_value_ = 0, scroll_range_ = 0;
  const cabana::Signal *current_sig_ = nullptr;  // currentIndex
  SignalModel::Item::Type current_type_ = SignalModel::Item::Root;
  const cabana::Signal *scroll_to_sig_ = nullptr;
  const cabana::Signal *hovered_sig_ = nullptr;
  std::function<void()> pending_action_;  // button clicks that destroy rows run after the tree is drawn
  std::string sparkline_label;
  const int sparkline_range_max = 30;
  std::string filter_edit;
  ChartsWidget *charts;
  std::string signal_count_lb;
  std::unique_ptr<SignalItemDelegate> delegate;
  Connections connections_;
};
