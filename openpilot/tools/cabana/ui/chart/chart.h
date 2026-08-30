#pragma once

#include <cfloat>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "tools/cabana/ui/chart/tiplabel.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/utils/util.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/imgui_util.h"

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};

class ChartsWidget;
class ChartView {
public:
  ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent = nullptr);
  void addSignal(const MessageId &msg_id, const cabana::Signal *sig);
  bool hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const;
  void updateSeries(const cabana::Signal *sig = nullptr, const MessageEventsMap *msg_new_events = nullptr);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(SeriesType type);
  void showTip(double sec);
  void hideTip();
  void draw(float width);  // one chart of settings.chart_height
  void drawGhost(float width);  // the same tile rendered again, without handling any input
  double secondsAtPoint(const ImVec2 &pt) const {
    return x_min + (pt.x - layout_.plot_area.Min.x) * (x_max - x_min) / std::max(layout_.plot_area.GetWidth(), 1.0f);
  }

  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    CabanaColor color;
    bool visible = true;
    std::vector<ImPlotPoint> vals;
    std::vector<ImPlotPoint> step_vals;
    ImPlotPoint track_pt{};
    SegmentTree segment_tree;
    double min = 0;
    double max = 0;
  };

private:
  void signalUpdated(const cabana::Signal *sig);
  void manageSignals();
  void msgRemoved(MessageId id) { removeIf([=](auto &s) { return s.msg_id.address == id.address && !dbc()->msg(id); }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

  void appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                       std::vector<ImPlotPoint> &vals, std::vector<ImPlotPoint> &step_vals);
  void createToolButtons();
  void contextMenuEvent();
  void mousePressEvent();
  void mouseMoveEvent();
  void mouseReleaseEvent();
  void resizeEvent();
  ImVec2 sizeHint() const;
  void updateAxisY();
  void paintEvent();
  void drawStaticLayer();
  void drawAxes();
  void drawLegend();
  void drawSeries();
  void drawForeground();
  void drawSignalValue();
  void drawTimeline();
  void drawRubberBandTimeRange();
  void drawMenuActions();  // the series type / manage / split entries shared by the menu button and the context menu
  int xAxisPrecision() const;
  std::tuple<double, double, int> getNiceAxisNumbers(double min, double max, int tick_count);
  double niceNumber(double x, bool ceiling);
  CabanaColor uniqueColor(CabanaColor color, const cabana::Signal *exclude = nullptr) const;
  void removeIf(std::function<bool(const SigItem &)> predicate);
  void takeSignalsFrom(ChartView *source);
  void setDropHighlight(bool highlight) { can_drop = highlight; }
  inline void clearTrackPoints() { for (auto &s : sigs) s.track_pt = {}; }
  inline float xPos(double sec) const { return layout_.plot_area.Min.x + (sec - x_min) / (x_max - x_min) * layout_.plot_area.GetWidth(); }
  inline float yPos(double val) const { return layout_.plot_area.Max.y - (val - y_min) / (y_max - y_min) * layout_.plot_area.GetHeight(); }

  // layout
  struct Layout {
    ImRect rect;  // the whole chart widget, screen coordinates
    ImRect plot_area;
    ImRect move_icon_rect;
    ImRect close_btn_rect;
    ImRect manage_btn_rect;
    std::vector<ImRect> legend_rects;
    float header_bottom = 0;
    bool plot_hovered = false;
  } layout_;
  // axes
  double x_min;
  double x_max;
  double y_min = 0;
  double y_max = 1;
  int y_tick_count = 3;
  int y_precision = 0;
  std::string y_unit;
  // interaction
  enum class MouseMode { None, Rubber, Scrub };
  MouseMode mouse_mode = MouseMode::None;
  ImVec2 press_pos;
  ImRect rubber_rect;
  bool resume_after_scrub = false;
  bool drawing_ghost = false;  // drawing the drag preview: no mouse handling, no tip
  ImGuiID context_menu_id = 0;

  TipLabel tip_label;
  std::vector<SigItem> sigs;
  double cur_sec = 0;
  SeriesType series_type = SeriesType::Line;
  bool can_drop = false;
  double tooltip_x = -1;
  ChartsWidget *charts_widget;
  Connections connections_;
  friend class ChartsWidget;
  friend class ChartsContainer;
};
