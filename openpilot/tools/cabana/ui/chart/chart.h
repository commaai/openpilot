#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "tools/cabana/core/analysis.h"
#include "tools/cabana/ui/widgets/cameraview.h"
#include "tools/cabana/ui/chart/mapwidget.h"
#include "tools/cabana/ui/chart/recordedcamera.h"
#include "tools/cabana/ui/chart/tiplabel.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/utils/util.h"

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};
inline constexpr const char *SERIES_TYPE_NAMES[] = {"Line", "Step Line", "Scatter"};

// the message part of a legend entry, drawn after the signal name
inline std::string msgLabel(const MessageId &id) { return " " + msgName(id) + " " + id.toString(); }

class ChartsWidget;
class ChartView {
public:
  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    std::string cereal_path;
    std::string display_name;
    std::optional<cabana::PythonSeries> python;
    std::string key() const { return python ? cabana::pythonSeriesKey(*python) : cereal_path; }
    bool isCereal() const { return !cereal_path.empty(); }
    std::string name() const { if (!display_name.empty()) return display_name; return isCereal() ? (python ? cereal_path : "/" + cereal_path) : sig->name; }
    std::string unit() const { return isCereal() ? "" : sig->unit; }
    std::string label() const { return isCereal() ? " cereal" : msgLabel(msg_id); }
    std::string formatValue(double value, bool with_unit = true) const {
      if (!std::isfinite(value)) return "--";
      if (!isCereal() && !transform.derivative && transform.scale == 1 && transform.offset == 0) {
        return sig->formatValue(value, with_unit);
      }
      if (!isCereal()) return doubleToString(value);
      if (transform.derivative || transform.scale != 1 || transform.offset != 0 ||
          value < 0 || value > 65535 || std::floor(value) != value) return doubleToString(value);
      auto label = enum_labels.find(static_cast<uint16_t>(value));
      return label != enum_labels.end() ? label->second : doubleToString(value);
    }
    std::map<uint16_t, std::string> enum_labels;
    std::map<std::string, uint16_t> text_codes;
    cabana::CerealPoints raw_points;
    cabana::SeriesTransform transform;
    std::string error;
    CabanaColor color;
    bool visible = true;
    std::vector<ImPlotPoint> vals;
    std::vector<ImPlotPoint> step_vals;
    ImPlotPoint track_pt{};
    SegmentTree segment_tree;
    double min = 0;
    double max = 0;
  };

  ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent);
  void addSignal(const MessageId &msg_id, const cabana::Signal *sig);
  void addCerealSignal(const std::string &path);
  void addPythonSignal(const std::string &name, const cabana::PythonSeries &spec);
  void applyLayout(const cabana::LayoutChart &layout);
  cabana::LayoutChart analysisLayout() const;
  bool hasCerealSignal(const std::string &path) const;
  void setCerealSeries(const std::string &path, const cabana::CerealPoints &points);
  void setCerealError(const std::string &key, const std::string &error);
  bool hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const;
  void updateSeries(const cabana::Signal *sig = nullptr, const MessageEventsMap *msg_new_events = nullptr);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(SeriesType type) { series_type_ = type; }
  void showTip(double sec);
  void hideTip();
  void draw(float width, float height = -1);
  bool isPlot() const { return kind_ == "plot"; }
  bool isMap() const { return kind_ == "map"; }
  void setMapError(const std::string &error) { pane_error_ = error; }
  void setMapSeries(const std::map<std::string, cabana::CerealPoints> &series);
  void hidePane() { if (camera_) camera_->setVisible(false); }  // one chart of settings.chart_height
  void drawGhost(float width);  // the same tile rendered again, without handling any input
  void removeIf(std::function<bool(const SigItem &)> predicate);
  void takeSignalsFrom(ChartView *source);
  // every signal but the first, with its original color, for a split into one chart per signal
  std::vector<SigItem> takeExtraSignals();
  void adoptSignal(SigItem s);
  void setDropHighlight(bool highlight) { can_drop_ = highlight; }
  const std::vector<SigItem> &signals() const { return sigs_; }
  const ImRect &rect() const { return layout_.rect; }  // the whole chart widget, screen coordinates
  bool plotHovered() const { return layout_.plot_hovered; }
  double secondsAtPoint(const ImVec2 &pt) const {
    const double fraction = (pt.x - layout_.plot_area.Min.x) / std::max(layout_.plot_area.GetWidth(), 1.0f);
    return x_min_ + (flip_x_ ? 1 - fraction : fraction) * (x_max_ - x_min_);
  }

private:
  using PointIter = std::vector<ImPlotPoint>::const_iterator;

  void signalUpdated(const cabana::Signal *sig);
  void manageSignals();
  void msgRemoved(MessageId id) { removeIf([=](auto &s) { return !s.isCereal() && s.msg_id.address == id.address && !dbc()->msg(id); }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

  void appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                       cabana::CerealPoints &points);
  void rebuildSeries(SigItem &series);
  void createToolButtons();
  void drawContextMenu();
  void handleMousePress();
  void handleMouseMove();
  void handleMouseRelease();
  void updateLayout();
  void updateAxisY();
  void paint();
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
  // the last sample at or before sec, nullptr when there is none inside the visible range
  const ImPlotPoint *lastPointBefore(const SigItem &s, double sec) const;
  // the samples inside [x_min_, x_max_)
  std::pair<PointIter, PointIter> visibleRange(const std::vector<ImPlotPoint> &points) const;
  inline void clearTrackPoints() { for (auto &s : sigs_) s.track_pt = {}; }
  inline float xPos(double sec) const {
    const double fraction = (sec - x_min_) / (x_max_ - x_min_);
    return layout_.plot_area.Min.x + (flip_x_ ? 1 - fraction : fraction) * layout_.plot_area.GetWidth();
  }
  inline float yPos(double val) const {
    const double fraction = (val - y_min_) / (y_max_ - y_min_);
    return layout_.plot_area.Max.y - (flip_y_ ? 1 - fraction : fraction) * layout_.plot_area.GetHeight();
  }

  // layout
  struct Layout {
    ImRect rect;  // the whole chart widget, screen coordinates
    ImRect content_rect;  // the same inset on all four sides, including during a drag
    ImRect plot_area;
    ImRect move_icon_rect;
    ImRect close_btn_rect;
    ImRect manage_btn_rect;
    std::vector<ImRect> legend_rects;
    float header_bottom = 0;
    bool plot_hovered = false;
  } layout_;
  // axes
  double x_min_;
  double x_max_;
  double y_min_ = 0;
  double y_max_ = 1;
  int y_tick_count_ = 3;
  int y_precision_ = 1;
  std::string y_unit_;
  // interaction
  enum class MouseMode { None, Rubber, Scrub };
  MouseMode mouse_mode_ = MouseMode::None;
  ImVec2 press_pos_;
  ImRect rubber_rect_;
  bool resume_after_scrub_ = false;
  bool drawing_ghost_ = false;  // drawing the drag preview: no mouse handling, no tip
  ImGuiID context_menu_id_ = 0;

  std::string title_;
  std::optional<std::pair<double, double>> fixed_y_limits_;
  TipLabel tip_label_;
  std::vector<SigItem> sigs_;
  double cur_sec_ = 0;
  std::string pane_error_;
  bool flip_x_ = false, flip_y_ = false;
  std::string kind_ = "plot", camera_view_ = "road";
  std::optional<std::pair<double, double>> fixed_x_limits_;
  std::unique_ptr<CameraWidget> camera_;
  std::unique_ptr<cabana::RecordedCameraWidget> recorded_camera_;
  std::unique_ptr<cabana::MapWidget> map_;
  SeriesType series_type_ = SeriesType::Line;
  bool can_drop_ = false;
  double tooltip_x_ = -1;
  ChartsWidget *charts_widget_;
  Connections connections_;
};
