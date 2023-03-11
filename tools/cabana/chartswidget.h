#pragma once

#include <QDragEnterEvent>
#include <QGridLayout>
#include <QLabel>
#include <QListWidget>
#include <QGraphicsPixmapItem>
#include <QGraphicsProxyWidget>
#include <QSlider>
#include <QtCharts/QChartView>
#include <QtCharts/QLegendMarker>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>

#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
using namespace QtCharts;

const int CHART_MIN_WIDTH = 300;

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(QWidget *parent = nullptr);
  void addSeries(const MessageId &msg_id, const cabana::Signal *sig);
  bool hasSeries(const MessageId &msg_id, const cabana::Signal *sig) const;
  void updateSeries(const cabana::Signal *sig = nullptr, const std::vector<Event*> *events = nullptr, bool clear = true);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(SeriesType type);
  void updatePlotArea(int left);

  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    QXYSeries *series = nullptr;
    QVector<QPointF> vals;
    QVector<QPointF> step_vals;
    uint64_t last_value_mono_time = 0;
    QPointF track_pt{};
    SegmentTree segment_tree;
  };

signals:
  void seriesRemoved(const MessageId &id, const cabana::Signal *sig);
  void seriesAdded(const MessageId &id, const cabana::Signal *sig);
  void zoomIn(double min, double max);
  void zoomReset();
  void remove();
  void axisYLabelWidthChanged(int w);

private slots:
  void msgUpdated(uint32_t address);
  void signalUpdated(const cabana::Signal *sig);
  void manageSeries();
  void handleMarkerClicked();
  void msgRemoved(uint32_t address) { removeIf([=](auto &s) { return s.msg_id.address == address; }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

private:
  void createToolButtons();
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void dragMoveEvent(QDragMoveEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  QSize sizeHint() const override { return {CHART_MIN_WIDTH, settings.chart_height}; }
  void updateAxisY();
  void updateTitle();
  void drawForeground(QPainter *painter, const QRectF &rect) override;
  std::tuple<double, double, int> getNiceAxisNumbers(qreal min, qreal max, int tick_count);
  qreal niceNumber(qreal x, bool ceiling);
  QXYSeries *createSeries(SeriesType type, QColor color);
  void updateSeriesPoints();
  void removeIf(std::function<bool(const SigItem &)> predicate);
  inline void clearTrackPoints() { for (auto &s : sigs) s.track_pt = {}; }

  int y_label_width = 0;
  int align_to = 0;
  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QGraphicsPixmapItem *move_icon;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  QGraphicsRectItem *background;
  QList<SigItem> sigs;
  double cur_sec = 0;
  const QString mime_type = "application/x-cabanachartview";
  SeriesType series_type = SeriesType::Line;
  bool is_scrubbing = false;
  bool resume_after_scrub = false;
  friend class ChartsWidget;
 };

class ChartsWidget : public QFrame {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);
  inline bool hasSignal(const MessageId &id, const cabana::Signal *sig) { return findChart(id, sig) != nullptr; }

public slots:
  void setColumnCount(int n);
  void removeAll();

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void seriesChanged();

private:
  void resizeEvent(QResizeEvent *event) override;
  void alignCharts();
  void newChart();
  ChartView * createChart();
  void removeChart(ChartView *chart);
  void eventsMerged();
  void updateState();
  void zoomIn(double min, double max);
  void zoomReset();
  void updateToolBar();
  void setMaxChartRange(int value);
  void updateLayout();
  void settingChanged();
  bool eventFilter(QObject *obj, QEvent *event) override;
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);

  QLabel *title_label;
  QLabel *range_lb;
  LogSlider *range_slider;
  QAction *range_lb_action;
  QAction *range_slider_action;
  bool docking = true;
  QAction *dock_btn;
  QAction *reset_zoom_action;
  QAction *remove_all_btn;
  QGridLayout *charts_layout;
  QList<ChartView *> charts;
  uint32_t max_chart_range = 0;
  bool is_zoomed = false;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
  bool use_dark_theme = false;
  QAction *columns_action;
  int column_count = 1;
  int current_column_count = 0;
};

class SeriesSelector : public QDialog {
public:
  struct ListItem : public QListWidgetItem {
    ListItem(const MessageId &msg_id, const cabana::Signal *sig, QListWidget *parent) : msg_id(msg_id), sig(sig), QListWidgetItem(parent) {}
    MessageId msg_id;
    const cabana::Signal *sig;
  };

  SeriesSelector(QString title, QWidget *parent);
  QList<ListItem *> seletedItems();
  inline void addSelected(const MessageId &id, const cabana::Signal *sig) { addItemToList(selected_list, id, sig, true); }

private:
  void updateAvailableList(int index);
  void addItemToList(QListWidget *parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name = false);
  void add(QListWidgetItem *item);
  void remove(QListWidgetItem *item);

  QComboBox *msgs_combo;
  QListWidget *available_list;
  QListWidget *selected_list;
};
