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

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(QWidget *parent = nullptr);
  void addSeries(const QString &msg_id, const Signal *sig);
  bool hasSeries(const QString &msg_id, const Signal *sig) const;
  void updateSeries(const Signal *sig = nullptr, const std::vector<Event*> *events = nullptr, bool clear = true);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(QAbstractSeries::SeriesType type);
  void updatePlotArea(int left);

  struct SigItem {
    QString msg_id;
    uint8_t source = 0;
    uint32_t address = 0;
    const Signal *sig = nullptr;
    QXYSeries *series = nullptr;
    QVector<QPointF> vals;
    uint64_t last_value_mono_time = 0;
  };

signals:
  void seriesRemoved(const QString &id, const Signal *sig);
  void seriesAdded(const QString &id, const Signal *sig);
  void zoomIn(double min, double max);
  void zoomReset();
  void remove();
  void axisYLabelWidthChanged(int w);

private slots:
  void msgUpdated(uint32_t address);
  void signalUpdated(const Signal *sig);
  void manageSeries();
  void handleMarkerClicked();
  void msgRemoved(uint32_t address) { removeIf([=](auto &s) { return s.address == address; }); }
  void signalRemoved(const Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

private:
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
  QXYSeries *createSeries(QAbstractSeries::SeriesType type, QColor color);
  void updateSeriesPoints();
  void removeIf(std::function<bool(const SigItem &)> predicate);

  int y_label_width = 0;
  int align_to = 0;
  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QVector<QPointF> track_pts;
  QGraphicsPixmapItem *move_icon;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  QGraphicsRectItem *background;
  QList<SigItem> sigs;
  double cur_sec = 0;
  const QString mime_type = "application/x-cabanachartview";
  QAbstractSeries::SeriesType series_type = QAbstractSeries::SeriesTypeLine;
  QAction *line_series_action;
  QAction *scatter_series_action;
  friend class ChartsWidget;
 };

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show, bool merge);
  inline bool hasSignal(const QString &id, const Signal *sig) { return findChart(id, sig) != nullptr; }

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
  ChartView *findChart(const QString &id, const Signal *sig);

  QLabel *title_label;
  QLabel *range_lb;
  QSlider *range_slider;
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
    ListItem(const QString &msg_id, const Signal *sig, QListWidget *parent) : msg_id(msg_id), sig(sig), QListWidgetItem(parent) {}
    QString msg_id;
    const Signal *sig;
  };

  SeriesSelector(QString title, QWidget *parent);
  QList<ListItem *> seletedItems();
  inline void addSelected(const QString &id, const Signal *sig) { addItemToList(selected_list, id, sig, true); }

private:
  void updateAvailableList(int index);
  void addItemToList(QListWidget *parent, const QString id, const Signal *sig, bool show_msg_name = false);
  void add(QListWidgetItem *item);
  void remove(QListWidgetItem *item);

  QComboBox *msgs_combo;
  QListWidget *available_list;
  QListWidget *selected_list;
};
