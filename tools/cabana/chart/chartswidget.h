#pragma once

#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QSlider>
#include <QTimer>

#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/chart/chartview.h"
#include "tools/cabana/streams/abstractstream.h"

using namespace QtCharts;

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show, bool merge);
  inline bool hasSignal(const QString &id, const Signal *sig) { return findChart(id, sig) != nullptr; }

public slots:
  void setColumnCount(int n);

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
  void removeAll();
  void setMaxChartRange(int value);
  void updateLayout();
  void settingChanged();
  bool eventFilter(QObject *obj, QEvent *event) override;
  ChartView *findChart(const QString &id, const Signal *sig);

  QLabel *title_label;
  QLabel *zoom_range_lb;
  QLabel *range_lb;
  QSlider *range_slider;
  bool docking = true;
  QAction *dock_btn;
  QAction *reset_zoom_btn;
  QAction *remove_all_btn;
  QTimer *align_charts_timer;
  QGridLayout *charts_layout;
  QList<ChartView *> charts;
  uint32_t max_chart_range = 0;
  bool is_zoomed = false;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
  bool use_dark_theme = false;
  QComboBox *columns_cb;
  int column_count = 1;
  const int CHART_MIN_WIDTH = 300;
};
