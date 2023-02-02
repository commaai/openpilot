#pragma once

#include <QComboBox>
#include <QDragEnterEvent>
#include <QGridLayout>
#include <QLabel>
#include <QListWidget>
#include <QGraphicsProxyWidget>
#include <QSlider>
#include <QTimer>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>

#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(QWidget *parent = nullptr);
  void addSeries(const QString &msg_id, const Signal *sig);
  void addSeries(const QList<QStringList> &series_list);
  void removeSeries(const QString &msg_id, const Signal *sig);
  bool hasSeries(const QString &msg_id, const Signal *sig) const;
  void updateSeries(const Signal *sig = nullptr, const std::vector<Event*> *events = nullptr, bool clear = true);
  void updatePlot(double cur, double min, double max);
  void setPlotAreaLeftPosition(int pos);
  qreal getYAsixLabelWidth() const;
  void setSeriesType(QAbstractSeries::SeriesType type);

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
  void axisYUpdated();

private slots:
  void msgRemoved(uint32_t address);
  void msgUpdated(uint32_t address);
  void signalUpdated(const Signal *sig);
  void signalRemoved(const Signal *sig);
  void manageSeries();

private:
  QList<ChartView::SigItem>::iterator removeItem(const QList<ChartView::SigItem>::iterator &it);
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void dragMoveEvent(QDragMoveEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void updateAxisY();
  void updateTitle();
  void drawForeground(QPainter *painter, const QRectF &rect) override;
  void applyNiceNumbers(qreal min, qreal max);
  qreal niceNumber(qreal x, bool ceiling);
  QXYSeries *createSeries(QAbstractSeries::SeriesType type);

  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QVector<QPointF> track_pts;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  QList<SigItem> sigs;
  double cur_sec = 0;
  const QString mime_type = "application/x-cabanachartview";
  QAbstractSeries::SeriesType series_type = QAbstractSeries::SeriesTypeLine;
  QAction *line_series_action;
  QAction *scatter_series_action;
 };

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
  QAction *columns_lb_action;
  QAction *columns_cb_action;
  QComboBox *columns_cb;
  int column_count = 1;
  const int CHART_MIN_WIDTH = 300;
};

class SeriesSelector : public QDialog {
  Q_OBJECT

public:
  SeriesSelector(QWidget *parent);
  void addSeries(const QString &id, const QString& msg_name, const QString &sig_name);
  QList<QStringList> series();

private slots:
  void msgSelected(int index);
  void addSignal(QListWidgetItem *item);

private:
  QComboBox *msgs_combo;
  QListWidget *sig_list;
  QListWidget *chart_series;
};
