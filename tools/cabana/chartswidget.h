#pragma once

#include <QComboBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QListWidget>
#include <QGraphicsProxyWidget>
#include <QVBoxLayout>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(QWidget *parent = nullptr);
  ~ChartView();
  void addSeries(const QString &msg_id, const Signal *sig);
  void removeSeries(const QString &msg_id, const Signal *sig);
  bool hasSeries(const QString &msg_id, const Signal *sig) const;
  void updateSeries(const Signal *sig = nullptr);
  void setEventsRange(const std::pair<double, double> &range);
  void setDisplayRange(double min, double max);

  struct SigItem {
    QString msg_id;
    uint8_t source = 0;
    uint32_t address = 0;
    const Signal *sig = nullptr;
    QLineSeries *series = nullptr;
    double min_y = 0;
    double max_y = 0;
    QVector<QPointF> vals;
  };

signals:
  void seriesRemoved(const QString &id, const Signal *sig);
  void zoomIn(double min, double max);
  void zoomReset();
  void remove();

private slots:
  void msgRemoved(uint32_t address);
  void msgUpdated(uint32_t address);
  void signalUpdated(const Signal *sig);
  void signalRemoved(const Signal *sig);
  void manageSeries();

private:
  QList<ChartView::SigItem>::iterator removeSeries(const QList<ChartView::SigItem>::iterator &it);
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void adjustChartMargins();
  void updateAxisY();
  void updateTitle();
  void updateFromSettings();
  void drawForeground(QPainter *painter, const QRectF &rect) override;
  void applyNiceNumbers(qreal min, qreal max);
  qreal niceNumber(qreal x, bool ceiling);

  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QPointF track_pt;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  std::pair<double, double> events_range = {0, 0};
  QList<SigItem> sigs;
 };

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show, bool merge);
  void removeChart(ChartView *chart);
  inline bool isChartOpened(const QString &id, const Signal *sig) { return findChart(id, sig) != nullptr; }

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void chartOpened(const QString &id, const Signal *sig);
  void chartClosed(const QString &id, const Signal *sig);

private:
  void eventsMerged();
  void updateState();
  void updateDisplayRange();
  void zoomIn(double min, double max);
  void zoomReset();
  void updateToolBar();
  void removeAll();
  void showAllData();
  bool eventFilter(QObject *obj, QEvent *event) override;
  ChartView *findChart(const QString &id, const Signal *sig);

  QLabel *title_label;
  QLabel *range_label;
  bool docking = true;
  QAction *show_all_values_btn;
  QAction *dock_btn;
  QAction *reset_zoom_btn;
  QAction *remove_all_btn;
  QVBoxLayout *charts_layout;
  QList<ChartView *> charts;
  uint32_t max_chart_range = 0;
  bool is_zoomed = false;
  std::pair<double, double> event_range;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
  bool use_dark_theme = false;
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
