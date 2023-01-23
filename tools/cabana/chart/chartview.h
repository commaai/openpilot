#pragma once

#include <QGraphicsProxyWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
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

  struct SigItem {
    QString msg_id;
    uint8_t source = 0;
    uint32_t address = 0;
    const Signal *sig = nullptr;
    QLineSeries *series = nullptr;
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

  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QPointF track_pt;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  QList<SigItem> sigs;
  double cur_sec = 0;
  const QString mime_type = "application/x-cabanachartview";
};
