#pragma once

#include <map>

#include <QLabel>
#include <QGraphicsLineItem>
#include <QGraphicsSimpleTextItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const QString &id, const Signal *sig, QWidget *parent = nullptr);
  void updateSeries();

private:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;

  void rangeChanged(qreal min, qreal max);
  void updateAxisY();
  void updateState();

  QGraphicsLineItem *track_line;
  QGraphicsSimpleTextItem *value_text;
  QGraphicsLineItem *line_marker;
  QList<QPointF> vals;
  QString id;
  const Signal *signal;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const Signal *sig, QWidget *parent);
  void updateTitle();
  void setHeight(int height);

signals:
  void remove();

public:
  QString id;
  const Signal *signal;
  QLabel *title;
  ChartView *chart_view = nullptr;
};

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(const QString &id, const Signal *sig);
  void removeChart(const Signal *sig);

signals:
  void dock(bool floating);

private:
  void updateState();
  void updateTitleBar();
  void removeAll();
  bool eventFilter(QObject *obj, QEvent *event);

  QWidget *title_bar;
  QLabel *title_label;
  QLabel *range_label;
  bool docking = true;
  QPushButton *dock_btn;
  QPushButton *reset_zoom_btn;
  QPushButton *remove_all_btn;
  QVBoxLayout *charts_layout;
  QHash<const Signal *, ChartWidget *> charts;
};
