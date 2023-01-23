#include "tools/cabana/chart/chartview.h"

#include <QDrag>
#include <QGraphicsLayout>
#include <QMimeData>
#include <QRubberBand>
#include <QtMath>
#include <QToolButton>
#include <QToolTip>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/chart/seriesdlg.h"

ChartView::ChartView(QWidget *parent) : QChartView(nullptr, parent) {
  QChart *chart = new QChart();
  chart->setBackgroundRoundness(0);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart->addAxis(axis_x, Qt::AlignBottom);
  chart->addAxis(axis_y, Qt::AlignLeft);
  chart->legend()->setShowToolTips(true);
  chart->layout()->setContentsMargins(0, 0, 0, 0);
  chart->setMargins({20, 11, 11, 11});

  QToolButton *remove_btn = new QToolButton();
  remove_btn->setIcon(bootstrapPixmap("x"));
  remove_btn->setAutoRaise(true);
  remove_btn->setToolTip(tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart);
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart->zValue() + 11);

  QToolButton *manage_btn = new QToolButton();
  manage_btn->setIcon(bootstrapPixmap("gear"));
  manage_btn->setAutoRaise(true);
  manage_btn->setToolTip(tr("Manage series"));
  manage_btn_proxy = new QGraphicsProxyWidget(chart);
  manage_btn_proxy->setWidget(manage_btn);
  manage_btn_proxy->setZValue(chart->zValue() + 11);

  setChart(chart);
  setRenderHint(QPainter::Antialiasing);
  // TODO: enable zoomIn/seekTo in live streaming mode.
  setRubberBand(can->liveStreaming() ? QChartView::NoRubberBand : QChartView::HorizontalRubberBand);

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &ChartView::msgUpdated);
  QObject::connect(remove_btn, &QToolButton::clicked, this, &ChartView::remove);
  QObject::connect(manage_btn, &QToolButton::clicked, this, &ChartView::manageSeries);
}

qreal ChartView::getYAsixLabelWidth() const {
  QFontMetrics fm(axis_y->labelsFont());
  int n = qMax(int(-  qFloor(std::log10((axis_y->max() - axis_y->min()) / (axis_y->tickCount() - 1)))), 0) + 1;
  return qMax(fm.width(QString::number(axis_y->min(), 'f', n)), fm.width(QString::number(axis_y->max(), 'f', n))) + 20;
}

void ChartView::setPlotAreaLeftPosition(int pos) {
  if (std::ceil(chart()->plotArea().left()) != pos) {
    const float left_margin = chart()->margins().left() + pos - chart()->plotArea().left();
    chart()->setMargins(QMargins(left_margin, 11, 11, 11));
  }
}

void ChartView::addSeries(const QString &msg_id, const Signal *sig) {
  QLineSeries *series = new QLineSeries(this);

  // TODO: Due to a bug in CameraWidget the camera frames
  // are drawn instead of the graphs on MacOS. Re-enable OpenGL when fixed
#ifndef __APPLE__
  series->setUseOpenGL(true);
#endif
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);
  auto [source, address] = DBCManager::parseId(msg_id);
  sigs.push_back({.msg_id = msg_id, .address = address, .source = source, .sig = sig, .series = series});
  updateTitle();
  updateSeries(sig);
  emit seriesAdded(msg_id, sig);
}

void ChartView::removeSeries(const QString &msg_id, const Signal *sig) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
  if (it != sigs.end()) {
    it = removeItem(it);
  }
}

bool ChartView::hasSeries(const QString &msg_id, const Signal *sig) const {
  return std::any_of(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

QList<ChartView::SigItem>::iterator ChartView::removeItem(const QList<ChartView::SigItem>::iterator &it) {
  chart()->removeSeries(it->series);
  it->series->deleteLater();
  QString msg_id = it->msg_id;
  const Signal *sig = it->sig;
  auto ret = sigs.erase(it);
  emit seriesRemoved(msg_id, sig);
  if (!sigs.isEmpty()) {
    updateAxisY();
  } else {
    emit remove();
  }
  return ret;
}

void ChartView::signalUpdated(const Signal *sig) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.sig == sig; })) {
    updateTitle();
    // TODO: don't update series if only name changed.
    updateSeries(sig);
  }
}

void ChartView::signalRemoved(const Signal *sig) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->sig == sig) ? removeItem(it) : ++it;
  }
}

void ChartView::msgUpdated(uint32_t address) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.address == address; }))
    updateTitle();
}

void ChartView::msgRemoved(uint32_t address) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->address == address) ? removeItem(it) : ++it;
  }
}

void ChartView::addSeries(const QList<QStringList> &series_list) {
  for (auto &s : series_list) {
    if (auto m = dbc()->msg(s[0])) {
      auto it = m->sigs.find(s[2]);
      if (it != m->sigs.end() && !hasSeries(s[0], &(it->second))) {
        addSeries(s[0], &(it->second));
      }
    }
  }
}

void ChartView::manageSeries() {
  SeriesSelector dlg(this);
  for (auto &s : sigs) {
    dlg.addSeries(s.msg_id, msgName(s.msg_id), QString::fromStdString(s.sig->name));
  }

  int ret = dlg.exec();
  if (ret == QDialog::Accepted) {
    QList<QStringList> series_list = dlg.series();
    if (series_list.isEmpty()) {
      emit remove();
    } else {
      addSeries(series_list);
      for (auto it = sigs.begin(); it != sigs.end(); /**/) {
        bool exists = std::any_of(series_list.cbegin(), series_list.cend(), [&](auto &s) {
          return s[0] == it->msg_id && s[2] == it->sig->name.c_str();
        });
        it = exists ? ++it : removeItem(it);
      }
    }
  }
}

void ChartView::resizeEvent(QResizeEvent *event) {
  QChartView::resizeEvent(event);
  int x = event->size().width() - close_btn_proxy->size().width() - 11;
  close_btn_proxy->setPos(x, 8);
  manage_btn_proxy->setPos(x - manage_btn_proxy->size().width() - 5, 8);
}

void ChartView::updateTitle() {
  for (auto &s : sigs) {
    s.series->setName(QString("<b>%1</b> <font color=\"gray\">%2 %3</font>").arg(s.sig->name.c_str()).arg(msgName(s.msg_id)).arg(s.msg_id));
  }
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != axis_x->min() || max != axis_x->max()) {
    axis_x->setRange(min, max);
    updateAxisY();
  }
  scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
}

void ChartView::updateSeries(const Signal *sig, const std::vector<Event *> *events, bool clear) {
  events = events ? events : can->events();
  for (auto &s : sigs) {
    if (!sig || s.sig == sig) {
      if (clear) {
        s.vals.clear();
        s.vals.reserve(settings.max_cached_minutes * 60 * 100);  // [n]seconds * 100hz
        s.last_value_mono_time = 0;
      }
      double route_start_time = can->routeStartTime();
      Event begin_event(cereal::Event::Which::INIT_DATA, s.last_value_mono_time);
      auto begin = std::upper_bound(events->begin(), events->end(), &begin_event, Event::lessThan());
      for (auto it = begin; it != events->end(); ++it) {
        if ((*it)->which == cereal::Event::Which::CAN) {
          for (const auto &c : (*it)->event.getCan()) {
            if (s.source == c.getSrc() && s.address == c.getAddress()) {
              auto dat = c.getDat();
              double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *s.sig);
              double ts = ((*it)->mono_time / (double)1e9) - route_start_time;  // seconds
              s.vals.push_back({ts, value});
            }
          }
        }
      }
      if (events->size()) {
        s.last_value_mono_time = events->back()->mono_time;
      }
      s.series->replace(s.vals);
      updateAxisY();
    }
  }
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.isEmpty()) return;

  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (auto &s : sigs) {
    auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
    for (auto it = begin; it != s.vals.end() && it->x() <= axis_x->max(); ++it) {
      if (it->y() < min_y) min_y = it->y();
      if (it->y() > max_y) max_y = it->y();
    }
  }

  if (min_y == std::numeric_limits<double>::max()) min_y = 0;
  if (max_y == std::numeric_limits<double>::lowest()) max_y = 0;
  if (max_y == min_y) {
    axis_y->setRange(min_y - 1, max_y + 1);
  } else {
    double range = max_y - min_y;
    applyNiceNumbers(min_y - range * 0.05, max_y + range * 0.05);
  }
  emit axisYUpdated();
}

void ChartView::applyNiceNumbers(qreal min, qreal max) {
  int tick_count = axis_y->tickCount();
  qreal range = niceNumber((max - min), true);  // range with ceiling
  qreal step = niceNumber(range / (tick_count - 1), false);
  min = qFloor(min / step);
  max = qCeil(max / step);
  tick_count = int(max - min) + 1;
  axis_y->setRange(min * step, max * step);
  axis_y->setTickCount(tick_count);
}

// nice numbers can be expressed as form of 1*10^n, 2* 10^n or 5*10^n
qreal ChartView::niceNumber(qreal x, bool ceiling) {
  qreal z = qPow(10, qFloor(std::log10(x))); //find corresponding number of the form of 10^n than is smaller than x
  qreal q = x / z; //q<10 && q>=1;
  if (ceiling) {
    if (q <= 1.0) q = 1;
    else if (q <= 2.0) q = 2;
    else if (q <= 5.0) q = 5;
    else q = 10;
  } else {
    if (q < 1.5) q = 1;
    else if (q < 3.0) q = 2;
    else if (q < 7.0) q = 5;
    else q = 10;
  }
  return q * z;
}

void ChartView::leaveEvent(QEvent *event) {
  track_pt = {0, 0};
  scene()->update();
  QChartView::leaveEvent(event);
}

void ChartView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton && !chart()->plotArea().contains(event->pos()) &&
      !manage_btn_proxy->widget()->underMouse() && !close_btn_proxy->widget()->underMouse()) {
    QMimeData *mimeData = new QMimeData;
    mimeData->setData(mime_type, QByteArray::number((qulonglong)this));
    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setPixmap(grab());
    drag->setHotSpot(event->pos());
    Qt::DropAction dropAction = drag->exec(Qt::CopyAction | Qt::MoveAction, Qt::MoveAction);
    if (dropAction == Qt::MoveAction) {
      return;
    }
  }
  QChartView::mousePressEvent(event);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  auto rubber = findChild<QRubberBand *>();
  if (event->button() == Qt::LeftButton && rubber && rubber->isVisible()) {
    rubber->hide();
    QRectF rect = rubber->geometry().normalized();
    double min = std::floor(chart()->mapToValue(rect.topLeft()).x() * 10.0) / 10.0;
    double max = std::floor(chart()->mapToValue(rect.bottomRight()).x() * 10.0) / 10.0;
    if (rubber->width() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(min);
    } else if ((max - min) >= 0.5) {
      // zoom in if selected range is greater than 0.5s
      emit zoomIn(min, max);
    }
    event->accept();
  } else if (!can->liveStreaming() && event->button() == Qt::RightButton) {
    emit zoomReset();
    event->accept();
  } else {
    QGraphicsView::mouseReleaseEvent(event);
  }
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  auto rubber = findChild<QRubberBand *>();
  bool is_zooming = rubber && rubber->isVisible();
  const auto plot_area = chart()->plotArea();
  track_pt = {0, 0};
  if (!is_zooming && plot_area.contains(ev->pos())) {
    QStringList text_list;
    const double sec = chart()->mapToValue(ev->pos()).x();
    for (auto &s : sigs) {
      QString value = "--";
      // use reverse iterator to find last item <= sec.
      auto it = std::lower_bound(s.vals.rbegin(), s.vals.rend(), sec, [](auto &p, double x) { return p.x() > x; });
      if (it != s.vals.rend() && it->x() >= axis_x->min()) {
        value = QString::number(it->y());
        auto value_pos = chart()->mapToPosition(*it);
        if (value_pos.x() > track_pt.x()) track_pt = value_pos;
      }
      text_list.push_back(QString("&nbsp;%1 : %2&nbsp;").arg(sigs.size() > 1 ? s.sig->name.c_str() : "Value").arg(value));
    }
    if (track_pt.x() == 0) track_pt = ev->pos();
    QString text = QString("<div style=\"background-color: darkGray;color: white;\">&nbsp;Time: %1 &nbsp;<br />%2</div>")
                       .arg(chart()->mapToValue(track_pt).x(), 0, 'f', 3)
                       .arg(text_list.join("<br />"));
    QPoint pt((int)track_pt.x() + 20, plot_area.top() - 20);
    QToolTip::showText(mapToGlobal(pt), text, this, plot_area.toRect());
    scene()->update();
  } else {
    QToolTip::hideText();
  }
  QChartView::mouseMoveEvent(ev);
}

void ChartView::dragMoveEvent(QDragMoveEvent *event) {
  if (event->mimeData()->hasFormat(mime_type)) {
    event->setDropAction(event->source() == this ? Qt::MoveAction : Qt::CopyAction);
    event->accept();
  } else {
    event->ignore();
  }
}

void ChartView::dropEvent(QDropEvent *event) {
  if (event->mimeData()->hasFormat(mime_type)) {
    if (event->source() == this) {
      event->setDropAction(Qt::MoveAction);
      event->accept();
    } else {
      ChartView *source_chart = (ChartView *)event->source();
      QList<QStringList> series;
      for (auto &s : source_chart->sigs) {
        series.push_back({s.msg_id, msgName(s.msg_id), QString::fromStdString(s.sig->name)});
      }
      addSeries(series);
      emit source_chart->remove();
      event->acceptProposedAction();
    }
  } else {
    event->ignore();
  }
}

void ChartView::drawForeground(QPainter *painter, const QRectF &rect) {
  qreal x = chart()->mapToPosition(QPointF{cur_sec, 0}).x();
  x = std::clamp(x, chart()->plotArea().left(), chart()->plotArea().right());
  qreal y1 = chart()->plotArea().top() - 2;
  qreal y2 = chart()->plotArea().bottom() + 2;
  painter->setPen(QPen(chart()->titleBrush().color(), 2));
  painter->drawLine(QPointF{x, y1}, QPointF{x, y2});
  if (!track_pt.isNull()) {
    painter->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
    painter->drawLine(QPointF{track_pt.x(), y1}, QPointF{track_pt.x(), y2});
    painter->setBrush(Qt::darkGray);
    painter->drawEllipse(track_pt, 5, 5);
  }
}
