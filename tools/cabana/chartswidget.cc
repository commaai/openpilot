#include "tools/cabana/chartswidget.h"

#include <QCompleter>
#include <QLineEdit>
#include <QFutureSynchronizer>
#include <QGraphicsLayout>
#include <QRubberBand>
#include <QToolBar>
#include <QToolButton>
#include <QToolTip>
#include <QtConcurrent>

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // toolbar
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  title_label = new QLabel();
  title_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(title_label);
  show_all_values_btn = toolbar->addAction("");
  toolbar->addWidget(range_label = new QLabel());
  reset_zoom_btn = toolbar->addAction("⟲");
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  remove_all_btn = toolbar->addAction("✖");
  remove_all_btn->setToolTip(tr("Remove all charts"));
  dock_btn = toolbar->addAction("");
  main_layout->addWidget(toolbar);

  // charts
  QWidget *charts_container = new QWidget(this);
  charts_layout = new QVBoxLayout(charts_container);
  charts_layout->addStretch();

  QScrollArea *charts_scroll = new QScrollArea(this);
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(charts_scroll);

  max_chart_range = settings.max_chart_x_range;
  use_dark_theme = palette().color(QPalette::WindowText).value() > palette().color(QPalette::Background).value();
  updateToolBar();

  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(can, &CANMessages::eventsMerged, this, &ChartsWidget::eventsMerged);
  QObject::connect(can, &CANMessages::updated, this, &ChartsWidget::updateState);
  QObject::connect(show_all_values_btn, &QAction::triggered, this, &ChartsWidget::showAllData);
  QObject::connect(remove_all_btn, &QAction::triggered, this, &ChartsWidget::removeAll);
  QObject::connect(reset_zoom_btn, &QAction::triggered, this, &ChartsWidget::zoomReset);
  QObject::connect(dock_btn, &QAction::triggered, [this]() {
    emit dock(!docking);
    docking = !docking;
    updateToolBar();
  });
}

void ChartsWidget::eventsMerged() {
  if (auto events = can->events(); events && !events->empty()) {
    event_range.first = (events->front()->mono_time / (double)1e9) - can->routeStartTime();
    event_range.second = (events->back()->mono_time / (double)1e9) - can->routeStartTime();
    updateState();
  }
}

void ChartsWidget::updateDisplayRange() {
  auto prev_range = display_range;
  double current_sec = can->currentSec();
  if (current_sec < display_range.first || current_sec >= (display_range.second - 5)) {
    // reached the end, or seeked to a timestamp out of range.
    display_range.first = current_sec - 5;
  }
  display_range.first = std::floor(std::max(display_range.first, event_range.first) * 10.0) / 10.0;
  display_range.second = std::floor(std::min(display_range.first + max_chart_range, event_range.second) * 10.0) / 10.0;
  if (prev_range != display_range) {
    QFutureSynchronizer<void> future_synchronizer;
    for (auto c : charts)
      future_synchronizer.addFuture(QtConcurrent::run(c, &ChartView::setEventsRange, display_range));
  }
}

void ChartsWidget::zoomIn(double min, double max) {
  zoomed_range = {min, max};
  is_zoomed = zoomed_range != display_range;
  updateToolBar();
  updateState();
  emit rangeChanged(min, max, is_zoomed);
}

void ChartsWidget::zoomReset() {
  zoomIn(display_range.first, display_range.second);
}

void ChartsWidget::updateState() {
  if (charts.isEmpty()) return;

  if (!is_zoomed) {
    updateDisplayRange();
  } else if (can->currentSec() < zoomed_range.first || can->currentSec() >= zoomed_range.second) {
    can->seekTo(zoomed_range.first);
  }

  const auto &range = is_zoomed ? zoomed_range : display_range;
  setUpdatesEnabled(false);
  for (auto c : charts) {
    c->setDisplayRange(range.first, range.second);
    c->scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
  }
  setUpdatesEnabled(true);
}

void ChartsWidget::showAllData() {
  bool switch_to_show_all = max_chart_range == settings.max_chart_x_range;
  max_chart_range = switch_to_show_all ? settings.cached_segment_limit * 60
                                       : settings.max_chart_x_range;
  max_chart_range = std::min(max_chart_range, (uint32_t)can->totalSeconds());
  updateToolBar();
  updateState();
}

void ChartsWidget::updateToolBar() {
  int min_range = std::min(settings.max_chart_x_range, (int)can->totalSeconds());
  bool displaying_all = max_chart_range != min_range;
  show_all_values_btn->setText(tr("%1 minutes").arg(max_chart_range / 60));
  show_all_values_btn->setToolTip(tr("Click to display %1 data").arg(displaying_all ? tr("%1 minutes").arg(min_range / 60) : tr("ALL cached")));
  show_all_values_btn->setVisible(!is_zoomed);
  remove_all_btn->setEnabled(!charts.isEmpty());
  reset_zoom_btn->setEnabled(is_zoomed);
  range_label->setText(is_zoomed ? tr("%1 - %2").arg(zoomed_range.first, 0, 'f', 2).arg(zoomed_range.second, 0, 'f', 2) : "");
  title_label->setText(charts.size() > 0 ? tr("Charts (%1)").arg(charts.size()) : tr("Charts"));
  dock_btn->setText(docking ? "⬈" : "⬋");
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

ChartView *ChartsWidget::findChart(const QString &id, const Signal *sig) {
  for (auto c : charts)
    if (c->hasSeries(id, sig)) return c;
  return nullptr;
}

void ChartsWidget::showChart(const QString &id, const Signal *sig, bool show, bool merge) {
  setUpdatesEnabled(false);
  if (show) {
    ChartView *chart = merge && charts.size() > 0 ? charts.back() : nullptr;
    if (!chart) {
      chart = new ChartView(this);
      chart->chart()->setTheme(use_dark_theme ? QChart::QChart::ChartThemeDark : QChart::ChartThemeLight);
      chart->setEventsRange(display_range);
      auto range = is_zoomed ? zoomed_range : display_range;
      chart->setDisplayRange(range.first, range.second);
      QObject::connect(chart, &ChartView::remove, [=]() { removeChart(chart); });
      QObject::connect(chart, &ChartView::zoomIn, this, &ChartsWidget::zoomIn);
      QObject::connect(chart, &ChartView::zoomReset, this, &ChartsWidget::zoomReset);
      QObject::connect(chart, &ChartView::seriesRemoved, this, &ChartsWidget::seriesChanged);
      QObject::connect(chart, &ChartView::seriesAdded, this, &ChartsWidget::seriesChanged);
      QObject::connect(chart, &ChartView::axisYUpdated, this, &ChartsWidget::alignCharts);
      charts_layout->insertWidget(0, chart);
      charts.push_back(chart);
    }
    chart->addSeries(id, sig);
  } else if (ChartView *chart = findChart(id, sig)) {
    chart->removeSeries(id, sig);
  }
  updateToolBar();
  setUpdatesEnabled(true);
}

void ChartsWidget::removeChart(ChartView *chart) {
  charts.removeOne(chart);
  chart->deleteLater();
  updateToolBar();
  alignCharts();
  emit seriesChanged();
}

void ChartsWidget::removeAll() {
  for (auto c : charts) {
    c->deleteLater();
  }
  charts.clear();
  updateToolBar();
  emit seriesChanged();
}

void ChartsWidget::alignCharts() {
  int plot_left = 0;
  for (auto c : charts) {
    plot_left = qMax((qreal)plot_left, c->getYAsixLabelWidth());
  }
  for (auto c : charts) {
    c->setPlotAreaLeftPosition(plot_left);
  }
}

bool ChartsWidget::eventFilter(QObject *obj, QEvent *event) {
  if (obj != this && event->type() == QEvent::Close) {
    emit dock_btn->triggered();
    return true;
  }
  return false;
}

// ChartView

ChartView::ChartView(QWidget *parent) : QChartView(nullptr, parent) {
  QChart *chart = new QChart();
  chart->setBackgroundRoundness(0);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart->addAxis(axis_x, Qt::AlignBottom);
  chart->addAxis(axis_y, Qt::AlignLeft);
  chart->legend()->setShowToolTips(true);
  chart->layout()->setContentsMargins(0, 0, 0, 0);

  QToolButton *remove_btn = new QToolButton();
  remove_btn->setText("X");
  remove_btn->setAutoRaise(true);
  remove_btn->setToolTip(tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart);
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart->zValue() + 11);

  QToolButton *manage_btn = new QToolButton();
  manage_btn->setText("🔧");
  manage_btn->setAutoRaise(true);
  manage_btn->setToolTip(tr("Manage series"));
  manage_btn_proxy = new QGraphicsProxyWidget(chart);
  manage_btn_proxy->setWidget(manage_btn);
  manage_btn_proxy->setZValue(chart->zValue() + 11);

  setChart(chart);
  setRenderHint(QPainter::Antialiasing);
  setRubberBand(QChartView::HorizontalRubberBand);
  updateFromSettings();

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &ChartView::msgUpdated);
  QObject::connect(&settings, &Settings::changed, this, &ChartView::updateFromSettings);
  QObject::connect(remove_btn, &QToolButton::clicked, this, &ChartView::remove);
  QObject::connect(manage_btn, &QToolButton::clicked, this, &ChartView::manageSeries);
}

qreal ChartView::getYAsixLabelWidth() const {
  QFontMetrics fm(axis_y->labelsFont());
  int n = qMax(int(-qFloor(std::log10((axis_y->max() - axis_y->min()) / (axis_y->tickCount() - 1)))), 0) + 1;
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
  series->setUseOpenGL(true);
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);
  auto [source, address] = DBCManager::parseId(msg_id);
  sigs.push_back({.msg_id = msg_id, .address = address, .source = source, .sig = sig, .series = series});
  updateTitle();
  updateSeries(sig);
  updateAxisY();
  emit seriesAdded(msg_id, sig);
}

void ChartView::removeSeries(const QString &msg_id, const Signal *sig) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
  if (it != sigs.end()) {
    it = removeSeries(it);
  }
}

bool ChartView::hasSeries(const QString &msg_id, const Signal *sig) const {
  return std::any_of(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

QList<ChartView::SigItem>::iterator ChartView::removeSeries(const QList<ChartView::SigItem>::iterator &it) {
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
    updateAxisY();
  }
}

void ChartView::signalRemoved(const Signal *sig) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->sig == sig) ? removeSeries(it) : ++it;
  }
}

void ChartView::msgUpdated(uint32_t address) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.address == address; }))
    updateTitle();
}

void ChartView::msgRemoved(uint32_t address) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->address == address) ? removeSeries(it) : ++it;
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
      for (auto &s : series_list) {
        if (auto m = dbc()->msg(s[0])) {
          auto it = m->sigs.find(s[2]);
          if (it != m->sigs.end() && !hasSeries(s[0], &(it->second))) {
            addSeries(s[0], &(it->second));
          }
        }
      }
      for (auto it = sigs.begin(); it != sigs.end(); /**/) {
        bool exists = std::any_of(series_list.cbegin(), series_list.cend(), [&](auto &s) {
          return s[0] == it->msg_id && s[2] == it->sig->name.c_str();
        });
        it = exists ? ++it : removeSeries(it);
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

void ChartView::updateFromSettings() {
  setFixedHeight(settings.chart_height);
}

void ChartView::setEventsRange(const std::pair<double, double> &range) {
  if (range != events_range) {
    events_range = range;
    updateSeries();
  }
}

void ChartView::setDisplayRange(double min, double max) {
  if (min != axis_x->min() || max != axis_x->max()) {
    axis_x->setRange(min, max);
    updateAxisY();
  }
}

void ChartView::updateSeries(const Signal *sig) {
  auto events = can->events();
  if (!events || sigs.isEmpty()) return;

  for (auto &s : sigs) {
    if (!sig || s.sig == sig) {
      s.vals.clear();
      s.vals.reserve((events_range.second - events_range.first) * 1000);  // [n]seconds * 1000hz
      s.min_y = std::numeric_limits<double>::max();
      s.max_y = std::numeric_limits<double>::lowest();

      double route_start_time = can->routeStartTime();
      Event begin_event(cereal::Event::Which::INIT_DATA, (route_start_time + events_range.first) * 1e9);
      auto begin = std::lower_bound(events->begin(), events->end(), &begin_event, Event::lessThan());
      double end_ns = (route_start_time + events_range.second) * 1e9;

      for (auto it = begin; it != events->end() && (*it)->mono_time <= end_ns; ++it) {
        if ((*it)->which == cereal::Event::Which::CAN) {
          for (const auto &c : (*it)->event.getCan()) {
            if (s.source == c.getSrc() && s.address == c.getAddress()) {
              auto dat = c.getDat();
              double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *s.sig);
              double ts = ((*it)->mono_time / (double)1e9) - route_start_time;  // seconds
              s.vals.push_back({ts, value});

              if (value < s.min_y) s.min_y = value;
              if (value > s.max_y) s.max_y = value;
            }
          }
        }
      }
      s.series->replace(s.vals);
    }
  }
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.isEmpty()) return;

  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  if (events_range == std::pair{axis_x->min(), axis_x->max()}) {
    for (auto &s : sigs) {
      if (s.min_y < min_y) min_y = s.min_y;
      if (s.max_y > max_y) max_y = s.max_y;
    }
  } else {
    for (auto &s : sigs) {
      auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
      for (auto it = begin; it != s.vals.end() && it->x() <= axis_x->max(); ++it) {
        if (it->y() < min_y) min_y = it->y();
        if (it->y() > max_y) max_y = it->y();
      }
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
  QTimer::singleShot(0, this, &ChartView::axisYUpdated);
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
  } else if (event->button() == Qt::RightButton) {
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

void ChartView::drawForeground(QPainter *painter, const QRectF &rect) {
  qreal x = chart()->mapToPosition(QPointF{can->currentSec(), 0}).x();
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

// SeriesSelector

SeriesSelector::SeriesSelector(QWidget *parent) {
  setWindowTitle(tr("Manage Chart Series"));
  QHBoxLayout *contents_layout = new QHBoxLayout();

  QVBoxLayout *left_layout = new QVBoxLayout();
  left_layout->addWidget(new QLabel(tr("Select Signals:")));

  msgs_combo = new QComboBox(this);
  msgs_combo->setEditable(true);
  msgs_combo->lineEdit()->setPlaceholderText(tr("Select Msg"));
  msgs_combo->setInsertPolicy(QComboBox::NoInsert);
  msgs_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  msgs_combo->completer()->setFilterMode(Qt::MatchContains);

  left_layout->addWidget(msgs_combo);
  sig_list = new QListWidget(this);
  sig_list->setSortingEnabled(true);
  sig_list->setToolTip(tr("Double click on an item to add signal to chart"));
  left_layout->addWidget(sig_list);

  QVBoxLayout *right_layout = new QVBoxLayout();
  right_layout->addWidget(new QLabel(tr("Chart Signals:")));
  chart_series = new QListWidget(this);
  chart_series->setSortingEnabled(true);
  chart_series->setToolTip(tr("Double click on an item to remove signal from chart"));
  right_layout->addWidget(chart_series);
  contents_layout->addLayout(left_layout);
  contents_layout->addLayout(right_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(contents_layout);
  main_layout->addWidget(buttonBox);

  for (auto it = can->can_msgs.cbegin(); it != can->can_msgs.cend(); ++it) {
    if (auto m = dbc()->msg(it.key())) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(it.key()), it.key());
    }
  }
  msgs_combo->model()->sort(0);

  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(msgs_combo, SIGNAL(currentIndexChanged(int)), SLOT(msgSelected(int)));
  QObject::connect(sig_list, &QListWidget::itemDoubleClicked, this, &SeriesSelector::addSignal);
  QObject::connect(chart_series, &QListWidget::itemDoubleClicked, [](QListWidgetItem *item) { delete item; });

  if (int index = msgs_combo->currentIndex(); index >= 0) {
    msgSelected(index);
  }
}

void SeriesSelector::msgSelected(int index) {
  QString msg_id = msgs_combo->itemData(index).toString();
  sig_list->clear();
  if (auto m = dbc()->msg(msg_id)) {
    for (auto &[name, s] : m->sigs) {
      QStringList data({msg_id, m->name, name});
      QListWidgetItem *item = new QListWidgetItem(name, sig_list);
      item->setData(Qt::UserRole, data);
      sig_list->addItem(item);
    }
  }
}

void SeriesSelector::addSignal(QListWidgetItem *item) {
  QStringList data = item->data(Qt::UserRole).toStringList();
  addSeries(data[0], data[1], data[2]);
}

void SeriesSelector::addSeries(const QString &id, const QString &msg_name, const QString &sig_name) {
  QStringList data({id, msg_name, sig_name});
  for (int i = 0; i < chart_series->count(); ++i) {
    if (chart_series->item(i)->data(Qt::UserRole).toStringList() == data) {
      return;
    }
  }
  QListWidgetItem *new_item = new QListWidgetItem(chart_series);
  new_item->setData(Qt::UserRole, data);
  chart_series->addItem(new_item);
  QLabel *label = new QLabel(QString("%0 <font color=\"gray\">%1 %2</font>").arg(data[2]).arg(data[1]).arg(data[0]), chart_series);
  label->setContentsMargins(5, 0, 5, 0);
  new_item->setSizeHint(label->sizeHint());
  chart_series->setItemWidget(new_item, label);
}

QList<QStringList> SeriesSelector::series() {
  QList<QStringList> ret;
  for (int i = 0; i < chart_series->count(); ++i) {
    ret.push_back(chart_series->item(i)->data(Qt::UserRole).toStringList());
  }
  return ret;
}
