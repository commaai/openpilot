#include "tools/cabana/chartswidget.h"

#include <QGraphicsLayout>
#include <QLabel>
#include <QRubberBand>
#include <QStackedLayout>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title bar
  title_bar = new QWidget(this);
  QHBoxLayout *title_layout = new QHBoxLayout(title_bar);
  title_layout->setContentsMargins(0, 0, 0, 0);
  title_label = new QLabel(tr("Charts"));

  title_layout->addWidget(title_label);
  title_layout->addStretch();

  range_label = new QLabel();
  title_layout->addWidget(range_label);

  reset_zoom_btn = new QPushButton("⟲", this);
  reset_zoom_btn->setVisible(false);
  reset_zoom_btn->setFixedSize(30, 30);
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  title_layout->addWidget(reset_zoom_btn);

  remove_all_btn = new QPushButton("✖", this);
  remove_all_btn->setVisible(false);
  remove_all_btn->setToolTip(tr("Remove all charts"));
  remove_all_btn->setFixedSize(30, 30);
  title_layout->addWidget(remove_all_btn);

  dock_btn = new QPushButton();
  dock_btn->setFixedSize(30, 30);
  title_layout->addWidget(dock_btn);

  main_layout->addWidget(title_bar, 0, Qt::AlignTop);

  // charts
  QWidget *charts_container = new QWidget(this);
  QVBoxLayout *charts_main = new QVBoxLayout(charts_container);
  charts_layout = new QVBoxLayout();
  charts_main->addLayout(charts_layout);
  charts_main->addStretch();

  QScrollArea *charts_scroll = new QScrollArea(this);
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  main_layout->addWidget(charts_scroll);

  updateTitleBar();

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartsWidget::removeChart);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(can, &CANMessages::rangeChanged, [this]() { updateTitleBar(); });
  QObject::connect(reset_zoom_btn, &QPushButton::clicked, can, &CANMessages::resetRange);
  QObject::connect(remove_all_btn, &QPushButton::clicked, this, &ChartsWidget::removeAll);
  QObject::connect(dock_btn, &QPushButton::clicked, [this]() {
    emit dock(!docking);
    docking = !docking;
    updateTitleBar();
  });
}

void ChartsWidget::updateTitleBar() {
  if (!charts.size()) {
    title_bar->setVisible(false);
    return;
  }

  title_label->setText(tr("Charts (%1)").arg(charts.size()));

  // show select range
  if (can->isZoomed()) {
    auto [min, max] = can->range();
    range_label->setText(tr("%1 - %2").arg(min, 0, 'f', 2).arg(max, 0, 'f', 2));
    range_label->setVisible(true);
    reset_zoom_btn->setEnabled(true);
  } else {
    reset_zoom_btn->setEnabled(false);
    range_label->setVisible(false);
  }

  dock_btn->setText(docking ? "⬈" : "⬋");
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
  remove_all_btn->setVisible(!charts.empty());
  reset_zoom_btn->setVisible(!charts.empty());
  title_bar->setVisible(true);
}

void ChartsWidget::addChart(const QString &id, const QString &sig_name) {
  const QString char_name = id + ":" + sig_name;
  if (charts.find(char_name) == charts.end()) {
    auto chart = new ChartWidget(id, sig_name, this);
    QObject::connect(chart, &ChartWidget::remove, [=]() {
      removeChart(id, sig_name);
    });
    charts_layout->insertWidget(0, chart);
    charts[char_name] = chart;
  }
  updateTitleBar();
}

void ChartsWidget::removeChart(const QString &id, const QString &sig_name) {
  if (auto it = charts.find(id + ":" + sig_name); it != charts.end()) {
    it->second->deleteLater();
    charts.erase(it);
  }
  updateTitleBar();
}

void ChartsWidget::removeAll() {
  for (auto [_, chart] : charts)
    chart->deleteLater();
  charts.clear();
  updateTitleBar();
}

bool ChartsWidget::eventFilter(QObject *obj, QEvent *event) {
  if (obj != this && event->type() == QEvent::Close) {
    emit dock_btn->clicked();
    return true;
  }
  return false;
}

// ChartWidget

ChartWidget::ChartWidget(const QString &id, const QString &sig_name, QWidget *parent) : id(id), sig_name(sig_name), QWidget(parent) {
  QStackedLayout *stacked = new QStackedLayout(this);
  stacked->setStackingMode(QStackedLayout::StackAll);

  QWidget *chart_widget = new QWidget(this);
  QVBoxLayout *chart_layout = new QVBoxLayout(chart_widget);
  chart_layout->setSpacing(0);
  chart_layout->setContentsMargins(0, 0, 0, 0);

  QWidget *header = new QWidget(this);
  header->setStyleSheet("background-color:white");
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(11, 11, 11, 0);
  QLabel *title = new QLabel(tr("%1 %2").arg(dbc()->msg(id)->name.c_str()).arg(id));
  header_layout->addWidget(title);
  header_layout->addStretch();

  QPushButton *remove_btn = new QPushButton("✖", this);
  remove_btn->setFixedSize(30, 30);
  remove_btn->setToolTip(tr("Remove chart"));
  QObject::connect(remove_btn, &QPushButton::clicked, this, &ChartWidget::remove);
  header_layout->addWidget(remove_btn);
  chart_layout->addWidget(header);

  QLineSeries *series = new QLineSeries();
  series->setUseOpenGL(true);
  auto chart = new QChart();
  chart->setTitle(sig_name);
  chart->addSeries(series);
  chart->createDefaultAxes();
  chart->legend()->hide();
  QFont font;
  font.setBold(true);
  chart->setTitleFont(font);
  chart->setMargins({0, 0, 0, 0});
  chart->layout()->setContentsMargins(0, 0, 0, 0);

  chart_view = new ChartView(chart);
  chart_view->setFixedHeight(300);
  chart_view->setRenderHint(QPainter::Antialiasing);
  chart_view->setRubberBand(QChartView::HorizontalRubberBand);
  if (auto rubber = chart_view->findChild<QRubberBand *>()) {
    QPalette pal;
    pal.setBrush(QPalette::Base, QColor(0, 0, 0, 80));
    rubber->setPalette(pal);
  }
  chart_layout->addWidget(chart_view);
  chart_layout->addStretch();

  stacked->addWidget(chart_widget);
  line_marker = new LineMarker(this);
  stacked->addWidget(line_marker);
  line_marker->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  line_marker->raise();

  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  QObject::connect(can, &CANMessages::updated, this, &ChartWidget::updateState);
  QObject::connect(can, &CANMessages::rangeChanged, this, &ChartWidget::rangeChanged);
  QObject::connect(can, &CANMessages::eventsMerged, this, &ChartWidget::updateSeries);
  QObject::connect(dynamic_cast<QValueAxis *>(chart->axisX()), &QValueAxis::rangeChanged, can, &CANMessages::setRange);
  QObject::connect(dbc(), &DBCManager::signalUpdated, [this](const QString &msg_id, const QString &sig_name) {
    if (this->id == msg_id && this->sig_name == sig_name)
      updateSeries();
  });
  updateSeries();
}

void ChartWidget::updateState() {
  auto chart = chart_view->chart();
  auto axis_x = dynamic_cast<QValueAxis *>(chart->axisX());

  int x = chart->plotArea().left() + chart->plotArea().width() * (can->currentSec() - axis_x->min()) / (axis_x->max() - axis_x->min());
  line_marker->setX(x);
}

void ChartWidget::updateSeries() {
  const Signal *sig = dbc()->signal(id, sig_name);
  auto events = can->events();
  if (!sig || !events) return;

  auto l = id.split(':');
  int bus = l[0].toInt();
  uint32_t address = l[1].toUInt(nullptr, 16);

  vals.clear();
  vals.reserve(3 * 60 * 100);
  uint64_t route_start_time = can->routeStartTime();
  for (auto &evt : *events) {
    if (evt->which == cereal::Event::Which::CAN) {
      for (auto c : evt->event.getCan()) {
        if (bus == c.getSrc() && address == c.getAddress()) {
          auto dat = c.getDat();
          double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *sig);
          double ts = (evt->mono_time / (double)1e9) - route_start_time;  // seconds
          vals.push_back({ts, value});
        }
      }
    }
  }
  QLineSeries *series = (QLineSeries *)chart_view->chart()->series()[0];
  series->replace(vals);
  auto [begin, end] = can->range();
  chart_view->chart()->axisX()->setRange(begin, end);
  updateAxisY();
}

void ChartWidget::rangeChanged(qreal min, qreal max) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart_view->chart()->axisX());
  if (axis_x->min() != min || axis_x->max() != max) {
    axis_x->setRange(min, max);
  }
  updateAxisY();
}

// auto zoom on yaxis
void ChartWidget::updateAxisY() {
  const auto axis_x = dynamic_cast<QValueAxis *>(chart_view->chart()->axisX());
  const auto axis_y = dynamic_cast<QValueAxis *>(chart_view->chart()->axisY());
  // vals is a sorted list
  auto begin = std::lower_bound(vals.begin(), vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
  if (begin == vals.end())
    return;

  auto end = std::upper_bound(vals.begin(), vals.end(), axis_x->max(), [](double x, auto &p) { return x < p.x(); });
  const auto [min, max] = std::minmax_element(begin, end, [](auto &p1, auto &p2) { return p1.y() < p2.y(); });
  if (min->y() == max->y()) {
    if (max->y() < 0) {
      axis_y->setRange(max->y(), 0);
    } else {
      axis_y->setRange(0, max->y() == 0 ? 1 : max->y());
    }
  } else {
    axis_y->setRange(min->y(), max->y());
  }
}

// ChartView

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  auto rubber = findChild<QRubberBand *>();
  if (event->button() == Qt::LeftButton && rubber && rubber->isVisible()) {
    auto [begin, end] = can->range();
    if (rubber->width() <= 0) {
      double seek_to = begin + ((event->pos().x() - chart()->plotArea().x()) / chart()->plotArea().width()) * (end - begin);
      can->seekTo(seek_to);
    } else if (((double)rubber->width() / chart()->plotArea().width()) * (end - begin) < 0.5) {
      // don't zoom if selected range is less than 0.5s
      rubber->hide();
      event->accept();
      return;
    }
  }
  // TODO: right-click to reset zoom
  QChartView::mouseReleaseEvent(event);
}


// LineMarker

void LineMarker::setX(double x) {
  if (x != x_pos) {
    x_pos = x;
    update();
  }
}

void LineMarker::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(QPen(Qt::black, 2));
  p.drawLine(QPointF{x_pos, 50.}, QPointF{x_pos, (qreal)height() - 11});
}
