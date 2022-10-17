#include "tools/cabana/chartswidget.h"

#include <QGraphicsLayout>
#include <QRubberBand>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title bar
  title_bar = new QWidget(this);
  title_bar->setVisible(false);
  QHBoxLayout *title_layout = new QHBoxLayout(title_bar);
  title_layout->setContentsMargins(0, 0, 0, 0);
  title_label = new QLabel(tr("Charts"));

  title_layout->addWidget(title_label);
  title_layout->addStretch();

  range_label = new QLabel();
  title_layout->addWidget(range_label);

  reset_zoom_btn = new QPushButton("⟲", this);
  reset_zoom_btn->setFixedSize(30, 30);
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  title_layout->addWidget(reset_zoom_btn);

  remove_all_btn = new QPushButton("✖", this);
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

  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartsWidget::removeChart);
  QObject::connect(dbc(), &DBCManager::signalUpdated, [this](const Signal *sig) {
    if (auto it = charts.find(sig); it != charts.end()) {
      it.value()->chart_view->updateSeries();
    }
  });
  QObject::connect(dbc(), &DBCManager::msgUpdated, [this](const QString &id) {
    for (auto chart : charts) {
      if (chart->id == id)
        chart->updateTitle();
    }
  });

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
  title_bar->setVisible(!charts.isEmpty());
  if (charts.isEmpty()) return;

  // show select range
  range_label->setVisible(can->isZoomed());
  reset_zoom_btn->setEnabled(can->isZoomed());
  if (can->isZoomed()) {
    auto [min, max] = can->range();
    range_label->setText(tr("%1 - %2").arg(min, 0, 'f', 2).arg(max, 0, 'f', 2));
  }

  title_label->setText(tr("Charts (%1)").arg(charts.size()));
  dock_btn->setText(docking ? "⬈" : "⬋");
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

void ChartsWidget::addChart(const QString &id, const Signal *sig) {
  if (!charts.contains(sig)) {
    auto chart = new ChartWidget(id, sig, this);
    QObject::connect(chart, &ChartWidget::remove, [=]() { removeChart(sig); });
    charts_layout->insertWidget(0, chart);
    charts.insert(sig, chart);
  }
  updateTitleBar();
}

void ChartsWidget::removeChart(const Signal *sig) {
  auto it = charts.find(sig);
  if (it != charts.end()) {
    it.value()->deleteLater();
    charts.remove(sig);
  }
  updateTitleBar();
}

void ChartsWidget::removeAll() {
  for (auto chart : charts)
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

ChartWidget::ChartWidget(const QString &id, const Signal *sig, QWidget *parent) : id(id), signal(sig), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setSpacing(0);
  main_layout->setContentsMargins(0, 0, 0, 0);

  QWidget *header = new QWidget(this);
  header->setStyleSheet("background-color:white");
  QHBoxLayout *header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(11, 11, 11, 0);
  title = new QLabel(tr("%1 %2").arg(dbc()->msg(id)->name.c_str()).arg(id));
  header_layout->addWidget(title);
  header_layout->addStretch();

  QPushButton *remove_btn = new QPushButton("✖", this);
  remove_btn->setFixedSize(30, 30);
  remove_btn->setToolTip(tr("Remove chart"));
  QObject::connect(remove_btn, &QPushButton::clicked, this, &ChartWidget::remove);
  header_layout->addWidget(remove_btn);
  main_layout->addWidget(header);

  chart_view = new ChartView(id, sig, this);
  chart_view->setFixedHeight(300);
  main_layout->addWidget(chart_view);
  main_layout->addStretch();

  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void ChartWidget::updateTitle() {
  title->setText(tr("%1 %2").arg(dbc()->msg(id)->name.c_str()).arg(id));
}

// ChartView

ChartView::ChartView(const QString &id, const Signal *sig, QWidget *parent)
    : id(id), signal(sig), QChartView(nullptr, parent) {
  QLineSeries *series = new QLineSeries();
  series->setUseOpenGL(true);
  QChart *chart = new QChart();
  chart->setTitle(sig->name.c_str());
  chart->addSeries(series);
  chart->createDefaultAxes();
  chart->legend()->hide();
  QFont font;
  font.setBold(true);
  chart->setTitleFont(font);
  chart->setMargins({0, 0, 0, 0});
  chart->layout()->setContentsMargins(0, 0, 0, 0);

  track_line = new QGraphicsLineItem(chart);
  track_line->setPen(QPen(Qt::gray, 1, Qt::DashLine));
  value_text = new QGraphicsSimpleTextItem(chart);
  value_text->setBrush(Qt::gray);
  line_marker = new QGraphicsLineItem(chart);
  line_marker->setPen(QPen(Qt::black, 2));

  setChart(chart);

  setRenderHint(QPainter::Antialiasing);
  setRubberBand(QChartView::HorizontalRubberBand);
  if (auto rubber = findChild<QRubberBand *>()) {
    QPalette pal;
    pal.setBrush(QPalette::Base, QColor(0, 0, 0, 80));
    rubber->setPalette(pal);
  }

  QObject::connect(can, &CANMessages::updated, this, &ChartView::updateState);
  QObject::connect(can, &CANMessages::rangeChanged, this, &ChartView::rangeChanged);
  QObject::connect(can, &CANMessages::eventsMerged, this, &ChartView::updateSeries);
  QObject::connect(dynamic_cast<QValueAxis *>(chart->axisX()), &QValueAxis::rangeChanged, can, &CANMessages::setRange);

  updateSeries();
}

void ChartView::updateState() {
  auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  int x = chart()->plotArea().left() + chart()->plotArea().width() * (can->currentSec() - axis_x->min()) / (axis_x->max() - axis_x->min());
  line_marker->setLine(x, 0, x, height());
}

void ChartView::updateSeries() {
  chart()->setTitle(signal->name.c_str());
  auto events = can->events();
  if (!events) return;

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
          double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *signal);
          double ts = (evt->mono_time / (double)1e9) - route_start_time;  // seconds
          vals.push_back({ts, value});
        }
      }
    }
  }
  QLineSeries *series = (QLineSeries *)chart()->series()[0];
  series->replace(vals);
  series->setPointLabelsColor(Qt::black);
  auto [begin, end] = can->range();
  chart()->axisX()->setRange(begin, end);
  updateAxisY();
}

void ChartView::rangeChanged(qreal min, qreal max) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  if (axis_x->min() != min || axis_x->max() != max) {
    axis_x->setRange(min, max);
  }
  updateAxisY();
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  const auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  const auto axis_y = dynamic_cast<QValueAxis *>(chart()->axisY());
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

void ChartView::enterEvent(QEvent *event) {
  track_line->setVisible(true);
  value_text->setVisible(true);
  QChartView::enterEvent(event);
}

void ChartView::leaveEvent(QEvent *event) {
  track_line->setVisible(false);
  value_text->setVisible(false);
  QChartView::leaveEvent(event);
}

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
  } else if (event->button() == Qt::RightButton) {
    // reset zoom
    if (can->isZoomed()) {
      can->resetRange();
      event->accept();
      return;
    }
  }
  QChartView::mouseReleaseEvent(event);
  line_marker->setVisible(true);
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  auto rubber = findChild<QRubberBand *>();
  bool show = !(rubber && rubber->isVisible());

  if (show) {
    const auto plot_area = chart()->plotArea();
    float x = std::clamp((float)ev->pos().x(), (float)plot_area.left(), (float)plot_area.right());
    track_line->setLine(x, plot_area.top(), x, plot_area.bottom());

    auto [begin, end] = can->range();
    double sec = begin + ((x - plot_area.x()) / plot_area.width()) * (end - begin);
    auto value = std::lower_bound(vals.begin(), vals.end(), sec, [](auto &p, double x) { return p.x() < x; });
    value_text->setPos(x + 6, plot_area.bottom() - 25);
    if (value != vals.end()) {
      value_text->setText(QString("(%1, %2)").arg(value->x(), 0, 'f', 3).arg(value->y()));
    } else {
      value_text->setText("(--, --)");
    }
  }

  value_text->setVisible(show);
  track_line->setVisible(show);
  line_marker->setVisible(show);
  QChartView::mouseMoveEvent(ev);
}
