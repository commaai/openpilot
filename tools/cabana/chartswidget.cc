#include "tools/cabana/chartswidget.h"

#include <QGraphicsLayout>
#include <QLabel>
#include <QRubberBand>
#include <QStackedLayout>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

int64_t get_raw_value(uint8_t *data, size_t data_size, const Signal &sig) {
  int64_t ret = 0;

  int i = sig.msb / 8;
  int bits = sig.size;
  while (i >= 0 && i < data_size && bits > 0) {
    int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i * 8;
    int msb = (int)(sig.msb / 8) == i ? sig.msb : (i + 1) * 8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (data[i] >> (lsb - (i * 8))) & ((1ULL << size) - 1);
    ret |= d << (bits - size);

    bits -= size;
    i = sig.is_little_endian ? i - 1 : i + 1;
  }
  return ret;
}

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title bar
  title_bar = new QWidget(this);
  QHBoxLayout *title_layout = new QHBoxLayout(title_bar);
  title_label = new QLabel(tr("Charts"));

  title_layout->addWidget(title_label);
  title_layout->addStretch();

  reset_zoom_btn = new QPushButton("⟲", this);
  reset_zoom_btn->setVisible(false);
  reset_zoom_btn->setFixedSize(30, 30);
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  title_layout->addWidget(reset_zoom_btn);

  remove_all_btn = new QPushButton(tr("✖"));
  remove_all_btn->setVisible(false);
  remove_all_btn->setToolTip(tr("Remove all charts"));
  remove_all_btn->setFixedSize(30, 30);
  title_layout->addWidget(remove_all_btn);

  dock_btn = new QPushButton();
  dock_btn->setFixedSize(30, 30);
  updateDockButton();
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
  charts_scroll->setFrameShape(QFrame::NoFrame);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  main_layout->addWidget(charts_scroll);

  QObject::connect(parser, &Parser::showPlot, this, &ChartsWidget::addChart);
  QObject::connect(parser, &Parser::hidePlot, this, &ChartsWidget::removeChart);
  QObject::connect(parser, &Parser::signalRemoved, this, &ChartsWidget::removeChart);
  QObject::connect(reset_zoom_btn, &QPushButton::clicked, parser, &Parser::resetRange);
  QObject::connect(remove_all_btn, &QPushButton::clicked, this, &ChartsWidget::removeAll);
  QObject::connect(dock_btn, &QPushButton::clicked, [=]() {
    emit dock(!docking);
    docking = !docking;
    updateDockButton();
  });
}

void ChartsWidget::updateDockButton() {
  dock_btn->setText(docking ? "⬈" : "⬋");
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

void ChartsWidget::addChart(const QString &id, const QString &sig_name) {
  const QString char_name = id + sig_name;
  if (charts.find(char_name) == charts.end()) {
    auto chart = new ChartWidget(id, sig_name, this);
    charts_layout->insertWidget(0, chart);
    charts[char_name] = chart;
  }
  remove_all_btn->setVisible(true);
  reset_zoom_btn->setVisible(true);
  title_label->setText(tr("Charts (%1)").arg(charts.size()));
}

void ChartsWidget::removeChart(const QString &id, const QString &sig_name) {
  if (auto it = charts.find(id + sig_name); it != charts.end()) {
    it->second->deleteLater();
    charts.erase(it);
    if (charts.empty()) {
      remove_all_btn->setVisible(false);
      reset_zoom_btn->setVisible(false);
    }
  }
  title_label->setText(tr("Charts (%1)").arg(charts.size()));
}

void ChartsWidget::removeAll() {
  for (auto [_, chart] : charts)
    chart->deleteLater();
  charts.clear();
  remove_all_btn->setVisible(false);
  reset_zoom_btn->setVisible(false);
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
  QLabel *title = new QLabel(tr("%1 %2").arg(parser->getMsg(id)->name.c_str()).arg(id));
  header_layout->addWidget(title);
  header_layout->addStretch();

  QPushButton *remove_btn = new QPushButton("✖", this);
  remove_btn->setFixedSize(30, 30);
  remove_btn->setToolTip(tr("Remove chart"));
  QObject::connect(remove_btn, &QPushButton::clicked, [=]() {
    emit parser->hidePlot(id, sig_name);
  });
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
  QObject::connect(dynamic_cast<QValueAxis *>(chart->axisX()), &QValueAxis::rangeChanged, parser, &Parser::setRange);

  chart_view = new QChartView(chart);
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
  QObject::connect(parser, &Parser::updated, this, &ChartWidget::updateState);
  QObject::connect(parser, &Parser::rangeChanged, this, &ChartWidget::rangeChanged);
  QObject::connect(parser, &Parser::eventsMerged, this, &ChartWidget::updateSeries);

  updateSeries();
}

void ChartWidget::updateState() {
  auto chart = chart_view->chart();
  auto axis_x = dynamic_cast<QValueAxis *>(chart->axisX());
  int x = chart->plotArea().left() + chart->plotArea().width() * (parser->currentSec() - axis_x->min()) / (axis_x->max() - axis_x->min());
  if (line_marker_x != x) {
    line_marker->setX(x);
    line_marker_x = x;
  }
}

void ChartWidget::updateSeries() {
  const Signal *sig = parser->getSig(id, sig_name);
  auto events = parser->replay->events();
  if (!sig || !events) return;

  auto l = id.split(':');
  int bus = l[0].toInt();
  uint32_t address = l[1].toUInt(nullptr, 16);

  vals.clear();
  vals.reserve(3 * 60 * 100);
  uint64_t route_start_time = parser->replay->routeStartTime();
  for (auto &evt : *events) {
    if (evt->which == cereal::Event::Which::CAN) {
      for (auto c : evt->event.getCan()) {
        if (bus == c.getSrc() && address == c.getAddress()) {
          auto dat = c.getDat();
          int64_t val = get_raw_value((uint8_t *)dat.begin(), dat.size(), *sig);
          if (sig->is_signed) {
            val -= ((val >> (sig->size - 1)) & 0x1) ? (1ULL << sig->size) : 0;
          }
          double value = val * sig->factor + sig->offset;
          double ts = (evt->mono_time - route_start_time) / (double)1e9;  // seconds
          vals.push_back({ts, value});
        }
      }
    }
  }
  QLineSeries *series = (QLineSeries *)chart_view->chart()->series()[0];
  series->replace(vals);
  auto [begin, end] = parser->range();
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
  // vals is a sorted list
  auto begin = std::lower_bound(vals.begin(), vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
  if (begin == vals.end())
    return;

  auto end = std::upper_bound(vals.begin(), vals.end(), axis_x->max(), [](double x, auto &p) { return x < p.x(); });
  const auto [min, max] = std::minmax_element(begin, end, [](auto &p1, auto &p2) { return p1.y() < p2.y(); });
  chart_view->chart()->axisY()->setRange(min->y(), max->y());
}

// LineMarker

void LineMarker::setX(double x) {
  x_pos = x;
  update();
}

void LineMarker::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(QPen(Qt::black, 2));
  p.drawLine(QPointF{x_pos, 50.}, QPointF{x_pos, (qreal)height() - 11});
}
