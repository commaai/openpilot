#include "tools/cabana/chartswidget.h"

#include <QGraphicsLayout>
#include <QLabel>
#include <QPushButton>
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

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  connect(parser, &Parser::showPlot, this, &ChartsWidget::addChart);
  connect(parser, &Parser::hidePlot, this, &ChartsWidget::removeChart);
  connect(parser, &Parser::signalRemoved, this, &ChartsWidget::removeChart);
}

void ChartsWidget::addChart(const QString &id, const QString &sig_name) {
  const QString char_name = id + sig_name;
  if (charts.find(char_name) == charts.end()) {
    auto chart = new ChartWidget(id, sig_name, this);
    main_layout->addWidget(chart);
    charts[char_name] = chart;
  }
}

void ChartsWidget::removeChart(const QString &id, const QString &sig_name) {
  if (auto it = charts.find(id + sig_name); it != charts.end()) {
    it->second->deleteLater();
    charts.erase(it);
  }
}

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
  auto title = new QLabel(tr("%1 %2").arg(parser->getMsg(id)->name.c_str()).arg(id));
  header_layout->addWidget(title);
  header_layout->addStretch();
  zoom_label = new QLabel("", this);
  header_layout->addWidget(zoom_label);
  QPushButton *zoom_in = new QPushButton("↺", this);
  zoom_in->setToolTip(tr("reset zoom"));
  QObject::connect(zoom_in, &QPushButton::clicked, []() { parser->resetRange(); });
  header_layout->addWidget(zoom_in);

  QPushButton *remove_btn = new QPushButton("✖", this);
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
  line_marker = new LineMarker(chart, this);
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
  line_marker->update();
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
}

void ChartWidget::rangeChanged(qreal min, qreal max) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart_view->chart()->axisX());
  if (axis_x->min() != min || axis_x->max() != max) {
    axis_x->setRange(min, max);
  }
  // auto zoom on yaxis
  double min_y = 0, max_y = 0;
  for (auto &p : vals) {
    if (p.x() > max) break;

    if (p.x() >= min) {
      if (p.y() < min_y) min_y = p.y();
      if (p.y() > max_y) max_y = p.y();
    }
  }
  chart_view->chart()->axisY()->setRange(min_y * 0.95, max_y * 1.05);
}

LineMarker::LineMarker(QChart *chart, QWidget *parent) : chart(chart), QWidget(parent) {}

void LineMarker::paintEvent(QPaintEvent *event) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart->axisX());
  if (axis_x->max() <= axis_x->min()) return;

  double x = chart->plotArea().left() + chart->plotArea().width() * (parser->currentSec() - axis_x->min()) / (axis_x->max() - axis_x->min());
  QPainter p(this);
  QPen pen = QPen(Qt::black);
  pen.setWidth(2);
  p.setPen(pen);
  p.drawLine(QPointF{x, 50.}, QPointF{x, (qreal)height() - 11});
}
