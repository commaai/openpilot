#include "tools/cabana/chart/chart.h"
#include "tools/cabana/dbc/dbcqt.h"

#include <algorithm>
#include <limits>
#include <random>

#include <QActionGroup>
#include <QContextMenuEvent>
#include <QMouseEvent>
#include <QPainterPath>

#include "tools/cabana/chart/chartswidget.h"

const int AXIS_X_TOP_MARGIN = 4;
const int X_TICK_COUNT = 5;
const double MIN_ZOOM_SECONDS = 0.01;  // 10ms
// Define a small value of epsilon to compare double values
const float EPSILON = 0.000001;
static inline bool xLessThan(const QPointF &p, float x) { return p.x() < (x - EPSILON); }

static QMargins layoutMargins(const QStyle *style) {
  return {
    style->pixelMetric(QStyle::PM_LayoutLeftMargin),
    style->pixelMetric(QStyle::PM_LayoutTopMargin),
    style->pixelMetric(QStyle::PM_LayoutRightMargin),
    style->pixelMetric(QStyle::PM_LayoutBottomMargin),
  };
}

ChartView::ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent)
    : x_min(x_range.first), x_max(x_range.second), charts_widget(parent), QWidget(parent) {
  series_type = (SeriesType)settings.chart_series_type;
  align_to = 50;
  setMouseTracking(true);
  tip_label = new TipLabel(this);
  createToolButtons();
  signal_value_font.setPointSize(9);

  QObject::connect(dbcNotifier(), &QtDBCNotifier::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbcNotifier(), &QtDBCNotifier::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbcNotifier(), &QtDBCNotifier::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbcNotifier(), &QtDBCNotifier::msgUpdated, this, &ChartView::msgUpdated);
}

void ChartView::createToolButtons() {
  close_btn = new ToolButton("x", tr("Remove Chart"), this);

  menu = new QMenu(this);
  // series types
  auto change_series_group = new QActionGroup(menu);
  change_series_group->setExclusive(true);
  QStringList types{tr("Line"), tr("Step Line"), tr("Scatter")};
  for (int i = 0; i < types.size(); ++i) {
    QAction *act = new QAction(types[i], change_series_group);
    act->setData(i);
    act->setCheckable(true);
    act->setChecked(i == (int)series_type);
    menu->addAction(act);
  }
  menu->addSeparator();
  menu->addAction(tr("Manage Signals"), this, &ChartView::manageSignals);
  split_chart_act = menu->addAction(tr("Split Chart"), [this]() { charts_widget->splitChart(this); });

  manage_btn = new ToolButton("list", "", this);
  manage_btn->setMenu(menu);
  manage_btn->setPopupMode(QToolButton::InstantPopup);
  manage_btn->setStyleSheet("QToolButton::menu-indicator { image: none; }");

  close_act = new QAction(tr("Close"), this);
  QObject::connect(close_act, &QAction::triggered, [this] () { charts_widget->removeChart(this); });
  QObject::connect(close_btn, &QToolButton::clicked, close_act, &QAction::triggered);
  QObject::connect(change_series_group, &QActionGroup::triggered, [this](QAction *action) {
    setSeriesType((SeriesType)action->data().toInt());
  });
}

QSize ChartView::sizeHint() const {
  return {CHART_MIN_WIDTH, settings.chart_height};
}

void ChartView::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  if (hasSignal(msg_id, sig)) return;

  sigs.push_back({.msg_id = msg_id, .sig = sig, .color = uniqueColor(toQColor(sig->color))});
  updateSeries(sig);
  updateTitle();
  emit charts_widget->seriesChanged();
}

bool ChartView::hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs.cbegin(), sigs.cend(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs.size();
  sigs.erase(std::remove_if(sigs.begin(), sigs.end(), predicate), sigs.end());
  if (sigs.empty()) {
    charts_widget->removeChart(this);
  } else if (sigs.size() != prev_size) {
    emit charts_widget->seriesChanged();
    updateAxisY();
    updateTitle();
  }
}

void ChartView::signalUpdated(const cabana::Signal *sig) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [sig](auto &s) { return s.sig == sig; });
  if (it != sigs.end()) {
    if (it->color != toQColor(sig->color)) {
      it->color = uniqueColor(toQColor(sig->color), sig);
    }
    updateTitle();
    updateSeries(sig);
  }
}

void ChartView::msgUpdated(MessageId id) {
  if (std::any_of(sigs.cbegin(), sigs.cend(), [=](auto &s) { return s.msg_id.address == id.address; })) {
    updateTitle();
  }
}

void ChartView::manageSignals() {
  SignalSelector dlg(tr("Manage Chart"), this);
  for (auto &s : sigs) {
    dlg.addSelected(s.msg_id, s.sig);
  }
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    for (auto s : items) {
      addSignal(s->msg_id, s->sig);
    }
    removeIf([&](auto &s) {
      return std::none_of(items.cbegin(), items.cend(), [&](auto &it) { return s.msg_id == it->msg_id && s.sig == it->sig; });
    });
  }
}

void ChartView::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  const auto margins = layoutMargins(style());
  QPixmap grip = utils::icon("grip-horizontal");
  move_icon_rect = QRect(QPoint(margins.left(), margins.top()), grip.size() / grip.devicePixelRatio());
  close_btn->resize(close_btn->sizeHint());
  manage_btn->resize(manage_btn->sizeHint());
  close_btn->move(rect().right() - margins.right() - close_btn->width(), margins.top());
  manage_btn->move(close_btn->x() - manage_btn->width() - style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing), margins.top());
  updatePlotArea(align_to, true);
}

void ChartView::updatePlotArea(int left_pos, bool force) {
  if (align_to != left_pos || force) {
    align_to = left_pos;

    const auto margins = layoutMargins(style());
    QFont bold_font = font();
    bold_font.setBold(true);
    QFontMetrics fm(font()), bfm(bold_font);
    const int marker_size = fm.height() - 4;
    const int row_height = std::max(marker_size, fm.height()) + QFontMetrics(signal_value_font).height() + 3;
    const int legend_left = move_icon_rect.right() + margins.left();
    const int legend_right = std::max(manage_btn->x() - margins.right(), legend_left + 10);

    // layout legend entries left-to-right, wrapping between the move icon and the buttons
    legend_rects.clear();
    int x = legend_left, y = margins.top();
    for (auto &s : sigs) {
      int w = marker_size + 5 + bfm.horizontalAdvance(QString::fromStdString(s.sig->name)) +
              fm.horizontalAdvance(QString::fromStdString(" " + msgName(s.msg_id) + " " + s.msg_id.toString()));
      w = std::min(w, legend_right - legend_left);  // keep oversized entries clear of the header buttons
      if (x + w > legend_right && x > legend_left) {
        x = legend_left;
        y += row_height;
      }
      legend_rects.emplace_back(x, y, w, std::max(marker_size, fm.height()));
      x += w + 12;
    }

    // add top space for the legend and signal values
    int adjust_top = (y + row_height) - margins.top();
    adjust_top = std::max(adjust_top, manage_btn->geometry().bottom() + style()->pixelMetric(QStyle::PM_LayoutTopMargin));
    // add right space for x-axis label
    QSizeF x_label_size = fm.size(Qt::TextSingleLine, QString::number(x_max, 'f', xAxisPrecision())) + QSizeF{5, 5};
    plot_area = rect().adjusted(align_to + margins.left(), adjust_top + margins.top(),
                                -x_label_size.width() / 2 - margins.right(),
                                -x_label_size.height() - margins.bottom());
    resetChartCache();
  }
}

void ChartView::updateTitle() {
  split_chart_act->setEnabled(sigs.size() > 1);
  updatePlotArea(align_to, true);
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != x_min || max != x_max) {
    x_min = min;
    x_max = max;
    updateAxisY();
    // update tooltip
    if (tooltip_x >= 0) {
      showTip(secondsAtPoint({tooltip_x, 0}));
    }
    resetChartCache();
  }
  update();
}

void ChartView::appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                                std::vector<QPointF> &vals, std::vector<QPointF> &step_vals) {
  vals.reserve(vals.size() + events.capacity());
  step_vals.reserve(step_vals.size() + events.capacity() * 2);

  double value = 0;
  for (const CanEvent *e : events) {
    if (sig->getValue(e->dat, e->size, &value)) {
      const double ts = can->toSeconds(e->mono_time);
      vals.emplace_back(ts, value);
      if (!step_vals.empty())
        step_vals.emplace_back(ts, step_vals.back().y());
      step_vals.emplace_back(ts, value);
    }
  }
}

void ChartView::updateSeries(const cabana::Signal *sig, const MessageEventsMap *msg_new_events) {
  for (auto &s : sigs) {
    if (!sig || s.sig == sig) {
      if (!msg_new_events) {
        s.vals.clear();
        s.step_vals.clear();
      }
      auto events = msg_new_events ? msg_new_events : &can->eventsMap();
      auto it = events->find(s.msg_id);
      if (it == events->end() || it->second.empty()) continue;

      if (s.vals.empty() || can->toSeconds(it->second.back()->mono_time) > s.vals.back().x()) {
        appendCanEvents(s.sig, it->second, s.vals, s.step_vals);
      } else {
        std::vector<QPointF> vals, step_vals;
        appendCanEvents(s.sig, it->second, vals, step_vals);
        s.vals.insert(std::lower_bound(s.vals.begin(), s.vals.end(), vals.front().x(), xLessThan),
                      vals.begin(), vals.end());
        s.step_vals.insert(std::lower_bound(s.step_vals.begin(), s.step_vals.end(), step_vals.front().x(), xLessThan),
                           step_vals.begin(), step_vals.end());
      }

      if (!can->liveStreaming()) {
        s.segment_tree.build(s.vals);
      }
    }
  }
  updateAxisY();
  // invoke resetChartCache in ui thread
  QMetaObject::invokeMethod(this, &ChartView::resetChartCache, Qt::QueuedConnection);
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.empty()) return;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  QString unit = QString::fromStdString(sigs[0].sig->unit);

  for (auto &s : sigs) {
    if (!s.visible) continue;

    // Only show unit when all signals have the same unit
    if (unit != QString::fromStdString(s.sig->unit)) {
      unit.clear();
    }

    auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), x_min, xLessThan);
    auto last = std::lower_bound(first, s.vals.cend(), x_max, xLessThan);
    s.min = std::numeric_limits<double>::max();
    s.max = std::numeric_limits<double>::lowest();
    if (can->liveStreaming()) {
      for (auto it = first; it != last; ++it) {
        if (it->y() < s.min) s.min = it->y();
        if (it->y() > s.max) s.max = it->y();
      }
    } else {
      std::tie(s.min, s.max) = s.segment_tree.minmax(std::distance(s.vals.cbegin(), first), std::distance(s.vals.cbegin(), last));
    }
    min = std::min(min, s.min);
    max = std::max(max, s.max);
  }
  if (min == std::numeric_limits<double>::max()) min = 0;
  if (max == std::numeric_limits<double>::lowest()) max = 0;

  if (y_unit != unit) {
    y_unit = unit;
    y_label_width = 0;  // recalc width
  }

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, 3);
  if (min_y != y_min || max_y != y_max || y_label_width == 0) {
    y_min = min_y;
    y_max = max_y;
    y_tick_count = tick_count;
    y_precision = std::max(int(-std::floor(std::log10((max_y - min_y) / (tick_count - 1)))), 0);

    QFontMetrics fm(font());
    int max_label_width = 0;
    for (int i = 0; i < tick_count; i++) {
      qreal value = min_y + (i * (max_y - min_y) / (tick_count - 1));
      max_label_width = std::max(max_label_width, fm.horizontalAdvance(QString::number(value, 'f', y_precision)));
    }

    int title_spacing = y_unit.isEmpty() ? 0 : fm.size(Qt::TextSingleLine, y_unit).height();
    y_label_width = title_spacing + max_label_width + 15;
    emit axisYLabelWidthChanged(y_label_width);
  }
}

std::tuple<double, double, int> ChartView::getNiceAxisNumbers(qreal min, qreal max, int tick_count) {
  qreal range = niceNumber((max - min), true);  // range with ceiling
  qreal step = niceNumber(range / (tick_count - 1), false);
  min = std::floor(min / step);
  max = std::ceil(max / step);
  tick_count = int(max - min) + 1;
  return {min * step, max * step, tick_count};
}

int ChartView::xAxisPrecision() const {
  return std::max(int(-std::floor(std::log10((x_max - x_min) / (X_TICK_COUNT - 1)))), 2);
}

// nice numbers can be expressed as form of 1*10^n, 2* 10^n or 5*10^n
qreal ChartView::niceNumber(qreal x, bool ceiling) {
  qreal z = std::pow(10, std::floor(std::log10(x))); //find corresponding number of the form of 10^n than is smaller than x
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

void ChartView::contextMenuEvent(QContextMenuEvent *event) {
  QMenu context_menu(this);
  context_menu.addActions(menu->actions());
  context_menu.addSeparator();
  context_menu.addAction(charts_widget->undo_zoom_action);
  context_menu.addAction(charts_widget->redo_zoom_action);
  context_menu.addSeparator();
  context_menu.addAction(close_act);
  context_menu.exec(event->globalPos());
}

void ChartView::mousePressEvent(QMouseEvent *event) {
  press_pos = event->pos();
  if (event->button() == Qt::LeftButton && move_icon_rect.contains(event->pos())) {
    charts_widget->startChartDrag(this, event->globalPos());
  } else if (event->button() == Qt::LeftButton && event->modifiers().testFlag(Qt::ShiftModifier)) {
    // Save current playback state when scrubbing
    resume_after_scrub = !can->isPaused();
    if (resume_after_scrub) {
      can->pause(true);
    }
    mouse_mode = MouseMode::Scrub;
  } else if (event->button() == Qt::LeftButton && plot_area.contains(event->pos())) {
    mouse_mode = MouseMode::Rubber;
    rubber_rect = QRect();
  } else {
    QWidget::mousePressEvent(event);
  }
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  // Scrubbing
  if (mouse_mode == MouseMode::Scrub && ev->modifiers().testFlag(Qt::ShiftModifier)) {
    if (plot_area.contains(ev->pos())) {
      can->seekTo(std::clamp(secondsAtPoint(ev->pos()), can->minSeconds(), can->maxSeconds()));
    }
  }

  if (mouse_mode == MouseMode::Rubber) {
    // horizontal selection, clamped to the plot area
    int left = std::clamp(std::min(press_pos.x(), ev->pos().x()), plot_area.left(), plot_area.right());
    int right = std::clamp(std::max(press_pos.x(), ev->pos().x()), plot_area.left(), plot_area.right());
    rubber_rect = QRect(left, plot_area.top(), right - left, plot_area.height());
    update();
  }

  clearTrackPoints();
  if (mouse_mode != MouseMode::Rubber && plot_area.contains(ev->pos()) && isActiveWindow()) {
    charts_widget->showValueTip(secondsAtPoint(ev->pos()));
  } else if (tip_label->isVisible()) {
    charts_widget->showValueTip(-1);
  }
  QWidget::mouseMoveEvent(ev);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton && mouse_mode == MouseMode::Rubber) {
    mouse_mode = MouseMode::None;
    // Prevent zooming/seeking past the end of the route
    double min = std::clamp(secondsAtPoint(rubber_rect.topLeft()), can->minSeconds(), can->maxSeconds());
    double max = std::clamp(secondsAtPoint(rubber_rect.bottomRight()), can->minSeconds(), can->maxSeconds());
    if (rubber_rect.width() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(std::clamp(secondsAtPoint(press_pos), can->minSeconds(), can->maxSeconds()));
    } else if (rubber_rect.width() > 10 && (max - min) > MIN_ZOOM_SECONDS) {
      charts_widget->zoom_undo_stack.push(new ZoomCommand({min, max}));
    }
    rubber_rect = QRect();
    update();
  } else if (event->button() == Qt::LeftButton && mouse_mode == MouseMode::None && sigs.size() > 1) {
    // toggle series visibility by clicking its legend entry
    for (int i = 0; i < sigs.size() && i < legend_rects.size(); ++i) {
      if (legend_rects[i].contains(press_pos) && legend_rects[i].contains(event->pos())) {
        sigs[i].visible = !sigs[i].visible;
        updateAxisY();
        updateTitle();
        break;
      }
    }
  } else if (event->button() == Qt::RightButton) {
    charts_widget->zoom_undo_stack.undo();
  } else {
    QWidget::mouseReleaseEvent(event);
  }

  // Resume playback if we were scrubbing
  if (mouse_mode == MouseMode::Scrub) {
    mouse_mode = MouseMode::None;
    if (resume_after_scrub) {
      can->pause(false);
      resume_after_scrub = false;
    }
  }
}

void ChartView::takeSignalsFrom(ChartView *source) {
  for (auto &s : source->sigs) {
    sigs.push_back(std::move(s));
    sigs.back().color = uniqueColor(sigs.back().color, sigs.back().sig);
  }
  source->sigs.clear();
  updateAxisY();
  updateTitle();
  charts_widget->removeChart(source);
}

void ChartView::showTip(double sec) {
  QRect tip_area(0, plot_area.top(), rect().width(), plot_area.height());
  QRect visible_rect = charts_widget->chartVisibleRect(this).intersected(tip_area);
  if (visible_rect.isEmpty()) {
    tip_label->hide();
    return;
  }

  tooltip_x = xPos(sec);
  qreal x = -1;
  QStringList text_list;
  for (auto &s : sigs) {
    if (s.visible) {
      QString value = "--";
      // use reverse iterator to find last item <= sec.
      auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), sec, [](auto &p, double v) { return p.x() > v; });
      if (it != s.vals.crend() && it->x() >= x_min) {
        value = QString::fromStdString(s.sig->formatValue(it->y(), false));
        s.track_pt = *it;
        x = std::max(x, xPos(it->x()));
      }
      QString name = sigs.size() > 1 ? QString::fromStdString(s.sig->name) + ": " : "";
      QString min = s.min == std::numeric_limits<double>::max() ? "--" : QString::number(s.min);
      QString max = s.max == std::numeric_limits<double>::lowest() ? "--" : QString::number(s.max);
      text_list << QString("<span style=\"color:%1;\">■ </span>%2<b>%3</b> (%4, %5)")
                       .arg(s.color.name(), name, value, min, max);
    }
  }
  if (x < 0) {
    x = tooltip_x;
  }
  QPoint pt(x, plot_area.top());
  text_list.push_front(QString::number(secondsAtPoint({x, 0}), 'f', 3));
  QString text = "<p style='white-space:pre'>" % text_list.join("<br />") % "</p>";
  tip_label->showText(pt, text, this, visible_rect);
  update();
}

void ChartView::hideTip() {
  clearTrackPoints();
  tooltip_x = -1;
  tip_label->hide();
  update();
}

void ChartView::resetChartCache() {
  chart_pixmap = QPixmap();
  update();
}

void ChartView::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHints(QPainter::Antialiasing);

  // the static layer is invalidated on x-range change and data merge, so cache it in live mode too
  const qreal dpr = devicePixelRatioF();
  if (chart_pixmap.isNull() || chart_pixmap.size() != size() * dpr) {
    chart_pixmap = QPixmap(size() * dpr);
    chart_pixmap.setDevicePixelRatio(dpr);
    QPainter p(&chart_pixmap);
    p.setRenderHints(QPainter::Antialiasing);
    p.setFont(font());
    drawStaticLayer(&p);
  }
  painter.drawPixmap(QPoint(), chart_pixmap);

  if (can_drop) {
    painter.setPen(QPen(palette().color(QPalette::Highlight), 4));
    painter.drawRect(rect());
  }
  drawForeground(&painter);
}

void ChartView::drawStaticLayer(QPainter *painter) {
  painter->fillRect(rect(), palette().color(QPalette::Base));
  painter->drawPixmap(move_icon_rect.topLeft(), utils::icon("grip-horizontal"));
  drawAxes(painter);
  drawLegend(painter);
  drawSeries(painter);
}

void ChartView::drawAxes(QPainter *painter) {
  const QColor text_color = palette().color(QPalette::Text);
  QColor grid_color = text_color;
  grid_color.setAlpha(50);
  QFontMetrics fm(font());
  painter->setFont(font());

  // y grid lines and tick labels
  for (int i = 0; i < y_tick_count; ++i) {
    double value = y_min + i * (y_max - y_min) / (y_tick_count - 1);
    qreal y = yPos(value);
    painter->setPen(grid_color);
    painter->drawLine(QPointF(plot_area.left(), y), QPointF(plot_area.right(), y));
    painter->setPen(text_color);
    QRectF label_rect(0, y - fm.height() / 2.0, plot_area.left() - 6, fm.height());
    painter->drawText(label_rect, Qt::AlignRight | Qt::AlignVCenter, QString::number(value, 'f', y_precision));
  }

  // rotated y axis title (unit)
  if (!y_unit.isEmpty()) {
    painter->save();
    painter->translate(plot_area.left() - y_label_width + fm.height() / 2.0, plot_area.center().y());
    painter->rotate(-90);
    painter->drawText(QRectF(-plot_area.height() / 2.0, -fm.height() / 2.0, plot_area.height(), fm.height()),
                      Qt::AlignCenter, y_unit);
    painter->restore();
  }

  // x grid lines and tick labels
  const int x_precision = xAxisPrecision();
  for (int i = 0; i < X_TICK_COUNT; ++i) {
    double sec = x_min + i * (x_max - x_min) / (X_TICK_COUNT - 1);
    qreal x = xPos(sec);
    painter->setPen(grid_color);
    painter->drawLine(QPointF(x, plot_area.top()), QPointF(x, plot_area.bottom()));
    painter->setPen(text_color);
    QString label = QString::number(sec, 'f', x_precision);
    QRectF label_rect(x - 100, plot_area.bottom() + AXIS_X_TOP_MARGIN, 200, fm.height());
    painter->drawText(label_rect, Qt::AlignHCenter | Qt::AlignTop, label);
  }
}

void ChartView::drawLegend(QPainter *painter) {
  QColor title_color = palette().color(QPalette::WindowText);
  // Draw message details in similar color, but slightly fade it to the background
  QColor msg_color = title_color;
  msg_color.setAlpha(180);
  QFont bold_font = font();
  bold_font.setBold(true);
  const int marker_size = QFontMetrics(font()).height() - 4;

  for (int i = 0; i < sigs.size() && i < legend_rects.size(); ++i) {
    const auto &s = sigs[i];
    const QRect &r = legend_rects[i];
    painter->setPen(Qt::NoPen);
    painter->setBrush(s.color);
    QRectF marker_rect(r.left(), r.center().y() - marker_size / 2.0, marker_size, marker_size);
    series_type == SeriesType::Scatter ? painter->drawEllipse(marker_rect) : painter->drawRect(marker_rect);

    bold_font.setStrikeOut(!s.visible);
    QFont normal_font = font();
    normal_font.setStrikeOut(!s.visible);

    qreal x = r.left() + marker_size + 5;
    painter->setFont(bold_font);
    painter->setPen(title_color);
    QString name = QFontMetrics(bold_font).elidedText(QString::fromStdString(s.sig->name), Qt::ElideRight, r.right() - x);
    painter->drawText(QRectF(x, r.top(), r.right() - x, r.height()), Qt::AlignLeft | Qt::AlignVCenter, name);
    x += QFontMetrics(bold_font).horizontalAdvance(name);
    painter->setFont(normal_font);
    painter->setPen(msg_color);
    QString msg = QFontMetrics(normal_font).elidedText(QString::fromStdString(" " + msgName(s.msg_id) + " " + s.msg_id.toString()),
                                                       Qt::ElideRight, r.right() - x);
    painter->drawText(QRectF(x, r.top(), r.right() - x, r.height()), Qt::AlignLeft | Qt::AlignVCenter, msg);
  }
}

void ChartView::drawSeries(QPainter *painter) {
  painter->save();
  painter->setClipRect(plot_area);
  for (auto &s : sigs) {
    if (!s.visible) continue;

    // visible points in vals to compute point density
    auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), x_min, xLessThan);
    auto last = std::lower_bound(first, s.vals.cend(), x_max, xLessThan);
    int num_points = std::max<int>(last - first, 1);
    double pixels_per_point = 0;
    if (first != last) {
      const QPointF &right_pt = last == s.vals.cend() ? s.vals.back() : *last;
      pixels_per_point = (xPos(right_pt.x()) - xPos(first->x())) / num_points;
    }

    if (series_type == SeriesType::Scatter) {
      qreal radius = std::clamp(pixels_per_point / 2.0, 2.0, 8.0) / 2.0;
      painter->setPen(Qt::NoPen);
      painter->setBrush(s.color);
      for (auto it = first; it != last; ++it) {
        painter->drawEllipse(QPointF(xPos(it->x()), yPos(it->y())), radius, radius);
      }
    } else {
      const auto &points = series_type == SeriesType::StepLine ? s.step_vals : s.vals;
      auto begin = std::lower_bound(points.cbegin(), points.cend(), x_min, xLessThan);
      if (begin != points.cbegin()) --begin;
      auto end = std::lower_bound(begin, points.cend(), x_max, xLessThan);
      if (end != points.cend()) ++end;
      if (begin == end) continue;

      std::vector<QPointF> polyline;
      polyline.reserve(end - begin);
      for (auto it = begin; it != end; ++it) {
        polyline.emplace_back(xPos(it->x()), yPos(it->y()));
      }
      painter->setPen(QPen(s.color, 2));
      painter->setBrush(Qt::NoBrush);
      painter->drawPolyline(polyline.data(), polyline.size());

      // show points when zoomed in enough
      if (num_points == 1 || pixels_per_point > 20) {
        painter->setPen(Qt::NoPen);
        painter->setBrush(s.color);
        for (auto it = first; it != last; ++it) {
          painter->drawEllipse(QPointF(xPos(it->x()), yPos(it->y())), 4, 4);
        }
      }
    }
  }
  painter->restore();
}

void ChartView::drawForeground(QPainter *painter) {
  drawTimeline(painter);
  drawSignalValue(painter);
  // draw track points
  painter->setPen(Qt::NoPen);
  qreal track_line_x = -1;
  for (auto &s : sigs) {
    if (!s.track_pt.isNull() && s.visible) {
      painter->setBrush(s.color.darker(125));
      QPointF pos(xPos(s.track_pt.x()), yPos(s.track_pt.y()));
      painter->drawEllipse(pos, 5.5, 5.5);
      track_line_x = std::max(track_line_x, pos.x());
    }
  }
  if (track_line_x > 0) {
    painter->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
    painter->drawLine(QPointF{track_line_x, (qreal)plot_area.top()}, QPointF{track_line_x, (qreal)plot_area.bottom()});
  }

  drawRubberBandTimeRange(painter);
}

void ChartView::drawRubberBandTimeRange(QPainter *painter) {
  if (rubber_rect.width() <= 1) return;

  // selection rect
  QColor highlight = palette().color(QPalette::Highlight);
  QColor fill = highlight;
  fill.setAlpha(50);
  painter->fillRect(rubber_rect, fill);
  painter->setPen(highlight);
  painter->setBrush(Qt::NoBrush);
  painter->drawRect(rubber_rect);

  // time labels at the bottom corners
  painter->setPen(Qt::white);
  painter->setFont(font());
  for (const auto &pt : {rubber_rect.bottomLeft(), rubber_rect.bottomRight()}) {
    QString sec = QString::number(secondsAtPoint(pt), 'f', 2);
    auto r = painter->fontMetrics().boundingRect(sec).adjusted(-6, -AXIS_X_TOP_MARGIN, 6, AXIS_X_TOP_MARGIN);
    pt == rubber_rect.bottomLeft() ? r.moveTopRight(pt + QPoint{0, 2}) : r.moveTopLeft(pt + QPoint{0, 2});
    painter->fillRect(r, Qt::gray);
    painter->drawText(r, Qt::AlignCenter, sec);
  }
}

void ChartView::drawTimeline(QPainter *painter) {
  // draw vertical time line
  qreal x = std::clamp(xPos(cur_sec), (qreal)plot_area.left(), (qreal)plot_area.right());
  painter->setPen(QPen(palette().color(QPalette::Text), 1));
  painter->drawLine(QPointF{x, plot_area.top() - 1.0}, QPointF{x, plot_area.bottom() + 1.0});

  // draw current time under the axis-x
  QString time_str = QString::number(cur_sec, 'f', 2);
  QSize time_str_size = QFontMetrics(font()).size(Qt::TextSingleLine, time_str) + QSize(8, 2);
  QRectF time_str_rect(QPointF(x - time_str_size.width() / 2.0, plot_area.bottom() + AXIS_X_TOP_MARGIN), time_str_size);
  QPainterPath path;
  path.addRoundedRect(time_str_rect, 3, 3);
  painter->fillPath(path, utils::isDarkTheme() ? Qt::darkGray : Qt::gray);
  painter->setPen(palette().color(QPalette::BrightText));
  painter->setFont(font());
  painter->drawText(time_str_rect, Qt::AlignCenter, time_str);
}

void ChartView::drawSignalValue(QPainter *painter) {
  painter->setFont(signal_value_font);
  painter->setPen(palette().color(QPalette::Text));
  for (int i = 0; i < sigs.size() && i < legend_rects.size(); ++i) {
    const auto &s = sigs[i];
    auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), cur_sec,
                               [](auto &p, double x) { return p.x() > x + EPSILON; });
    QString value = (it != s.vals.crend() && it->x() >= x_min) ? QString::fromStdString(s.sig->formatValue(it->y())) : "--";
    QRectF value_rect(legend_rects[i].bottomLeft() - QPoint(0, 1), legend_rects[i].size());
    QString elided_val = painter->fontMetrics().elidedText(value, Qt::ElideRight, value_rect.width());
    painter->drawText(value_rect, Qt::AlignHCenter | Qt::AlignTop, elided_val);
  }
}

QColor ChartView::uniqueColor(QColor color, const cabana::Signal *exclude) const {
  for (auto &s : sigs) {
    if (s.sig != exclude && std::abs(color.hueF() - s.color.hueF()) < 0.1) {
      // use different color to distinguish it from others.
      auto last_color = sigs.back().color;
      static thread_local std::mt19937 rng{std::random_device{}()};
      std::uniform_int_distribution<int> sat(35, 99);
      std::uniform_int_distribution<int> val(85, 99);
      color.setHsvF(std::fmod(last_color.hueF() + 60 / 360.0, 1.0),
                    sat(rng) / 100.0,
                    val(rng) / 100.0);
      break;
    }
  }
  return color;
}

void ChartView::setSeriesType(SeriesType type) {
  if (type != series_type) {
    series_type = type;
    menu->actions()[(int)type]->setChecked(true);
    updateTitle();
  }
}
