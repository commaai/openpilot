#include "tools/cabana/chart/chart.h"

#include <algorithm>
#include <limits>

#include <QActionGroup>
#include <QApplication>
#include <QDrag>
#include <QGraphicsLayout>
#include <QGraphicsDropShadowEffect>
#include <QGraphicsItemGroup>
#include <QGraphicsOpacityEffect>
#include <QMimeData>
#include <QOpenGLWidget>
#include <QPropertyAnimation>
#include <QRandomGenerator>
#include <QRubberBand>
#include <QScreen>
#include <QWindow>

#include "tools/cabana/chart/chartswidget.h"

// ChartAxisElement's padding is 4 (https://codebrowser.dev/qt5/qtcharts/src/charts/axis/chartaxiselement_p.h.html)
const int AXIS_X_TOP_MARGIN = 4;
// Define a small value of epsilon to compare double values
const float EPSILON = 0.000001;
static inline bool xLessThan(const QPointF &p, float x) { return p.x() < (x - EPSILON); }

ChartView::ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent)
    : charts_widget(parent), QChartView(parent) {
  series_type = (SeriesType)settings.chart_series_type;
  chart()->setBackgroundVisible(false);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart()->addAxis(axis_x, Qt::AlignBottom);
  chart()->addAxis(axis_y, Qt::AlignLeft);
  chart()->legend()->layout()->setContentsMargins(0, 0, 0, 0);
  chart()->legend()->setShowToolTips(true);
  chart()->setMargins({0, 0, 0, 0});

  axis_x->setRange(x_range.first, x_range.second);

  tip_label = new TipLabel(this);
  createToolButtons();
  setRubberBand(QChartView::HorizontalRubberBand);
  setMouseTracking(true);
  setTheme(settings.theme == DARK_THEME ? QChart::QChart::ChartThemeDark : QChart::ChartThemeLight);
  signal_value_font.setPointSize(9);

  QObject::connect(axis_y, &QValueAxis::rangeChanged, this, &ChartView::resetChartCache);
  QObject::connect(axis_y, &QAbstractAxis::titleTextChanged, this, &ChartView::resetChartCache);
  QObject::connect(window()->windowHandle(), &QWindow::screenChanged, this, &ChartView::resetChartCache);

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &ChartView::msgUpdated);
}

void ChartView::createToolButtons() {
  move_icon = new QGraphicsPixmapItem(utils::icon("grip-horizontal"), chart());
  move_icon->setToolTip(tr("Drag and drop to move chart"));

  QToolButton *remove_btn = new ToolButton("x", tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart());
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart()->zValue() + 11);

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

  QToolButton *manage_btn = new ToolButton("list", "");
  manage_btn->setMenu(menu);
  manage_btn->setPopupMode(QToolButton::InstantPopup);
  manage_btn->setStyleSheet("QToolButton::menu-indicator { image: none; }");
  manage_btn_proxy = new QGraphicsProxyWidget(chart());
  manage_btn_proxy->setWidget(manage_btn);
  manage_btn_proxy->setZValue(chart()->zValue() + 11);

  close_act = new QAction(tr("Close"), this);
  QObject::connect(close_act, &QAction::triggered, [this] () { charts_widget->removeChart(this); });
  QObject::connect(remove_btn, &QToolButton::clicked, close_act, &QAction::triggered);
  QObject::connect(change_series_group, &QActionGroup::triggered, [this](QAction *action) {
    setSeriesType((SeriesType)action->data().toInt());
  });
}

QSize ChartView::sizeHint() const {
  return {CHART_MIN_WIDTH, settings.chart_height};
}

void ChartView::setTheme(QChart::ChartTheme theme) {
  chart()->setTheme(theme);
  if (theme == QChart::ChartThemeDark) {
    axis_x->setTitleBrush(palette().text());
    axis_x->setLabelsBrush(palette().text());
    axis_y->setTitleBrush(palette().text());
    axis_y->setLabelsBrush(palette().text());
    chart()->legend()->setLabelColor(palette().color(QPalette::Text));
  }
  axis_x->setLineVisible(false);
  axis_y->setLineVisible(false);
  for (auto &s : sigs) {
    s.series->setColor(s.sig->color);
  }
}

void ChartView::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  if (hasSignal(msg_id, sig)) return;

  QXYSeries *series = createSeries(series_type, sig->color);
  sigs.push_back({.msg_id = msg_id, .sig = sig, .series = series});
  updateSeries(sig);
  updateSeriesPoints();
  updateTitle();
  emit charts_widget->seriesChanged();
}

bool ChartView::hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs.cbegin(), sigs.cend(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs.size();
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    if (predicate(*it)) {
      chart()->removeSeries(it->series);
      it->series->deleteLater();
      it = sigs.erase(it);
    } else {
      ++it;
    }
  }
  if (sigs.empty()) {
    charts_widget->removeChart(this);
  } else if (sigs.size() != prev_size) {
    emit charts_widget->seriesChanged();
    updateAxisY();
    resetChartCache();
  }
}

void ChartView::signalUpdated(const cabana::Signal *sig) {
  if (std::any_of(sigs.cbegin(), sigs.cend(), [=](auto &s) { return s.sig == sig; })) {
    for (const auto &s : sigs) {
      if (s.sig == sig && s.series->color() != sig->color) {
        setSeriesColor(s.series, sig->color);
      }
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
  qreal left, top, right, bottom;
  chart()->layout()->getContentsMargins(&left, &top, &right, &bottom);
  move_icon->setPos(left, top);
  close_btn_proxy->setPos(rect().right() - right - close_btn_proxy->size().width(), top);
  int x = close_btn_proxy->pos().x() - manage_btn_proxy->size().width() - style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing);
  manage_btn_proxy->setPos(x, top);
  if (align_to > 0) {
    updatePlotArea(align_to, true);
  }
  QChartView::resizeEvent(event);
}

void ChartView::updatePlotArea(int left_pos, bool force) {
  if (align_to != left_pos || force) {
    align_to = left_pos;

    qreal left, top, right, bottom;
    chart()->layout()->getContentsMargins(&left, &top, &right, &bottom);
    QSizeF legend_size = chart()->legend()->layout()->minimumSize();
    legend_size.setWidth(manage_btn_proxy->sceneBoundingRect().left() - move_icon->sceneBoundingRect().right());
    chart()->legend()->setGeometry({move_icon->sceneBoundingRect().topRight(), legend_size});

    // add top space for signal value
    int adjust_top = chart()->legend()->geometry().height() + QFontMetrics(signal_value_font).height() + 3;
    adjust_top = std::max<int>(adjust_top, manage_btn_proxy->sceneBoundingRect().height() + style()->pixelMetric(QStyle::PM_LayoutTopMargin));
    // add right space for x-axis label
    QSizeF x_label_size = QFontMetrics(axis_x->labelsFont()).size(Qt::TextSingleLine, QString::number(axis_x->max(), 'f', 2));
    x_label_size += QSizeF{5, 5};
    chart()->setPlotArea(rect().adjusted(align_to + left, adjust_top + top, -x_label_size.width() / 2 - right, -x_label_size.height() - bottom));
    chart()->layout()->invalidate();
    resetChartCache();
  }
}

void ChartView::updateTitle() {
  for (QLegendMarker *marker : chart()->legend()->markers()) {
    QObject::connect(marker, &QLegendMarker::clicked, this, &ChartView::handleMarkerClicked, Qt::UniqueConnection);
  }

  // Use CSS to draw titles in the WindowText color
  auto tmp = palette().color(QPalette::WindowText);
  auto titleColorCss = tmp.name(QColor::HexArgb);
  // Draw message details in similar color, but slightly fade it to the background
  tmp.setAlpha(180);
  auto msgColorCss = tmp.name(QColor::HexArgb);

  for (auto &s : sigs) {
    auto decoration = s.series->isVisible() ? "none" : "line-through";
    s.series->setName(QString("<span style=\"text-decoration:%1; color:%2\"><b>%3</b> <font color=\"%4\">%5 %6</font></span>")
                      .arg(decoration, titleColorCss, s.sig->name,
                           msgColorCss, msgName(s.msg_id), s.msg_id.toString()));
  }
  split_chart_act->setEnabled(sigs.size() > 1);
  resetChartCache();
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != axis_x->min() || max != axis_x->max()) {
    axis_x->setRange(min, max);
    updateAxisY();
    updateSeriesPoints();
    // update tooltip
    if (tooltip_x >= 0) {
      showTip(chart()->mapToValue({tooltip_x, 0}).x());
    }
    resetChartCache();
  }
  viewport()->update();
}

void ChartView::updateSeriesPoints() {
  // Show points when zoomed in enough
  for (auto &s : sigs) {
    auto begin = std::lower_bound(s.vals.cbegin(), s.vals.cend(), axis_x->min(), xLessThan);
    auto end = std::lower_bound(begin, s.vals.cend(), axis_x->max(), xLessThan);
    if (begin != end) {
      int num_points = std::max<int>((end - begin), 1);
      QPointF right_pt = end == s.vals.cend() ? s.vals.back() : *end;
      double pixels_per_point = (chart()->mapToPosition(right_pt).x() - chart()->mapToPosition(*begin).x()) / num_points;

      if (series_type == SeriesType::Scatter) {
        qreal size = std::clamp(pixels_per_point / 2.0, 2.0, 8.0);
        if (s.series->useOpenGL()) {
          size *= devicePixelRatioF();
        }
        ((QScatterSeries *)s.series)->setMarkerSize(size);
      } else {
        s.series->setPointsVisible(pixels_per_point > 20);
      }
    }
  }
}

void ChartView::appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                                std::vector<QPointF> &vals, std::vector<QPointF> &step_vals) {
  vals.reserve(vals.size() + events.capacity());
  step_vals.reserve(step_vals.size() + events.capacity() * 2);

  double value = 0;
  const uint64_t begin_mono_time = can->routeStartTime() * 1e9;
  for (const CanEvent *e : events) {
    if (sig->getValue(e->dat, e->size, &value)) {
      const double ts = (e->mono_time - std::min(e->mono_time, begin_mono_time)) / 1e9;
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

      if (s.vals.empty() || (it->second.back()->mono_time / 1e9 - can->routeStartTime()) > s.vals.back().x()) {
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
      s.series->replace(QVector<QPointF>::fromStdVector(series_type == SeriesType::StepLine ? s.step_vals : s.vals));
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
  QString unit = sigs[0].sig->unit;

  for (auto &s : sigs) {
    if (!s.series->isVisible()) continue;

    // Only show unit when all signals have the same unit
    if (unit != s.sig->unit) {
      unit.clear();
    }

    auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), axis_x->min(), xLessThan);
    auto last = std::lower_bound(first, s.vals.cend(), axis_x->max(), xLessThan);
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

  if (axis_y->titleText() != unit) {
    axis_y->setTitleText(unit);
    y_label_width = 0;  // recalc width
  }

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, 3);
  if (min_y != axis_y->min() || max_y != axis_y->max() || y_label_width == 0) {
    axis_y->setRange(min_y, max_y);
    axis_y->setTickCount(tick_count);

    int n = std::max(int(-std::floor(std::log10((max_y - min_y) / (tick_count - 1)))), 0);
    int max_label_width = 0;
    QFontMetrics fm(axis_y->labelsFont());
    for (int i = 0; i < tick_count; i++) {
      qreal value = min_y + (i * (max_y - min_y) / (tick_count - 1));
      max_label_width = std::max(max_label_width, fm.width(QString::number(value, 'f', n)));
    }

    int title_spacing = unit.isEmpty() ? 0 : QFontMetrics(axis_y->titleFont()).size(Qt::TextSingleLine, unit).height();
    y_label_width = title_spacing + max_label_width + 15;
    axis_y->setLabelFormat(QString("%.%1f").arg(n));
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

void ChartView::leaveEvent(QEvent *event) {
  if (tip_label->isVisible()) {
    charts_widget->showValueTip(-1);
  }
  QChartView::leaveEvent(event);
}

QPixmap getBlankShadowPixmap(const QPixmap &px, int radius) {
  QGraphicsDropShadowEffect *e = new QGraphicsDropShadowEffect;
  e->setColor(QColor(40, 40, 40, 245));
  e->setOffset(0, 0);
  e->setBlurRadius(radius);

  qreal dpr = px.devicePixelRatio();
  QPixmap blank(px.size());
  blank.setDevicePixelRatio(dpr);
  blank.fill(Qt::white);

  QGraphicsScene scene;
  QGraphicsPixmapItem item(blank);
  item.setGraphicsEffect(e);
  scene.addItem(&item);

  QPixmap shadow(px.size() + QSize(radius * dpr * 2, radius * dpr * 2));
  shadow.setDevicePixelRatio(dpr);
  shadow.fill(Qt::transparent);
  QPainter p(&shadow);
  scene.render(&p, {QPoint(), shadow.size() / dpr}, item.boundingRect().adjusted(-radius, -radius, radius, radius));
  return shadow;
}

static QPixmap getDropPixmap(const QPixmap &src) {
  static QPixmap shadow_px;
  const int radius = 10;
  if (shadow_px.size() != src.size() + QSize(radius * 2, radius * 2)) {
    shadow_px = getBlankShadowPixmap(src, radius);
  }
  QPixmap px = shadow_px;
  QPainter p(&px);
  QRectF target_rect(QPointF(radius, radius), src.size() / src.devicePixelRatio());
  p.drawPixmap(target_rect.topLeft(), src);
  p.setCompositionMode(QPainter::CompositionMode_DestinationIn);
  p.fillRect(target_rect, QColor(0, 0, 0, 200));
  return px;
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
  if (event->button() == Qt::LeftButton && move_icon->sceneBoundingRect().contains(event->pos())) {
    QMimeData *mimeData = new QMimeData;
    mimeData->setData(CHART_MIME_TYPE, QByteArray::number((qulonglong)this));
    QPixmap px = grab().scaledToWidth(CHART_MIN_WIDTH * viewport()->devicePixelRatio(), Qt::SmoothTransformation);
    charts_widget->stopAutoScroll();
    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setPixmap(getDropPixmap(px));
    drag->setHotSpot(-QPoint(5, 5));
    drag->exec(Qt::CopyAction | Qt::MoveAction, Qt::MoveAction);
  } else if (event->button() == Qt::LeftButton && QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier)) {
    // Save current playback state when scrubbing
    resume_after_scrub = !can->isPaused();
    if (resume_after_scrub) {
      can->pause(true);
    }
    is_scrubbing = true;
  } else {
    QChartView::mousePressEvent(event);
  }
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  auto rubber = findChild<QRubberBand *>();
  if (event->button() == Qt::LeftButton && rubber && rubber->isVisible()) {
    rubber->hide();
    auto rect = rubber->geometry().normalized();
    // Prevent zooming/seeking past the end of the route
    double min = std::clamp(chart()->mapToValue(rect.topLeft()).x(), 0., can->totalSeconds());
    double max = std::clamp(chart()->mapToValue(rect.bottomRight()).x(), 0., can->totalSeconds());
    if (rubber->width() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(min);
    } else if (rubber->width() > 10 && (max - min) > 0.01) { // Minimum range is 10 milliseconds.
      charts_widget->zoom_undo_stack->push(new ZoomCommand(charts_widget, {min, max}));
    } else {
      viewport()->update();
    }
    event->accept();
  } else if (event->button() == Qt::RightButton) {
    charts_widget->zoom_undo_stack->undo();
    event->accept();
  } else {
    QGraphicsView::mouseReleaseEvent(event);
  }

  // Resume playback if we were scrubbing
  is_scrubbing = false;
  if (resume_after_scrub) {
    can->pause(false);
    resume_after_scrub = false;
  }
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  const auto plot_area = chart()->plotArea();
  // Scrubbing
  if (is_scrubbing && QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier)) {
    if (plot_area.contains(ev->pos())) {
      can->seekTo(std::clamp(chart()->mapToValue(ev->pos()).x(), 0., can->totalSeconds()));
    }
  }

  auto rubber = findChild<QRubberBand *>();
  bool is_zooming = rubber && rubber->isVisible();
  clearTrackPoints();

  if (!is_zooming && plot_area.contains(ev->pos())) {
    const double sec = chart()->mapToValue(ev->pos()).x();
    charts_widget->showValueTip(sec);
  } else if (tip_label->isVisible()) {
    charts_widget->showValueTip(-1);
  }

  QChartView::mouseMoveEvent(ev);
  if (is_zooming) {
    QRect rubber_rect = rubber->geometry();
    rubber_rect.setLeft(std::max(rubber_rect.left(), (int)plot_area.left()));
    rubber_rect.setRight(std::min(rubber_rect.right(), (int)plot_area.right()));
    if (rubber_rect != rubber->geometry()) {
      rubber->setGeometry(rubber_rect);
    }
    viewport()->update();
  }
}

void ChartView::showTip(double sec) {
  QRect tip_area(0, chart()->plotArea().top(), rect().width(), chart()->plotArea().height());
  QRect visible_rect = charts_widget->chartVisibleRect(this).intersected(tip_area);
  if (visible_rect.isEmpty()) {
    tip_label->hide();
    return;
  }

  tooltip_x = chart()->mapToPosition({sec, 0}).x();
  qreal x = -1;
  QStringList text_list;
  for (auto &s : sigs) {
    if (s.series->isVisible()) {
      QString value = "--";
      // use reverse iterator to find last item <= sec.
      auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), sec, [](auto &p, double x) { return p.x() > x; });
      if (it != s.vals.crend() && it->x() >= axis_x->min()) {
        value = s.sig->formatValue(it->y(), false);
        s.track_pt = *it;
        x = std::max(x, chart()->mapToPosition(*it).x());
      }
      QString name = sigs.size() > 1 ? s.sig->name + ": " : "";
      QString min = s.min == std::numeric_limits<double>::max() ? "--" : QString::number(s.min);
      QString max = s.max == std::numeric_limits<double>::lowest() ? "--" : QString::number(s.max);
      text_list << QString("<span style=\"color:%1;\">■ </span>%2<b>%3</b> (%4, %5)")
                       .arg(s.series->color().name(), name, value, min, max);
    }
  }
  if (x < 0) {
    x = tooltip_x;
  }
  QPoint pt(x, chart()->plotArea().top());
  text_list.push_front(QString::number(chart()->mapToValue({x, 0}).x(), 'f', 3));
  QString text = "<p style='white-space:pre'>" % text_list.join("<br />") % "</p>";
  tip_label->showText(pt, text, this, visible_rect);
  viewport()->update();
}

void ChartView::hideTip() {
  clearTrackPoints();
  tooltip_x = -1;
  tip_label->hide();
  viewport()->update();
}

void ChartView::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasFormat(CHART_MIME_TYPE)) {
    drawDropIndicator(event->source() != this);
    event->acceptProposedAction();
  }
}

void ChartView::dragMoveEvent(QDragMoveEvent *event) {
  if (event->mimeData()->hasFormat(CHART_MIME_TYPE)) {
    event->setDropAction(event->source() == this ? Qt::MoveAction : Qt::CopyAction);
    event->accept();
  }
  charts_widget->startAutoScroll();
}

void ChartView::dropEvent(QDropEvent *event) {
  if (event->mimeData()->hasFormat(CHART_MIME_TYPE)) {
    if (event->source() != this) {
      ChartView *source_chart = (ChartView *)event->source();
      for (auto &s : source_chart->sigs) {
        source_chart->chart()->removeSeries(s.series);
        addSeries(s.series);
      }
      sigs.insert(sigs.end(), std::move_iterator(source_chart->sigs.begin()), std::move_iterator(source_chart->sigs.end()));
      updateAxisY();
      updateTitle();
      startAnimation();

      source_chart->sigs.clear();
      charts_widget->removeChart(source_chart);
      event->acceptProposedAction();
    }
    can_drop = false;
  }
}

void ChartView::resetChartCache() {
  chart_pixmap = QPixmap();
  viewport()->update();
}

void ChartView::startAnimation() {
  QGraphicsOpacityEffect *eff = new QGraphicsOpacityEffect(this);
  viewport()->setGraphicsEffect(eff);
  QPropertyAnimation *a = new QPropertyAnimation(eff, "opacity");
  a->setDuration(250);
  a->setStartValue(0.3);
  a->setEndValue(1);
  a->setEasingCurve(QEasingCurve::InBack);
  a->start(QPropertyAnimation::DeleteWhenStopped);
}

void ChartView::paintEvent(QPaintEvent *event) {
  if (!can->liveStreaming()) {
    if (chart_pixmap.isNull()) {
      const qreal dpr = viewport()->devicePixelRatioF();
      chart_pixmap = QPixmap(viewport()->size() * dpr);
      chart_pixmap.setDevicePixelRatio(dpr);
      QPainter p(&chart_pixmap);
      p.setRenderHints(QPainter::Antialiasing);
      drawBackground(&p, viewport()->rect());
      scene()->setSceneRect(viewport()->rect());
      scene()->render(&p, viewport()->rect());
    }

    QPainter painter(viewport());
    painter.setRenderHints(QPainter::Antialiasing);
    painter.drawPixmap(QPoint(), chart_pixmap);
    if (can_drop) {
      painter.setPen(QPen(palette().color(QPalette::Highlight), 4));
      painter.drawRect(viewport()->rect());
    }
    QRectF exposed_rect = mapToScene(event->region().boundingRect()).boundingRect();
    drawForeground(&painter, exposed_rect);
  } else {
    QChartView::paintEvent(event);
  }
}

void ChartView::drawBackground(QPainter *painter, const QRectF &rect) {
  painter->fillRect(rect, palette().color(QPalette::Base));
}

void ChartView::drawForeground(QPainter *painter, const QRectF &rect) {
  drawTimeline(painter);
  drawSignalValue(painter);
  // draw track points
  painter->setPen(Qt::NoPen);
  qreal track_line_x = -1;
  for (auto &s : sigs) {
    if (!s.track_pt.isNull() && s.series->isVisible()) {
      painter->setBrush(s.series->color().darker(125));
      QPointF pos = chart()->mapToPosition(s.track_pt);
      painter->drawEllipse(pos, 5.5, 5.5);
      track_line_x = std::max(track_line_x, pos.x());
    }
  }
  if (track_line_x > 0) {
    auto plot_area = chart()->plotArea();
    painter->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
    painter->drawLine(QPointF{track_line_x, plot_area.top()}, QPointF{track_line_x, plot_area.bottom()});
  }

  // paint points. OpenGL mode lacks certain features (such as showing points)
  painter->setPen(Qt::NoPen);
  for (auto &s : sigs) {
    if (s.series->useOpenGL() && s.series->isVisible() && s.series->pointsVisible()) {
      auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), axis_x->min(), xLessThan);
      auto last = std::lower_bound(first, s.vals.cend(), axis_x->max(), xLessThan);
      painter->setBrush(s.series->color());
      for (auto it = first; it != last; ++it) {
        painter->drawEllipse(chart()->mapToPosition(*it), 4, 4);
      }
    }
  }

  drawRubberBandTimeRange(painter);
}

void ChartView::drawRubberBandTimeRange(QPainter *painter) {
  auto rubber = findChild<QRubberBand *>();
  if (rubber && rubber->isVisible() && rubber->width() > 1) {
    painter->setPen(Qt::white);
    auto rubber_rect = rubber->geometry().normalized();
    for (const auto &pt : {rubber_rect.bottomLeft(), rubber_rect.bottomRight()}) {
      QString sec = QString::number(chart()->mapToValue(pt).x(), 'f', 2);
      auto r = painter->fontMetrics().boundingRect(sec).adjusted(-6, -AXIS_X_TOP_MARGIN, 6, AXIS_X_TOP_MARGIN);
      pt == rubber_rect.bottomLeft() ? r.moveTopRight(pt + QPoint{0, 2}) : r.moveTopLeft(pt + QPoint{0, 2});
      painter->fillRect(r, Qt::gray);
      painter->drawText(r, Qt::AlignCenter, sec);
    }
  }
}

void ChartView::drawTimeline(QPainter *painter) {
  const auto plot_area = chart()->plotArea();
  // draw vertical time line
  qreal x = std::clamp(chart()->mapToPosition(QPointF{cur_sec, 0}).x(), plot_area.left(), plot_area.right());
  painter->setPen(QPen(chart()->titleBrush().color(), 1));
  painter->drawLine(QPointF{x, plot_area.top() - 1}, QPointF{x, plot_area.bottom() + 1});

  // draw current time under the axis-x
  QString time_str = QString::number(cur_sec, 'f', 2);
  QSize time_str_size = QFontMetrics(axis_x->labelsFont()).size(Qt::TextSingleLine, time_str) + QSize(8, 2);
  QRectF time_str_rect(QPointF(x - time_str_size.width() / 2.0, plot_area.bottom() + AXIS_X_TOP_MARGIN), time_str_size);
  QPainterPath path;
  path.addRoundedRect(time_str_rect, 3, 3);
  painter->fillPath(path, settings.theme == DARK_THEME ? Qt::darkGray : Qt::gray);
  painter->setPen(palette().color(QPalette::BrightText));
  painter->setFont(axis_x->labelsFont());
  painter->drawText(time_str_rect, Qt::AlignCenter, time_str);
}

void ChartView::drawSignalValue(QPainter *painter) {
  auto item_group = qgraphicsitem_cast<QGraphicsItemGroup *>(chart()->legend()->childItems()[0]);
  assert(item_group != nullptr);
  auto legend_markers = item_group->childItems();
  assert(legend_markers.size() == sigs.size());

  painter->setFont(signal_value_font);
  painter->setPen(chart()->legend()->labelColor());
  int i = 0;
  for (auto &s : sigs) {
    auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), cur_sec,
                               [](auto &p, double x) { return p.x() > x + EPSILON; });
    QString value = (it != s.vals.crend() && it->x() >= axis_x->min()) ? s.sig->formatValue(it->y()) : "--";
    QRectF marker_rect = legend_markers[i++]->sceneBoundingRect();
    QRectF value_rect(marker_rect.bottomLeft() - QPoint(0, 1), marker_rect.size());
    QString elided_val = painter->fontMetrics().elidedText(value, Qt::ElideRight, value_rect.width());
    painter->drawText(value_rect, Qt::AlignHCenter | Qt::AlignTop, elided_val);
  }
}

QXYSeries *ChartView::createSeries(SeriesType type, QColor color) {
  QXYSeries *series = nullptr;
  if (type == SeriesType::Line) {
    series = new QLineSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeRectangle);
  } else if (type == SeriesType::StepLine) {
    series = new QLineSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
  } else {
    series = new QScatterSeries(this);
    static_cast<QScatterSeries*>(series)->setBorderColor(color);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeCircle);
  }
  series->setColor(color);
  // TODO: Due to a bug in CameraWidget the camera frames
  // are drawn instead of the graphs on MacOS. Re-enable OpenGL when fixed
#ifndef __APPLE__
  series->setUseOpenGL(true);
  // Qt doesn't properly apply device pixel ratio in OpenGL mode
  QPen pen = series->pen();
  pen.setWidthF(2.0 * devicePixelRatioF());
  series->setPen(pen);
#endif
  addSeries(series);
  return series;
}

void ChartView::addSeries(QXYSeries *series) {
  setSeriesColor(series, series->color());
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);

  // disables the delivery of mouse events to the opengl widget.
  // this enables the user to select the zoom area when the mouse press on the data point.
  auto glwidget = findChild<QOpenGLWidget *>();
  if (glwidget && !glwidget->testAttribute(Qt::WA_TransparentForMouseEvents)) {
    glwidget->setAttribute(Qt::WA_TransparentForMouseEvents);
  }
}

void ChartView::setSeriesColor(QXYSeries *series, QColor color) {
  auto existing_series = chart()->series();
  for (auto s : existing_series) {
    if (s != series && std::abs(color.hueF() - qobject_cast<QXYSeries *>(s)->color().hueF()) < 0.1) {
      // use different color to distinguish it from others.
      auto last_color = qobject_cast<QXYSeries *>(existing_series.back())->color();
      color.setHsvF(std::fmod(last_color.hueF() + 60 / 360.0, 1.0),
                    QRandomGenerator::global()->bounded(35, 100) / 100.0,
                    QRandomGenerator::global()->bounded(85, 100) / 100.0);
      break;
    }
  }
  series->setColor(color);
}

void ChartView::setSeriesType(SeriesType type) {
  if (type != series_type) {
    series_type = type;
    for (auto &s : sigs) {
      chart()->removeSeries(s.series);
      s.series->deleteLater();
    }
    for (auto &s : sigs) {
      s.series = createSeries(series_type, s.sig->color);
      s.series->replace(QVector<QPointF>::fromStdVector(series_type == SeriesType::StepLine ? s.step_vals : s.vals));
    }
    updateSeriesPoints();
    updateTitle();
  }
}

void ChartView::handleMarkerClicked() {
  auto marker = qobject_cast<QLegendMarker *>(sender());
  Q_ASSERT(marker);
  if (sigs.size() > 1) {
    auto series = marker->series();
    series->setVisible(!series->isVisible());
    marker->setVisible(true);
    updateAxisY();
    updateTitle();
  }
}
