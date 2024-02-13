#include "tools/cabana/chart/tiplabel.h"

#include <utility>

#include <QApplication>
#include <QStylePainter>
#include <QToolTip>

#include "tools/cabana/settings.h"

TipLabel::TipLabel(QWidget *parent) : QLabel(parent, Qt::ToolTip | Qt::FramelessWindowHint) {
  setForegroundRole(QPalette::ToolTipText);
  setBackgroundRole(QPalette::ToolTipBase);
  QFont font;
  font.setPointSizeF(8.34563465);
  setFont(font);
  auto palette = QToolTip::palette();
  if (settings.theme != DARK_THEME) {
    palette.setColor(QPalette::ToolTipBase, QApplication::palette().color(QPalette::Base));
    palette.setColor(QPalette::ToolTipText, QRgb(0x404044));  // same color as chart label brush
  }
  setPalette(palette);
  ensurePolished();
  setMargin(1 + style()->pixelMetric(QStyle::PM_ToolTipLabelFrameWidth, nullptr, this));
  setAttribute(Qt::WA_ShowWithoutActivating);
  setTextFormat(Qt::RichText);
  setVisible(false);
}

void TipLabel::showText(const QPoint &pt, const QString &text, QWidget *w, const QRect &rect) {
  setText(text);
  if (!text.isEmpty()) {
    QSize extra(1, 1);
    resize(sizeHint() + extra);
    QPoint tip_pos(pt.x() + 8, rect.top() + 2);
    if (tip_pos.x() + size().width() >= rect.right()) {
      tip_pos.rx() = pt.x() - size().width() - 8;
    }
    if (rect.contains({tip_pos, size()})) {
      move(w->mapToGlobal(tip_pos));
      setVisible(true);
      return;
    }
  }
  setVisible(false);
}

void TipLabel::paintEvent(QPaintEvent *ev) {
  QStylePainter p(this);
  QStyleOptionFrame opt;
  opt.init(this);
  p.drawPrimitive(QStyle::PE_PanelTipLabel, opt);
  p.end();
  QLabel::paintEvent(ev);
}
