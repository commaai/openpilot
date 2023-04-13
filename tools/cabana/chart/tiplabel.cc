#include "tools/cabana/chart/tiplabel.h"

#include <QApplication>
#include <QStylePainter>
#include <QToolTip>

#include "tools/cabana/settings.h"

TipLabel::TipLabel(QWidget *parent) : QLabel(parent, Qt::ToolTip | Qt::FramelessWindowHint) {
  setForegroundRole(QPalette::ToolTipText);
  setBackgroundRole(QPalette::ToolTipBase);
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

void TipLabel::showText(const QPoint &pt, const QString &text, int right_edge) {
  setText(text);
  if (!text.isEmpty()) {
    QSize extra(1, 1);
    resize(sizeHint() + extra);
    QPoint tip_pos(pt.x() + 12, pt.y());
    if (tip_pos.x() + size().width() >= right_edge) {
      tip_pos.rx() = pt.x() - size().width() - 12;
    }
    move(tip_pos);
  }
  setVisible(!text.isEmpty());
}

void TipLabel::paintEvent(QPaintEvent *ev) {
  QStylePainter p(this);
  QStyleOptionFrame opt;
  opt.init(this);
  p.drawPrimitive(QStyle::PE_PanelTipLabel, opt);
  p.end();
  QLabel::paintEvent(ev);
}
