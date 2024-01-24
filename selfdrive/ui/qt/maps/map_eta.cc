#include "selfdrive/ui/qt/maps/map_eta.h"

#include <QDateTime>
#include <QPainter>

#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/ui.h"

const float MANEUVER_TRANSITION_THRESHOLD = 10;

MapETA::MapETA(QWidget *parent) : QWidget(parent) {
  setVisible(false);
  setAttribute(Qt::WA_TranslucentBackground);
  eta_doc.setUndoRedoEnabled(false);
  eta_doc.setDefaultStyleSheet("body {font-family:Inter;font-size:70px;color:white;} b{font-weight:600;} td{padding:0 3px;}");
}

void MapETA::paintEvent(QPaintEvent *event) {
  if (!eta_doc.isEmpty()) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 255));
    QSizeF txt_size = eta_doc.size();
    p.drawRoundedRect((width() - txt_size.width()) / 2 - UI_BORDER_SIZE, 0, txt_size.width() + UI_BORDER_SIZE * 2, height() + 25, 25, 25);
    p.translate((width() - txt_size.width()) / 2, (height() - txt_size.height()) / 2);
    eta_doc.drawContents(&p);
  }
}

void MapETA::updateETA(float s, float s_typical, float d) {
  // ETA
  auto eta_t = QDateTime::currentDateTime().addSecs(s).time();
  auto eta = format_24h ? std::pair{eta_t.toString("HH:mm"), tr("eta")}
                        : std::pair{eta_t.toString("h:mm a").split(' ')[0], eta_t.toString("a")};

  // Remaining time
  auto remaining = s < 3600 ? std::pair{QString::number(int(s / 60)), tr("min")}
                            : std::pair{QString("%1:%2").arg((int)s / 3600).arg(((int)s % 3600) / 60, 2, 10, QLatin1Char('0')), tr("hr")};
  QString color = "#25DA6E";
  if (s / s_typical > 1.5)
    color = "#DA3025";
  else if (s / s_typical > 1.2)
    color = "#DAA725";

  // Distance
  auto distance = map_format_distance(d, uiState()->scene.is_metric);

  eta_doc.setHtml(QString(R"(<body><table><tr style="vertical-align:bottom;"><td><b>%1</b></td><td>%2</td>
                             <td style="padding-left:40px;color:%3;"><b>%4</b></td><td style="padding-right:40px;color:%3;">%5</td>
                             <td><b>%6</b></td><td>%7</td></tr></body>)")
                      .arg(eta.first, eta.second, color, remaining.first, remaining.second, distance.first, distance.second));

  setVisible(d >= MANEUVER_TRANSITION_THRESHOLD);
  update();
}
