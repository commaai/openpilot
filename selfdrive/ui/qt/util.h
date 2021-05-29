#pragma once

#include <QtWidgets>

inline void configFont(QPainter &p, QString family, int size, const QString &style) {
  QFont f(family);
  f.setPixelSize(size);
  f.setStyleName(style);
  p.setFont(f);
}

inline void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

inline QString timeAgo(const QDateTime &date) {
  int diff = date.secsTo(QDateTime::currentDateTime());

  QString s;
  if (diff < 60) {
    s = "now";
  } else if (diff < 60 * 60) {
    int minutes = diff / 60;
    s = QString("%1 minute%2 ago").arg(minutes).arg(minutes > 1 ? "s" : "");
  } else if (diff < 60 * 60 * 24) {
    int hours = diff / (60 * 60);
    s = QString("%1 hour%2 ago").arg(hours).arg(hours > 1 ? "s" : "");
  } else if (diff < 3600 * 24 * 7) {
    int days = diff / (60 * 60 * 24);
    s = QString("%1 day%2 ago").arg(days).arg(days > 1 ? "s" : "");
  } else {
    s = date.date().toString();
  }

  return s;
}
