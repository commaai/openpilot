#pragma once

#include <QPaintEvent>
#include <QTextDocument>
#include <QWidget>

#include "common/params.h"

class MapETA : public QWidget {
  Q_OBJECT

public:
  MapETA(QWidget * parent=nullptr);
  void updateETA(float seconds, float seconds_typical, float distance);

private:
  void paintEvent(QPaintEvent *event) override;
  QSizeF adjustSize();
  void showEvent(QShowEvent *event) override { format_24h = param.getBool("NavSettingTime24h"); }

  bool format_24h = false;
  int font_size = 70;
  QTextDocument eta_doc;
  Params param;
};
