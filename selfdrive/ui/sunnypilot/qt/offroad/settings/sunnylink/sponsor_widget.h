/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QrCode.hpp>
#include <QtCore/qjsonobject.h>

#include "common/util.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

const QString SUNNYLINK_BASE_URL = util::getenv("SUNNYLINK_API_HOST", "https://stg.api.sunnypilot.ai").c_str();

class SunnylinkSponsorQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit SunnylinkSponsorQRWidget(bool sponsor_pair = false, QWidget* parent = 0);
  void paintEvent(QPaintEvent*) override;

private:
  QPixmap img;
  QTimer *timer;
  void updateQrCode(const QString &text);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

  bool sponsor_pair = false;

private slots:
  void refresh();
};

// sponsor popup widget
class SunnylinkSponsorPopup : public DialogBase {
  Q_OBJECT

public:
  explicit SunnylinkSponsorPopup(bool sponsor_pair = false, QWidget* parent = 0);

private:
  static QStringList getInstructions(bool sponsor_pair);
  bool sponsor_pair = false;
};
