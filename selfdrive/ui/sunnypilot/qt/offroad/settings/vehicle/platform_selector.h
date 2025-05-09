/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"

static const QString GREEN_PLATFORM = "#00F100";
static const QString BLUE_PLATFORM = "#0086E9";
static const QString YELLOW_PLATFORM = "#FFD500";

enum class FingerprintStatus {
  AUTO_FINGERPRINT,
  MANUAL_FINGERPRINT,
  UNRECOGNIZED,
};

class PlatformSelector : public ButtonControl {
  Q_OBJECT

public:
  PlatformSelector();
  QVariant getPlatformBundle(const QString &key);

  QString platform;
  QString brand;

public slots:
  void refresh(bool _offroad);

signals:
  void refreshPanel();

private:
  void searchPlatforms(const QString &query);
  void setPlatform(const QString &platform = "");
  QMap<QString, QVariantMap> platforms;

  Params params;
  bool offroad;

  QString unrecognized_str = tr("Unrecognized Vehicle");

  static QString platformDescription(FingerprintStatus status = FingerprintStatus::UNRECOGNIZED) {
    QString auto_str = "🟢 - " + tr("Fingerprinted automatically");
    QString manual_str = "🔵 - " + tr("Manually selected");
    QString unrecognized_str = "🟡 - " + tr("Not fingerprinted or manually selected");

    if (status == FingerprintStatus::AUTO_FINGERPRINT) {
      auto_str = "<font color='white'><b>" + auto_str + "</b></font>";
    } else if (status == FingerprintStatus::MANUAL_FINGERPRINT) {
      manual_str = "<font color='white'><b>" + manual_str + "</b></font>";
    } else {
      unrecognized_str = "<font color='white'><b>" + unrecognized_str + "</b></font>";
    }

    return QString("%1<br>%2<br><br>%3<br>%4<br>%5")
             .arg(tr("Select vehicle to force fingerprint manually."))
             .arg(tr("Colors represent fingerprint status:"))
             .arg(auto_str)
             .arg(manual_str)
             .arg(unrecognized_str);
  }
};
