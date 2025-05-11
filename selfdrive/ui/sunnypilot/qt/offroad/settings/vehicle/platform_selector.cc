/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/platform_selector.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>

#include "selfdrive/ui/sunnypilot/qt/util.h"

QVariant PlatformSelector::getPlatformBundle(const QString &key) {
  QString platform_bundle = QString::fromStdString(params.get("CarPlatformBundle"));
  if (!platform_bundle.isEmpty()) {
    QJsonDocument json = QJsonDocument::fromJson(platform_bundle.toUtf8());
    if (!json.isNull() && json.isObject()) {
      return json.object().value(key).toVariant();
    }
  }
  return {};
}

PlatformSelector::PlatformSelector() : ButtonControl(tr("Vehicle"), "", "") {
  platforms = loadPlatformList();

  QObject::connect(this, &ButtonControl::clicked, [=]() {
    if (text() == tr("SEARCH")) {
      QString query = InputDialog::getText(tr("Search your vehicle"), this, tr("Enter model year (e.g., 2021) and model name (Toyota Corolla):"), false);
      if (query.length() > 0) {
        setText(tr("SEARCHING"));
        setEnabled(false);
        searchPlatforms(query);
        refresh(offroad);
      }
    } else {
      params.remove("CarPlatformBundle");
      refresh(offroad);
    }
  });

  main_layout->addStretch(0);
  refresh(offroad);
}

void PlatformSelector::refresh(bool _offroad) {
  QString name = getPlatformBundle("name").toString();
  platform = unrecognized_str;
  QString platform_color = YELLOW_PLATFORM;

  if (!name.isEmpty()) {
    platform = name;
    platform_color = BLUE_PLATFORM;
    brand = getPlatformBundle("brand").toString();
    setText(tr("REMOVE"));
  } else {
    setText(tr("SEARCH"));

    platform = unrecognized_str;
    brand = "";
    auto cp_bytes = params.get("CarParamsPersistent");
    if (!cp_bytes.empty()) {
      AlignedBuffer aligned_buf;
      capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
      cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

      platform = QString::fromStdString(CP.getCarFingerprint().cStr());

      for (auto it = platforms.constBegin(); it != platforms.constEnd(); ++it) {
        if (it.value()["platform"].toString() == platform) {
          brand = it.value()["brand"].toString();
          break;
        }
      }

      if (platform == "MOCK") {
        platform = unrecognized_str;
      } else {
        platform_color = GREEN_PLATFORM;
      }
    }
  }
  setValue(platform, platform_color);
  setEnabled(true);
  emit refreshPanel();

  offroad = _offroad;

  FingerprintStatus cur_status;
  if (platform_color == GREEN_PLATFORM) {
    cur_status = FingerprintStatus::AUTO_FINGERPRINT;
  } else if (platform_color == BLUE_PLATFORM) {
    cur_status = FingerprintStatus::MANUAL_FINGERPRINT;
  } else {
    cur_status = FingerprintStatus::UNRECOGNIZED;
  }

  setDescription(platformDescription(cur_status));
  showDescription();
}

void PlatformSelector::setPlatform(const QString &_platform) {
  QVariantMap platform_data = platforms[_platform];

  const QString offroad_msg = offroad ? tr("This setting will take effect immediately.") :
                                        tr("This setting will take effect once the device enters offroad state.");
  const QString msg = QString("<b>%1</b><br><br>%2")
                      .arg(_platform, offroad_msg);

  QString content("<body><h2 style=\"text-align: center;\">" + tr("Vehicle Selector") + "</h2><br>"
                  "<p style=\"text-align: center; margin: 0 128px; font-size: 50px;\">" + msg + "</p></body>");

  if (ConfirmationDialog(content, tr("Confirm"), tr("Cancel"), true, this).exec()) {
    QJsonObject json_bundle;
    json_bundle["platform"] = platform_data["platform"].toString();
    json_bundle["name"] = _platform;
    json_bundle["make"] = platform_data["make"].toString();
    json_bundle["brand"] = platform_data["brand"].toString();
    json_bundle["model"] = platform_data["model"].toString();
    json_bundle["package"] = platform_data["package"].toString();

    QVariantList yearList = platform_data["year"].toList();
    QJsonArray yearArray;
    for (const QVariant &year : yearList) {
      yearArray.append(year.toString());
    }
    json_bundle["year"] = yearArray;

    QString json_bundle_str = QString::fromUtf8(QJsonDocument(json_bundle).toJson(QJsonDocument::Compact));

    params.put("CarPlatformBundle", json_bundle_str.toStdString());
  }
}

void PlatformSelector::searchPlatforms(const QString &query) {
  if (query.isEmpty()) {
    return;
  }

  QSet<QString> matched_cars;

  QString normalized_query = query.simplified().toLower();
  QStringList tokens = normalized_query.split(" ", QString::SkipEmptyParts);

  int search_year = -1;
  QStringList search_terms;

  for (const QString &token : tokens) {
    bool ok;
    int year = token.toInt(&ok);
    if (ok && year >= 1900 && year <= 2100) {
      search_year = year;
    } else {
      search_terms << token;
    }
  }

  for (auto it = platforms.constBegin(); it != platforms.constEnd(); ++it) {
    QString platform_name = it.key();
    QVariantMap platform_data = it.value();

    if (search_year != -1) {
      QVariantList year_list = platform_data["year"].toList();
      bool year_match = false;
      for (const QVariant &year_var : year_list) {
        int year = year_var.toString().toInt();
        if (year == search_year) {
          year_match = true;
          break;
        }
      }
      if (!year_match) continue;
    }

    QString normalized_make = platform_data["make"].toString().normalized(QString::NormalizationForm_KD).toLower();
    QString normalized_model = platform_data["model"].toString().normalized(QString::NormalizationForm_KD).toLower();
    normalized_make.remove(QRegExp("[^a-zA-Z0-9\\s]"));
    normalized_model.remove(QRegExp("[^a-zA-Z0-9\\s]"));

    bool all_terms_match = true;
    for (const QString &term : search_terms) {
      QString normalized_term = term.normalized(QString::NormalizationForm_KD).toLower();
      normalized_term.remove(QRegExp("[^a-zA-Z0-9\\s]"));

      bool term_matched = false;

      if (normalized_make.contains(normalized_term, Qt::CaseInsensitive)) {
        term_matched = true;
      }

      if (!term_matched) {
        if (term.contains(QRegExp("[a-z]\\d|\\d[a-z]", Qt::CaseInsensitive))) {
          QString clean_model = normalized_model;
          QString clean_term = normalized_term;
          clean_model.remove(" ");
          clean_term.remove(" ");
          if (clean_model.contains(clean_term, Qt::CaseInsensitive)) {
            term_matched = true;
          }
        } else {
          if (normalized_model.contains(normalized_term, Qt::CaseInsensitive)) {
            term_matched = true;
          }
        }
      }

      if (!term_matched) {
        all_terms_match = false;
        break;
      }
    }

    if (all_terms_match) {
      matched_cars.insert(platform_name);
    }
  }

  QStringList results = matched_cars.toList();
  results.sort();

  if (results.isEmpty()) {
    ConfirmationDialog::alert(tr("No vehicles found for query: %1").arg(query), this);
    return;
  }

  QString selected_platform = MultiOptionDialog::getSelection(tr("Select a vehicle"), results, "", this);

  if (!selected_platform.isEmpty()) {
    setPlatform(selected_platform);
  }
}
