/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <map>

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"

class NeuralNetworkLateralControl : public ParamControl {
  Q_OBJECT

public:
  NeuralNetworkLateralControl();

public slots:
  void updateToggle();

private:
  Params params;

  // Status messages
  const QString STATUS_NOT_AVAILABLE = tr("NNLC is currently not available on this platform.");
  const QString STATUS_CHECK_COMPATIBILITY = tr("Start the car to check car compatibility");
  const QString STATUS_NOT_LOADED = tr("NNLC Not Loaded");
  const QString STATUS_LOADED = tr("NNLC Loaded");
  const QString STATUS_MATCH = tr("Match");
  const QString STATUS_MATCH_EXACT = tr("Exact");
  const QString STATUS_MATCH_FUZZY = tr("Fuzzy");

  // Explanations
  const QString EXPLANATION_MATCH = tr("Match: \"Exact\" is ideal, but \"Fuzzy\" is fine too.");
  const QString EXPLANATION_FEATURE = tr("Formerly known as <b>\"NNFF\"</b>, this replaces the lateral <b>\"torque\"</b> controller, "
                                            "with one using a neural network trained on each car's (actually, each separate EPS firmware) driving data for increased controls accuracy.");

  // Support information
  const QString SUPPORT_CHANNEL = "<font color='white'><b>#tuning-nnlc</b></font>";
  const QString SUPPORT_REACH_OUT = tr("Reach out to the sunnypilot team in the following channel at the sunnypilot Discord server");
  const QString SUPPORT_FEEDBACK = tr("with feedback, or to provide log data for your car if your car is currently unsupported:");
  const QString SUPPORT_ISSUES = tr("if there are any issues:");
  const QString SUPPORT_DONATE_LOGS = tr("and donate logs to get NNLC loaded for your car:");

  // Description builders
  QString buildSupportText(const QString& context) const {
    return SUPPORT_REACH_OUT + " " + context + " " + SUPPORT_CHANNEL;
  }

  QString nnffDescriptionBuilder(const QString &custom_description) const {
    return "<b>" + custom_description + "</b><br><br>" + getBaseDescription();
  }

  QString getBaseDescription() const {
    return EXPLANATION_FEATURE + "<br><br>" + buildSupportText(SUPPORT_FEEDBACK);
  }
};
