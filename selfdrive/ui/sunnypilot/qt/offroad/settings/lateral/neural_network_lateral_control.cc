/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/neural_network_lateral_control.h"

NeuralNetworkLateralControl::NeuralNetworkLateralControl() :
  ParamControl("NeuralNetworkLateralControl", tr("Neural Network Lateral Control (NNLC)"),  "", "") {
  setConfirmation(true, false);
  updateToggle();
}

void NeuralNetworkLateralControl::updateToggle() {
  QString statusInitText = "<font color='yellow'>" + STATUS_CHECK_COMPATIBILITY + "</font>";
  QString notLoadedText = "<font color='yellow'>" + STATUS_NOT_LOADED + "</font>";
  QString loadedText = "<font color=#00ff00>" + STATUS_LOADED + "</font>";

  auto cp_bytes = params.get("CarParamsPersistent");
  auto cp_sp_bytes = params.get("CarParamsSPPersistent");
  if (!cp_bytes.empty() && !cp_sp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    AlignedBuffer aligned_buf_sp;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    capnp::FlatArrayMessageReader cmsg_sp(aligned_buf_sp.align(cp_sp_bytes.data(), cp_sp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
    cereal::CarParamsSP::Reader CP_SP = cmsg_sp.getRoot<cereal::CarParamsSP>();

    if (CP.getSteerControlType() == cereal::CarParams::SteerControlType::ANGLE) {
      params.remove("NeuralNetworkLateralControl");
      setDescription(nnffDescriptionBuilder(STATUS_NOT_AVAILABLE));
      setEnabled(false);
    } else {
      QString nn_model_name = QString::fromStdString(CP_SP.getNeuralNetworkLateralControl().getModel().getName());
      QString nn_fuzzy = CP_SP.getNeuralNetworkLateralControl().getFuzzyFingerprint() ?
                         STATUS_MATCH_FUZZY : STATUS_MATCH_EXACT;

      if (nn_model_name.isEmpty()) {
        setDescription(nnffDescriptionBuilder(statusInitText));
      } else if (nn_model_name == "MOCK") {
        setDescription(nnffDescriptionBuilder(
          notLoadedText + "<br>" + buildSupportText(SUPPORT_DONATE_LOGS)
        ));
      } else {
        QString statusText = loadedText + " | " + STATUS_MATCH + " = " + nn_fuzzy + " | " + nn_model_name;
        QString explanationText = EXPLANATION_MATCH + " " + buildSupportText(SUPPORT_ISSUES);
        setDescription(nnffDescriptionBuilder(statusText + "<br><br>" + explanationText));
      }
    }
  } else {
    setDescription(nnffDescriptionBuilder(statusInitText));
  }

  if (getDescription() != getBaseDescription()) {
    showDescription();
  }
}
