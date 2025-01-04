/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/software_panel.h"

#include <algorithm>
#include <QJsonDocument>

/**
 * @brief Constructs the software panel with model bundle selection functionality
 * @param parent Parent widget
 */
SoftwarePanelSP::SoftwarePanelSP(QWidget *parent) : SoftwarePanel(parent) {
  const auto current_model = GetActiveModelName();
  currentModelLblBtn = new ButtonControlSP(tr("Current Model"), tr("SELECT"), current_model);
  currentModelLblBtn->setValue(current_model);

  connect(currentModelLblBtn, &ButtonControlSP::clicked, this, &SoftwarePanelSP::handleCurrentModelLblBtnClicked);
  QObject::connect(uiStateSP(), &UIStateSP::uiUpdate, this, &SoftwarePanelSP::updateLabels);
  AddWidgetAt(0, currentModelLblBtn);
}

/**
 * @brief Updates the UI with bundle download progress information
 * Reads status from modelManagerSP cereal message and displays status for all models
 */
void SoftwarePanelSP::handleBundleDownloadProgress() {
  const SubMaster &sm = *(uiStateSP()->sm);
  const auto model_manager = sm["modelManagerSP"].getModelManagerSP();

  if (!model_manager.hasSelectedBundle()) {
    currentModelLblBtn->setDescription("");
    return;
  }

  const auto &bundle = model_manager.getSelectedBundle();
  const auto &models = bundle.getModels();
  QStringList status;

  // Get status for each model type in order
  for (const auto &model: models) {
    QString typeName;
    QString modelName;

    switch (model.getType()) {
      case cereal::ModelManagerSP::Type::DRIVE:
        typeName = tr("Driving");
        modelName = QString::fromStdString(bundle.getDisplayName());
        break;
      case cereal::ModelManagerSP::Type::NAVIGATION:
        typeName = tr("Navigation");
        modelName = QString::fromStdString(model.getFullName());
        break;
      case cereal::ModelManagerSP::Type::METADATA:
        typeName = tr("Metadata");
        modelName = QString::fromStdString(model.getFullName());
        break;
    }

    const auto &progress = model.getDownloadProgress();
    QString line;

    if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::DOWNLOADING) {
      line = tr("Downloading %1 model [%2]... (%3%)").arg(typeName, modelName).arg(progress.getProgress(), 0, 'f', 2);
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::DOWNLOADED) {
      line = tr("%1 model [%2] downloaded").arg(typeName, modelName);
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::CACHED) {
      line = tr("%1 model [%2] from cache").arg(typeName, modelName);
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::FAILED) {
      line = tr("%1 model [%2] download failed").arg(typeName, modelName);
    } else {
      line = tr("%1 model [%2] pending...").arg(typeName, modelName);
    }
    status.append(line);
  }

  currentModelLblBtn->setDescription(status.join("\n"));

  if (bundle.getStatus() == cereal::ModelManagerSP::DownloadStatus::DOWNLOADING) {
    currentModelLblBtn->showDescription();
  }

  currentModelLblBtn->setEnabled(!is_onroad && !isDownloading());
}

/**
 * @brief Gets the name of the currently selected model bundle
 * @return Display name of the selected bundle or default model name
 */
QString SoftwarePanelSP::GetActiveModelName() {
  const SubMaster &sm = *(uiStateSP()->sm);
  const auto model_manager = sm["modelManagerSP"].getModelManagerSP();

  if (model_manager.hasActiveBundle()) {
    return QString::fromStdString(model_manager.getActiveBundle().getDisplayName());
  }

  return "";
}

/**
 * @brief Handles the model bundle selection button click
 * Displays available bundles, allows selection, and initiates download
 */
void SoftwarePanelSP::handleCurrentModelLblBtnClicked() {
  currentModelLblBtn->setEnabled(false);
  currentModelLblBtn->setValue(tr("Fetching models..."));

  const SubMaster &sm = *(uiStateSP()->sm);
  const auto model_manager = sm["modelManagerSP"].getModelManagerSP();

  // Create mapping of bundle indices to display names
  QMap<uint32_t, QString> index_to_bundle;
  const auto bundles = model_manager.getAvailableBundles();
  for (const auto &bundle: bundles) {
    index_to_bundle.insert(bundle.getIndex(), QString::fromStdString(bundle.getDisplayName()));
  }

  // Sort bundles by index in descending order
  QStringList bundleNames;
  auto indices = index_to_bundle.keys();
  std::sort(indices.begin(), indices.end(), std::greater<uint32_t>());
  for (const auto &index: indices) {
    bundleNames.append(index_to_bundle[index]);
  }

  currentModelLblBtn->setEnabled(!is_onroad);
  currentModelLblBtn->setValue(GetActiveModelName());

  const QString selectedBundleName = MultiOptionDialog::getSelection(
    tr("Select a Model"), bundleNames, GetActiveModelName(), this);

  if (selectedBundleName.isEmpty() || !canContinueOnMeteredDialog()) {
    return;
  }

  // Find selected bundle and initiate download
  for (const auto &bundle: bundles) {
    if (QString::fromStdString(bundle.getDisplayName()) == selectedBundleName) {
      params.put("ModelManager_DownloadIndex", std::to_string(bundle.getIndex()));
      if (bundle.getGeneration() != model_manager.getActiveBundle().getGeneration()) {
        showResetParamsDialog();
      }
      break;
    }
  }

  updateLabels();
}

/**
 * @brief Updates the UI elements based on current state
 */
void SoftwarePanelSP::updateLabels() {
  if (!isVisible()) {
    return;
  }

  handleBundleDownloadProgress();
  currentModelLblBtn->setValue(GetActiveModelName());
  SoftwarePanel::updateLabels();
}

/**
 * @brief Shows dialog prompting user to reset calibration after model download
 */
void SoftwarePanelSP::showResetParamsDialog() {
  const auto confirmMsg = tr("Model download has started in the background.") + "\n" +
                          tr("We STRONGLY suggest you to reset calibration. Would you like to do that now?");
  const auto button_text = tr("Reset Calibration");

  if (showConfirmationDialog(confirmMsg, button_text, false)) {
    params.remove("CalibrationParams");
    params.remove("LiveTorqueParameters");
  }
}
