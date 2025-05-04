/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/software_panel.h"

#include <algorithm>
#include <QJsonDocument>

#include "common/model.h"

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
  using DS = cereal::ModelManagerSP::DownloadStatus;
  if (!model_manager.hasSelectedBundle() && !model_manager.hasActiveBundle()) {
    currentModelLblBtn->setDescription(tr("No custom model selected!"));
    return;
  }

  const bool showSelectedBundle = model_manager.hasSelectedBundle() && (isDownloading() || model_manager.getSelectedBundle().getStatus() == DS::FAILED);
  const auto &bundle = showSelectedBundle ? model_manager.getSelectedBundle() : model_manager.getActiveBundle();
  const auto &models = bundle.getModels();
  download_status = bundle.getStatus();
  const auto download_status_changed = prev_download_status != download_status;
  QStringList status;

  // Get status for each model type in order
  for (const auto &model: models) {
    QString typeName;
    QString modelName = QString::fromStdString(bundle.getDisplayName());

    switch (model.getType()) {
      case cereal::ModelManagerSP::Model::Type::SUPERCOMBO:
        typeName = tr("Driving");
        break;
      case cereal::ModelManagerSP::Model::Type::NAVIGATION:
        typeName = tr("Navigation");
        break;
      case cereal::ModelManagerSP::Model::Type::VISION:
        typeName = tr("Vision");
        break;
      case cereal::ModelManagerSP::Model::Type::POLICY:
        typeName = tr("Policy");
        break;
    }

    const auto &progress = model.getArtifact().getDownloadProgress();
    QString line;

    if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::DOWNLOADING) {
      line = tr("Downloading %1 model [%2]... (%3%)").arg(typeName, modelName).arg(progress.getProgress(), 0, 'f', 2);
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::DOWNLOADED) {
      line = tr("%1 model [%2] %3").arg(typeName, modelName, download_status_changed ? tr("downloaded") : tr("ready"));
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::CACHED) {
      line = tr("%1 model [%2] %3").arg(typeName, modelName, download_status_changed ? tr("from cache") : tr("ready"));
    } else if (progress.getStatus() == cereal::ModelManagerSP::DownloadStatus::FAILED) {
      line = tr("%1 model [%2] download failed").arg(typeName, modelName);
    } else {
      line = tr("%1 model [%2] pending...").arg(typeName, modelName);
    }
    status.append(line);
  }

  currentModelLblBtn->setDescription(status.join("\n"));

  if (prev_download_status != download_status) {
    switch (bundle.getStatus()) {
      case cereal::ModelManagerSP::DownloadStatus::DOWNLOADING:
      case cereal::ModelManagerSP::DownloadStatus::CACHED:
      case cereal::ModelManagerSP::DownloadStatus::DOWNLOADED:
        currentModelLblBtn->showDescription();
        break;
      case cereal::ModelManagerSP::DownloadStatus::FAILED:
      default:
        break;
    }
  }
  prev_download_status = download_status;
}

/**
 * @brief Gets the name of the currently selected model bundle
 * @return Display name of the selected bundle or default model name
 */
QString SoftwarePanelSP::GetActiveModelName() {
  if (model_manager.hasActiveBundle()) {
    return QString::fromStdString(model_manager.getActiveBundle().getDisplayName());
  }

  return DEFAULT_MODEL;
}

void SoftwarePanelSP::updateModelManagerState() {
  const SubMaster &sm = *(uiStateSP()->sm);
  model_manager = sm["modelManagerSP"].getModelManagerSP();
}

/**
 * @brief Handles the model bundle selection button click
 * Displays available bundles, allows selection, and initiates download
 */
void SoftwarePanelSP::handleCurrentModelLblBtnClicked() {
  currentModelLblBtn->setEnabled(false);
  currentModelLblBtn->setValue(tr("Fetching models..."));

  // Create mapping of bundle indices to display names
  QMap<uint32_t, QString> index_to_bundle;
  const auto bundles = model_manager.getAvailableBundles();
  for (const auto &bundle: bundles) {
    index_to_bundle.insert(bundle.getIndex(), QString::fromStdString(bundle.getDisplayName()));
  }

  // Sort bundles by index in descending order
  QStringList bundleNames;
  // Add "Default" as the first option
  bundleNames.append(tr("Use Default"));

  auto indices = index_to_bundle.keys();
  std::sort(indices.begin(), indices.end(), std::greater<uint32_t>());
  for (const auto &index: indices) {
    bundleNames.append(index_to_bundle[index]);
  }

  currentModelLblBtn->setValue(GetActiveModelName());

  const QString selectedBundleName = MultiOptionDialog::getSelection(
    tr("Select a Model"), bundleNames, GetActiveModelName(), this);

  if (selectedBundleName.isEmpty() || !canContinueOnMeteredDialog()) {
    return;
  }

  // Handle "Stock" selection differently
  if (selectedBundleName == tr("Use Default")) {
    params.remove("ModelManager_ActiveBundle");
    currentModelLblBtn->setValue(tr("Default"));
    showResetParamsDialog();
  } else {
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

  updateModelManagerState();
  handleBundleDownloadProgress();
  currentModelLblBtn->setEnabled(!is_onroad && !isDownloading());
  currentModelLblBtn->setValue(GetActiveModelName());

  SoftwarePanel::updateLabels();
}

/**
 * @brief Shows dialog prompting user to reset calibration after model download
 */
void SoftwarePanelSP::showResetParamsDialog() {
  const auto confirmMsg = QString("%1<br><br><b>%2</b><br><br><b>%3</b>")
                          .arg(tr("Model download has started in the background."))
                          .arg(tr("We STRONGLY suggest you to reset calibration."))
                          .arg(tr("Would you like to do that now?"));
  const auto button_text = tr("Reset Calibration");

  QString content("<body><h2 style=\"text-align: center;\">" + tr("Driving Model Selector") + "</h2><br>"
                  "<p style=\"text-align: center; margin: 0 128px; font-size: 50px;\">" + confirmMsg + "</p></body>");

  if (showConfirmationDialog(content, button_text, false)) {
    params.remove("CalibrationParams");
    params.remove("LiveTorqueParameters");
  }
}
