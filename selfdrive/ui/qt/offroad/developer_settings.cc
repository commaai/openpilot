#include "selfdrive/ui/qt/offroad/settings.h"

#include <cassert>
#include <cmath>
#include <string>

#include <QDebug>
#include <QLabel>

#include "common/params.h"
#include "common/util.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "system/hardware/hw.h"

DeveloperPanel::DeveloperPanel(QWidget* parent) : ListWidget(parent) {
  addItem(new SshToggle());

  // if (!params.getBool("IsTestedBranch")) {
  //   addItme(joystickModeToggle);
  //   addItme(longitudinalReportButton);
  // }

  // fs_watch = new ParamWatcher(this);
  // QObject::connect(fs_watch, &ParamWatcher::paramChanged, [=](const QString &param_name, const QString &param_value) {
  //   updateLabels();
  // });

  // connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
  //   is_onroad = !offroad;
  //   updateLabels();
  // });

  // updateLabels();
}

// Add this function implementation
void DeveloperPanel::showEvent(QShowEvent* event) {
  ListWidget::showEvent(event);
  // Add any custom behavior for when the panel is shown
}
