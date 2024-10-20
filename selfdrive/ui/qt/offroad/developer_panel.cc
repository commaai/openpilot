#include <QDebug>

#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
  hellolBtn = new ButtonControl("Hello World", "HAI WRLD");
  addItem(hellolBtn);

  // SSH keys
  addItem(new SshToggle());
  addItem(new SshControl());
}