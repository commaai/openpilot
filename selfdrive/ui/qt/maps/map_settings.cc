#include "map_settings.h"

#include "selfdrive/ui/qt/widgets/controls.h"


MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  Params params = Params();

  QString dongle = QString::fromStdString(params.get("DongleId", false));
  // TODO: Add buttons for home/work shortcuts

  main_layout->addWidget(new ParamControl("NavSettingTime24h",
                                    "Show ETA in 24h format",
                                    "Use 24h format instead of am/pm",
                                    "",
                                    this));
  main_layout->addStretch();
}
