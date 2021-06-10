#include "map_settings.h"

#include "selfdrive/ui/qt/widgets/controls.h"


MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *layout = new QVBoxLayout;
  Params params = Params();

  QString dongle = QString::fromStdString(params.get("DongleId", false));
  // TODO: Add buttons for home/work shortcuts

  layout->addWidget(new ParamControl("NavSettingTime24h",
                                    "Show ETA in 24h format",
                                    "Use 24h format instead of am/pm",
                                    "",
                                    this));
  layout->addStretch();
  setLayout(layout);
}