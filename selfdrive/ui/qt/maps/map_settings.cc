#include "map_settings.h"

#include "selfdrive/ui/qt/widgets/controls.h"


MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  Params params = Params();

  QPixmap home_pix("../assets/navigation/home_inactive.png");
  QPixmap work_pix("../assets/navigation/work.png");

  QHBoxLayout *home_layout = new QHBoxLayout;
  QLabel *home_icon = new QLabel;
  home_icon->setPixmap(home_pix);
  home_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  home_layout->addWidget(home_icon);

  QLabel *home_address = new QLabel("No home\nlocation set");
  home_address->setWordWrap(true);
  home_address->setStyleSheet(R"(font-size: 30px; color: grey;)");
  home_layout->addSpacing(20);
  home_layout->addWidget(home_address);

  QHBoxLayout *work_layout = new QHBoxLayout;
  QLabel *work_icon = new QLabel;
  work_icon->setPixmap(work_pix);
  work_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  work_layout->addWidget(work_icon);

  QLabel *work_address = new QLabel("41 Santa Monica Way, San Francisco, CA");
  work_address->setWordWrap(true);
  work_address->setStyleSheet(R"(font-size: 30px;)");
  work_layout->addSpacing(20);
  work_layout->addWidget(work_address);

  QHBoxLayout *home_work_layout = new QHBoxLayout;
  home_work_layout->addLayout(home_layout, 1);
  home_work_layout->addSpacing(50);
  home_work_layout->addLayout(work_layout, 1);

  main_layout->addLayout(home_work_layout);
  main_layout->addSpacing(50);
  main_layout->addWidget(horizontal_line());

  QString dongle = QString::fromStdString(params.get("DongleId", false));
  // TODO: Add buttons for home/work shortcuts

  main_layout->addWidget(new ParamControl("NavSettingTime24h",
                                    "Show ETA in 24h format",
                                    "Use 24h format instead of am/pm",
                                    "",
                                    this));
  main_layout->addStretch();
}
