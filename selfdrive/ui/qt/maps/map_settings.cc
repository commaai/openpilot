#include "map_settings.h"

#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/common/util.h"

MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  Params params = Params();

  QHBoxLayout *home_layout = new QHBoxLayout;
  home_icon = new QLabel;
  home_icon->setPixmap(QPixmap("../assets/navigation/home_inactive.png"));
  home_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  home_layout->addWidget(home_icon);

  home_address = new QLabel("No home\nlocation set");
  home_address->setWordWrap(true);
  home_address->setStyleSheet(R"(font-size: 30px; color: grey;)");
  home_layout->addSpacing(20);
  home_layout->addWidget(home_address);

  QHBoxLayout *work_layout = new QHBoxLayout;
  work_icon = new QLabel;
  work_icon->setPixmap(QPixmap("../assets/navigation/work_inactive.png"));
  work_icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  work_layout->addWidget(work_icon);

  work_address = new QLabel("No work\nlocation set");
  work_address->setWordWrap(true);
  work_address->setStyleSheet(R"(font-size: 30px; color: grey;)");
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

  main_layout->addWidget(new ParamControl("NavSettingTime24h",
                                    "Show ETA in 24h format",
                                    "Use 24h format instead of am/pm",
                                    "",
                                    this));
  main_layout->addStretch();
  parseResponse(QString::fromStdString(util::read_file("/home/batman/example_response.txt")));
}

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

void MapPanel::parseResponse(const QString& response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on navigation locations";
    return;
  }

  for (auto location : doc.array()) {
    auto obj = location.toObject();
    qDebug() << obj;

    auto type = obj["save_type"].toString();
    auto label = obj["label"].toString();
    auto name = obj["place_name"].toString();
    auto details = obj["place_details"].toString();

    if (type == "favorite") {

      if (label == "home") {
        home_address->setText(shorten(name, 15) + "\n" + shorten(details, 50));
        home_address->setStyleSheet(R"(font-size: 30px; color: white;)");
        home_icon->setPixmap(QPixmap("../assets/navigation/home.png"));
      } else if (label == "work") {
        work_address->setText(shorten(name, 15) + "\n" + shorten(details, 50));
        work_address->setStyleSheet(R"(font-size: 30px; color: white;)");
        work_icon->setPixmap(QPixmap("../assets/navigation/work.png"));
      }
    }
  }
}
