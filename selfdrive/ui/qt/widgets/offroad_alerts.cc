#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVBoxLayout>

#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(50);
  main_layout->setSpacing(30);

  QWidget *alerts_widget = new QWidget(this);
  alerts_layout = new QVBoxLayout(alerts_widget);
  alerts_layout->setMargin(0);
  alerts_layout->setSpacing(30);
  alerts_widget->setStyleSheet("background-color: transparent;");

  // release notes
  releaseNotes.setWordWrap(true);
  releaseNotes.setVisible(false);
  releaseNotes.setAlignment(Qt::AlignTop);

  releaseNotesScroll = new ScrollView(&releaseNotes, this);
  main_layout->addWidget(releaseNotesScroll);

  alertsScroll = new ScrollView(alerts_widget, this);
  main_layout->addWidget(alertsScroll);

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, &QPushButton::released, this, &OffroadAlert::closeAlerts);

  rebootBtn.setText("Reboot and Update");
  rebootBtn.setObjectName("rebootBtn");
  rebootBtn.setFixedSize(600, 125);
  rebootBtn.setVisible(false);
  footer_layout->addWidget(&rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(&rebootBtn, &QPushButton::released, [=]() { Hardware::reboot(); });

  setStyleSheet(R"(
    * {
      font-size: 48px;
      color: white;
    }
    QFrame {
      border-radius: 30px;
      background-color: #393939;
    }
    QPushButton {
      color: black;
      font-weight: 500;
      border-radius: 30px;
      background-color: white;
    }
    #rebootBtn {
      background-color: #E22C2C;
    }
  )");
}

void OffroadAlert::refresh() {
  if (alerts.empty()) {
    // setup labels for each alert
    QString json = util::read_file("../controls/lib/alerts_offroad.json").c_str();
    QJsonObject obj = QJsonDocument::fromJson(json.toUtf8()).object();
    // descending sort labels by severity
    std::vector<std::pair<std::string, int>> sorted;
    for (auto it = obj.constBegin(); it != obj.constEnd(); ++it) {
      sorted.push_back({it.key().toStdString(), it.value()["severity"].toInt()});
    }
    std::sort(sorted.begin(), sorted.end(), [=](auto &l, auto &r) {
      return l.second > r.second;
    });

    for (auto it = sorted.begin(); it != sorted.end(); ++it) {
      QLabel *l = new QLabel(this);
      alerts[it->first] = l;
      l->setMargin(60);
      l->setWordWrap(true);
      l->setVisible(false);
      l->setStyleSheet(QString("background-color: %1").arg(it->second ? "#E22C2C" : "#292929"));
      alerts_layout->addWidget(l);
    }
    alerts_layout->addStretch(1);
  }

  updateAlerts();
}

void OffroadAlert::updateAlerts() {
  alertCount = 0;
  for (const auto& [key, label] : alerts) {
    QString text;
    std::string bytes = params.get(key);
    if (bytes.size()) {
      auto doc_par = QJsonDocument::fromJson(bytes.c_str());
      text = doc_par["text"].toString();
    }
    label->setText(text);
    label->setVisible(!text.isEmpty());
    alertCount += !text.isEmpty();
  }
  updateAvailable = params.getBool("UpdateAvailable") && alertCount < 1;
}

void OffroadAlert::setCurrentIndex(int idx) {
  // show release notes when idx = 0.
  releaseNotesScroll->setVisible(idx == 0);
  rebootBtn.setVisible(idx == 0);
  alertsScroll->setVisible(idx == 1);
}
