#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

AbstractAlert::AbstractAlert(bool hasRebootBtn, QWidget *parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(50);
  main_layout->setSpacing(30);

  QWidget *widget = new QWidget;
  scrollable_layout = new QVBoxLayout(widget);
  widget->setStyleSheet("background-color: transparent;");
  main_layout->addWidget(new ScrollView(widget));

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton(tr("Close"));
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, &QPushButton::clicked, this, &AbstractAlert::dismiss);

  snooze_btn = new QPushButton(tr("Snooze Update"));
  snooze_btn->setVisible(false);
  snooze_btn->setFixedSize(550, 125);
  footer_layout->addWidget(snooze_btn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(snooze_btn, &QPushButton::clicked, [=]() {
    params.putBool("SnoozeUpdate", true);
  });
  QObject::connect(snooze_btn, &QPushButton::clicked, this, &AbstractAlert::dismiss);
  snooze_btn->setStyleSheet(R"(color: white; background-color: #4F4F4F;)");

  if (hasRebootBtn) {
    QPushButton *rebootBtn = new QPushButton(tr("Reboot and Update"));
    rebootBtn->setFixedSize(600, 125);
    footer_layout->addWidget(rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
    QObject::connect(rebootBtn, &QPushButton::clicked, [=]() { Hardware::reboot(); });
  }

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
  )");
}

OffroadAlert::OffroadAlert(QWidget *parent) : AbstractAlert(parent) {
  for (auto &[key, severity, text] : offroad_alerts) {
    QLabel *l = new QLabel(this);
    l->setMargin(60);
    l->setWordWrap(true);
    l->setStyleSheet(QString("background-color: %1").arg(severity == Severity::high ? "#E22C2C" : "#292929"));
    l->setVisible(false);
    scrollable_layout->addWidget(l);

    alerts[key] = {.text = text, .label = l};
  }
  scrollable_layout->addStretch(1);
}

int OffroadAlert::refresh() {
  Params params;
  int alertCount = 0;

  for (const auto &[key, a] : alerts) {
    if (params.exists(key)) {
      std::string extra = params.get(key);
      a.label->setText(extra.empty() ? a.text : a.text.arg(extra.c_str()));
      a.label->setVisible(true);
      ++alertCount;
    } else {
      a.label->setVisible(false);
    }
  }

  snooze_btn->setVisible(alerts["Offroad_ConnectivityNeeded"].label->isVisible());
  return alertCount;
}

UpdateAlert::UpdateAlert(QWidget *parent) : AbstractAlert(true, parent) {
  releaseNotes = new QLabel(this);
  releaseNotes->setWordWrap(true);
  releaseNotes->setAlignment(Qt::AlignTop);
  scrollable_layout->addWidget(releaseNotes);
}

bool UpdateAlert::refresh() {
  bool updateAvailable = params.getBool("UpdateAvailable");
  if (updateAvailable) {
    releaseNotes->setText(params.get("UpdaterNewReleaseNotes").c_str());
  }
  return updateAvailable;
}
