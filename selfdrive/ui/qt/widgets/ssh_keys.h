#pragma once

#include <QTimer>
#include <QPushButton>
#include <QNetworkAccessManager>

#include "widgets/controls.h"
#include "selfdrive/hardware/hw.h"

// SSH enable toggle
class SshToggle : public ToggleControl {
  Q_OBJECT

public:
  SshToggle() : ToggleControl("Enable SSH", "", "", Hardware::get_ssh_enabled()) {
    QObject::connect(this, &SshToggle::toggleFlipped, [=](bool state) {
      Hardware::set_ssh_enabled(state);
    });
  }
};

// SSH key management widget
class SshControl : public AbstractControl {
  Q_OBJECT

public:
  SshControl();

private:
  Params params;

  QPushButton btn;
  QString username;
  QLabel username_label;

  // networking
  QTimer* networkTimer;
  QNetworkReply* reply;
  QNetworkAccessManager* manager;

  void refresh();
  void getUserKeys(QString username);

signals:
  void failedResponse(QString errorString);

private slots:
  void timeout();
  void parseResponse();
};
