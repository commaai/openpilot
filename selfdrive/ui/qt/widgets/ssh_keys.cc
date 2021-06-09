#include "selfdrive/ui/qt/widgets/ssh_keys.h"

#include <QHBoxLayout>
#include <QNetworkReply>

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/input.h"

SshControl::SshControl() : AbstractControl("SSH Keys", "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.", "") {

  // setup widget
  hlayout->addStretch(1);

  username_label.setAlignment(Qt::AlignVCenter);
  username_label.setStyleSheet("color: #aaaaaa");
  hlayout->addWidget(&username_label);

  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 50px;
    font-size: 35px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  btn.setFixedSize(250, 100);
  hlayout->addWidget(&btn);

  QObject::connect(&btn, &QPushButton::released, [=]() {
    if (btn.text() == "ADD") {
      QString username = InputDialog::getText("Enter your GitHub username");
      if (username.length() > 0) {
        btn.setText("LOADING");
        btn.setEnabled(false);
        getUserKeys(username);
      }
    } else {
      qParams.remove("GithubUsername");
      qParams.remove("GithubSshKeys");
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = qParams.Get("GithubSshKeys");
  if (param.length()) {
    username_label.setText(qParams.Get("GithubUsername"));
    btn.setText("REMOVE");
  } else {
    username_label.setText("");
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(const QString &username) {
  HttpRequest *request = new HttpRequest(this, "https://github.com/" + username + ".keys", "", false);
  QObject::connect(request, &HttpRequest::receivedResponse, [=](const QString &resp) {
    if (!resp.isEmpty()) {
      qParams.put("GithubUsername", username);
      qParams.put("GithubSshKeys", resp);
    } else {
      ConfirmationDialog::alert("Username '" + username + "' has no keys on GitHub");
    }
    refresh();
    request->deleteLater();
  });
  QObject::connect(request, &HttpRequest::failedResponse, [=] {
    ConfirmationDialog::alert("Username '" + username + "' doesn't exist on GitHub");
    refresh();
    request->deleteLater();
  });
  QObject::connect(request, &HttpRequest::timeoutResponse, [=] {
    ConfirmationDialog::alert("Request timed out");
    refresh();
    request->deleteLater();
  });
}
