#include <QNetworkReply>

#include "widgets/input.hpp"
#include "widgets/ssh_keys.hpp"
#include "common/params.h"
#include "api.hpp"

SshControl::SshControl() : AbstractControl("SSH Keys", "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.", "") {
  // setup widget
  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 50px;
    font-size: 40px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  btn.setFixedSize(250, 100);
  hlayout->addWidget(&btn);

  QObject::connect(&btn, &QPushButton::released, [=]() {
    if (btn.text() == "ADD") {
      username = InputDialog::getText("Enter your GitHub username");
      if (username.length() > 0) {
        btn.setText("LOADING");
        btn.setEnabled(false);
        getUserKeys(username);
      }
    } else {
      Params().delete_db_value("GithubSshKeys");
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(Params().get("GithubSshKeys"));
  if (param.length()) {
    btn.setText("REMOVE");
  } else {
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(QString username) {
  QString url = "https://github.com/" + username + ".keys";
  TimeoutRequest::get(this, url, 5000, nullptr, [=](const QString &resp, bool err) {
    QString errmsg = "";
    if (err) {
      errmsg = resp;
    } else {
      if (resp.length()) {
        Params().write_db_value("GithubSshKeys", resp.toStdString());
      } else {
        errmsg = "Username '" + username + "' doesn't exist or has no keys on GitHub";
      }
    }
    if (errmsg.length()) {
      ConfirmationDialog::alert(errmsg);
    }
    refresh();
  });
}
