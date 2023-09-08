#include "selfdrive/ui/qt/widgets/ssh_keys.h"

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/widgets/input.h"

SshControl::SshControl() :
  ButtonControl(tr("SSH Keys"), "", tr("Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username "
                                       "other than your own. A comma employee will NEVER ask you to add their GitHub username.")) {

  QObject::connect(this, &ButtonControl::clicked, [=]() {
    if (text() == tr("ADD")) {
      QString username = InputDialog::getText(tr("Enter your GitHub username"), this);
      if (username.length() > 0) {
        setText(tr("LOADING"));
        setEnabled(false);
        getUserKeys(username);
      }
    } else {
      UIState::params.remove("GithubUsername");
      UIState::params.remove("GithubSshKeys");
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(UIState::params.get("GithubSshKeys"));
  if (param.length()) {
    setValue(QString::fromStdString(UIState::params.get("GithubUsername")));
    setText(tr("REMOVE"));
  } else {
    setValue("");
    setText(tr("ADD"));
  }
  setEnabled(true);
}

void SshControl::getUserKeys(const QString &username) {
  HttpRequest *request = new HttpRequest(this, false);
  QObject::connect(request, &HttpRequest::requestDone, [=](const QString &resp, bool success) {
    if (success) {
      if (!resp.isEmpty()) {
        UIState::params.put("GithubUsername", username.toStdString());
        UIState::params.put("GithubSshKeys", resp.toStdString());
      } else {
        ConfirmationDialog::alert(tr("Username '%1' has no keys on GitHub").arg(username), this);
      }
    } else {
      if (request->timeout()) {
        ConfirmationDialog::alert(tr("Request timed out"), this);
      } else {
        ConfirmationDialog::alert(tr("Username '%1' doesn't exist on GitHub").arg(username), this);
      }
    }

    refresh();
    request->deleteLater();
  });

  request->sendRequest("https://github.com/" + username + ".keys");
}
