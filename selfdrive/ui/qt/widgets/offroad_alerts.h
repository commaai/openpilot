#pragma once

#include <map>

#include <QLabel>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"

class AbstractAlert : public QFrame {
  Q_OBJECT

protected:
  AbstractAlert(bool hasRebootBtn, QWidget *parent = nullptr);
  QVBoxLayout *scrollable_layout;
  Params params;

signals:
  void dismiss();
};

class UpdateAlert : public AbstractAlert {
  Q_OBJECT

public:
  UpdateAlert(QWidget *parent = 0);
  bool refresh();

private:
  QLabel *releaseNotes = nullptr;
};

class OffroadAlert : public AbstractAlert {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0) : AbstractAlert(false, parent) {}
  int refresh();

private:
  std::map<std::string, QLabel*> alerts;
};
