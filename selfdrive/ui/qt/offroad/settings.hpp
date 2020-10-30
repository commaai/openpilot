#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>
#include <QPushButton>


class ParamsToggle : public QFrame {
  Q_OBJECT

public:
  explicit ParamsToggle(QString param, QString title, QString description, QString icon, QWidget *parent = 0);

private:
  QCheckBox *checkbox;
  QString param;

public slots:
  void checkboxClicked(int state);
};


class SettingsWindow : public QWidget {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

signals:
  void closeSettings();
};


