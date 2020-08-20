#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>

class ParamsToggle : public QFrame {
  Q_OBJECT

private:
  QCheckBox *checkbox;
  QString param;
public:
  explicit ParamsToggle(QString param, QString title, QString description, QString icon, QWidget *parent = 0);
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
