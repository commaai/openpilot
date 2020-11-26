#pragma once

#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>
#include <QStackedLayout>


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

private:
  std::map<QString, QWidget *> panels;
  QStackedLayout *panel_layout;

private slots:
  void setActivePanel();
};
