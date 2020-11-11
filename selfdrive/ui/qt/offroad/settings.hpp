#pragma once

#include <QCheckBox>
#include <QFrame>
#include <QLabel>
#include <QStackedLayout>
#include <QTimer>
#include <QWidget>


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

private:
  inline QWidget* newLinebreakWidget();
  void initGeneralSettingsWidget();
  void initDeviceSettingsWidget();

signals:
  void closeSettings();

private:
  std::map<QString, QWidget *> panels;
  QStackedLayout *panel_layout = nullptr;
  QWidget *general_settings_widget = nullptr;
  QWidget *device_settings_widget = nullptr;
  QLabel *general_settings_label, *device_settings_label,
         *network_settings_label, *developer_settings_label;

private slots:
  void setActivePanel();
  void selected(int);
};
