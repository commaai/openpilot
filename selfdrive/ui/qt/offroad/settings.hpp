#pragma once

#include <QCheckBox>
#include <QFrame>
#include <QLabel>
#include <QStackedLayout>
#include <QTimer>
#include <QWidget>

#define MAX_PANELS 4

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


class ClickableLabel;

class SettingsWindow : public QWidget {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

private:
  inline QWidget* newLinebreakWidget();
  inline QWidget* newSettingsPanelBaseWidget();

  void initGeneralSettingsWidget();
  void initDeviceSettingsWidget();
  void initNetworkSettingsWidget();
  void initDeveloperSettingsWidget();

signals:
  void closeSettings();

private:
  ClickableLabel* panels[MAX_PANELS];
  std::map<QString, QWidget *> _panels;
  QStackedLayout *panel_layout = nullptr;
  QWidget *general_settings_widget, *device_settings_widget,
          *network_settings_widget, *developer_settings_widget;
  ClickableLabel *general_settings_label, *device_settings_label,
         *network_settings_label, *developer_settings_label;

private slots:
  void setActivePanel();
  void selected(int);
};
