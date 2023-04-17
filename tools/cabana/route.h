#pragma once

#include <QButtonGroup>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>

#include "tools/cabana/streams/abstractstream.h"

class AbstractOpenWidget : public QWidget {
  Q_OBJECT
public:
  AbstractOpenWidget(AbstractStream **stream, QWidget *parent) : stream(stream), QWidget(parent) {}
  virtual bool open() = 0;

protected:
  AbstractStream **stream = nullptr;
};

class OpenRouteWidget : public AbstractOpenWidget {
  Q_OBJECT

public:
  OpenRouteWidget(AbstractStream **stream, QWidget *parent);
  bool open() override;
  inline bool failedToLoad() const { return failed_to_load; }

private:
  QLineEdit *route_edit;
  QComboBox *choose_video_cb;
  bool failed_to_load = false;
};

class OpenPandaWidget : public AbstractOpenWidget {
  Q_OBJECT

public:
  OpenPandaWidget(AbstractStream **stream, QWidget *parent);
  bool open() override;

private:
  QLineEdit *serial_edit;
};

class OpenDeviceWidget : public AbstractOpenWidget {
  Q_OBJECT

public:
  OpenDeviceWidget(AbstractStream **stream, QWidget *parent);
  bool open() override;

private:
  QLineEdit *ip_address;
  QButtonGroup *group;
};

class StreamDialog : public QDialog {
  Q_OBJECT

public:
  StreamDialog(AbstractStream **stream, QWidget *parent);

 private:
  OpenRouteWidget *route_widget;
  OpenPandaWidget *panda_widget;
  OpenDeviceWidget *device_widget;
  QDialogButtonBox *btn_box;
};

class OpenRouteDialog : public QDialog {
  Q_OBJECT

 public:
  OpenRouteDialog(QWidget *parent);
  inline bool failedToLoad() const { return route_widget->failedToLoad(); }

 private:
  OpenRouteWidget *route_widget;
};
