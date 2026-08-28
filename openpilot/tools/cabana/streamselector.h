#pragma once

#include <vector>

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDialog>
#include <QFormLayout>
#include <QLineEdit>
#include <QTabWidget>

#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/streams/pandastream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif

class AbstractOpenStreamWidget : public QWidget {
  Q_OBJECT
public:
  AbstractOpenStreamWidget(QWidget *parent = nullptr) : QWidget(parent) {}
  virtual AbstractStream *open() = 0;

signals:
  void enableOpenButton(bool);
};

class OpenReplayWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenReplayWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  QLineEdit *route_edit;
  std::vector<QCheckBox *> cameras;
};

class OpenPandaWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenPandaWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  void refreshSerials();
  void buildConfigForm();

  QComboBox *serial_edit;
  QFormLayout *form_layout;
  PandaStreamConfig config = {};
};

class OpenDeviceWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenDeviceWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  QLineEdit *ip_address;
  QButtonGroup *group;
};

#ifdef __linux__
// no Q_OBJECT: moc does not define __linux__ and would otherwise skip this class
class OpenSocketCanWidget : public AbstractOpenStreamWidget {
public:
  OpenSocketCanWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  void refreshDevices();

  QComboBox *device_edit;
  SocketCanStreamConfig config = {};
};
#endif

class StreamSelector : public QDialog {
  Q_OBJECT

public:
  StreamSelector(QWidget *parent = nullptr);
  void addStreamWidget(AbstractOpenStreamWidget *w, const QString &title);
  QString dbcFile() const { return dbc_file->text(); }
  AbstractStream *stream() const { return stream_; }

private:
  AbstractStream *stream_ = nullptr;
  QLineEdit *dbc_file;
  QTabWidget *tab;
  QDialogButtonBox *btn_box;
};
