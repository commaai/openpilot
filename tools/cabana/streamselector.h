#pragma once

#include <QDialog>
#include <QLineEdit>

#include "tools/cabana/streams/abstractstream.h"

class OpenPandaWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenPandaWidget(AbstractStream **stream, QWidget *parent);
  bool open() override;
  QString title() override { return tr("Panda"); }

private:
  QLineEdit *serial_edit;
};


class StreamSelector : public QDialog {
  Q_OBJECT

public:
  StreamSelector(AbstractStream **stream, QWidget *parent);
  QString dbcFile() const { return dbc_file->text(); }

 private:
  QLineEdit *dbc_file;
};
