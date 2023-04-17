#pragma once

#include <QDialog>
#include <QLineEdit>

#include "tools/cabana/streams/abstractstream.h"

class StreamSelector : public QDialog {
  Q_OBJECT

public:
  StreamSelector(AbstractStream **stream, QWidget *parent);
  QString dbcFile() const { return dbc_file->text(); }

 private:
  QLineEdit *dbc_file;
};
