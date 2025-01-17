#pragma once

#include <QDialogButtonBox>
#include <QDialog>
#include <QLineEdit>
#include <QTabWidget>

#include "tools/cabana/streams/abstractstream.h"

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
