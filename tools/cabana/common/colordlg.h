#pragma once

#include <QColorDialog>
#include <QDialog>
#include <QListWidget>
#include <QListWidgetItem>

#include "tools/cabana/dbc/dbc.h"

class SignalColorDlg : public QDialog {
  Q_OBJECT

public:
  explicit SignalColorDlg(QWidget *parent);
  void addSignal(const MessageId &msg_id, const cabana::Signal *sig);

private:
  struct ListItem : public QListWidgetItem {
    ListItem(const MessageId &id, const cabana::Signal *s, QListWidget *parent)
        : msg_id(id), sig(s), color(s->color), QListWidgetItem(parent) {}
    MessageId msg_id;
    const cabana::Signal *sig;
    QColor color;
  };

  QListWidget *signal_list;
  QColorDialog *color_picker;
};
