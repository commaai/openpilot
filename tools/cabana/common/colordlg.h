#pragma once

#include <QColorDialog>
#include <QListWidget>

#include "tools/cabana/dbc/dbcmanager.h"

class SignalColorDlg : public QDialog {
public:
  explicit SignalColorDlg(QWidget *parent);
  void addSignal(const MessageId &msg_id, const cabana::Signal *sig);

private:
  struct Item : public QListWidgetItem {
    Item(const MessageId &id, const cabana::Signal *s, QListWidget *parent)
        : msg_id(id), QListWidgetItem(s->name, parent) { setColor(s->color); }
    void setColor(const QColor &c) {
      color = c;
      QPixmap pm(12, 12);
      pm.fill(color);
      setIcon(pm);
    }
    MessageId msg_id;
    QColor color;
  };

  QListWidget *list;
  QColorDialog *color_picker;
};
