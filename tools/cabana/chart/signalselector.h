#pragma once

#include <QComboBox>
#include <QDialog>
#include <QListWidget>

#include "tools/cabana/dbc/dbcmanager.h"

class SignalSelector : public QDialog {
public:
  struct ListItem : public QListWidgetItem {
    ListItem(const MessageId &msg_id, const cabana::Signal *sig, QListWidget *parent) : msg_id(msg_id), sig(sig), QListWidgetItem(parent) {}
    MessageId msg_id;
    const cabana::Signal *sig;
  };

  SignalSelector(QString title, QWidget *parent);
  QList<ListItem *> seletedItems();
  inline void addSelected(const MessageId &id, const cabana::Signal *sig) { addItemToList(selected_list, id, sig, true); }

private:
  void updateAvailableList(int index);
  void addItemToList(QListWidget *parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name = false);
  void add(QListWidgetItem *item);
  void remove(QListWidgetItem *item);

  QComboBox *msgs_combo;
  QListWidget *available_list;
  QListWidget *selected_list;
};
