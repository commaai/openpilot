#pragma once

#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>

#include "selfdrive/ui/qt/widgets/controls.h"

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class SignalForm : public QWidget {
public:
  SignalForm(const Signal &sig, QWidget *parent);
  Signal getSignal();

  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *offset;
  QDoubleSpinBox *factor, *min_val, *max_val;
  QComboBox *sign, *endianness;
  int start_bit = 0;
};

class SignalEdit : public QWidget {
  Q_OBJECT

public:
  SignalEdit(int index, const QString &msg_id, const Signal &sig, QWidget *parent = nullptr);
  void setFormVisible(bool show);
  inline bool isFormVisible() const { return form_container->isVisible(); }
  QString sig_name;
  SignalForm *form;

signals:
  void showChart();
  void showFormClicked();
  void remove();
  void save();

protected:
  ElidedLabel *title;
  QWidget *form_container;
  QLabel *icon;
};

class AddSignalDialog : public QDialog {
public:
  AddSignalDialog(const QString &id, int start_bit, int size, QWidget *parent);
  SignalForm *form;
};
