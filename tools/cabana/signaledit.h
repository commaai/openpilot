#pragma once

#include <optional>

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
  Q_OBJECT

public:
  SignalForm(const Signal &sig, QWidget *parent);
  std::optional<Signal> getSignal();

  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *msb, *lsb, *offset;
  QDoubleSpinBox *factor, *min_val, *max_val;
  QComboBox *sign, *endianness;
  int start_bit = 0;
};

class SignalEdit : public QWidget {
  Q_OBJECT

public:
  SignalEdit(int index, const QString &id, const Signal &sig, const QString &color, QWidget *parent = nullptr);
  void setFormVisible(bool show);
  inline bool isFormVisible() const { return form_container->isVisible(); }
  void save();

signals:
  void showChart(const QString &msg_id, const QString &sig_name);
  void showFormClicked();

protected:
  void remove();

  QString id;
  QString name_;
  QPushButton *plot_btn;
  ElidedLabel *title;
  SignalForm *form;
  QWidget *form_container;
  QPushButton *remove_btn;
  QLabel *icon;
};

class AddSignalDialog : public QDialog {
  Q_OBJECT

public:
  AddSignalDialog(const QString &id, QWidget *parent);
};
