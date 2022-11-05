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
  SignalForm(QWidget *parent);

  QLineEdit *name, *unit, *comment, *val_desc, *offset, *factor, *min_val, *max_val;
  QLabel *lsb, *msb;
  QSpinBox *size;
  QComboBox *sign, *endianness;
};

class SignalEdit : public QWidget {
  Q_OBJECT

public:
  SignalEdit(int index, QWidget *parent = nullptr);
  void setSignal(const QString &msg_id, const Signal *sig, bool show_form);
  void setChartOpened(bool opened);
  void setFormVisible(bool show);
  void signalHovered(const Signal *sig);
  inline bool isFormVisible() const { return form_container->isVisible(); }
  const Signal *sig = nullptr;
  QString msg_id;

signals:
  void highlight(const Signal *sig);
  void showChart(const QString &name, const Signal *sig, bool show);
  void showFormClicked();
  void remove(const Signal *sig);
  void save(const Signal *sig, const Signal &new_sig);

protected:
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void saveSignal();

  SignalForm *form = nullptr;
  ElidedLabel *title;
  QWidget *form_container;
  QLabel *icon;
  int form_idx = 0;
  bool chart_opened = false;
  QPushButton *plot_btn;
};

class SignalFindDlg : public QDialog {
  Q_OBJECT

public:
  SignalFindDlg(const QString &id, const Signal *signal, QWidget *parent);
};
