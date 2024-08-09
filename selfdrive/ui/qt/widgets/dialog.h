#pragma once

#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/qt_window.h"

class DialogBase : public QDialog {
  Q_OBJECT

protected:
  DialogBase(QWidget *parent);
  bool eventFilter(QObject *o, QEvent *e) override;

public slots:
  int exec() override;
};

class ConfirmationDialog : public DialogBase {
  Q_OBJECT

public:
  explicit ConfirmationDialog(const QString &prompt_text, const QString &confirm_text,
                              const QString &cancel_text, const bool rich, QWidget* parent);
  static bool alert(const QString &prompt_text, QWidget *parent);
  static bool confirm(const QString &prompt_text, const QString &confirm_text, QWidget *parent);
  static bool rich(const QString &prompt_text, QWidget *parent);
};

class MultiOptionDialog : public DialogBase {
  Q_OBJECT

public:
  explicit MultiOptionDialog(const QString &prompt_text, const QStringList &l, const QString &current, QWidget *parent);
  static QString getSelection(const QString &prompt_text, const QStringList &l, const QString &current, QWidget *parent);
  QString selection;
};
