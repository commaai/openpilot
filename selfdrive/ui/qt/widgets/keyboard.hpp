#pragma once

#include <QFrame>
#include <QString>
#include <QStackedLayout>
#include <QAbstractButton>

class Keyboard : public QFrame {
  Q_OBJECT

public:
  explicit Keyboard(QWidget *parent = 0);

private:
  QStackedLayout* main_layout;

private slots:
  void handleButton(QAbstractButton* m_button);

signals:
  void emitButton(const QString &s);
};
