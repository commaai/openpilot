#pragma once

#include <vector>

#include <QFrame>
#include <QPushButton>
#include <QStackedLayout>

class KeyButton : public QPushButton {
  Q_OBJECT

public:
  KeyButton(const QString &text, QWidget *parent = 0);
  bool event(QEvent *event) override;
};

class KeyboardLayout : public QWidget {
  Q_OBJECT

public:
  explicit KeyboardLayout(QWidget* parent, const std::vector<QVector<QString>>& layout);
};

class Keyboard : public QFrame {
  Q_OBJECT

public:
  explicit Keyboard(QWidget *parent = 0);

private:
  QStackedLayout* main_layout;
  int shift_state = 0;

private slots:
  void handleButton(QAbstractButton* m_button);
  void handleCapsPress();

signals:
  void emitKey(const QString &s);
  void emitBackspace();
  void emitEnter();
};
