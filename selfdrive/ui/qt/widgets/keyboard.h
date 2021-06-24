#pragma once

#include <vector>

#include <QAbstractButton>
#include <QFrame>
#include <QStackedLayout>
#include <QString>
#include <QWidget>

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

private slots:
  void handleButton(QAbstractButton* m_button);

signals:
  void emitButton(const QString &s);
};
