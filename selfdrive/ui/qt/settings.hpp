#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QScrollArea>
#include <QCheckBox>
#include <QScroller>
#include <QScrollerProperties>


class SettingsWindow : public QWidget
{
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

signals:
  void closeSettings();
};
