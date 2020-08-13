#pragma once

#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QLabel>
#include <QDebug>

class Window : public QWidget
{
  Q_OBJECT

public:
  explicit Window(QWidget *parent = 0);

};
