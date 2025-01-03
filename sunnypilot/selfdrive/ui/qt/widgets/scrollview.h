/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/widgets/scrollview.h"

class ScrollViewSP : public ScrollView {
  Q_OBJECT

public:
  explicit ScrollViewSP(QWidget *w = nullptr, QWidget *parent = nullptr) : ScrollView(w, parent) {}

public slots:
  void setLastScrollPosition();
  void restoreScrollPosition();

private:
  int lastScrollPosition = 0;
};
