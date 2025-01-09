/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/widgets/prime.h"

SetupWidgetSP::SetupWidgetSP(QWidget *parent) : SetupWidget(parent) {
  PrimeUserWidget *primeUser = new PrimeUserWidget;
  content_layout->insertWidget(0, primeUser);

  primeUser->setVisible(uiState()->prime_state->isSubscribed());

  QObject::connect(uiState()->prime_state, &PrimeState::changed, [=]() {
    primeUser->setVisible(uiState()->prime_state->isSubscribed());
  });
}
