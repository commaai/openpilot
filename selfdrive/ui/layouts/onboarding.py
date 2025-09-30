import os
import re
from enum import IntEnum

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app

from selfdrive.ui.ui_state import ui_state

DEBUG = True

STEP_RECTS = [rl.Rectangle(104.0, 800.0, 633.0, 175.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2156.0, 1080.0),
              rl.Rectangle(1526.0, 473.0, 427.0, 472.0), rl.Rectangle(1643.0, 441.0, 217.0, 223.0), rl.Rectangle(1835.0, 0.0, 2155.0, 1080.0),
              rl.Rectangle(1786.0, 591.0, 267.0, 236.0), rl.Rectangle(1353.0, 0.0, 804.0, 1080.0), rl.Rectangle(1458.0, 485.0, 633.0, 211.0),
              rl.Rectangle(95.0, 794.0, 1158.0, 187.0), rl.Rectangle(1560.0, 170.0, 392.0, 397.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(1351.0, 0.0, 807.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2158.0, 1080.0), rl.Rectangle(1531.0, 82.0, 441.0, 920.0),
              rl.Rectangle(1336.0, 438.0, 490.0, 393.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(612.0 - 525, 795.0, 662.0 + 525, 186.0)]

DM_RECORD_STEP = 9
DM_RECORD_YES_RECT = rl.Rectangle(695, 794, 558, 187)

RESTART_TRAINING_RECT = rl.Rectangle(612.0 - 525, 795.0, 662.0 - 190, 186.0)


class OnboardingDialog(Widget):
  def __init__(self, completed_callback=None):
    super().__init__()
    self._completed_callback = completed_callback

    self._step = 18
    self._load_images()

  def _load_images(self):
    self._images = []
    paths = [fn for fn in os.listdir(os.path.join(BASEDIR, "selfdrive/assets/training")) if re.match(r'^step\d*\.png$', fn)]
    paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x).group()))
    for fn in paths:
      path = os.path.join(BASEDIR, "selfdrive/assets/training", fn)
      self._images.append(gui_app.texture(path, gui_app.width, gui_app.height))

  def _handle_mouse_release(self, mouse_pos):
    if rl.check_collision_point_rec(mouse_pos, STEP_RECTS[self._step]):
      if self._step == DM_RECORD_STEP:
        yes = rl.check_collision_point_rec(mouse_pos, DM_RECORD_YES_RECT)
        print(f"putting RecordFront to {yes}")
        ui_state.params.put_bool("RecordFront", yes)

      elif self._step == len(self._images) - 1:
        if rl.check_collision_point_rec(mouse_pos, RESTART_TRAINING_RECT):
          self._step = -1

      self._step += 1

      if self._step >= len(self._images):
        self._completed_training()
        if self._completed_callback:
          self._completed_callback()
        return

  def _completed_training(self):
    self._step = 0
    current_training_version = ui_state.params.get("TrainingVersion")
    ui_state.params.put("CompletedTrainingVersion", current_training_version)

  def _render(self, _):
    rl.draw_texture(self._images[self._step], 0, 0, rl.WHITE)

    # progress bar
    if 0 < self._step < len(STEP_RECTS) - 1:
      h = 20
      w = int((self._step / (len(STEP_RECTS) - 1)) * self._rect.width)
      rl.draw_rectangle(int(self._rect.x), int(self._rect.y + self._rect.height - h),
                        w, h, rl.Color(70, 91, 234, 255))

    if DEBUG:
      rl.draw_rectangle_lines_ex(STEP_RECTS[self._step], 3, rl.RED)


from openpilot.system.ui.lib.text_measure import measure_text_cached  # noqa: F401
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import Label, TextAlignment
from openpilot.common.params_pyx import Params


"""
void TermsPage::showEvent(QShowEvent *event) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(45, 35, 45, 45);
  main_layout->setSpacing(0);

  QVBoxLayout *vlayout = new QVBoxLayout();
  vlayout->setContentsMargins(165, 165, 165, 0);
  main_layout->addLayout(vlayout);

  QLabel *title = new QLabel(tr("Welcome to openpilot"));
  title->setStyleSheet("font-size: 90px; font-weight: 500;");
  vlayout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  vlayout->addSpacing(90);
  QLabel *desc = new QLabel(tr("You must accept the Terms and Conditions to use openpilot. Read the latest terms at <span style='color: #465BEA;'>https://comma.ai/terms</span> before continuing."));
  desc->setWordWrap(true);
  desc->setStyleSheet("font-size: 80px; font-weight: 300;");
  vlayout->addWidget(desc, 0);

  vlayout->addStretch();

  QHBoxLayout* buttons = new QHBoxLayout;
  buttons->setMargin(0);
  buttons->setSpacing(45);
  main_layout->addLayout(buttons);

  QPushButton *decline_btn = new QPushButton(tr("Decline"));
  buttons->addWidget(decline_btn);
  QObject::connect(decline_btn, &QPushButton::clicked, this, &TermsPage::declinedTerms);

  accept_btn = new QPushButton(tr("Agree"));
  accept_btn->setStyleSheet(R"(
    QPushButton {
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  buttons->addWidget(accept_btn);
  QObject::connect(accept_btn, &QPushButton::clicked, this, &TermsPage::acceptedTerms);
}
"""


class TermsPage(Widget):
  def __init__(self, on_accept=None, on_decline=None):
    super().__init__()
    self._on_accept = on_accept
    self._on_decline = on_decline

    self._title = Label("Welcome to openpilot", font_size=90, font_weight=FontWeight.BOLD, text_alignment=TextAlignment.LEFT)
    self._desc = Label("You must accept the Terms and Conditions to use openpilot. Read the latest terms at https://comma.ai/terms before continuing.",
                       font_size=90, font_weight=FontWeight.MEDIUM, text_alignment=TextAlignment.LEFT)

    self._decline_btn = Button("Decline", click_callback=self._on_decline_clicked)
    self._accept_btn = Button("Agree", button_style=ButtonStyle.PRIMARY, click_callback=self._on_accept_clicked)

  def _on_accept_clicked(self):
    if self._on_accept:
      self._on_accept()

  def _on_decline_clicked(self):
    if self._on_decline:
      self._on_decline()

  def _render(self, _):
    welcome_x = self._rect.x + 165
    welcome_y = self._rect.y + 165

    # self._title.set_position(welcome_x, welcome_y)
    welcome_rect = rl.Rectangle(welcome_x, welcome_y, self._rect.width - welcome_x, 90)
    rl.draw_rectangle_lines_ex(welcome_rect, 3, rl.RED)
    self._title.render(welcome_rect)

    desc_x = welcome_x
    # TODO: Label doesn't top align when wrapping
    desc_y = welcome_y - 100
    desc_rect = rl.Rectangle(desc_x, desc_y, self._rect.width - desc_x, self._rect.height - desc_y - 250)
    rl.draw_rectangle_lines_ex(desc_rect, 3, rl.RED)
    self._desc.render(desc_rect)

    btn_y = self._rect.y + self._rect.height - 160 - 45
    btn_spacing = 45
    btn_width = (self._rect.width - 45 * 3) / 2
    self._decline_btn.render(rl.Rectangle(self._rect.x + 45, btn_y, btn_width, 160))
    self._accept_btn.render(rl.Rectangle(self._rect.x + 45 * 2 + btn_width, btn_y, btn_width, 160))

    return -1


# --- Onboarding helpers ---
# def completed(params: _Params | None = None) -> bool:
#   p = params or _Params()
#   current_terms_version = p.get("TermsVersion")
#   current_training_version = p.get("TrainingVersion")
#   accepted_terms = p.get("HasAcceptedTerms") == current_terms_version
#   training_done = p.get("CompletedTrainingVersion") == current_training_version
#   return accepted_terms and training_done


# def show_training_guide():
#   gui_app.set_modal_overlay(OnboardingDialog())
#
#
# def maybe_show_onboarding():
#   p = _Params()
#   current_terms_version = p.get("TermsVersion")
#   current_training_version = p.get("TrainingVersion")
#   accepted_terms = p.get("HasAcceptedTerms") == current_terms_version
#   training_done = p.get("CompletedTrainingVersion") == current_training_version
#
#   if not accepted_terms:
#     def _on_accept():
#       p.put("HasAcceptedTerms", current_terms_version)
#       if p.get("CompletedTrainingVersion") != current_training_version:
#         show_training_guide()
#
#     def _on_decline():
#       p.put_bool("DoUninstall", True)
#
#     gui_app.set_modal_overlay(TermsPage(on_accept=_on_accept, on_decline=_on_decline))
#   elif not training_done:
#     show_training_guide()



"""
void DeclinePage::showEvent(QShowEvent *event) {
  if (layout()) {
    return;
  }

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(45);
  main_layout->setSpacing(40);

  QLabel *text = new QLabel(this);
  text->setText(tr("You must accept the Terms and Conditions in order to use openpilot."));
  text->setStyleSheet(R"(font-size: 80px; font-weight: 300; margin: 200px;)");
  text->setWordWrap(true);
  main_layout->addWidget(text, 0, Qt::AlignCenter);

  QHBoxLayout* buttons = new QHBoxLayout;
  buttons->setSpacing(45);
  main_layout->addLayout(buttons);

  QPushButton *back_btn = new QPushButton(tr("Back"));
  buttons->addWidget(back_btn);

  QObject::connect(back_btn, &QPushButton::clicked, this, &DeclinePage::getBack);

  QPushButton *uninstall_btn = new QPushButton(tr("Decline, uninstall %1").arg(getBrand()));
  uninstall_btn->setStyleSheet("background-color: #B73D3D");
  buttons->addWidget(uninstall_btn);
  QObject::connect(uninstall_btn, &QPushButton::clicked, [=]() {
    Params().putBool("DoUninstall", true);
  });
}
"""


class DeclinePage(Widget):
  def __init__(self, back_callback=None):
    super().__init__()
    self._back_callback = back_callback
    self._text = Label("You must accept the Terms and Conditions in order to use openpilot.",
                       font_size=90, font_weight=FontWeight.MEDIUM, text_alignment=TextAlignment.LEFT)
    self._back_btn = Button("Back", click_callback=self._on_back_clicked)
    self._uninstall_btn = Button("Decline, uninstall openpilot", button_style=ButtonStyle.DANGER,
                                 click_callback=self._on_uninstall_clicked)

  def _on_uninstall_clicked(self):
    ui_state.params.put_bool("DoUninstall", True)
    gui_app.request_close()

  def _on_back_clicked(self):
    if self._back_callback:
      self._back_callback()

  def _render(self, _):
    # btns:
    btn_y = self._rect.y + self._rect.height - 160 - 45
    btn_width = (self._rect.width - 45 * 3) / 2
    self._back_btn.render(rl.Rectangle(self._rect.x + 45, btn_y, btn_width, 160))
    self._uninstall_btn.render(rl.Rectangle(self._rect.x + 45 * 2 + btn_width, btn_y, btn_width, 160))

    # text rect in middle of top and buttony
    text_height = btn_y - (200 + 45)
    text_rect = rl.Rectangle(self._rect.x + 165, self._rect.y + (btn_y - text_height) / 2, self._rect.width - (165 * 2), text_height)
    rl.draw_rectangle_lines_ex(text_rect, 3, rl.RED)
    self._text.render(text_rect)


"""
OnboardingWindow::OnboardingWindow(QWidget *parent) : QStackedWidget(parent) {
  std::string current_terms_version = params.get("TermsVersion");
  std::string current_training_version = params.get("TrainingVersion");
  accepted_terms = params.get("HasAcceptedTerms") == current_terms_version;
  training_done = params.get("CompletedTrainingVersion") == current_training_version;

  TermsPage* terms = new TermsPage(this);
  addWidget(terms);
  connect(terms, &TermsPage::acceptedTerms, [=]() {
    params.put("HasAcceptedTerms", current_terms_version);
    accepted_terms = true;
    updateActiveScreen();
  });
  connect(terms, &TermsPage::declinedTerms, [=]() { setCurrentIndex(2); });

  TrainingGuide* tr = new TrainingGuide(this);
  addWidget(tr);
  connect(tr, &TrainingGuide::completedTraining, [=]() {
    training_done = true;
    params.put("CompletedTrainingVersion", current_training_version);
    updateActiveScreen();
  });

  DeclinePage* declinePage = new DeclinePage(this);
  addWidget(declinePage);
  connect(declinePage, &DeclinePage::getBack, [=]() { updateActiveScreen(); });

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      height: 160px;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #4F4F4F;
    }
  )");
  updateActiveScreen();
}
"""


class OnboardingState(IntEnum):
  TERMS = 0
  ONBOARDING = 1
  DECLINE = 2


class OnboardingWindow(Widget):
  def __init__(self):
    super().__init__()
    self._current_terms_version = ui_state.params.get("TermsVersion")
    self._current_training_version = ui_state.params.get("TrainingVersion")
    self._accepted_terms = ui_state.params.get("HasAcceptedTerms") == self._current_terms_version
    self._training_done = ui_state.params.get("CompletedTrainingVersion") == self._current_training_version

    self._state = OnboardingState.TERMS if not self._accepted_terms else OnboardingState.ONBOARDING

    self._terms = TermsPage(on_accept=self._on_terms_accepted, on_decline=self._on_terms_declined)
    self._training_guide = OnboardingDialog(completed_callback=self._on_completed_training)
    self._decline_page = DeclinePage(back_callback=self._on_decline_back)

  @property
  def completed(self) -> bool:
    return self._accepted_terms and self._training_done

  def _on_terms_declined(self):
    self._state = OnboardingState.DECLINE

  def _on_decline_back(self):
    self._state = OnboardingState.TERMS

  def _on_terms_accepted(self):
    ui_state.params.put("HasAcceptedTerms", self._current_terms_version)
    self._state = OnboardingState.ONBOARDING

  def _on_completed_training(self):
    ui_state.params.put("CompletedTrainingVersion", self._current_training_version)
    gui_app.set_modal_overlay(None)

  def _render(self, _):
    print(f"OnboardingWindow state: {self._state}, accepted_terms: {self._accepted_terms}, training_done: {self._training_done}")
    if self._state == OnboardingState.TERMS:
      self._terms.render(self._rect)
    if self._state == OnboardingState.ONBOARDING:
      self._training_guide.render(self._rect)
    elif self._state == OnboardingState.DECLINE:
      self._decline_page.render(self._rect)
    return -1
