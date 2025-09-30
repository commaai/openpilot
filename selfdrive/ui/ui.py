#!/usr/bin/env python3
import pyray as rl
from openpilot.common.watchdog import kick_watchdog
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.common.params_pyx import Params
from openpilot.selfdrive.ui.layouts.onboarding import TermsPage, OnboardingDialog


def main():
  # TODO: https://github.com/commaai/agnos-builder/pull/490
  # os.nice(-20)

  gui_app.init_window("UI")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  # Show onboarding if needed (terms/training)
  def _start_training():
    gui_app.set_modal_overlay(OnboardingDialog())

  params = Params()
  current_terms_version = params.get("TermsVersion")
  current_training_version = params.get("TrainingVersion")
  accepted_terms = params.get("HasAcceptedTerms") == current_terms_version
  training_done = params.get("CompletedTrainingVersion") == current_training_version

  if not accepted_terms:
    def _on_accept():
      params.put("HasAcceptedTerms", current_terms_version)
      if params.get("CompletedTrainingVersion") != current_training_version:
        _start_training()

    def _on_decline():
      params.put_bool("DoUninstall", True)

    gui_app.set_modal_overlay(TermsPage(on_accept=_on_accept, on_decline=_on_decline))
  elif not training_done:
    _start_training()
  for showing_dialog in gui_app.render():
    ui_state.update()

    kick_watchdog()

    if not showing_dialog:
      main_layout.render()


if __name__ == "__main__":
  main()
