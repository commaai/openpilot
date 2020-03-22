#include <stdlib.h>
#include "sound.hpp"

#include "common/swaglog.h"

typedef struct {
  AudibleAlert alert;
  const char* uri;
  bool loop;
} sound_file;

extern "C"{
#include "slplay.h"
}

int last_volume = 0;

void set_volume(int volume) {
  if (last_volume != volume) {
    char volume_change_cmd[64];
    sprintf(volume_change_cmd, "service call audio 3 i32 3 i32 %d i32 1 &", volume);

    // 5 second timeout at 60fps
    int volume_changed = system(volume_change_cmd);
    last_volume = volume;
  }
}


sound_file sound_table[] = {
  { cereal_CarControl_HUDControl_AudibleAlert_chimeDisengage, "../assets/sounds/disengaged.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_chimeEngage, "../assets/sounds/engaged.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_chimeWarning1, "../assets/sounds/warning_1.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_chimeWarning2, "../assets/sounds/warning_2.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_chimeWarningRepeat, "../assets/sounds/warning_repeat.wav", true },
  { cereal_CarControl_HUDControl_AudibleAlert_chimeError, "../assets/sounds/error.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_chimePrompt, "../assets/sounds/error.wav", false },
  { cereal_CarControl_HUDControl_AudibleAlert_none, NULL, false },
};

sound_file* get_sound_file(AudibleAlert alert) {
  for (sound_file *s = sound_table; s->alert != cereal_CarControl_HUDControl_AudibleAlert_none; s++) {
    if (s->alert == alert) {
      return s;
    }
  }

  return NULL;
}

void play_alert_sound(AudibleAlert alert) {
  sound_file* sound = get_sound_file(alert);
  char* error = NULL;

  slplay_play(sound->uri, sound->loop, &error);
  if(error) {
    LOGW("error playing sound: %s", error);
  }
}

void stop_alert_sound(AudibleAlert alert) {
  sound_file* sound = get_sound_file(alert);
  char* error = NULL;

  slplay_stop_uri(sound->uri, &error);
  if(error) {
    LOGW("error stopping sound: %s", error);
  }
}

void ui_sound_init() {
  char *error = NULL;
  slplay_setup(&error);
  if (error) goto fail;

  for (sound_file *s = sound_table; s->alert != cereal_CarControl_HUDControl_AudibleAlert_none; s++) {
    slplay_create_player_for_uri(s->uri, &error);
    if (error) goto fail;
  }
  return;

fail:
  LOGW(error);
  exit(1);
}

void ui_sound_destroy() {
  slplay_destroy();
}

