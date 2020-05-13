#ifndef __SOUND_HPP
#define __SOUND_HPP

#include "cereal/gen/cpp/log.capnp.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

void ui_sound_init();
void ui_sound_destroy();

void set_volume(int volume);

void play_alert_sound(AudibleAlert alert);
void stop_alert_sound(AudibleAlert alert);

#endif

