#ifndef __SOUND_HPP
#define __SOUND_HPP

#include "cereal/gen/c/log.capnp.h"

typedef enum cereal_CarControl_HUDControl_AudibleAlert AudibleAlert;

void ui_sound_init();
void ui_sound_destroy();

void set_volume(int volume);

void play_alert_sound(AudibleAlert alert);
void stop_alert_sound(AudibleAlert alert);

#endif

