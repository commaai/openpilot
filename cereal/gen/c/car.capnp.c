#include "car.capnp.h"
/* AUTO GENERATED - DO NOT EDIT */

cereal_CarState_ptr cereal_new_CarState(struct capn_segment *s) {
	cereal_CarState_ptr p;
	p.p = capn_new_struct(s, 24, 5);
	return p;
}
cereal_CarState_list cereal_new_CarState_list(struct capn_segment *s, int len) {
	cereal_CarState_list p;
	p.p = capn_new_list(s, len, 24, 5);
	return p;
}
void cereal_read_CarState(struct cereal_CarState *s, cereal_CarState_ptr p) {
	capn_resolve(&p.p);
	s->errors.p = capn_getp(p.p, 0, 0);
	s->vEgo = capn_to_f32(capn_read32(p.p, 0));
	s->wheelSpeeds.p = capn_getp(p.p, 1, 0);
	s->gas = capn_to_f32(capn_read32(p.p, 4));
	s->gasPressed = (capn_read8(p.p, 8) & 1) != 0;
	s->brake = capn_to_f32(capn_read32(p.p, 12));
	s->brakePressed = (capn_read8(p.p, 8) & 2) != 0;
	s->steeringAngle = capn_to_f32(capn_read32(p.p, 16));
	s->steeringTorque = capn_to_f32(capn_read32(p.p, 20));
	s->steeringPressed = (capn_read8(p.p, 8) & 4) != 0;
	s->cruiseState.p = capn_getp(p.p, 2, 0);
	s->buttonEvents.p = capn_getp(p.p, 3, 0);
	s->canMonoTimes.p = capn_getp(p.p, 4, 0);
}
void cereal_write_CarState(const struct cereal_CarState *s, cereal_CarState_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->errors.p);
	capn_write32(p.p, 0, capn_from_f32(s->vEgo));
	capn_setp(p.p, 1, s->wheelSpeeds.p);
	capn_write32(p.p, 4, capn_from_f32(s->gas));
	capn_write1(p.p, 64, s->gasPressed != 0);
	capn_write32(p.p, 12, capn_from_f32(s->brake));
	capn_write1(p.p, 65, s->brakePressed != 0);
	capn_write32(p.p, 16, capn_from_f32(s->steeringAngle));
	capn_write32(p.p, 20, capn_from_f32(s->steeringTorque));
	capn_write1(p.p, 66, s->steeringPressed != 0);
	capn_setp(p.p, 2, s->cruiseState.p);
	capn_setp(p.p, 3, s->buttonEvents.p);
	capn_setp(p.p, 4, s->canMonoTimes.p);
}
void cereal_get_CarState(struct cereal_CarState *s, cereal_CarState_list l, int i) {
	cereal_CarState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarState(s, p);
}
void cereal_set_CarState(const struct cereal_CarState *s, cereal_CarState_list l, int i) {
	cereal_CarState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarState(s, p);
}

cereal_CarState_WheelSpeeds_ptr cereal_new_CarState_WheelSpeeds(struct capn_segment *s) {
	cereal_CarState_WheelSpeeds_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarState_WheelSpeeds_list cereal_new_CarState_WheelSpeeds_list(struct capn_segment *s, int len) {
	cereal_CarState_WheelSpeeds_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds *s, cereal_CarState_WheelSpeeds_ptr p) {
	capn_resolve(&p.p);
	s->fl = capn_to_f32(capn_read32(p.p, 0));
	s->fr = capn_to_f32(capn_read32(p.p, 4));
	s->rl = capn_to_f32(capn_read32(p.p, 8));
	s->rr = capn_to_f32(capn_read32(p.p, 12));
}
void cereal_write_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds *s, cereal_CarState_WheelSpeeds_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->fl));
	capn_write32(p.p, 4, capn_from_f32(s->fr));
	capn_write32(p.p, 8, capn_from_f32(s->rl));
	capn_write32(p.p, 12, capn_from_f32(s->rr));
}
void cereal_get_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds *s, cereal_CarState_WheelSpeeds_list l, int i) {
	cereal_CarState_WheelSpeeds_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarState_WheelSpeeds(s, p);
}
void cereal_set_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds *s, cereal_CarState_WheelSpeeds_list l, int i) {
	cereal_CarState_WheelSpeeds_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarState_WheelSpeeds(s, p);
}

cereal_CarState_CruiseState_ptr cereal_new_CarState_CruiseState(struct capn_segment *s) {
	cereal_CarState_CruiseState_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_CarState_CruiseState_list cereal_new_CarState_CruiseState_list(struct capn_segment *s, int len) {
	cereal_CarState_CruiseState_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_CarState_CruiseState(struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_ptr p) {
	capn_resolve(&p.p);
	s->enabled = (capn_read8(p.p, 0) & 1) != 0;
	s->speed = capn_to_f32(capn_read32(p.p, 4));
}
void cereal_write_CarState_CruiseState(const struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->enabled != 0);
	capn_write32(p.p, 4, capn_from_f32(s->speed));
}
void cereal_get_CarState_CruiseState(struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_list l, int i) {
	cereal_CarState_CruiseState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarState_CruiseState(s, p);
}
void cereal_set_CarState_CruiseState(const struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_list l, int i) {
	cereal_CarState_CruiseState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarState_CruiseState(s, p);
}

cereal_CarState_ButtonEvent_ptr cereal_new_CarState_ButtonEvent(struct capn_segment *s) {
	cereal_CarState_ButtonEvent_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_CarState_ButtonEvent_list cereal_new_CarState_ButtonEvent_list(struct capn_segment *s, int len) {
	cereal_CarState_ButtonEvent_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent *s, cereal_CarState_ButtonEvent_ptr p) {
	capn_resolve(&p.p);
	s->pressed = (capn_read8(p.p, 0) & 1) != 0;
	s->type = (enum cereal_CarState_ButtonEvent_Type)(int) capn_read16(p.p, 2);
}
void cereal_write_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent *s, cereal_CarState_ButtonEvent_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->pressed != 0);
	capn_write16(p.p, 2, (uint16_t) (s->type));
}
void cereal_get_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent *s, cereal_CarState_ButtonEvent_list l, int i) {
	cereal_CarState_ButtonEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarState_ButtonEvent(s, p);
}
void cereal_set_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent *s, cereal_CarState_ButtonEvent_list l, int i) {
	cereal_CarState_ButtonEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarState_ButtonEvent(s, p);
}

cereal_RadarState_ptr cereal_new_RadarState(struct capn_segment *s) {
	cereal_RadarState_ptr p;
	p.p = capn_new_struct(s, 0, 3);
	return p;
}
cereal_RadarState_list cereal_new_RadarState_list(struct capn_segment *s, int len) {
	cereal_RadarState_list p;
	p.p = capn_new_list(s, len, 0, 3);
	return p;
}
void cereal_read_RadarState(struct cereal_RadarState *s, cereal_RadarState_ptr p) {
	capn_resolve(&p.p);
	s->errors.p = capn_getp(p.p, 0, 0);
	s->points.p = capn_getp(p.p, 1, 0);
	s->canMonoTimes.p = capn_getp(p.p, 2, 0);
}
void cereal_write_RadarState(const struct cereal_RadarState *s, cereal_RadarState_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->errors.p);
	capn_setp(p.p, 1, s->points.p);
	capn_setp(p.p, 2, s->canMonoTimes.p);
}
void cereal_get_RadarState(struct cereal_RadarState *s, cereal_RadarState_list l, int i) {
	cereal_RadarState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_RadarState(s, p);
}
void cereal_set_RadarState(const struct cereal_RadarState *s, cereal_RadarState_list l, int i) {
	cereal_RadarState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_RadarState(s, p);
}

cereal_RadarState_RadarPoint_ptr cereal_new_RadarState_RadarPoint(struct capn_segment *s) {
	cereal_RadarState_RadarPoint_ptr p;
	p.p = capn_new_struct(s, 32, 0);
	return p;
}
cereal_RadarState_RadarPoint_list cereal_new_RadarState_RadarPoint_list(struct capn_segment *s, int len) {
	cereal_RadarState_RadarPoint_list p;
	p.p = capn_new_list(s, len, 32, 0);
	return p;
}
void cereal_read_RadarState_RadarPoint(struct cereal_RadarState_RadarPoint *s, cereal_RadarState_RadarPoint_ptr p) {
	capn_resolve(&p.p);
	s->trackId = capn_read64(p.p, 0);
	s->dRel = capn_to_f32(capn_read32(p.p, 8));
	s->yRel = capn_to_f32(capn_read32(p.p, 12));
	s->vRel = capn_to_f32(capn_read32(p.p, 16));
	s->aRel = capn_to_f32(capn_read32(p.p, 20));
	s->yvRel = capn_to_f32(capn_read32(p.p, 24));
}
void cereal_write_RadarState_RadarPoint(const struct cereal_RadarState_RadarPoint *s, cereal_RadarState_RadarPoint_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->trackId);
	capn_write32(p.p, 8, capn_from_f32(s->dRel));
	capn_write32(p.p, 12, capn_from_f32(s->yRel));
	capn_write32(p.p, 16, capn_from_f32(s->vRel));
	capn_write32(p.p, 20, capn_from_f32(s->aRel));
	capn_write32(p.p, 24, capn_from_f32(s->yvRel));
}
void cereal_get_RadarState_RadarPoint(struct cereal_RadarState_RadarPoint *s, cereal_RadarState_RadarPoint_list l, int i) {
	cereal_RadarState_RadarPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_RadarState_RadarPoint(s, p);
}
void cereal_set_RadarState_RadarPoint(const struct cereal_RadarState_RadarPoint *s, cereal_RadarState_RadarPoint_list l, int i) {
	cereal_RadarState_RadarPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_RadarState_RadarPoint(s, p);
}

cereal_CarControl_ptr cereal_new_CarControl(struct capn_segment *s) {
	cereal_CarControl_ptr p;
	p.p = capn_new_struct(s, 16, 2);
	return p;
}
cereal_CarControl_list cereal_new_CarControl_list(struct capn_segment *s, int len) {
	cereal_CarControl_list p;
	p.p = capn_new_list(s, len, 16, 2);
	return p;
}
void cereal_read_CarControl(struct cereal_CarControl *s, cereal_CarControl_ptr p) {
	capn_resolve(&p.p);
	s->enabled = (capn_read8(p.p, 0) & 1) != 0;
	s->gas = capn_to_f32(capn_read32(p.p, 4));
	s->brake = capn_to_f32(capn_read32(p.p, 8));
	s->steeringTorque = capn_to_f32(capn_read32(p.p, 12));
	s->cruiseControl.p = capn_getp(p.p, 0, 0);
	s->hudControl.p = capn_getp(p.p, 1, 0);
}
void cereal_write_CarControl(const struct cereal_CarControl *s, cereal_CarControl_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->enabled != 0);
	capn_write32(p.p, 4, capn_from_f32(s->gas));
	capn_write32(p.p, 8, capn_from_f32(s->brake));
	capn_write32(p.p, 12, capn_from_f32(s->steeringTorque));
	capn_setp(p.p, 0, s->cruiseControl.p);
	capn_setp(p.p, 1, s->hudControl.p);
}
void cereal_get_CarControl(struct cereal_CarControl *s, cereal_CarControl_list l, int i) {
	cereal_CarControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarControl(s, p);
}
void cereal_set_CarControl(const struct cereal_CarControl *s, cereal_CarControl_list l, int i) {
	cereal_CarControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarControl(s, p);
}

cereal_CarControl_CruiseControl_ptr cereal_new_CarControl_CruiseControl(struct capn_segment *s) {
	cereal_CarControl_CruiseControl_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarControl_CruiseControl_list cereal_new_CarControl_CruiseControl_list(struct capn_segment *s, int len) {
	cereal_CarControl_CruiseControl_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl *s, cereal_CarControl_CruiseControl_ptr p) {
	capn_resolve(&p.p);
	s->cancel = (capn_read8(p.p, 0) & 1) != 0;
	s->override = (capn_read8(p.p, 0) & 2) != 0;
	s->speedOverride = capn_to_f32(capn_read32(p.p, 4));
	s->accelOverride = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl *s, cereal_CarControl_CruiseControl_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->cancel != 0);
	capn_write1(p.p, 1, s->override != 0);
	capn_write32(p.p, 4, capn_from_f32(s->speedOverride));
	capn_write32(p.p, 8, capn_from_f32(s->accelOverride));
}
void cereal_get_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl *s, cereal_CarControl_CruiseControl_list l, int i) {
	cereal_CarControl_CruiseControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarControl_CruiseControl(s, p);
}
void cereal_set_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl *s, cereal_CarControl_CruiseControl_list l, int i) {
	cereal_CarControl_CruiseControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarControl_CruiseControl(s, p);
}

cereal_CarControl_HUDControl_ptr cereal_new_CarControl_HUDControl(struct capn_segment *s) {
	cereal_CarControl_HUDControl_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarControl_HUDControl_list cereal_new_CarControl_HUDControl_list(struct capn_segment *s, int len) {
	cereal_CarControl_HUDControl_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarControl_HUDControl(struct cereal_CarControl_HUDControl *s, cereal_CarControl_HUDControl_ptr p) {
	capn_resolve(&p.p);
	s->speedVisible = (capn_read8(p.p, 0) & 1) != 0;
	s->setSpeed = capn_to_f32(capn_read32(p.p, 4));
	s->lanesVisible = (capn_read8(p.p, 0) & 2) != 0;
	s->leadVisible = (capn_read8(p.p, 0) & 4) != 0;
	s->visualAlert = (enum cereal_CarControl_HUDControl_VisualAlert)(int) capn_read16(p.p, 2);
	s->audibleAlert = (enum cereal_CarControl_HUDControl_AudibleAlert)(int) capn_read16(p.p, 8);
}
void cereal_write_CarControl_HUDControl(const struct cereal_CarControl_HUDControl *s, cereal_CarControl_HUDControl_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->speedVisible != 0);
	capn_write32(p.p, 4, capn_from_f32(s->setSpeed));
	capn_write1(p.p, 1, s->lanesVisible != 0);
	capn_write1(p.p, 2, s->leadVisible != 0);
	capn_write16(p.p, 2, (uint16_t) (s->visualAlert));
	capn_write16(p.p, 8, (uint16_t) (s->audibleAlert));
}
void cereal_get_CarControl_HUDControl(struct cereal_CarControl_HUDControl *s, cereal_CarControl_HUDControl_list l, int i) {
	cereal_CarControl_HUDControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarControl_HUDControl(s, p);
}
void cereal_set_CarControl_HUDControl(const struct cereal_CarControl_HUDControl *s, cereal_CarControl_HUDControl_list l, int i) {
	cereal_CarControl_HUDControl_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarControl_HUDControl(s, p);
}
