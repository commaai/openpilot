#include "car.capnp.h"
/* AUTO GENERATED - DO NOT EDIT */
static const capn_text capn_val0 = {0,""};

cereal_CarEvent_ptr cereal_new_CarEvent(struct capn_segment *s) {
	cereal_CarEvent_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_CarEvent_list cereal_new_CarEvent_list(struct capn_segment *s, int len) {
	cereal_CarEvent_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_CarEvent(struct cereal_CarEvent *s, cereal_CarEvent_ptr p) {
	capn_resolve(&p.p);
	s->name = (enum cereal_CarEvent_EventName)(int) capn_read16(p.p, 0);
	s->enable = (capn_read8(p.p, 2) & 1) != 0;
	s->noEntry = (capn_read8(p.p, 2) & 2) != 0;
	s->warning = (capn_read8(p.p, 2) & 4) != 0;
	s->userDisable = (capn_read8(p.p, 2) & 8) != 0;
	s->softDisable = (capn_read8(p.p, 2) & 16) != 0;
	s->immediateDisable = (capn_read8(p.p, 2) & 32) != 0;
	s->preEnable = (capn_read8(p.p, 2) & 64) != 0;
	s->permanent = (capn_read8(p.p, 2) & 128) != 0;
}
void cereal_write_CarEvent(const struct cereal_CarEvent *s, cereal_CarEvent_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, (uint16_t) (s->name));
	capn_write1(p.p, 16, s->enable != 0);
	capn_write1(p.p, 17, s->noEntry != 0);
	capn_write1(p.p, 18, s->warning != 0);
	capn_write1(p.p, 19, s->userDisable != 0);
	capn_write1(p.p, 20, s->softDisable != 0);
	capn_write1(p.p, 21, s->immediateDisable != 0);
	capn_write1(p.p, 22, s->preEnable != 0);
	capn_write1(p.p, 23, s->permanent != 0);
}
void cereal_get_CarEvent(struct cereal_CarEvent *s, cereal_CarEvent_list l, int i) {
	cereal_CarEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarEvent(s, p);
}
void cereal_set_CarEvent(const struct cereal_CarEvent *s, cereal_CarEvent_list l, int i) {
	cereal_CarEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarEvent(s, p);
}

cereal_CarState_ptr cereal_new_CarState(struct capn_segment *s) {
	cereal_CarState_ptr p;
	p.p = capn_new_struct(s, 40, 6);
	return p;
}
cereal_CarState_list cereal_new_CarState_list(struct capn_segment *s, int len) {
	cereal_CarState_list p;
	p.p = capn_new_list(s, len, 40, 6);
	return p;
}
void cereal_read_CarState(struct cereal_CarState *s, cereal_CarState_ptr p) {
	capn_resolve(&p.p);
	s->errorsDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->events.p = capn_getp(p.p, 5, 0);
	s->vEgo = capn_to_f32(capn_read32(p.p, 0));
	s->aEgo = capn_to_f32(capn_read32(p.p, 28));
	s->vEgoRaw = capn_to_f32(capn_read32(p.p, 32));
	s->yawRate = capn_to_f32(capn_read32(p.p, 36));
	s->standstill = (capn_read8(p.p, 8) & 8) != 0;
	s->wheelSpeeds.p = capn_getp(p.p, 1, 0);
	s->gas = capn_to_f32(capn_read32(p.p, 4));
	s->gasPressed = (capn_read8(p.p, 8) & 1) != 0;
	s->brake = capn_to_f32(capn_read32(p.p, 12));
	s->brakePressed = (capn_read8(p.p, 8) & 2) != 0;
	s->brakeLights = (capn_read8(p.p, 8) & 16) != 0;
	s->steeringAngle = capn_to_f32(capn_read32(p.p, 16));
	s->steeringRate = capn_to_f32(capn_read32(p.p, 24));
	s->steeringTorque = capn_to_f32(capn_read32(p.p, 20));
	s->steeringPressed = (capn_read8(p.p, 8) & 4) != 0;
	s->cruiseState.p = capn_getp(p.p, 2, 0);
	s->gearShifter = (enum cereal_CarState_GearShifter)(int) capn_read16(p.p, 10);
	s->buttonEvents.p = capn_getp(p.p, 3, 0);
	s->leftBlinker = (capn_read8(p.p, 8) & 32) != 0;
	s->rightBlinker = (capn_read8(p.p, 8) & 64) != 0;
	s->genericToggle = (capn_read8(p.p, 8) & 128) != 0;
	s->doorOpen = (capn_read8(p.p, 9) & 1) != 0;
	s->seatbeltUnlatched = (capn_read8(p.p, 9) & 2) != 0;
	s->canValid = (capn_read8(p.p, 9) & 4) != 0;
	s->canMonoTimes.p = capn_getp(p.p, 4, 0);
}
void cereal_write_CarState(const struct cereal_CarState *s, cereal_CarState_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->errorsDEPRECATED.p);
	capn_setp(p.p, 5, s->events.p);
	capn_write32(p.p, 0, capn_from_f32(s->vEgo));
	capn_write32(p.p, 28, capn_from_f32(s->aEgo));
	capn_write32(p.p, 32, capn_from_f32(s->vEgoRaw));
	capn_write32(p.p, 36, capn_from_f32(s->yawRate));
	capn_write1(p.p, 67, s->standstill != 0);
	capn_setp(p.p, 1, s->wheelSpeeds.p);
	capn_write32(p.p, 4, capn_from_f32(s->gas));
	capn_write1(p.p, 64, s->gasPressed != 0);
	capn_write32(p.p, 12, capn_from_f32(s->brake));
	capn_write1(p.p, 65, s->brakePressed != 0);
	capn_write1(p.p, 68, s->brakeLights != 0);
	capn_write32(p.p, 16, capn_from_f32(s->steeringAngle));
	capn_write32(p.p, 24, capn_from_f32(s->steeringRate));
	capn_write32(p.p, 20, capn_from_f32(s->steeringTorque));
	capn_write1(p.p, 66, s->steeringPressed != 0);
	capn_setp(p.p, 2, s->cruiseState.p);
	capn_write16(p.p, 10, (uint16_t) (s->gearShifter));
	capn_setp(p.p, 3, s->buttonEvents.p);
	capn_write1(p.p, 69, s->leftBlinker != 0);
	capn_write1(p.p, 70, s->rightBlinker != 0);
	capn_write1(p.p, 71, s->genericToggle != 0);
	capn_write1(p.p, 72, s->doorOpen != 0);
	capn_write1(p.p, 73, s->seatbeltUnlatched != 0);
	capn_write1(p.p, 74, s->canValid != 0);
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
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarState_CruiseState_list cereal_new_CarState_CruiseState_list(struct capn_segment *s, int len) {
	cereal_CarState_CruiseState_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarState_CruiseState(struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_ptr p) {
	capn_resolve(&p.p);
	s->enabled = (capn_read8(p.p, 0) & 1) != 0;
	s->speed = capn_to_f32(capn_read32(p.p, 4));
	s->available = (capn_read8(p.p, 0) & 2) != 0;
	s->speedOffset = capn_to_f32(capn_read32(p.p, 8));
	s->standstill = (capn_read8(p.p, 0) & 4) != 0;
}
void cereal_write_CarState_CruiseState(const struct cereal_CarState_CruiseState *s, cereal_CarState_CruiseState_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->enabled != 0);
	capn_write32(p.p, 4, capn_from_f32(s->speed));
	capn_write1(p.p, 1, s->available != 0);
	capn_write32(p.p, 8, capn_from_f32(s->speedOffset));
	capn_write1(p.p, 2, s->standstill != 0);
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

cereal_RadarData_ptr cereal_new_RadarData(struct capn_segment *s) {
	cereal_RadarData_ptr p;
	p.p = capn_new_struct(s, 0, 3);
	return p;
}
cereal_RadarData_list cereal_new_RadarData_list(struct capn_segment *s, int len) {
	cereal_RadarData_list p;
	p.p = capn_new_list(s, len, 0, 3);
	return p;
}
void cereal_read_RadarData(struct cereal_RadarData *s, cereal_RadarData_ptr p) {
	capn_resolve(&p.p);
	s->errors.p = capn_getp(p.p, 0, 0);
	s->points.p = capn_getp(p.p, 1, 0);
	s->canMonoTimes.p = capn_getp(p.p, 2, 0);
}
void cereal_write_RadarData(const struct cereal_RadarData *s, cereal_RadarData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->errors.p);
	capn_setp(p.p, 1, s->points.p);
	capn_setp(p.p, 2, s->canMonoTimes.p);
}
void cereal_get_RadarData(struct cereal_RadarData *s, cereal_RadarData_list l, int i) {
	cereal_RadarData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_RadarData(s, p);
}
void cereal_set_RadarData(const struct cereal_RadarData *s, cereal_RadarData_list l, int i) {
	cereal_RadarData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_RadarData(s, p);
}

cereal_RadarData_RadarPoint_ptr cereal_new_RadarData_RadarPoint(struct capn_segment *s) {
	cereal_RadarData_RadarPoint_ptr p;
	p.p = capn_new_struct(s, 32, 0);
	return p;
}
cereal_RadarData_RadarPoint_list cereal_new_RadarData_RadarPoint_list(struct capn_segment *s, int len) {
	cereal_RadarData_RadarPoint_list p;
	p.p = capn_new_list(s, len, 32, 0);
	return p;
}
void cereal_read_RadarData_RadarPoint(struct cereal_RadarData_RadarPoint *s, cereal_RadarData_RadarPoint_ptr p) {
	capn_resolve(&p.p);
	s->trackId = capn_read64(p.p, 0);
	s->dRel = capn_to_f32(capn_read32(p.p, 8));
	s->yRel = capn_to_f32(capn_read32(p.p, 12));
	s->vRel = capn_to_f32(capn_read32(p.p, 16));
	s->aRel = capn_to_f32(capn_read32(p.p, 20));
	s->yvRel = capn_to_f32(capn_read32(p.p, 24));
	s->measured = (capn_read8(p.p, 28) & 1) != 0;
}
void cereal_write_RadarData_RadarPoint(const struct cereal_RadarData_RadarPoint *s, cereal_RadarData_RadarPoint_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->trackId);
	capn_write32(p.p, 8, capn_from_f32(s->dRel));
	capn_write32(p.p, 12, capn_from_f32(s->yRel));
	capn_write32(p.p, 16, capn_from_f32(s->vRel));
	capn_write32(p.p, 20, capn_from_f32(s->aRel));
	capn_write32(p.p, 24, capn_from_f32(s->yvRel));
	capn_write1(p.p, 224, s->measured != 0);
}
void cereal_get_RadarData_RadarPoint(struct cereal_RadarData_RadarPoint *s, cereal_RadarData_RadarPoint_list l, int i) {
	cereal_RadarData_RadarPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_RadarData_RadarPoint(s, p);
}
void cereal_set_RadarData_RadarPoint(const struct cereal_RadarData_RadarPoint *s, cereal_RadarData_RadarPoint_list l, int i) {
	cereal_RadarData_RadarPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_RadarData_RadarPoint(s, p);
}

cereal_CarControl_ptr cereal_new_CarControl(struct capn_segment *s) {
	cereal_CarControl_ptr p;
	p.p = capn_new_struct(s, 16, 3);
	return p;
}
cereal_CarControl_list cereal_new_CarControl_list(struct capn_segment *s, int len) {
	cereal_CarControl_list p;
	p.p = capn_new_list(s, len, 16, 3);
	return p;
}
void cereal_read_CarControl(struct cereal_CarControl *s, cereal_CarControl_ptr p) {
	capn_resolve(&p.p);
	s->enabled = (capn_read8(p.p, 0) & 1) != 0;
	s->active = (capn_read8(p.p, 0) & 2) != 0;
	s->gasDEPRECATED = capn_to_f32(capn_read32(p.p, 4));
	s->brakeDEPRECATED = capn_to_f32(capn_read32(p.p, 8));
	s->steeringTorqueDEPRECATED = capn_to_f32(capn_read32(p.p, 12));
	s->actuators.p = capn_getp(p.p, 2, 0);
	s->cruiseControl.p = capn_getp(p.p, 0, 0);
	s->hudControl.p = capn_getp(p.p, 1, 0);
}
void cereal_write_CarControl(const struct cereal_CarControl *s, cereal_CarControl_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->enabled != 0);
	capn_write1(p.p, 1, s->active != 0);
	capn_write32(p.p, 4, capn_from_f32(s->gasDEPRECATED));
	capn_write32(p.p, 8, capn_from_f32(s->brakeDEPRECATED));
	capn_write32(p.p, 12, capn_from_f32(s->steeringTorqueDEPRECATED));
	capn_setp(p.p, 2, s->actuators.p);
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

cereal_CarControl_Actuators_ptr cereal_new_CarControl_Actuators(struct capn_segment *s) {
	cereal_CarControl_Actuators_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarControl_Actuators_list cereal_new_CarControl_Actuators_list(struct capn_segment *s, int len) {
	cereal_CarControl_Actuators_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarControl_Actuators(struct cereal_CarControl_Actuators *s, cereal_CarControl_Actuators_ptr p) {
	capn_resolve(&p.p);
	s->gas = capn_to_f32(capn_read32(p.p, 0));
	s->brake = capn_to_f32(capn_read32(p.p, 4));
	s->steer = capn_to_f32(capn_read32(p.p, 8));
	s->steerAngle = capn_to_f32(capn_read32(p.p, 12));
}
void cereal_write_CarControl_Actuators(const struct cereal_CarControl_Actuators *s, cereal_CarControl_Actuators_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->gas));
	capn_write32(p.p, 4, capn_from_f32(s->brake));
	capn_write32(p.p, 8, capn_from_f32(s->steer));
	capn_write32(p.p, 12, capn_from_f32(s->steerAngle));
}
void cereal_get_CarControl_Actuators(struct cereal_CarControl_Actuators *s, cereal_CarControl_Actuators_list l, int i) {
	cereal_CarControl_Actuators_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarControl_Actuators(s, p);
}
void cereal_set_CarControl_Actuators(const struct cereal_CarControl_Actuators *s, cereal_CarControl_Actuators_list l, int i) {
	cereal_CarControl_Actuators_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarControl_Actuators(s, p);
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
	s->rightLaneVisible = (capn_read8(p.p, 0) & 8) != 0;
	s->leftLaneVisible = (capn_read8(p.p, 0) & 16) != 0;
	s->rightLaneDepart = (capn_read8(p.p, 0) & 32) != 0;
	s->leftLaneDepart = (capn_read8(p.p, 0) & 64) != 0;
}
void cereal_write_CarControl_HUDControl(const struct cereal_CarControl_HUDControl *s, cereal_CarControl_HUDControl_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->speedVisible != 0);
	capn_write32(p.p, 4, capn_from_f32(s->setSpeed));
	capn_write1(p.p, 1, s->lanesVisible != 0);
	capn_write1(p.p, 2, s->leadVisible != 0);
	capn_write16(p.p, 2, (uint16_t) (s->visualAlert));
	capn_write16(p.p, 8, (uint16_t) (s->audibleAlert));
	capn_write1(p.p, 3, s->rightLaneVisible != 0);
	capn_write1(p.p, 4, s->leftLaneVisible != 0);
	capn_write1(p.p, 5, s->rightLaneDepart != 0);
	capn_write1(p.p, 6, s->leftLaneDepart != 0);
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

cereal_CarParams_ptr cereal_new_CarParams(struct capn_segment *s) {
	cereal_CarParams_ptr p;
	p.p = capn_new_struct(s, 72, 11);
	return p;
}
cereal_CarParams_list cereal_new_CarParams_list(struct capn_segment *s, int len) {
	cereal_CarParams_list p;
	p.p = capn_new_list(s, len, 72, 11);
	return p;
}
void cereal_read_CarParams(struct cereal_CarParams *s, cereal_CarParams_ptr p) {
	capn_resolve(&p.p);
	s->carName = capn_get_text(p.p, 0, capn_val0);
	s->carFingerprint = capn_get_text(p.p, 1, capn_val0);
	s->enableGasInterceptor = (capn_read8(p.p, 0) & 1) != 0;
	s->enableCruise = (capn_read8(p.p, 0) & 2) != 0;
	s->enableCamera = (capn_read8(p.p, 0) & 4) != 0;
	s->enableDsu = (capn_read8(p.p, 0) & 8) != 0;
	s->enableApgs = (capn_read8(p.p, 0) & 16) != 0;
	s->minEnableSpeed = capn_to_f32(capn_read32(p.p, 4));
	s->minSteerSpeed = capn_to_f32(capn_read32(p.p, 8));
	s->safetyModel = (enum cereal_CarParams_SafetyModel)(int) capn_read16(p.p, 2);
	s->safetyParam = (int16_t) ((int16_t)capn_read16(p.p, 12));
	s->steerMaxBP.p = capn_getp(p.p, 2, 0);
	s->steerMaxV.p = capn_getp(p.p, 3, 0);
	s->gasMaxBP.p = capn_getp(p.p, 4, 0);
	s->gasMaxV.p = capn_getp(p.p, 5, 0);
	s->brakeMaxBP.p = capn_getp(p.p, 6, 0);
	s->brakeMaxV.p = capn_getp(p.p, 7, 0);
	s->mass = capn_to_f32(capn_read32(p.p, 16));
	s->wheelbase = capn_to_f32(capn_read32(p.p, 20));
	s->centerToFront = capn_to_f32(capn_read32(p.p, 24));
	s->steerRatio = capn_to_f32(capn_read32(p.p, 28));
	s->steerRatioRear = capn_to_f32(capn_read32(p.p, 32));
	s->rotationalInertia = capn_to_f32(capn_read32(p.p, 36));
	s->tireStiffnessFront = capn_to_f32(capn_read32(p.p, 40));
	s->tireStiffnessRear = capn_to_f32(capn_read32(p.p, 44));
	s->longitudinalTuning.p = capn_getp(p.p, 8, 0);
	s->lateralTuning_which = (enum cereal_CarParams_lateralTuning_which)(int) capn_read16(p.p, 14);
	switch (s->lateralTuning_which) {
	case cereal_CarParams_lateralTuning_pid:
	case cereal_CarParams_lateralTuning_indi:
		s->lateralTuning.indi.p = capn_getp(p.p, 9, 0);
		break;
	default:
		break;
	}
	s->steerLimitAlert = (capn_read8(p.p, 0) & 32) != 0;
	s->vEgoStopping = capn_to_f32(capn_read32(p.p, 48));
	s->directAccelControl = (capn_read8(p.p, 0) & 64) != 0;
	s->stoppingControl = (capn_read8(p.p, 0) & 128) != 0;
	s->startAccel = capn_to_f32(capn_read32(p.p, 52));
	s->steerRateCost = capn_to_f32(capn_read32(p.p, 56));
	s->steerControlType = (enum cereal_CarParams_SteerControlType)(int) capn_read16(p.p, 60);
	s->radarOffCan = (capn_read8(p.p, 1) & 1) != 0;
	s->steerActuatorDelay = capn_to_f32(capn_read32(p.p, 64));
	s->openpilotLongitudinalControl = (capn_read8(p.p, 1) & 2) != 0;
	s->carVin = capn_get_text(p.p, 10, capn_val0);
}
void cereal_write_CarParams(const struct cereal_CarParams *s, cereal_CarParams_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->carName);
	capn_set_text(p.p, 1, s->carFingerprint);
	capn_write1(p.p, 0, s->enableGasInterceptor != 0);
	capn_write1(p.p, 1, s->enableCruise != 0);
	capn_write1(p.p, 2, s->enableCamera != 0);
	capn_write1(p.p, 3, s->enableDsu != 0);
	capn_write1(p.p, 4, s->enableApgs != 0);
	capn_write32(p.p, 4, capn_from_f32(s->minEnableSpeed));
	capn_write32(p.p, 8, capn_from_f32(s->minSteerSpeed));
	capn_write16(p.p, 2, (uint16_t) (s->safetyModel));
	capn_write16(p.p, 12, (uint16_t) (s->safetyParam));
	capn_setp(p.p, 2, s->steerMaxBP.p);
	capn_setp(p.p, 3, s->steerMaxV.p);
	capn_setp(p.p, 4, s->gasMaxBP.p);
	capn_setp(p.p, 5, s->gasMaxV.p);
	capn_setp(p.p, 6, s->brakeMaxBP.p);
	capn_setp(p.p, 7, s->brakeMaxV.p);
	capn_write32(p.p, 16, capn_from_f32(s->mass));
	capn_write32(p.p, 20, capn_from_f32(s->wheelbase));
	capn_write32(p.p, 24, capn_from_f32(s->centerToFront));
	capn_write32(p.p, 28, capn_from_f32(s->steerRatio));
	capn_write32(p.p, 32, capn_from_f32(s->steerRatioRear));
	capn_write32(p.p, 36, capn_from_f32(s->rotationalInertia));
	capn_write32(p.p, 40, capn_from_f32(s->tireStiffnessFront));
	capn_write32(p.p, 44, capn_from_f32(s->tireStiffnessRear));
	capn_setp(p.p, 8, s->longitudinalTuning.p);
	capn_write16(p.p, 14, s->lateralTuning_which);
	switch (s->lateralTuning_which) {
	case cereal_CarParams_lateralTuning_pid:
	case cereal_CarParams_lateralTuning_indi:
		capn_setp(p.p, 9, s->lateralTuning.indi.p);
		break;
	default:
		break;
	}
	capn_write1(p.p, 5, s->steerLimitAlert != 0);
	capn_write32(p.p, 48, capn_from_f32(s->vEgoStopping));
	capn_write1(p.p, 6, s->directAccelControl != 0);
	capn_write1(p.p, 7, s->stoppingControl != 0);
	capn_write32(p.p, 52, capn_from_f32(s->startAccel));
	capn_write32(p.p, 56, capn_from_f32(s->steerRateCost));
	capn_write16(p.p, 60, (uint16_t) (s->steerControlType));
	capn_write1(p.p, 8, s->radarOffCan != 0);
	capn_write32(p.p, 64, capn_from_f32(s->steerActuatorDelay));
	capn_write1(p.p, 9, s->openpilotLongitudinalControl != 0);
	capn_set_text(p.p, 10, s->carVin);
}
void cereal_get_CarParams(struct cereal_CarParams *s, cereal_CarParams_list l, int i) {
	cereal_CarParams_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarParams(s, p);
}
void cereal_set_CarParams(const struct cereal_CarParams *s, cereal_CarParams_list l, int i) {
	cereal_CarParams_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarParams(s, p);
}

cereal_CarParams_LateralPIDTuning_ptr cereal_new_CarParams_LateralPIDTuning(struct capn_segment *s) {
	cereal_CarParams_LateralPIDTuning_ptr p;
	p.p = capn_new_struct(s, 8, 4);
	return p;
}
cereal_CarParams_LateralPIDTuning_list cereal_new_CarParams_LateralPIDTuning_list(struct capn_segment *s, int len) {
	cereal_CarParams_LateralPIDTuning_list p;
	p.p = capn_new_list(s, len, 8, 4);
	return p;
}
void cereal_read_CarParams_LateralPIDTuning(struct cereal_CarParams_LateralPIDTuning *s, cereal_CarParams_LateralPIDTuning_ptr p) {
	capn_resolve(&p.p);
	s->kpBP.p = capn_getp(p.p, 0, 0);
	s->kpV.p = capn_getp(p.p, 1, 0);
	s->kiBP.p = capn_getp(p.p, 2, 0);
	s->kiV.p = capn_getp(p.p, 3, 0);
	s->kf = capn_to_f32(capn_read32(p.p, 0));
}
void cereal_write_CarParams_LateralPIDTuning(const struct cereal_CarParams_LateralPIDTuning *s, cereal_CarParams_LateralPIDTuning_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->kpBP.p);
	capn_setp(p.p, 1, s->kpV.p);
	capn_setp(p.p, 2, s->kiBP.p);
	capn_setp(p.p, 3, s->kiV.p);
	capn_write32(p.p, 0, capn_from_f32(s->kf));
}
void cereal_get_CarParams_LateralPIDTuning(struct cereal_CarParams_LateralPIDTuning *s, cereal_CarParams_LateralPIDTuning_list l, int i) {
	cereal_CarParams_LateralPIDTuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarParams_LateralPIDTuning(s, p);
}
void cereal_set_CarParams_LateralPIDTuning(const struct cereal_CarParams_LateralPIDTuning *s, cereal_CarParams_LateralPIDTuning_list l, int i) {
	cereal_CarParams_LateralPIDTuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarParams_LateralPIDTuning(s, p);
}

cereal_CarParams_LongitudinalPIDTuning_ptr cereal_new_CarParams_LongitudinalPIDTuning(struct capn_segment *s) {
	cereal_CarParams_LongitudinalPIDTuning_ptr p;
	p.p = capn_new_struct(s, 0, 6);
	return p;
}
cereal_CarParams_LongitudinalPIDTuning_list cereal_new_CarParams_LongitudinalPIDTuning_list(struct capn_segment *s, int len) {
	cereal_CarParams_LongitudinalPIDTuning_list p;
	p.p = capn_new_list(s, len, 0, 6);
	return p;
}
void cereal_read_CarParams_LongitudinalPIDTuning(struct cereal_CarParams_LongitudinalPIDTuning *s, cereal_CarParams_LongitudinalPIDTuning_ptr p) {
	capn_resolve(&p.p);
	s->kpBP.p = capn_getp(p.p, 0, 0);
	s->kpV.p = capn_getp(p.p, 1, 0);
	s->kiBP.p = capn_getp(p.p, 2, 0);
	s->kiV.p = capn_getp(p.p, 3, 0);
	s->deadzoneBP.p = capn_getp(p.p, 4, 0);
	s->deadzoneV.p = capn_getp(p.p, 5, 0);
}
void cereal_write_CarParams_LongitudinalPIDTuning(const struct cereal_CarParams_LongitudinalPIDTuning *s, cereal_CarParams_LongitudinalPIDTuning_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->kpBP.p);
	capn_setp(p.p, 1, s->kpV.p);
	capn_setp(p.p, 2, s->kiBP.p);
	capn_setp(p.p, 3, s->kiV.p);
	capn_setp(p.p, 4, s->deadzoneBP.p);
	capn_setp(p.p, 5, s->deadzoneV.p);
}
void cereal_get_CarParams_LongitudinalPIDTuning(struct cereal_CarParams_LongitudinalPIDTuning *s, cereal_CarParams_LongitudinalPIDTuning_list l, int i) {
	cereal_CarParams_LongitudinalPIDTuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarParams_LongitudinalPIDTuning(s, p);
}
void cereal_set_CarParams_LongitudinalPIDTuning(const struct cereal_CarParams_LongitudinalPIDTuning *s, cereal_CarParams_LongitudinalPIDTuning_list l, int i) {
	cereal_CarParams_LongitudinalPIDTuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarParams_LongitudinalPIDTuning(s, p);
}

cereal_CarParams_LateralINDITuning_ptr cereal_new_CarParams_LateralINDITuning(struct capn_segment *s) {
	cereal_CarParams_LateralINDITuning_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_CarParams_LateralINDITuning_list cereal_new_CarParams_LateralINDITuning_list(struct capn_segment *s, int len) {
	cereal_CarParams_LateralINDITuning_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_CarParams_LateralINDITuning(struct cereal_CarParams_LateralINDITuning *s, cereal_CarParams_LateralINDITuning_ptr p) {
	capn_resolve(&p.p);
	s->outerLoopGain = capn_to_f32(capn_read32(p.p, 0));
	s->innerLoopGain = capn_to_f32(capn_read32(p.p, 4));
	s->timeConstant = capn_to_f32(capn_read32(p.p, 8));
	s->actuatorEffectiveness = capn_to_f32(capn_read32(p.p, 12));
}
void cereal_write_CarParams_LateralINDITuning(const struct cereal_CarParams_LateralINDITuning *s, cereal_CarParams_LateralINDITuning_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->outerLoopGain));
	capn_write32(p.p, 4, capn_from_f32(s->innerLoopGain));
	capn_write32(p.p, 8, capn_from_f32(s->timeConstant));
	capn_write32(p.p, 12, capn_from_f32(s->actuatorEffectiveness));
}
void cereal_get_CarParams_LateralINDITuning(struct cereal_CarParams_LateralINDITuning *s, cereal_CarParams_LateralINDITuning_list l, int i) {
	cereal_CarParams_LateralINDITuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CarParams_LateralINDITuning(s, p);
}
void cereal_set_CarParams_LateralINDITuning(const struct cereal_CarParams_LateralINDITuning *s, cereal_CarParams_LateralINDITuning_list l, int i) {
	cereal_CarParams_LateralINDITuning_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CarParams_LateralINDITuning(s, p);
}
