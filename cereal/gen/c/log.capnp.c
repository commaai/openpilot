#include "log.capnp.h"
/* AUTO GENERATED - DO NOT EDIT */
static const capn_text capn_val0 = {0,""};
int32_t cereal_logVersion = 1;

cereal_InitData_ptr cereal_new_InitData(struct capn_segment *s) {
	cereal_InitData_ptr p;
	p.p = capn_new_struct(s, 0, 3);
	return p;
}
cereal_InitData_list cereal_new_InitData_list(struct capn_segment *s, int len) {
	cereal_InitData_list p;
	p.p = capn_new_list(s, len, 0, 3);
	return p;
}
void cereal_read_InitData(struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	s->kernelArgs = capn_getp(p.p, 0, 0);
	s->gctx = capn_get_text(p.p, 1, capn_val0);
	s->dongleId = capn_get_text(p.p, 2, capn_val0);
}
void cereal_write_InitData(const struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->kernelArgs);
	capn_set_text(p.p, 1, s->gctx);
	capn_set_text(p.p, 2, s->dongleId);
}
void cereal_get_InitData(struct cereal_InitData *s, cereal_InitData_list l, int i) {
	cereal_InitData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData(s, p);
}
void cereal_set_InitData(const struct cereal_InitData *s, cereal_InitData_list l, int i) {
	cereal_InitData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData(s, p);
}

cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment *s) {
	cereal_FrameData_ptr p;
	p.p = capn_new_struct(s, 32, 1);
	return p;
}
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment *s, int len) {
	cereal_FrameData_list p;
	p.p = capn_new_list(s, len, 32, 1);
	return p;
}
void cereal_read_FrameData(struct cereal_FrameData *s, cereal_FrameData_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->encodeId = capn_read32(p.p, 4);
	s->timestampEof = capn_read64(p.p, 8);
	s->frameLength = (int32_t) ((int32_t)capn_read32(p.p, 16));
	s->integLines = (int32_t) ((int32_t)capn_read32(p.p, 20));
	s->globalGain = (int32_t) ((int32_t)capn_read32(p.p, 24));
	s->image = capn_get_data(p.p, 0);
}
void cereal_write_FrameData(const struct cereal_FrameData *s, cereal_FrameData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write32(p.p, 4, s->encodeId);
	capn_write64(p.p, 8, s->timestampEof);
	capn_write32(p.p, 16, (uint32_t) (s->frameLength));
	capn_write32(p.p, 20, (uint32_t) (s->integLines));
	capn_write32(p.p, 24, (uint32_t) (s->globalGain));
	capn_setp(p.p, 0, s->image.p);
}
void cereal_get_FrameData(struct cereal_FrameData *s, cereal_FrameData_list l, int i) {
	cereal_FrameData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_FrameData(s, p);
}
void cereal_set_FrameData(const struct cereal_FrameData *s, cereal_FrameData_list l, int i) {
	cereal_FrameData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_FrameData(s, p);
}

cereal_GPSNMEAData_ptr cereal_new_GPSNMEAData(struct capn_segment *s) {
	cereal_GPSNMEAData_ptr p;
	p.p = capn_new_struct(s, 16, 1);
	return p;
}
cereal_GPSNMEAData_list cereal_new_GPSNMEAData_list(struct capn_segment *s, int len) {
	cereal_GPSNMEAData_list p;
	p.p = capn_new_list(s, len, 16, 1);
	return p;
}
void cereal_read_GPSNMEAData(struct cereal_GPSNMEAData *s, cereal_GPSNMEAData_ptr p) {
	capn_resolve(&p.p);
	s->timestamp = (int64_t) ((int64_t)(capn_read64(p.p, 0)));
	s->localWallTime = capn_read64(p.p, 8);
	s->nmea = capn_get_text(p.p, 0, capn_val0);
}
void cereal_write_GPSNMEAData(const struct cereal_GPSNMEAData *s, cereal_GPSNMEAData_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, (uint64_t) (s->timestamp));
	capn_write64(p.p, 8, s->localWallTime);
	capn_set_text(p.p, 0, s->nmea);
}
void cereal_get_GPSNMEAData(struct cereal_GPSNMEAData *s, cereal_GPSNMEAData_list l, int i) {
	cereal_GPSNMEAData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_GPSNMEAData(s, p);
}
void cereal_set_GPSNMEAData(const struct cereal_GPSNMEAData *s, cereal_GPSNMEAData_list l, int i) {
	cereal_GPSNMEAData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_GPSNMEAData(s, p);
}

cereal_SensorEventData_ptr cereal_new_SensorEventData(struct capn_segment *s) {
	cereal_SensorEventData_ptr p;
	p.p = capn_new_struct(s, 24, 1);
	return p;
}
cereal_SensorEventData_list cereal_new_SensorEventData_list(struct capn_segment *s, int len) {
	cereal_SensorEventData_list p;
	p.p = capn_new_list(s, len, 24, 1);
	return p;
}
void cereal_read_SensorEventData(struct cereal_SensorEventData *s, cereal_SensorEventData_ptr p) {
	capn_resolve(&p.p);
	s->version = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->sensor = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->type = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->timestamp = (int64_t) ((int64_t)(capn_read64(p.p, 16)));
	s->which = (enum cereal_SensorEventData_which)(int) capn_read16(p.p, 12);
	switch (s->which) {
	case cereal_SensorEventData_acceleration:
	case cereal_SensorEventData_magnetic:
	case cereal_SensorEventData_orientation:
	case cereal_SensorEventData_gyro:
		s->gyro.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_SensorEventData(const struct cereal_SensorEventData *s, cereal_SensorEventData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->version));
	capn_write32(p.p, 4, (uint32_t) (s->sensor));
	capn_write32(p.p, 8, (uint32_t) (s->type));
	capn_write64(p.p, 16, (uint64_t) (s->timestamp));
	capn_write16(p.p, 12, s->which);
	switch (s->which) {
	case cereal_SensorEventData_acceleration:
	case cereal_SensorEventData_magnetic:
	case cereal_SensorEventData_orientation:
	case cereal_SensorEventData_gyro:
		capn_setp(p.p, 0, s->gyro.p);
		break;
	default:
		break;
	}
}
void cereal_get_SensorEventData(struct cereal_SensorEventData *s, cereal_SensorEventData_list l, int i) {
	cereal_SensorEventData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_SensorEventData(s, p);
}
void cereal_set_SensorEventData(const struct cereal_SensorEventData *s, cereal_SensorEventData_list l, int i) {
	cereal_SensorEventData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_SensorEventData(s, p);
}

cereal_SensorEventData_SensorVec_ptr cereal_new_SensorEventData_SensorVec(struct capn_segment *s) {
	cereal_SensorEventData_SensorVec_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_SensorEventData_SensorVec_list cereal_new_SensorEventData_SensorVec_list(struct capn_segment *s, int len) {
	cereal_SensorEventData_SensorVec_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec *s, cereal_SensorEventData_SensorVec_ptr p) {
	capn_resolve(&p.p);
	s->v.p = capn_getp(p.p, 0, 0);
	s->status = (int8_t) ((int8_t)capn_read8(p.p, 0));
}
void cereal_write_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec *s, cereal_SensorEventData_SensorVec_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->v.p);
	capn_write8(p.p, 0, (uint8_t) (s->status));
}
void cereal_get_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec *s, cereal_SensorEventData_SensorVec_list l, int i) {
	cereal_SensorEventData_SensorVec_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_SensorEventData_SensorVec(s, p);
}
void cereal_set_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec *s, cereal_SensorEventData_SensorVec_list l, int i) {
	cereal_SensorEventData_SensorVec_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_SensorEventData_SensorVec(s, p);
}

cereal_CanData_ptr cereal_new_CanData(struct capn_segment *s) {
	cereal_CanData_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_CanData_list cereal_new_CanData_list(struct capn_segment *s, int len) {
	cereal_CanData_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_CanData(struct cereal_CanData *s, cereal_CanData_ptr p) {
	capn_resolve(&p.p);
	s->address = capn_read32(p.p, 0);
	s->busTime = capn_read16(p.p, 4);
	s->dat = capn_get_data(p.p, 0);
	s->src = (int8_t) ((int8_t)capn_read8(p.p, 6));
}
void cereal_write_CanData(const struct cereal_CanData *s, cereal_CanData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->address);
	capn_write16(p.p, 4, s->busTime);
	capn_setp(p.p, 0, s->dat.p);
	capn_write8(p.p, 6, (uint8_t) (s->src));
}
void cereal_get_CanData(struct cereal_CanData *s, cereal_CanData_list l, int i) {
	cereal_CanData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CanData(s, p);
}
void cereal_set_CanData(const struct cereal_CanData *s, cereal_CanData_list l, int i) {
	cereal_CanData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CanData(s, p);
}

cereal_ThermalData_ptr cereal_new_ThermalData(struct capn_segment *s) {
	cereal_ThermalData_ptr p;
	p.p = capn_new_struct(s, 24, 0);
	return p;
}
cereal_ThermalData_list cereal_new_ThermalData_list(struct capn_segment *s, int len) {
	cereal_ThermalData_list p;
	p.p = capn_new_list(s, len, 24, 0);
	return p;
}
void cereal_read_ThermalData(struct cereal_ThermalData *s, cereal_ThermalData_ptr p) {
	capn_resolve(&p.p);
	s->cpu0 = capn_read16(p.p, 0);
	s->cpu1 = capn_read16(p.p, 2);
	s->cpu2 = capn_read16(p.p, 4);
	s->cpu3 = capn_read16(p.p, 6);
	s->mem = capn_read16(p.p, 8);
	s->gpu = capn_read16(p.p, 10);
	s->bat = capn_read32(p.p, 12);
	s->freeSpace = capn_to_f32(capn_read32(p.p, 16));
}
void cereal_write_ThermalData(const struct cereal_ThermalData *s, cereal_ThermalData_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->cpu0);
	capn_write16(p.p, 2, s->cpu1);
	capn_write16(p.p, 4, s->cpu2);
	capn_write16(p.p, 6, s->cpu3);
	capn_write16(p.p, 8, s->mem);
	capn_write16(p.p, 10, s->gpu);
	capn_write32(p.p, 12, s->bat);
	capn_write32(p.p, 16, capn_from_f32(s->freeSpace));
}
void cereal_get_ThermalData(struct cereal_ThermalData *s, cereal_ThermalData_list l, int i) {
	cereal_ThermalData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ThermalData(s, p);
}
void cereal_set_ThermalData(const struct cereal_ThermalData *s, cereal_ThermalData_list l, int i) {
	cereal_ThermalData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ThermalData(s, p);
}

cereal_HealthData_ptr cereal_new_HealthData(struct capn_segment *s) {
	cereal_HealthData_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_HealthData_list cereal_new_HealthData_list(struct capn_segment *s, int len) {
	cereal_HealthData_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_HealthData(struct cereal_HealthData *s, cereal_HealthData_ptr p) {
	capn_resolve(&p.p);
	s->voltage = capn_read32(p.p, 0);
	s->current = capn_read32(p.p, 4);
	s->started = (capn_read8(p.p, 8) & 1) != 0;
	s->controlsAllowed = (capn_read8(p.p, 8) & 2) != 0;
	s->gasInterceptorDetected = (capn_read8(p.p, 8) & 4) != 0;
	s->startedSignalDetected = (capn_read8(p.p, 8) & 8) != 0;
}
void cereal_write_HealthData(const struct cereal_HealthData *s, cereal_HealthData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->voltage);
	capn_write32(p.p, 4, s->current);
	capn_write1(p.p, 64, s->started != 0);
	capn_write1(p.p, 65, s->controlsAllowed != 0);
	capn_write1(p.p, 66, s->gasInterceptorDetected != 0);
	capn_write1(p.p, 67, s->startedSignalDetected != 0);
}
void cereal_get_HealthData(struct cereal_HealthData *s, cereal_HealthData_list l, int i) {
	cereal_HealthData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_HealthData(s, p);
}
void cereal_set_HealthData(const struct cereal_HealthData *s, cereal_HealthData_list l, int i) {
	cereal_HealthData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_HealthData(s, p);
}

cereal_LiveUI_ptr cereal_new_LiveUI(struct capn_segment *s) {
	cereal_LiveUI_ptr p;
	p.p = capn_new_struct(s, 8, 2);
	return p;
}
cereal_LiveUI_list cereal_new_LiveUI_list(struct capn_segment *s, int len) {
	cereal_LiveUI_list p;
	p.p = capn_new_list(s, len, 8, 2);
	return p;
}
void cereal_read_LiveUI(struct cereal_LiveUI *s, cereal_LiveUI_ptr p) {
	capn_resolve(&p.p);
	s->rearViewCam = (capn_read8(p.p, 0) & 1) != 0;
	s->alertText1 = capn_get_text(p.p, 0, capn_val0);
	s->alertText2 = capn_get_text(p.p, 1, capn_val0);
	s->awarenessStatus = capn_to_f32(capn_read32(p.p, 4));
}
void cereal_write_LiveUI(const struct cereal_LiveUI *s, cereal_LiveUI_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->rearViewCam != 0);
	capn_set_text(p.p, 0, s->alertText1);
	capn_set_text(p.p, 1, s->alertText2);
	capn_write32(p.p, 4, capn_from_f32(s->awarenessStatus));
}
void cereal_get_LiveUI(struct cereal_LiveUI *s, cereal_LiveUI_list l, int i) {
	cereal_LiveUI_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveUI(s, p);
}
void cereal_set_LiveUI(const struct cereal_LiveUI *s, cereal_LiveUI_list l, int i) {
	cereal_LiveUI_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveUI(s, p);
}

cereal_Live20Data_ptr cereal_new_Live20Data(struct capn_segment *s) {
	cereal_Live20Data_ptr p;
	p.p = capn_new_struct(s, 32, 4);
	return p;
}
cereal_Live20Data_list cereal_new_Live20Data_list(struct capn_segment *s, int len) {
	cereal_Live20Data_list p;
	p.p = capn_new_list(s, len, 32, 4);
	return p;
}
void cereal_read_Live20Data(struct cereal_Live20Data *s, cereal_Live20Data_ptr p) {
	capn_resolve(&p.p);
	s->canMonoTimes.p = capn_getp(p.p, 3, 0);
	s->mdMonoTime = capn_read64(p.p, 16);
	s->ftMonoTime = capn_read64(p.p, 24);
	s->warpMatrixDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->angleOffsetDEPRECATED = capn_to_f32(capn_read32(p.p, 0));
	s->calStatusDEPRECATED = (int8_t) ((int8_t)capn_read8(p.p, 4));
	s->calCycleDEPRECATED = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->calPercDEPRECATED = (int8_t) ((int8_t)capn_read8(p.p, 5));
	s->leadOne.p = capn_getp(p.p, 1, 0);
	s->leadTwo.p = capn_getp(p.p, 2, 0);
	s->cumLagMs = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_Live20Data(const struct cereal_Live20Data *s, cereal_Live20Data_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 3, s->canMonoTimes.p);
	capn_write64(p.p, 16, s->mdMonoTime);
	capn_write64(p.p, 24, s->ftMonoTime);
	capn_setp(p.p, 0, s->warpMatrixDEPRECATED.p);
	capn_write32(p.p, 0, capn_from_f32(s->angleOffsetDEPRECATED));
	capn_write8(p.p, 4, (uint8_t) (s->calStatusDEPRECATED));
	capn_write32(p.p, 12, (uint32_t) (s->calCycleDEPRECATED));
	capn_write8(p.p, 5, (uint8_t) (s->calPercDEPRECATED));
	capn_setp(p.p, 1, s->leadOne.p);
	capn_setp(p.p, 2, s->leadTwo.p);
	capn_write32(p.p, 8, capn_from_f32(s->cumLagMs));
}
void cereal_get_Live20Data(struct cereal_Live20Data *s, cereal_Live20Data_list l, int i) {
	cereal_Live20Data_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Live20Data(s, p);
}
void cereal_set_Live20Data(const struct cereal_Live20Data *s, cereal_Live20Data_list l, int i) {
	cereal_Live20Data_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Live20Data(s, p);
}

cereal_Live20Data_LeadData_ptr cereal_new_Live20Data_LeadData(struct capn_segment *s) {
	cereal_Live20Data_LeadData_ptr p;
	p.p = capn_new_struct(s, 48, 0);
	return p;
}
cereal_Live20Data_LeadData_list cereal_new_Live20Data_LeadData_list(struct capn_segment *s, int len) {
	cereal_Live20Data_LeadData_list p;
	p.p = capn_new_list(s, len, 48, 0);
	return p;
}
void cereal_read_Live20Data_LeadData(struct cereal_Live20Data_LeadData *s, cereal_Live20Data_LeadData_ptr p) {
	capn_resolve(&p.p);
	s->dRel = capn_to_f32(capn_read32(p.p, 0));
	s->yRel = capn_to_f32(capn_read32(p.p, 4));
	s->vRel = capn_to_f32(capn_read32(p.p, 8));
	s->aRel = capn_to_f32(capn_read32(p.p, 12));
	s->vLead = capn_to_f32(capn_read32(p.p, 16));
	s->aLead = capn_to_f32(capn_read32(p.p, 20));
	s->dPath = capn_to_f32(capn_read32(p.p, 24));
	s->vLat = capn_to_f32(capn_read32(p.p, 28));
	s->vLeadK = capn_to_f32(capn_read32(p.p, 32));
	s->aLeadK = capn_to_f32(capn_read32(p.p, 36));
	s->fcw = (capn_read8(p.p, 40) & 1) != 0;
	s->status = (capn_read8(p.p, 40) & 2) != 0;
}
void cereal_write_Live20Data_LeadData(const struct cereal_Live20Data_LeadData *s, cereal_Live20Data_LeadData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->dRel));
	capn_write32(p.p, 4, capn_from_f32(s->yRel));
	capn_write32(p.p, 8, capn_from_f32(s->vRel));
	capn_write32(p.p, 12, capn_from_f32(s->aRel));
	capn_write32(p.p, 16, capn_from_f32(s->vLead));
	capn_write32(p.p, 20, capn_from_f32(s->aLead));
	capn_write32(p.p, 24, capn_from_f32(s->dPath));
	capn_write32(p.p, 28, capn_from_f32(s->vLat));
	capn_write32(p.p, 32, capn_from_f32(s->vLeadK));
	capn_write32(p.p, 36, capn_from_f32(s->aLeadK));
	capn_write1(p.p, 320, s->fcw != 0);
	capn_write1(p.p, 321, s->status != 0);
}
void cereal_get_Live20Data_LeadData(struct cereal_Live20Data_LeadData *s, cereal_Live20Data_LeadData_list l, int i) {
	cereal_Live20Data_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Live20Data_LeadData(s, p);
}
void cereal_set_Live20Data_LeadData(const struct cereal_Live20Data_LeadData *s, cereal_Live20Data_LeadData_list l, int i) {
	cereal_Live20Data_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Live20Data_LeadData(s, p);
}

cereal_LiveCalibrationData_ptr cereal_new_LiveCalibrationData(struct capn_segment *s) {
	cereal_LiveCalibrationData_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_LiveCalibrationData_list cereal_new_LiveCalibrationData_list(struct capn_segment *s, int len) {
	cereal_LiveCalibrationData_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_LiveCalibrationData(struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_ptr p) {
	capn_resolve(&p.p);
	s->warpMatrix.p = capn_getp(p.p, 0, 0);
	s->calStatus = (int8_t) ((int8_t)capn_read8(p.p, 0));
	s->calCycle = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->calPerc = (int8_t) ((int8_t)capn_read8(p.p, 1));
}
void cereal_write_LiveCalibrationData(const struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->warpMatrix.p);
	capn_write8(p.p, 0, (uint8_t) (s->calStatus));
	capn_write32(p.p, 4, (uint32_t) (s->calCycle));
	capn_write8(p.p, 1, (uint8_t) (s->calPerc));
}
void cereal_get_LiveCalibrationData(struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_list l, int i) {
	cereal_LiveCalibrationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveCalibrationData(s, p);
}
void cereal_set_LiveCalibrationData(const struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_list l, int i) {
	cereal_LiveCalibrationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveCalibrationData(s, p);
}

cereal_LiveTracks_ptr cereal_new_LiveTracks(struct capn_segment *s) {
	cereal_LiveTracks_ptr p;
	p.p = capn_new_struct(s, 40, 0);
	return p;
}
cereal_LiveTracks_list cereal_new_LiveTracks_list(struct capn_segment *s, int len) {
	cereal_LiveTracks_list p;
	p.p = capn_new_list(s, len, 40, 0);
	return p;
}
void cereal_read_LiveTracks(struct cereal_LiveTracks *s, cereal_LiveTracks_ptr p) {
	capn_resolve(&p.p);
	s->trackId = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->dRel = capn_to_f32(capn_read32(p.p, 4));
	s->yRel = capn_to_f32(capn_read32(p.p, 8));
	s->vRel = capn_to_f32(capn_read32(p.p, 12));
	s->aRel = capn_to_f32(capn_read32(p.p, 16));
	s->timeStamp = capn_to_f32(capn_read32(p.p, 20));
	s->status = capn_to_f32(capn_read32(p.p, 24));
	s->currentTime = capn_to_f32(capn_read32(p.p, 28));
	s->stationary = (capn_read8(p.p, 32) & 1) != 0;
	s->oncoming = (capn_read8(p.p, 32) & 2) != 0;
}
void cereal_write_LiveTracks(const struct cereal_LiveTracks *s, cereal_LiveTracks_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->trackId));
	capn_write32(p.p, 4, capn_from_f32(s->dRel));
	capn_write32(p.p, 8, capn_from_f32(s->yRel));
	capn_write32(p.p, 12, capn_from_f32(s->vRel));
	capn_write32(p.p, 16, capn_from_f32(s->aRel));
	capn_write32(p.p, 20, capn_from_f32(s->timeStamp));
	capn_write32(p.p, 24, capn_from_f32(s->status));
	capn_write32(p.p, 28, capn_from_f32(s->currentTime));
	capn_write1(p.p, 256, s->stationary != 0);
	capn_write1(p.p, 257, s->oncoming != 0);
}
void cereal_get_LiveTracks(struct cereal_LiveTracks *s, cereal_LiveTracks_list l, int i) {
	cereal_LiveTracks_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveTracks(s, p);
}
void cereal_set_LiveTracks(const struct cereal_LiveTracks *s, cereal_LiveTracks_list l, int i) {
	cereal_LiveTracks_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveTracks(s, p);
}

cereal_Live100Data_ptr cereal_new_Live100Data(struct capn_segment *s) {
	cereal_Live100Data_ptr p;
	p.p = capn_new_struct(s, 104, 3);
	return p;
}
cereal_Live100Data_list cereal_new_Live100Data_list(struct capn_segment *s, int len) {
	cereal_Live100Data_list p;
	p.p = capn_new_list(s, len, 104, 3);
	return p;
}
void cereal_read_Live100Data(struct cereal_Live100Data *s, cereal_Live100Data_ptr p) {
	capn_resolve(&p.p);
	s->canMonoTime = capn_read64(p.p, 64);
	s->canMonoTimes.p = capn_getp(p.p, 0, 0);
	s->l20MonoTime = capn_read64(p.p, 72);
	s->mdMonoTime = capn_read64(p.p, 80);
	s->vEgo = capn_to_f32(capn_read32(p.p, 0));
	s->aEgoDEPRECATED = capn_to_f32(capn_read32(p.p, 4));
	s->vPid = capn_to_f32(capn_read32(p.p, 8));
	s->vTargetLead = capn_to_f32(capn_read32(p.p, 12));
	s->upAccelCmd = capn_to_f32(capn_read32(p.p, 16));
	s->uiAccelCmd = capn_to_f32(capn_read32(p.p, 20));
	s->yActual = capn_to_f32(capn_read32(p.p, 24));
	s->yDes = capn_to_f32(capn_read32(p.p, 28));
	s->upSteer = capn_to_f32(capn_read32(p.p, 32));
	s->uiSteer = capn_to_f32(capn_read32(p.p, 36));
	s->aTargetMin = capn_to_f32(capn_read32(p.p, 40));
	s->aTargetMax = capn_to_f32(capn_read32(p.p, 44));
	s->jerkFactor = capn_to_f32(capn_read32(p.p, 48));
	s->angleSteers = capn_to_f32(capn_read32(p.p, 52));
	s->hudLeadDEPRECATED = (int32_t) ((int32_t)capn_read32(p.p, 56));
	s->cumLagMs = capn_to_f32(capn_read32(p.p, 60));
	s->enabled = (capn_read8(p.p, 88) & 1) != 0;
	s->steerOverride = (capn_read8(p.p, 88) & 2) != 0;
	s->vCruise = capn_to_f32(capn_read32(p.p, 92));
	s->rearViewCam = (capn_read8(p.p, 88) & 4) != 0;
	s->alertText1 = capn_get_text(p.p, 1, capn_val0);
	s->alertText2 = capn_get_text(p.p, 2, capn_val0);
	s->awarenessStatus = capn_to_f32(capn_read32(p.p, 96));
}
void cereal_write_Live100Data(const struct cereal_Live100Data *s, cereal_Live100Data_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 64, s->canMonoTime);
	capn_setp(p.p, 0, s->canMonoTimes.p);
	capn_write64(p.p, 72, s->l20MonoTime);
	capn_write64(p.p, 80, s->mdMonoTime);
	capn_write32(p.p, 0, capn_from_f32(s->vEgo));
	capn_write32(p.p, 4, capn_from_f32(s->aEgoDEPRECATED));
	capn_write32(p.p, 8, capn_from_f32(s->vPid));
	capn_write32(p.p, 12, capn_from_f32(s->vTargetLead));
	capn_write32(p.p, 16, capn_from_f32(s->upAccelCmd));
	capn_write32(p.p, 20, capn_from_f32(s->uiAccelCmd));
	capn_write32(p.p, 24, capn_from_f32(s->yActual));
	capn_write32(p.p, 28, capn_from_f32(s->yDes));
	capn_write32(p.p, 32, capn_from_f32(s->upSteer));
	capn_write32(p.p, 36, capn_from_f32(s->uiSteer));
	capn_write32(p.p, 40, capn_from_f32(s->aTargetMin));
	capn_write32(p.p, 44, capn_from_f32(s->aTargetMax));
	capn_write32(p.p, 48, capn_from_f32(s->jerkFactor));
	capn_write32(p.p, 52, capn_from_f32(s->angleSteers));
	capn_write32(p.p, 56, (uint32_t) (s->hudLeadDEPRECATED));
	capn_write32(p.p, 60, capn_from_f32(s->cumLagMs));
	capn_write1(p.p, 704, s->enabled != 0);
	capn_write1(p.p, 705, s->steerOverride != 0);
	capn_write32(p.p, 92, capn_from_f32(s->vCruise));
	capn_write1(p.p, 706, s->rearViewCam != 0);
	capn_set_text(p.p, 1, s->alertText1);
	capn_set_text(p.p, 2, s->alertText2);
	capn_write32(p.p, 96, capn_from_f32(s->awarenessStatus));
}
void cereal_get_Live100Data(struct cereal_Live100Data *s, cereal_Live100Data_list l, int i) {
	cereal_Live100Data_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Live100Data(s, p);
}
void cereal_set_Live100Data(const struct cereal_Live100Data *s, cereal_Live100Data_list l, int i) {
	cereal_Live100Data_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Live100Data(s, p);
}

cereal_LiveEventData_ptr cereal_new_LiveEventData(struct capn_segment *s) {
	cereal_LiveEventData_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_LiveEventData_list cereal_new_LiveEventData_list(struct capn_segment *s, int len) {
	cereal_LiveEventData_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_LiveEventData(struct cereal_LiveEventData *s, cereal_LiveEventData_ptr p) {
	capn_resolve(&p.p);
	s->name = capn_get_text(p.p, 0, capn_val0);
	s->value = (int32_t) ((int32_t)capn_read32(p.p, 0));
}
void cereal_write_LiveEventData(const struct cereal_LiveEventData *s, cereal_LiveEventData_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->name);
	capn_write32(p.p, 0, (uint32_t) (s->value));
}
void cereal_get_LiveEventData(struct cereal_LiveEventData *s, cereal_LiveEventData_list l, int i) {
	cereal_LiveEventData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveEventData(s, p);
}
void cereal_set_LiveEventData(const struct cereal_LiveEventData *s, cereal_LiveEventData_list l, int i) {
	cereal_LiveEventData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveEventData(s, p);
}

cereal_ModelData_ptr cereal_new_ModelData(struct capn_segment *s) {
	cereal_ModelData_ptr p;
	p.p = capn_new_struct(s, 8, 5);
	return p;
}
cereal_ModelData_list cereal_new_ModelData_list(struct capn_segment *s, int len) {
	cereal_ModelData_list p;
	p.p = capn_new_list(s, len, 8, 5);
	return p;
}
void cereal_read_ModelData(struct cereal_ModelData *s, cereal_ModelData_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->path.p = capn_getp(p.p, 0, 0);
	s->leftLane.p = capn_getp(p.p, 1, 0);
	s->rightLane.p = capn_getp(p.p, 2, 0);
	s->lead.p = capn_getp(p.p, 3, 0);
	s->settings.p = capn_getp(p.p, 4, 0);
}
void cereal_write_ModelData(const struct cereal_ModelData *s, cereal_ModelData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_setp(p.p, 0, s->path.p);
	capn_setp(p.p, 1, s->leftLane.p);
	capn_setp(p.p, 2, s->rightLane.p);
	capn_setp(p.p, 3, s->lead.p);
	capn_setp(p.p, 4, s->settings.p);
}
void cereal_get_ModelData(struct cereal_ModelData *s, cereal_ModelData_list l, int i) {
	cereal_ModelData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData(s, p);
}
void cereal_set_ModelData(const struct cereal_ModelData *s, cereal_ModelData_list l, int i) {
	cereal_ModelData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData(s, p);
}

cereal_ModelData_PathData_ptr cereal_new_ModelData_PathData(struct capn_segment *s) {
	cereal_ModelData_PathData_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_ModelData_PathData_list cereal_new_ModelData_PathData_list(struct capn_segment *s, int len) {
	cereal_ModelData_PathData_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_ModelData_PathData(struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_ptr p) {
	capn_resolve(&p.p);
	s->points.p = capn_getp(p.p, 0, 0);
	s->prob = capn_to_f32(capn_read32(p.p, 0));
	s->std = capn_to_f32(capn_read32(p.p, 4));
}
void cereal_write_ModelData_PathData(const struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->points.p);
	capn_write32(p.p, 0, capn_from_f32(s->prob));
	capn_write32(p.p, 4, capn_from_f32(s->std));
}
void cereal_get_ModelData_PathData(struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_list l, int i) {
	cereal_ModelData_PathData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData_PathData(s, p);
}
void cereal_set_ModelData_PathData(const struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_list l, int i) {
	cereal_ModelData_PathData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData_PathData(s, p);
}

cereal_ModelData_LeadData_ptr cereal_new_ModelData_LeadData(struct capn_segment *s) {
	cereal_ModelData_LeadData_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_ModelData_LeadData_list cereal_new_ModelData_LeadData_list(struct capn_segment *s, int len) {
	cereal_ModelData_LeadData_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_ModelData_LeadData(struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_ptr p) {
	capn_resolve(&p.p);
	s->dist = capn_to_f32(capn_read32(p.p, 0));
	s->prob = capn_to_f32(capn_read32(p.p, 4));
	s->std = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_ModelData_LeadData(const struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->dist));
	capn_write32(p.p, 4, capn_from_f32(s->prob));
	capn_write32(p.p, 8, capn_from_f32(s->std));
}
void cereal_get_ModelData_LeadData(struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_list l, int i) {
	cereal_ModelData_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData_LeadData(s, p);
}
void cereal_set_ModelData_LeadData(const struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_list l, int i) {
	cereal_ModelData_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData_LeadData(s, p);
}

cereal_ModelData_ModelSettings_ptr cereal_new_ModelData_ModelSettings(struct capn_segment *s) {
	cereal_ModelData_ModelSettings_ptr p;
	p.p = capn_new_struct(s, 8, 2);
	return p;
}
cereal_ModelData_ModelSettings_list cereal_new_ModelData_ModelSettings_list(struct capn_segment *s, int len) {
	cereal_ModelData_ModelSettings_list p;
	p.p = capn_new_list(s, len, 8, 2);
	return p;
}
void cereal_read_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings *s, cereal_ModelData_ModelSettings_ptr p) {
	capn_resolve(&p.p);
	s->bigBoxX = capn_read16(p.p, 0);
	s->bigBoxY = capn_read16(p.p, 2);
	s->bigBoxWidth = capn_read16(p.p, 4);
	s->bigBoxHeight = capn_read16(p.p, 6);
	s->boxProjection.p = capn_getp(p.p, 0, 0);
	s->yuvCorrection.p = capn_getp(p.p, 1, 0);
}
void cereal_write_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings *s, cereal_ModelData_ModelSettings_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->bigBoxX);
	capn_write16(p.p, 2, s->bigBoxY);
	capn_write16(p.p, 4, s->bigBoxWidth);
	capn_write16(p.p, 6, s->bigBoxHeight);
	capn_setp(p.p, 0, s->boxProjection.p);
	capn_setp(p.p, 1, s->yuvCorrection.p);
}
void cereal_get_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings *s, cereal_ModelData_ModelSettings_list l, int i) {
	cereal_ModelData_ModelSettings_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData_ModelSettings(s, p);
}
void cereal_set_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings *s, cereal_ModelData_ModelSettings_list l, int i) {
	cereal_ModelData_ModelSettings_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData_ModelSettings(s, p);
}

cereal_CalibrationFeatures_ptr cereal_new_CalibrationFeatures(struct capn_segment *s) {
	cereal_CalibrationFeatures_ptr p;
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_CalibrationFeatures_list cereal_new_CalibrationFeatures_list(struct capn_segment *s, int len) {
	cereal_CalibrationFeatures_list p;
	p.p = capn_new_list(s, len, 8, 3);
	return p;
}
void cereal_read_CalibrationFeatures(struct cereal_CalibrationFeatures *s, cereal_CalibrationFeatures_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->p0.p = capn_getp(p.p, 0, 0);
	s->p1.p = capn_getp(p.p, 1, 0);
	s->status.p = capn_getp(p.p, 2, 0);
}
void cereal_write_CalibrationFeatures(const struct cereal_CalibrationFeatures *s, cereal_CalibrationFeatures_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_setp(p.p, 0, s->p0.p);
	capn_setp(p.p, 1, s->p1.p);
	capn_setp(p.p, 2, s->status.p);
}
void cereal_get_CalibrationFeatures(struct cereal_CalibrationFeatures *s, cereal_CalibrationFeatures_list l, int i) {
	cereal_CalibrationFeatures_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CalibrationFeatures(s, p);
}
void cereal_set_CalibrationFeatures(const struct cereal_CalibrationFeatures *s, cereal_CalibrationFeatures_list l, int i) {
	cereal_CalibrationFeatures_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CalibrationFeatures(s, p);
}

cereal_EncodeIndex_ptr cereal_new_EncodeIndex(struct capn_segment *s) {
	cereal_EncodeIndex_ptr p;
	p.p = capn_new_struct(s, 24, 0);
	return p;
}
cereal_EncodeIndex_list cereal_new_EncodeIndex_list(struct capn_segment *s, int len) {
	cereal_EncodeIndex_list p;
	p.p = capn_new_list(s, len, 24, 0);
	return p;
}
void cereal_read_EncodeIndex(struct cereal_EncodeIndex *s, cereal_EncodeIndex_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->type = (enum cereal_EncodeIndex_Type)(int) capn_read16(p.p, 4);
	s->encodeId = capn_read32(p.p, 8);
	s->segmentNum = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->segmentId = capn_read32(p.p, 16);
}
void cereal_write_EncodeIndex(const struct cereal_EncodeIndex *s, cereal_EncodeIndex_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write16(p.p, 4, (uint16_t) (s->type));
	capn_write32(p.p, 8, s->encodeId);
	capn_write32(p.p, 12, (uint32_t) (s->segmentNum));
	capn_write32(p.p, 16, s->segmentId);
}
void cereal_get_EncodeIndex(struct cereal_EncodeIndex *s, cereal_EncodeIndex_list l, int i) {
	cereal_EncodeIndex_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_EncodeIndex(s, p);
}
void cereal_set_EncodeIndex(const struct cereal_EncodeIndex *s, cereal_EncodeIndex_list l, int i) {
	cereal_EncodeIndex_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_EncodeIndex(s, p);
}

cereal_AndroidLogEntry_ptr cereal_new_AndroidLogEntry(struct capn_segment *s) {
	cereal_AndroidLogEntry_ptr p;
	p.p = capn_new_struct(s, 24, 2);
	return p;
}
cereal_AndroidLogEntry_list cereal_new_AndroidLogEntry_list(struct capn_segment *s, int len) {
	cereal_AndroidLogEntry_list p;
	p.p = capn_new_list(s, len, 24, 2);
	return p;
}
void cereal_read_AndroidLogEntry(struct cereal_AndroidLogEntry *s, cereal_AndroidLogEntry_ptr p) {
	capn_resolve(&p.p);
	s->id = capn_read8(p.p, 0);
	s->ts = capn_read64(p.p, 8);
	s->priority = capn_read8(p.p, 1);
	s->pid = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->tid = (int32_t) ((int32_t)capn_read32(p.p, 16));
	s->tag = capn_get_text(p.p, 0, capn_val0);
	s->message = capn_get_text(p.p, 1, capn_val0);
}
void cereal_write_AndroidLogEntry(const struct cereal_AndroidLogEntry *s, cereal_AndroidLogEntry_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->id);
	capn_write64(p.p, 8, s->ts);
	capn_write8(p.p, 1, s->priority);
	capn_write32(p.p, 4, (uint32_t) (s->pid));
	capn_write32(p.p, 16, (uint32_t) (s->tid));
	capn_set_text(p.p, 0, s->tag);
	capn_set_text(p.p, 1, s->message);
}
void cereal_get_AndroidLogEntry(struct cereal_AndroidLogEntry *s, cereal_AndroidLogEntry_list l, int i) {
	cereal_AndroidLogEntry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidLogEntry(s, p);
}
void cereal_set_AndroidLogEntry(const struct cereal_AndroidLogEntry *s, cereal_AndroidLogEntry_list l, int i) {
	cereal_AndroidLogEntry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidLogEntry(s, p);
}

cereal_LogRotate_ptr cereal_new_LogRotate(struct capn_segment *s) {
	cereal_LogRotate_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_LogRotate_list cereal_new_LogRotate_list(struct capn_segment *s, int len) {
	cereal_LogRotate_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_LogRotate(struct cereal_LogRotate *s, cereal_LogRotate_ptr p) {
	capn_resolve(&p.p);
	s->segmentNum = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->path = capn_get_text(p.p, 0, capn_val0);
}
void cereal_write_LogRotate(const struct cereal_LogRotate *s, cereal_LogRotate_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->segmentNum));
	capn_set_text(p.p, 0, s->path);
}
void cereal_get_LogRotate(struct cereal_LogRotate *s, cereal_LogRotate_list l, int i) {
	cereal_LogRotate_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LogRotate(s, p);
}
void cereal_set_LogRotate(const struct cereal_LogRotate *s, cereal_LogRotate_list l, int i) {
	cereal_LogRotate_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LogRotate(s, p);
}

cereal_Event_ptr cereal_new_Event(struct capn_segment *s) {
	cereal_Event_ptr p;
	p.p = capn_new_struct(s, 16, 1);
	return p;
}
cereal_Event_list cereal_new_Event_list(struct capn_segment *s, int len) {
	cereal_Event_list p;
	p.p = capn_new_list(s, len, 16, 1);
	return p;
}
void cereal_read_Event(struct cereal_Event *s, cereal_Event_ptr p) {
	capn_resolve(&p.p);
	s->logMonoTime = capn_read64(p.p, 0);
	s->which = (enum cereal_Event_which)(int) capn_read16(p.p, 8);
	switch (s->which) {
	case cereal_Event_logMessage:
		s->logMessage = capn_get_text(p.p, 0, capn_val0);
		break;
	case cereal_Event_initData:
	case cereal_Event_frame:
	case cereal_Event_gpsNMEA:
	case cereal_Event_sensorEventDEPRECATED:
	case cereal_Event_can:
	case cereal_Event_thermal:
	case cereal_Event_live100:
	case cereal_Event_liveEventDEPRECATED:
	case cereal_Event_model:
	case cereal_Event_features:
	case cereal_Event_sensorEvents:
	case cereal_Event_health:
	case cereal_Event_live20:
	case cereal_Event_liveUIDEPRECATED:
	case cereal_Event_encodeIdx:
	case cereal_Event_liveTracks:
	case cereal_Event_sendcan:
	case cereal_Event_liveCalibration:
	case cereal_Event_androidLogEntry:
		s->androidLogEntry.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_Event(const struct cereal_Event *s, cereal_Event_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->logMonoTime);
	capn_write16(p.p, 8, s->which);
	switch (s->which) {
	case cereal_Event_logMessage:
		capn_set_text(p.p, 0, s->logMessage);
		break;
	case cereal_Event_initData:
	case cereal_Event_frame:
	case cereal_Event_gpsNMEA:
	case cereal_Event_sensorEventDEPRECATED:
	case cereal_Event_can:
	case cereal_Event_thermal:
	case cereal_Event_live100:
	case cereal_Event_liveEventDEPRECATED:
	case cereal_Event_model:
	case cereal_Event_features:
	case cereal_Event_sensorEvents:
	case cereal_Event_health:
	case cereal_Event_live20:
	case cereal_Event_liveUIDEPRECATED:
	case cereal_Event_encodeIdx:
	case cereal_Event_liveTracks:
	case cereal_Event_sendcan:
	case cereal_Event_liveCalibration:
	case cereal_Event_androidLogEntry:
		capn_setp(p.p, 0, s->androidLogEntry.p);
		break;
	default:
		break;
	}
}
void cereal_get_Event(struct cereal_Event *s, cereal_Event_list l, int i) {
	cereal_Event_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Event(s, p);
}
void cereal_set_Event(const struct cereal_Event *s, cereal_Event_list l, int i) {
	cereal_Event_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Event(s, p);
}
