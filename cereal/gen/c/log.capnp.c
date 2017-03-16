#include "log.capnp.h"
/* AUTO GENERATED - DO NOT EDIT */
static const capn_text capn_val0 = {0,""};
int32_t cereal_logVersion = 1;

cereal_Map_ptr cereal_new_Map(struct capn_segment *s) {
	cereal_Map_ptr p;
	p.p = capn_new_struct(s, 0, 1);
	return p;
}
cereal_Map_list cereal_new_Map_list(struct capn_segment *s, int len) {
	cereal_Map_list p;
	p.p = capn_new_list(s, len, 0, 1);
	return p;
}
void cereal_read_Map(struct cereal_Map *s, cereal_Map_ptr p) {
	capn_resolve(&p.p);
	s->entries.p = capn_getp(p.p, 0, 0);
}
void cereal_write_Map(const struct cereal_Map *s, cereal_Map_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->entries.p);
}
void cereal_get_Map(struct cereal_Map *s, cereal_Map_list l, int i) {
	cereal_Map_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Map(s, p);
}
void cereal_set_Map(const struct cereal_Map *s, cereal_Map_list l, int i) {
	cereal_Map_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Map(s, p);
}

cereal_Map_Entry_ptr cereal_new_Map_Entry(struct capn_segment *s) {
	cereal_Map_Entry_ptr p;
	p.p = capn_new_struct(s, 0, 2);
	return p;
}
cereal_Map_Entry_list cereal_new_Map_Entry_list(struct capn_segment *s, int len) {
	cereal_Map_Entry_list p;
	p.p = capn_new_list(s, len, 0, 2);
	return p;
}
void cereal_read_Map_Entry(struct cereal_Map_Entry *s, cereal_Map_Entry_ptr p) {
	capn_resolve(&p.p);
	s->key = capn_getp(p.p, 0, 0);
	s->value = capn_getp(p.p, 1, 0);
}
void cereal_write_Map_Entry(const struct cereal_Map_Entry *s, cereal_Map_Entry_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->key);
	capn_setp(p.p, 1, s->value);
}
void cereal_get_Map_Entry(struct cereal_Map_Entry *s, cereal_Map_Entry_list l, int i) {
	cereal_Map_Entry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Map_Entry(s, p);
}
void cereal_set_Map_Entry(const struct cereal_Map_Entry *s, cereal_Map_Entry_list l, int i) {
	cereal_Map_Entry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Map_Entry(s, p);
}

cereal_InitData_ptr cereal_new_InitData(struct capn_segment *s) {
	cereal_InitData_ptr p;
	p.p = capn_new_struct(s, 8, 7);
	return p;
}
cereal_InitData_list cereal_new_InitData_list(struct capn_segment *s, int len) {
	cereal_InitData_list p;
	p.p = capn_new_list(s, len, 8, 7);
	return p;
}
void cereal_read_InitData(struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	s->kernelArgs = capn_getp(p.p, 0, 0);
	s->gctx = capn_get_text(p.p, 1, capn_val0);
	s->dongleId = capn_get_text(p.p, 2, capn_val0);
	s->deviceType = (enum cereal_InitData_DeviceType)(int) capn_read16(p.p, 0);
	s->version = capn_get_text(p.p, 3, capn_val0);
	s->androidBuildInfo.p = capn_getp(p.p, 4, 0);
	s->androidSensors.p = capn_getp(p.p, 5, 0);
	s->chffrAndroidExtra.p = capn_getp(p.p, 6, 0);
}
void cereal_write_InitData(const struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->kernelArgs);
	capn_set_text(p.p, 1, s->gctx);
	capn_set_text(p.p, 2, s->dongleId);
	capn_write16(p.p, 0, (uint16_t) (s->deviceType));
	capn_set_text(p.p, 3, s->version);
	capn_setp(p.p, 4, s->androidBuildInfo.p);
	capn_setp(p.p, 5, s->androidSensors.p);
	capn_setp(p.p, 6, s->chffrAndroidExtra.p);
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

cereal_InitData_AndroidBuildInfo_ptr cereal_new_InitData_AndroidBuildInfo(struct capn_segment *s) {
	cereal_InitData_AndroidBuildInfo_ptr p;
	p.p = capn_new_struct(s, 16, 21);
	return p;
}
cereal_InitData_AndroidBuildInfo_list cereal_new_InitData_AndroidBuildInfo_list(struct capn_segment *s, int len) {
	cereal_InitData_AndroidBuildInfo_list p;
	p.p = capn_new_list(s, len, 16, 21);
	return p;
}
void cereal_read_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo *s, cereal_InitData_AndroidBuildInfo_ptr p) {
	capn_resolve(&p.p);
	s->board = capn_get_text(p.p, 0, capn_val0);
	s->bootloader = capn_get_text(p.p, 1, capn_val0);
	s->brand = capn_get_text(p.p, 2, capn_val0);
	s->device = capn_get_text(p.p, 3, capn_val0);
	s->display = capn_get_text(p.p, 4, capn_val0);
	s->fingerprint = capn_get_text(p.p, 5, capn_val0);
	s->hardware = capn_get_text(p.p, 6, capn_val0);
	s->host = capn_get_text(p.p, 7, capn_val0);
	s->id = capn_get_text(p.p, 8, capn_val0);
	s->manufacturer = capn_get_text(p.p, 9, capn_val0);
	s->model = capn_get_text(p.p, 10, capn_val0);
	s->product = capn_get_text(p.p, 11, capn_val0);
	s->radioVersion = capn_get_text(p.p, 12, capn_val0);
	s->serial = capn_get_text(p.p, 13, capn_val0);
	s->supportedAbis = capn_getp(p.p, 14, 0);
	s->tags = capn_get_text(p.p, 15, capn_val0);
	s->time = (int64_t) ((int64_t)(capn_read64(p.p, 0)));
	s->type = capn_get_text(p.p, 16, capn_val0);
	s->user = capn_get_text(p.p, 17, capn_val0);
	s->versionCodename = capn_get_text(p.p, 18, capn_val0);
	s->versionRelease = capn_get_text(p.p, 19, capn_val0);
	s->versionSdk = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->versionSecurityPatch = capn_get_text(p.p, 20, capn_val0);
}
void cereal_write_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo *s, cereal_InitData_AndroidBuildInfo_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->board);
	capn_set_text(p.p, 1, s->bootloader);
	capn_set_text(p.p, 2, s->brand);
	capn_set_text(p.p, 3, s->device);
	capn_set_text(p.p, 4, s->display);
	capn_set_text(p.p, 5, s->fingerprint);
	capn_set_text(p.p, 6, s->hardware);
	capn_set_text(p.p, 7, s->host);
	capn_set_text(p.p, 8, s->id);
	capn_set_text(p.p, 9, s->manufacturer);
	capn_set_text(p.p, 10, s->model);
	capn_set_text(p.p, 11, s->product);
	capn_set_text(p.p, 12, s->radioVersion);
	capn_set_text(p.p, 13, s->serial);
	capn_setp(p.p, 14, s->supportedAbis);
	capn_set_text(p.p, 15, s->tags);
	capn_write64(p.p, 0, (uint64_t) (s->time));
	capn_set_text(p.p, 16, s->type);
	capn_set_text(p.p, 17, s->user);
	capn_set_text(p.p, 18, s->versionCodename);
	capn_set_text(p.p, 19, s->versionRelease);
	capn_write32(p.p, 8, (uint32_t) (s->versionSdk));
	capn_set_text(p.p, 20, s->versionSecurityPatch);
}
void cereal_get_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo *s, cereal_InitData_AndroidBuildInfo_list l, int i) {
	cereal_InitData_AndroidBuildInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData_AndroidBuildInfo(s, p);
}
void cereal_set_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo *s, cereal_InitData_AndroidBuildInfo_list l, int i) {
	cereal_InitData_AndroidBuildInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData_AndroidBuildInfo(s, p);
}

cereal_InitData_AndroidSensor_ptr cereal_new_InitData_AndroidSensor(struct capn_segment *s) {
	cereal_InitData_AndroidSensor_ptr p;
	p.p = capn_new_struct(s, 48, 3);
	return p;
}
cereal_InitData_AndroidSensor_list cereal_new_InitData_AndroidSensor_list(struct capn_segment *s, int len) {
	cereal_InitData_AndroidSensor_list p;
	p.p = capn_new_list(s, len, 48, 3);
	return p;
}
void cereal_read_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor *s, cereal_InitData_AndroidSensor_ptr p) {
	capn_resolve(&p.p);
	s->id = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->name = capn_get_text(p.p, 0, capn_val0);
	s->vendor = capn_get_text(p.p, 1, capn_val0);
	s->version = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->handle = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->type = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->maxRange = capn_to_f32(capn_read32(p.p, 16));
	s->resolution = capn_to_f32(capn_read32(p.p, 20));
	s->power = capn_to_f32(capn_read32(p.p, 24));
	s->minDelay = (int32_t) ((int32_t)capn_read32(p.p, 28));
	s->fifoReservedEventCount = capn_read32(p.p, 32);
	s->fifoMaxEventCount = capn_read32(p.p, 36);
	s->stringType = capn_get_text(p.p, 2, capn_val0);
	s->maxDelay = (int32_t) ((int32_t)capn_read32(p.p, 40));
}
void cereal_write_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor *s, cereal_InitData_AndroidSensor_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->id));
	capn_set_text(p.p, 0, s->name);
	capn_set_text(p.p, 1, s->vendor);
	capn_write32(p.p, 4, (uint32_t) (s->version));
	capn_write32(p.p, 8, (uint32_t) (s->handle));
	capn_write32(p.p, 12, (uint32_t) (s->type));
	capn_write32(p.p, 16, capn_from_f32(s->maxRange));
	capn_write32(p.p, 20, capn_from_f32(s->resolution));
	capn_write32(p.p, 24, capn_from_f32(s->power));
	capn_write32(p.p, 28, (uint32_t) (s->minDelay));
	capn_write32(p.p, 32, s->fifoReservedEventCount);
	capn_write32(p.p, 36, s->fifoMaxEventCount);
	capn_set_text(p.p, 2, s->stringType);
	capn_write32(p.p, 40, (uint32_t) (s->maxDelay));
}
void cereal_get_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor *s, cereal_InitData_AndroidSensor_list l, int i) {
	cereal_InitData_AndroidSensor_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData_AndroidSensor(s, p);
}
void cereal_set_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor *s, cereal_InitData_AndroidSensor_list l, int i) {
	cereal_InitData_AndroidSensor_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData_AndroidSensor(s, p);
}

cereal_InitData_ChffrAndroidExtra_ptr cereal_new_InitData_ChffrAndroidExtra(struct capn_segment *s) {
	cereal_InitData_ChffrAndroidExtra_ptr p;
	p.p = capn_new_struct(s, 0, 1);
	return p;
}
cereal_InitData_ChffrAndroidExtra_list cereal_new_InitData_ChffrAndroidExtra_list(struct capn_segment *s, int len) {
	cereal_InitData_ChffrAndroidExtra_list p;
	p.p = capn_new_list(s, len, 0, 1);
	return p;
}
void cereal_read_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra *s, cereal_InitData_ChffrAndroidExtra_ptr p) {
	capn_resolve(&p.p);
	s->allCameraCharacteristics.p = capn_getp(p.p, 0, 0);
}
void cereal_write_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra *s, cereal_InitData_ChffrAndroidExtra_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->allCameraCharacteristics.p);
}
void cereal_get_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra *s, cereal_InitData_ChffrAndroidExtra_list l, int i) {
	cereal_InitData_ChffrAndroidExtra_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData_ChffrAndroidExtra(s, p);
}
void cereal_set_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra *s, cereal_InitData_ChffrAndroidExtra_list l, int i) {
	cereal_InitData_ChffrAndroidExtra_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData_ChffrAndroidExtra(s, p);
}

cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment *s) {
	cereal_FrameData_ptr p;
	p.p = capn_new_struct(s, 40, 2);
	return p;
}
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment *s, int len) {
	cereal_FrameData_list p;
	p.p = capn_new_list(s, len, 40, 2);
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
	s->frameType = (enum cereal_FrameData_FrameType)(int) capn_read16(p.p, 28);
	s->timestampSof = capn_read64(p.p, 32);
	s->androidCaptureResult.p = capn_getp(p.p, 1, 0);
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
	capn_write16(p.p, 28, (uint16_t) (s->frameType));
	capn_write64(p.p, 32, s->timestampSof);
	capn_setp(p.p, 1, s->androidCaptureResult.p);
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

cereal_FrameData_AndroidCaptureResult_ptr cereal_new_FrameData_AndroidCaptureResult(struct capn_segment *s) {
	cereal_FrameData_AndroidCaptureResult_ptr p;
	p.p = capn_new_struct(s, 32, 2);
	return p;
}
cereal_FrameData_AndroidCaptureResult_list cereal_new_FrameData_AndroidCaptureResult_list(struct capn_segment *s, int len) {
	cereal_FrameData_AndroidCaptureResult_list p;
	p.p = capn_new_list(s, len, 32, 2);
	return p;
}
void cereal_read_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult *s, cereal_FrameData_AndroidCaptureResult_ptr p) {
	capn_resolve(&p.p);
	s->sensitivity = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->frameDuration = (int64_t) ((int64_t)(capn_read64(p.p, 8)));
	s->exposureTime = (int64_t) ((int64_t)(capn_read64(p.p, 16)));
	s->rollingShutterSkew = capn_read64(p.p, 24);
	s->colorCorrectionTransform.p = capn_getp(p.p, 0, 0);
	s->colorCorrectionGains.p = capn_getp(p.p, 1, 0);
}
void cereal_write_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult *s, cereal_FrameData_AndroidCaptureResult_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->sensitivity));
	capn_write64(p.p, 8, (uint64_t) (s->frameDuration));
	capn_write64(p.p, 16, (uint64_t) (s->exposureTime));
	capn_write64(p.p, 24, s->rollingShutterSkew);
	capn_setp(p.p, 0, s->colorCorrectionTransform.p);
	capn_setp(p.p, 1, s->colorCorrectionGains.p);
}
void cereal_get_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult *s, cereal_FrameData_AndroidCaptureResult_list l, int i) {
	cereal_FrameData_AndroidCaptureResult_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_FrameData_AndroidCaptureResult(s, p);
}
void cereal_set_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult *s, cereal_FrameData_AndroidCaptureResult_list l, int i) {
	cereal_FrameData_AndroidCaptureResult_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_FrameData_AndroidCaptureResult(s, p);
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
	s->source = (enum cereal_SensorEventData_SensorSource)(int) capn_read16(p.p, 14);
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
	capn_write16(p.p, 14, (uint16_t) (s->source));
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

cereal_GpsLocationData_ptr cereal_new_GpsLocationData(struct capn_segment *s) {
	cereal_GpsLocationData_ptr p;
	p.p = capn_new_struct(s, 48, 0);
	return p;
}
cereal_GpsLocationData_list cereal_new_GpsLocationData_list(struct capn_segment *s, int len) {
	cereal_GpsLocationData_list p;
	p.p = capn_new_list(s, len, 48, 0);
	return p;
}
void cereal_read_GpsLocationData(struct cereal_GpsLocationData *s, cereal_GpsLocationData_ptr p) {
	capn_resolve(&p.p);
	s->flags = capn_read16(p.p, 0);
	s->latitude = capn_to_f64(capn_read64(p.p, 8));
	s->longitude = capn_to_f64(capn_read64(p.p, 16));
	s->altitude = capn_to_f64(capn_read64(p.p, 24));
	s->speed = capn_to_f32(capn_read32(p.p, 4));
	s->bearing = capn_to_f32(capn_read32(p.p, 32));
	s->accuracy = capn_to_f32(capn_read32(p.p, 36));
	s->timestamp = (int64_t) ((int64_t)(capn_read64(p.p, 40)));
	s->source = (enum cereal_GpsLocationData_SensorSource)(int) capn_read16(p.p, 2);
}
void cereal_write_GpsLocationData(const struct cereal_GpsLocationData *s, cereal_GpsLocationData_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->flags);
	capn_write64(p.p, 8, capn_from_f64(s->latitude));
	capn_write64(p.p, 16, capn_from_f64(s->longitude));
	capn_write64(p.p, 24, capn_from_f64(s->altitude));
	capn_write32(p.p, 4, capn_from_f32(s->speed));
	capn_write32(p.p, 32, capn_from_f32(s->bearing));
	capn_write32(p.p, 36, capn_from_f32(s->accuracy));
	capn_write64(p.p, 40, (uint64_t) (s->timestamp));
	capn_write16(p.p, 2, (uint16_t) (s->source));
}
void cereal_get_GpsLocationData(struct cereal_GpsLocationData *s, cereal_GpsLocationData_list l, int i) {
	cereal_GpsLocationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_GpsLocationData(s, p);
}
void cereal_set_GpsLocationData(const struct cereal_GpsLocationData *s, cereal_GpsLocationData_list l, int i) {
	cereal_GpsLocationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_GpsLocationData(s, p);
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
	p.p = capn_new_struct(s, 24, 1);
	return p;
}
cereal_ThermalData_list cereal_new_ThermalData_list(struct capn_segment *s, int len) {
	cereal_ThermalData_list p;
	p.p = capn_new_list(s, len, 24, 1);
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
	s->batteryPercent = (int16_t) ((int16_t)capn_read16(p.p, 20));
	s->batteryStatus = capn_get_text(p.p, 0, capn_val0);
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
	capn_write16(p.p, 20, (uint16_t) (s->batteryPercent));
	capn_set_text(p.p, 0, s->batteryStatus);
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
	s->segmentIdEncode = capn_read32(p.p, 20);
}
void cereal_write_EncodeIndex(const struct cereal_EncodeIndex *s, cereal_EncodeIndex_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write16(p.p, 4, (uint16_t) (s->type));
	capn_write32(p.p, 8, s->encodeId);
	capn_write32(p.p, 12, (uint32_t) (s->segmentNum));
	capn_write32(p.p, 16, s->segmentId);
	capn_write32(p.p, 20, s->segmentIdEncode);
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

cereal_Plan_ptr cereal_new_Plan(struct capn_segment *s) {
	cereal_Plan_ptr p;
	p.p = capn_new_struct(s, 24, 1);
	return p;
}
cereal_Plan_list cereal_new_Plan_list(struct capn_segment *s, int len) {
	cereal_Plan_list p;
	p.p = capn_new_list(s, len, 24, 1);
	return p;
}
void cereal_read_Plan(struct cereal_Plan *s, cereal_Plan_ptr p) {
	capn_resolve(&p.p);
	s->lateralValid = (capn_read8(p.p, 0) & 1) != 0;
	s->dPoly.p = capn_getp(p.p, 0, 0);
	s->longitudalValid = (capn_read8(p.p, 0) & 2) != 0;
	s->vTarget = capn_to_f32(capn_read32(p.p, 4));
	s->aTargetMin = capn_to_f32(capn_read32(p.p, 8));
	s->aTargetMax = capn_to_f32(capn_read32(p.p, 12));
	s->jerkFactor = capn_to_f32(capn_read32(p.p, 16));
}
void cereal_write_Plan(const struct cereal_Plan *s, cereal_Plan_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->lateralValid != 0);
	capn_setp(p.p, 0, s->dPoly.p);
	capn_write1(p.p, 1, s->longitudalValid != 0);
	capn_write32(p.p, 4, capn_from_f32(s->vTarget));
	capn_write32(p.p, 8, capn_from_f32(s->aTargetMin));
	capn_write32(p.p, 12, capn_from_f32(s->aTargetMax));
	capn_write32(p.p, 16, capn_from_f32(s->jerkFactor));
}
void cereal_get_Plan(struct cereal_Plan *s, cereal_Plan_list l, int i) {
	cereal_Plan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Plan(s, p);
}
void cereal_set_Plan(const struct cereal_Plan *s, cereal_Plan_list l, int i) {
	cereal_Plan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Plan(s, p);
}

cereal_LiveLocationData_ptr cereal_new_LiveLocationData(struct capn_segment *s) {
	cereal_LiveLocationData_ptr p;
	p.p = capn_new_struct(s, 48, 4);
	return p;
}
cereal_LiveLocationData_list cereal_new_LiveLocationData_list(struct capn_segment *s, int len) {
	cereal_LiveLocationData_list p;
	p.p = capn_new_list(s, len, 48, 4);
	return p;
}
void cereal_read_LiveLocationData(struct cereal_LiveLocationData *s, cereal_LiveLocationData_ptr p) {
	capn_resolve(&p.p);
	s->status = capn_read8(p.p, 0);
	s->lat = capn_to_f64(capn_read64(p.p, 8));
	s->lon = capn_to_f64(capn_read64(p.p, 16));
	s->alt = capn_to_f32(capn_read32(p.p, 4));
	s->speed = capn_to_f32(capn_read32(p.p, 24));
	s->vNED.p = capn_getp(p.p, 0, 0);
	s->roll = capn_to_f32(capn_read32(p.p, 28));
	s->pitch = capn_to_f32(capn_read32(p.p, 32));
	s->heading = capn_to_f32(capn_read32(p.p, 36));
	s->wanderAngle = capn_to_f32(capn_read32(p.p, 40));
	s->trackAngle = capn_to_f32(capn_read32(p.p, 44));
	s->gyro.p = capn_getp(p.p, 1, 0);
	s->accel.p = capn_getp(p.p, 2, 0);
	s->accuracy.p = capn_getp(p.p, 3, 0);
}
void cereal_write_LiveLocationData(const struct cereal_LiveLocationData *s, cereal_LiveLocationData_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->status);
	capn_write64(p.p, 8, capn_from_f64(s->lat));
	capn_write64(p.p, 16, capn_from_f64(s->lon));
	capn_write32(p.p, 4, capn_from_f32(s->alt));
	capn_write32(p.p, 24, capn_from_f32(s->speed));
	capn_setp(p.p, 0, s->vNED.p);
	capn_write32(p.p, 28, capn_from_f32(s->roll));
	capn_write32(p.p, 32, capn_from_f32(s->pitch));
	capn_write32(p.p, 36, capn_from_f32(s->heading));
	capn_write32(p.p, 40, capn_from_f32(s->wanderAngle));
	capn_write32(p.p, 44, capn_from_f32(s->trackAngle));
	capn_setp(p.p, 1, s->gyro.p);
	capn_setp(p.p, 2, s->accel.p);
	capn_setp(p.p, 3, s->accuracy.p);
}
void cereal_get_LiveLocationData(struct cereal_LiveLocationData *s, cereal_LiveLocationData_list l, int i) {
	cereal_LiveLocationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveLocationData(s, p);
}
void cereal_set_LiveLocationData(const struct cereal_LiveLocationData *s, cereal_LiveLocationData_list l, int i) {
	cereal_LiveLocationData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveLocationData(s, p);
}

cereal_LiveLocationData_Accuracy_ptr cereal_new_LiveLocationData_Accuracy(struct capn_segment *s) {
	cereal_LiveLocationData_Accuracy_ptr p;
	p.p = capn_new_struct(s, 24, 2);
	return p;
}
cereal_LiveLocationData_Accuracy_list cereal_new_LiveLocationData_Accuracy_list(struct capn_segment *s, int len) {
	cereal_LiveLocationData_Accuracy_list p;
	p.p = capn_new_list(s, len, 24, 2);
	return p;
}
void cereal_read_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy *s, cereal_LiveLocationData_Accuracy_ptr p) {
	capn_resolve(&p.p);
	s->pNEDError.p = capn_getp(p.p, 0, 0);
	s->vNEDError.p = capn_getp(p.p, 1, 0);
	s->rollError = capn_to_f32(capn_read32(p.p, 0));
	s->pitchError = capn_to_f32(capn_read32(p.p, 4));
	s->headingError = capn_to_f32(capn_read32(p.p, 8));
	s->ellipsoidSemiMajorError = capn_to_f32(capn_read32(p.p, 12));
	s->ellipsoidSemiMinorError = capn_to_f32(capn_read32(p.p, 16));
	s->ellipsoidOrientationError = capn_to_f32(capn_read32(p.p, 20));
}
void cereal_write_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy *s, cereal_LiveLocationData_Accuracy_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->pNEDError.p);
	capn_setp(p.p, 1, s->vNEDError.p);
	capn_write32(p.p, 0, capn_from_f32(s->rollError));
	capn_write32(p.p, 4, capn_from_f32(s->pitchError));
	capn_write32(p.p, 8, capn_from_f32(s->headingError));
	capn_write32(p.p, 12, capn_from_f32(s->ellipsoidSemiMajorError));
	capn_write32(p.p, 16, capn_from_f32(s->ellipsoidSemiMinorError));
	capn_write32(p.p, 20, capn_from_f32(s->ellipsoidOrientationError));
}
void cereal_get_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy *s, cereal_LiveLocationData_Accuracy_list l, int i) {
	cereal_LiveLocationData_Accuracy_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveLocationData_Accuracy(s, p);
}
void cereal_set_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy *s, cereal_LiveLocationData_Accuracy_list l, int i) {
	cereal_LiveLocationData_Accuracy_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveLocationData_Accuracy(s, p);
}

cereal_EthernetPacket_ptr cereal_new_EthernetPacket(struct capn_segment *s) {
	cereal_EthernetPacket_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_EthernetPacket_list cereal_new_EthernetPacket_list(struct capn_segment *s, int len) {
	cereal_EthernetPacket_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_EthernetPacket(struct cereal_EthernetPacket *s, cereal_EthernetPacket_ptr p) {
	capn_resolve(&p.p);
	s->pkt = capn_get_data(p.p, 0);
	s->ts = capn_to_f32(capn_read32(p.p, 0));
}
void cereal_write_EthernetPacket(const struct cereal_EthernetPacket *s, cereal_EthernetPacket_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->pkt.p);
	capn_write32(p.p, 0, capn_from_f32(s->ts));
}
void cereal_get_EthernetPacket(struct cereal_EthernetPacket *s, cereal_EthernetPacket_list l, int i) {
	cereal_EthernetPacket_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_EthernetPacket(s, p);
}
void cereal_set_EthernetPacket(const struct cereal_EthernetPacket *s, cereal_EthernetPacket_list l, int i) {
	cereal_EthernetPacket_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_EthernetPacket(s, p);
}

cereal_NavUpdate_ptr cereal_new_NavUpdate(struct capn_segment *s) {
	cereal_NavUpdate_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_NavUpdate_list cereal_new_NavUpdate_list(struct capn_segment *s, int len) {
	cereal_NavUpdate_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_NavUpdate(struct cereal_NavUpdate *s, cereal_NavUpdate_ptr p) {
	capn_resolve(&p.p);
	s->isNavigating = (capn_read8(p.p, 0) & 1) != 0;
	s->curSegment = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->segments.p = capn_getp(p.p, 0, 0);
}
void cereal_write_NavUpdate(const struct cereal_NavUpdate *s, cereal_NavUpdate_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->isNavigating != 0);
	capn_write32(p.p, 4, (uint32_t) (s->curSegment));
	capn_setp(p.p, 0, s->segments.p);
}
void cereal_get_NavUpdate(struct cereal_NavUpdate *s, cereal_NavUpdate_list l, int i) {
	cereal_NavUpdate_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_NavUpdate(s, p);
}
void cereal_set_NavUpdate(const struct cereal_NavUpdate *s, cereal_NavUpdate_list l, int i) {
	cereal_NavUpdate_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_NavUpdate(s, p);
}

cereal_NavUpdate_LatLng_ptr cereal_new_NavUpdate_LatLng(struct capn_segment *s) {
	cereal_NavUpdate_LatLng_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_NavUpdate_LatLng_list cereal_new_NavUpdate_LatLng_list(struct capn_segment *s, int len) {
	cereal_NavUpdate_LatLng_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng *s, cereal_NavUpdate_LatLng_ptr p) {
	capn_resolve(&p.p);
	s->lat = capn_to_f64(capn_read64(p.p, 0));
	s->lng = capn_to_f64(capn_read64(p.p, 8));
}
void cereal_write_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng *s, cereal_NavUpdate_LatLng_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, capn_from_f64(s->lat));
	capn_write64(p.p, 8, capn_from_f64(s->lng));
}
void cereal_get_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng *s, cereal_NavUpdate_LatLng_list l, int i) {
	cereal_NavUpdate_LatLng_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_NavUpdate_LatLng(s, p);
}
void cereal_set_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng *s, cereal_NavUpdate_LatLng_list l, int i) {
	cereal_NavUpdate_LatLng_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_NavUpdate_LatLng(s, p);
}

cereal_NavUpdate_Segment_ptr cereal_new_NavUpdate_Segment(struct capn_segment *s) {
	cereal_NavUpdate_Segment_ptr p;
	p.p = capn_new_struct(s, 24, 3);
	return p;
}
cereal_NavUpdate_Segment_list cereal_new_NavUpdate_Segment_list(struct capn_segment *s, int len) {
	cereal_NavUpdate_Segment_list p;
	p.p = capn_new_list(s, len, 24, 3);
	return p;
}
void cereal_read_NavUpdate_Segment(struct cereal_NavUpdate_Segment *s, cereal_NavUpdate_Segment_ptr p) {
	capn_resolve(&p.p);
	s->from.p = capn_getp(p.p, 0, 0);
	s->to.p = capn_getp(p.p, 1, 0);
	s->updateTime = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->distance = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->crossTime = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->exitNo = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->instruction = (enum cereal_NavUpdate_Segment_Instruction)(int) capn_read16(p.p, 16);
	s->parts.p = capn_getp(p.p, 2, 0);
}
void cereal_write_NavUpdate_Segment(const struct cereal_NavUpdate_Segment *s, cereal_NavUpdate_Segment_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->from.p);
	capn_setp(p.p, 1, s->to.p);
	capn_write32(p.p, 0, (uint32_t) (s->updateTime));
	capn_write32(p.p, 4, (uint32_t) (s->distance));
	capn_write32(p.p, 8, (uint32_t) (s->crossTime));
	capn_write32(p.p, 12, (uint32_t) (s->exitNo));
	capn_write16(p.p, 16, (uint16_t) (s->instruction));
	capn_setp(p.p, 2, s->parts.p);
}
void cereal_get_NavUpdate_Segment(struct cereal_NavUpdate_Segment *s, cereal_NavUpdate_Segment_list l, int i) {
	cereal_NavUpdate_Segment_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_NavUpdate_Segment(s, p);
}
void cereal_set_NavUpdate_Segment(const struct cereal_NavUpdate_Segment *s, cereal_NavUpdate_Segment_list l, int i) {
	cereal_NavUpdate_Segment_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_NavUpdate_Segment(s, p);
}

cereal_CellInfo_ptr cereal_new_CellInfo(struct capn_segment *s) {
	cereal_CellInfo_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_CellInfo_list cereal_new_CellInfo_list(struct capn_segment *s, int len) {
	cereal_CellInfo_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_CellInfo(struct cereal_CellInfo *s, cereal_CellInfo_ptr p) {
	capn_resolve(&p.p);
	s->timestamp = capn_read64(p.p, 0);
	s->repr = capn_get_text(p.p, 0, capn_val0);
}
void cereal_write_CellInfo(const struct cereal_CellInfo *s, cereal_CellInfo_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->timestamp);
	capn_set_text(p.p, 0, s->repr);
}
void cereal_get_CellInfo(struct cereal_CellInfo *s, cereal_CellInfo_list l, int i) {
	cereal_CellInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CellInfo(s, p);
}
void cereal_set_CellInfo(const struct cereal_CellInfo *s, cereal_CellInfo_list l, int i) {
	cereal_CellInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CellInfo(s, p);
}

cereal_WifiScan_ptr cereal_new_WifiScan(struct capn_segment *s) {
	cereal_WifiScan_ptr p;
	p.p = capn_new_struct(s, 40, 5);
	return p;
}
cereal_WifiScan_list cereal_new_WifiScan_list(struct capn_segment *s, int len) {
	cereal_WifiScan_list p;
	p.p = capn_new_list(s, len, 40, 5);
	return p;
}
void cereal_read_WifiScan(struct cereal_WifiScan *s, cereal_WifiScan_ptr p) {
	capn_resolve(&p.p);
	s->bssid = capn_get_text(p.p, 0, capn_val0);
	s->ssid = capn_get_text(p.p, 1, capn_val0);
	s->capabilities = capn_get_text(p.p, 2, capn_val0);
	s->frequency = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->level = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->timestamp = (int64_t) ((int64_t)(capn_read64(p.p, 8)));
	s->centerFreq0 = (int32_t) ((int32_t)capn_read32(p.p, 16));
	s->centerFreq1 = (int32_t) ((int32_t)capn_read32(p.p, 20));
	s->channelWidth = (enum cereal_WifiScan_ChannelWidth)(int) capn_read16(p.p, 24);
	s->operatorFriendlyName = capn_get_text(p.p, 3, capn_val0);
	s->venueName = capn_get_text(p.p, 4, capn_val0);
	s->is80211mcResponder = (capn_read8(p.p, 26) & 1) != 0;
	s->passpoint = (capn_read8(p.p, 26) & 2) != 0;
	s->distanceCm = (int32_t) ((int32_t)capn_read32(p.p, 28));
	s->distanceSdCm = (int32_t) ((int32_t)capn_read32(p.p, 32));
}
void cereal_write_WifiScan(const struct cereal_WifiScan *s, cereal_WifiScan_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->bssid);
	capn_set_text(p.p, 1, s->ssid);
	capn_set_text(p.p, 2, s->capabilities);
	capn_write32(p.p, 0, (uint32_t) (s->frequency));
	capn_write32(p.p, 4, (uint32_t) (s->level));
	capn_write64(p.p, 8, (uint64_t) (s->timestamp));
	capn_write32(p.p, 16, (uint32_t) (s->centerFreq0));
	capn_write32(p.p, 20, (uint32_t) (s->centerFreq1));
	capn_write16(p.p, 24, (uint16_t) (s->channelWidth));
	capn_set_text(p.p, 3, s->operatorFriendlyName);
	capn_set_text(p.p, 4, s->venueName);
	capn_write1(p.p, 208, s->is80211mcResponder != 0);
	capn_write1(p.p, 209, s->passpoint != 0);
	capn_write32(p.p, 28, (uint32_t) (s->distanceCm));
	capn_write32(p.p, 32, (uint32_t) (s->distanceSdCm));
}
void cereal_get_WifiScan(struct cereal_WifiScan *s, cereal_WifiScan_list l, int i) {
	cereal_WifiScan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_WifiScan(s, p);
}
void cereal_set_WifiScan(const struct cereal_WifiScan *s, cereal_WifiScan_list l, int i) {
	cereal_WifiScan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_WifiScan(s, p);
}

cereal_AndroidGnss_ptr cereal_new_AndroidGnss(struct capn_segment *s) {
	cereal_AndroidGnss_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_AndroidGnss_list cereal_new_AndroidGnss_list(struct capn_segment *s, int len) {
	cereal_AndroidGnss_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_AndroidGnss(struct cereal_AndroidGnss *s, cereal_AndroidGnss_ptr p) {
	capn_resolve(&p.p);
	s->which = (enum cereal_AndroidGnss_which)(int) capn_read16(p.p, 0);
	switch (s->which) {
	case cereal_AndroidGnss_measurements:
	case cereal_AndroidGnss_navigationMessage:
		s->navigationMessage.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_AndroidGnss(const struct cereal_AndroidGnss *s, cereal_AndroidGnss_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->which);
	switch (s->which) {
	case cereal_AndroidGnss_measurements:
	case cereal_AndroidGnss_navigationMessage:
		capn_setp(p.p, 0, s->navigationMessage.p);
		break;
	default:
		break;
	}
}
void cereal_get_AndroidGnss(struct cereal_AndroidGnss *s, cereal_AndroidGnss_list l, int i) {
	cereal_AndroidGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidGnss(s, p);
}
void cereal_set_AndroidGnss(const struct cereal_AndroidGnss *s, cereal_AndroidGnss_list l, int i) {
	cereal_AndroidGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidGnss(s, p);
}

cereal_AndroidGnss_Measurements_ptr cereal_new_AndroidGnss_Measurements(struct capn_segment *s) {
	cereal_AndroidGnss_Measurements_ptr p;
	p.p = capn_new_struct(s, 0, 2);
	return p;
}
cereal_AndroidGnss_Measurements_list cereal_new_AndroidGnss_Measurements_list(struct capn_segment *s, int len) {
	cereal_AndroidGnss_Measurements_list p;
	p.p = capn_new_list(s, len, 0, 2);
	return p;
}
void cereal_read_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements *s, cereal_AndroidGnss_Measurements_ptr p) {
	capn_resolve(&p.p);
	s->clock.p = capn_getp(p.p, 0, 0);
	s->measurements.p = capn_getp(p.p, 1, 0);
}
void cereal_write_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements *s, cereal_AndroidGnss_Measurements_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->clock.p);
	capn_setp(p.p, 1, s->measurements.p);
}
void cereal_get_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements *s, cereal_AndroidGnss_Measurements_list l, int i) {
	cereal_AndroidGnss_Measurements_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidGnss_Measurements(s, p);
}
void cereal_set_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements *s, cereal_AndroidGnss_Measurements_list l, int i) {
	cereal_AndroidGnss_Measurements_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidGnss_Measurements(s, p);
}

cereal_AndroidGnss_Measurements_Clock_ptr cereal_new_AndroidGnss_Measurements_Clock(struct capn_segment *s) {
	cereal_AndroidGnss_Measurements_Clock_ptr p;
	p.p = capn_new_struct(s, 72, 0);
	return p;
}
cereal_AndroidGnss_Measurements_Clock_list cereal_new_AndroidGnss_Measurements_Clock_list(struct capn_segment *s, int len) {
	cereal_AndroidGnss_Measurements_Clock_list p;
	p.p = capn_new_list(s, len, 72, 0);
	return p;
}
void cereal_read_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock *s, cereal_AndroidGnss_Measurements_Clock_ptr p) {
	capn_resolve(&p.p);
	s->timeNanos = (int64_t) ((int64_t)(capn_read64(p.p, 0)));
	s->hardwareClockDiscontinuityCount = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->hasTimeUncertaintyNanos = (capn_read8(p.p, 12) & 1) != 0;
	s->timeUncertaintyNanos = capn_to_f64(capn_read64(p.p, 16));
	s->hasLeapSecond = (capn_read8(p.p, 12) & 2) != 0;
	s->leapSecond = (int32_t) ((int32_t)capn_read32(p.p, 24));
	s->hasFullBiasNanos = (capn_read8(p.p, 12) & 4) != 0;
	s->fullBiasNanos = (int64_t) ((int64_t)(capn_read64(p.p, 32)));
	s->hasBiasNanos = (capn_read8(p.p, 12) & 8) != 0;
	s->biasNanos = capn_to_f64(capn_read64(p.p, 40));
	s->hasBiasUncertaintyNanos = (capn_read8(p.p, 12) & 16) != 0;
	s->biasUncertaintyNanos = capn_to_f64(capn_read64(p.p, 48));
	s->hasDriftNanosPerSecond = (capn_read8(p.p, 12) & 32) != 0;
	s->driftNanosPerSecond = capn_to_f64(capn_read64(p.p, 56));
	s->hasDriftUncertaintyNanosPerSecond = (capn_read8(p.p, 12) & 64) != 0;
	s->driftUncertaintyNanosPerSecond = capn_to_f64(capn_read64(p.p, 64));
}
void cereal_write_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock *s, cereal_AndroidGnss_Measurements_Clock_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, (uint64_t) (s->timeNanos));
	capn_write32(p.p, 8, (uint32_t) (s->hardwareClockDiscontinuityCount));
	capn_write1(p.p, 96, s->hasTimeUncertaintyNanos != 0);
	capn_write64(p.p, 16, capn_from_f64(s->timeUncertaintyNanos));
	capn_write1(p.p, 97, s->hasLeapSecond != 0);
	capn_write32(p.p, 24, (uint32_t) (s->leapSecond));
	capn_write1(p.p, 98, s->hasFullBiasNanos != 0);
	capn_write64(p.p, 32, (uint64_t) (s->fullBiasNanos));
	capn_write1(p.p, 99, s->hasBiasNanos != 0);
	capn_write64(p.p, 40, capn_from_f64(s->biasNanos));
	capn_write1(p.p, 100, s->hasBiasUncertaintyNanos != 0);
	capn_write64(p.p, 48, capn_from_f64(s->biasUncertaintyNanos));
	capn_write1(p.p, 101, s->hasDriftNanosPerSecond != 0);
	capn_write64(p.p, 56, capn_from_f64(s->driftNanosPerSecond));
	capn_write1(p.p, 102, s->hasDriftUncertaintyNanosPerSecond != 0);
	capn_write64(p.p, 64, capn_from_f64(s->driftUncertaintyNanosPerSecond));
}
void cereal_get_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock *s, cereal_AndroidGnss_Measurements_Clock_list l, int i) {
	cereal_AndroidGnss_Measurements_Clock_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidGnss_Measurements_Clock(s, p);
}
void cereal_set_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock *s, cereal_AndroidGnss_Measurements_Clock_list l, int i) {
	cereal_AndroidGnss_Measurements_Clock_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidGnss_Measurements_Clock(s, p);
}

cereal_AndroidGnss_Measurements_Measurement_ptr cereal_new_AndroidGnss_Measurements_Measurement(struct capn_segment *s) {
	cereal_AndroidGnss_Measurements_Measurement_ptr p;
	p.p = capn_new_struct(s, 120, 0);
	return p;
}
cereal_AndroidGnss_Measurements_Measurement_list cereal_new_AndroidGnss_Measurements_Measurement_list(struct capn_segment *s, int len) {
	cereal_AndroidGnss_Measurements_Measurement_list p;
	p.p = capn_new_list(s, len, 120, 0);
	return p;
}
void cereal_read_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement *s, cereal_AndroidGnss_Measurements_Measurement_ptr p) {
	capn_resolve(&p.p);
	s->svId = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->constellation = (enum cereal_AndroidGnss_Measurements_Measurement_Constellation)(int) capn_read16(p.p, 4);
	s->timeOffsetNanos = capn_to_f64(capn_read64(p.p, 8));
	s->state = (int32_t) ((int32_t)capn_read32(p.p, 16));
	s->receivedSvTimeNanos = (int64_t) ((int64_t)(capn_read64(p.p, 24)));
	s->receivedSvTimeUncertaintyNanos = (int64_t) ((int64_t)(capn_read64(p.p, 32)));
	s->cn0DbHz = capn_to_f64(capn_read64(p.p, 40));
	s->pseudorangeRateMetersPerSecond = capn_to_f64(capn_read64(p.p, 48));
	s->pseudorangeRateUncertaintyMetersPerSecond = capn_to_f64(capn_read64(p.p, 56));
	s->accumulatedDeltaRangeState = (int32_t) ((int32_t)capn_read32(p.p, 20));
	s->accumulatedDeltaRangeMeters = capn_to_f64(capn_read64(p.p, 64));
	s->accumulatedDeltaRangeUncertaintyMeters = capn_to_f64(capn_read64(p.p, 72));
	s->hasCarrierFrequencyHz = (capn_read8(p.p, 6) & 1) != 0;
	s->carrierFrequencyHz = capn_to_f32(capn_read32(p.p, 80));
	s->hasCarrierCycles = (capn_read8(p.p, 6) & 2) != 0;
	s->carrierCycles = (int64_t) ((int64_t)(capn_read64(p.p, 88)));
	s->hasCarrierPhase = (capn_read8(p.p, 6) & 4) != 0;
	s->carrierPhase = capn_to_f64(capn_read64(p.p, 96));
	s->hasCarrierPhaseUncertainty = (capn_read8(p.p, 6) & 8) != 0;
	s->carrierPhaseUncertainty = capn_to_f64(capn_read64(p.p, 104));
	s->hasSnrInDb = (capn_read8(p.p, 6) & 16) != 0;
	s->snrInDb = capn_to_f64(capn_read64(p.p, 112));
	s->multipathIndicator = (enum cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator)(int) capn_read16(p.p, 84);
}
void cereal_write_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement *s, cereal_AndroidGnss_Measurements_Measurement_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->svId));
	capn_write16(p.p, 4, (uint16_t) (s->constellation));
	capn_write64(p.p, 8, capn_from_f64(s->timeOffsetNanos));
	capn_write32(p.p, 16, (uint32_t) (s->state));
	capn_write64(p.p, 24, (uint64_t) (s->receivedSvTimeNanos));
	capn_write64(p.p, 32, (uint64_t) (s->receivedSvTimeUncertaintyNanos));
	capn_write64(p.p, 40, capn_from_f64(s->cn0DbHz));
	capn_write64(p.p, 48, capn_from_f64(s->pseudorangeRateMetersPerSecond));
	capn_write64(p.p, 56, capn_from_f64(s->pseudorangeRateUncertaintyMetersPerSecond));
	capn_write32(p.p, 20, (uint32_t) (s->accumulatedDeltaRangeState));
	capn_write64(p.p, 64, capn_from_f64(s->accumulatedDeltaRangeMeters));
	capn_write64(p.p, 72, capn_from_f64(s->accumulatedDeltaRangeUncertaintyMeters));
	capn_write1(p.p, 48, s->hasCarrierFrequencyHz != 0);
	capn_write32(p.p, 80, capn_from_f32(s->carrierFrequencyHz));
	capn_write1(p.p, 49, s->hasCarrierCycles != 0);
	capn_write64(p.p, 88, (uint64_t) (s->carrierCycles));
	capn_write1(p.p, 50, s->hasCarrierPhase != 0);
	capn_write64(p.p, 96, capn_from_f64(s->carrierPhase));
	capn_write1(p.p, 51, s->hasCarrierPhaseUncertainty != 0);
	capn_write64(p.p, 104, capn_from_f64(s->carrierPhaseUncertainty));
	capn_write1(p.p, 52, s->hasSnrInDb != 0);
	capn_write64(p.p, 112, capn_from_f64(s->snrInDb));
	capn_write16(p.p, 84, (uint16_t) (s->multipathIndicator));
}
void cereal_get_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement *s, cereal_AndroidGnss_Measurements_Measurement_list l, int i) {
	cereal_AndroidGnss_Measurements_Measurement_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidGnss_Measurements_Measurement(s, p);
}
void cereal_set_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement *s, cereal_AndroidGnss_Measurements_Measurement_list l, int i) {
	cereal_AndroidGnss_Measurements_Measurement_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidGnss_Measurements_Measurement(s, p);
}

cereal_AndroidGnss_NavigationMessage_ptr cereal_new_AndroidGnss_NavigationMessage(struct capn_segment *s) {
	cereal_AndroidGnss_NavigationMessage_ptr p;
	p.p = capn_new_struct(s, 24, 1);
	return p;
}
cereal_AndroidGnss_NavigationMessage_list cereal_new_AndroidGnss_NavigationMessage_list(struct capn_segment *s, int len) {
	cereal_AndroidGnss_NavigationMessage_list p;
	p.p = capn_new_list(s, len, 24, 1);
	return p;
}
void cereal_read_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage *s, cereal_AndroidGnss_NavigationMessage_ptr p) {
	capn_resolve(&p.p);
	s->type = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->svId = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->messageId = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->submessageId = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->data = capn_get_data(p.p, 0);
	s->status = (enum cereal_AndroidGnss_NavigationMessage_Status)(int) capn_read16(p.p, 16);
}
void cereal_write_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage *s, cereal_AndroidGnss_NavigationMessage_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->type));
	capn_write32(p.p, 4, (uint32_t) (s->svId));
	capn_write32(p.p, 8, (uint32_t) (s->messageId));
	capn_write32(p.p, 12, (uint32_t) (s->submessageId));
	capn_setp(p.p, 0, s->data.p);
	capn_write16(p.p, 16, (uint16_t) (s->status));
}
void cereal_get_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage *s, cereal_AndroidGnss_NavigationMessage_list l, int i) {
	cereal_AndroidGnss_NavigationMessage_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_AndroidGnss_NavigationMessage(s, p);
}
void cereal_set_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage *s, cereal_AndroidGnss_NavigationMessage_list l, int i) {
	cereal_AndroidGnss_NavigationMessage_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_AndroidGnss_NavigationMessage(s, p);
}

cereal_QcomGnss_ptr cereal_new_QcomGnss(struct capn_segment *s) {
	cereal_QcomGnss_ptr p;
	p.p = capn_new_struct(s, 16, 1);
	return p;
}
cereal_QcomGnss_list cereal_new_QcomGnss_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_list p;
	p.p = capn_new_list(s, len, 16, 1);
	return p;
}
void cereal_read_QcomGnss(struct cereal_QcomGnss *s, cereal_QcomGnss_ptr p) {
	capn_resolve(&p.p);
	s->logTs = capn_read64(p.p, 0);
	s->which = (enum cereal_QcomGnss_which)(int) capn_read16(p.p, 8);
	switch (s->which) {
	case cereal_QcomGnss_measurementReport:
	case cereal_QcomGnss_clockReport:
		s->clockReport.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_QcomGnss(const struct cereal_QcomGnss *s, cereal_QcomGnss_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->logTs);
	capn_write16(p.p, 8, s->which);
	switch (s->which) {
	case cereal_QcomGnss_measurementReport:
	case cereal_QcomGnss_clockReport:
		capn_setp(p.p, 0, s->clockReport.p);
		break;
	default:
		break;
	}
}
void cereal_get_QcomGnss(struct cereal_QcomGnss *s, cereal_QcomGnss_list l, int i) {
	cereal_QcomGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss(s, p);
}
void cereal_set_QcomGnss(const struct cereal_QcomGnss *s, cereal_QcomGnss_list l, int i) {
	cereal_QcomGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss(s, p);
}

cereal_QcomGnss_MeasurementReport_ptr cereal_new_QcomGnss_MeasurementReport(struct capn_segment *s) {
	cereal_QcomGnss_MeasurementReport_ptr p;
	p.p = capn_new_struct(s, 32, 1);
	return p;
}
cereal_QcomGnss_MeasurementReport_list cereal_new_QcomGnss_MeasurementReport_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_MeasurementReport_list p;
	p.p = capn_new_list(s, len, 32, 1);
	return p;
}
void cereal_read_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport *s, cereal_QcomGnss_MeasurementReport_ptr p) {
	capn_resolve(&p.p);
	s->source = (enum cereal_QcomGnss_MeasurementReport_Source)(int) capn_read16(p.p, 0);
	s->fCount = capn_read32(p.p, 4);
	s->gpsWeek = capn_read16(p.p, 2);
	s->glonassCycleNumber = capn_read8(p.p, 8);
	s->glonassNumberOfDays = capn_read16(p.p, 10);
	s->milliseconds = capn_read32(p.p, 12);
	s->timeBias = capn_to_f32(capn_read32(p.p, 16));
	s->clockTimeUncertainty = capn_to_f32(capn_read32(p.p, 20));
	s->clockFrequencyBias = capn_to_f32(capn_read32(p.p, 24));
	s->clockFrequencyUncertainty = capn_to_f32(capn_read32(p.p, 28));
	s->sv.p = capn_getp(p.p, 0, 0);
}
void cereal_write_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport *s, cereal_QcomGnss_MeasurementReport_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, (uint16_t) (s->source));
	capn_write32(p.p, 4, s->fCount);
	capn_write16(p.p, 2, s->gpsWeek);
	capn_write8(p.p, 8, s->glonassCycleNumber);
	capn_write16(p.p, 10, s->glonassNumberOfDays);
	capn_write32(p.p, 12, s->milliseconds);
	capn_write32(p.p, 16, capn_from_f32(s->timeBias));
	capn_write32(p.p, 20, capn_from_f32(s->clockTimeUncertainty));
	capn_write32(p.p, 24, capn_from_f32(s->clockFrequencyBias));
	capn_write32(p.p, 28, capn_from_f32(s->clockFrequencyUncertainty));
	capn_setp(p.p, 0, s->sv.p);
}
void cereal_get_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport *s, cereal_QcomGnss_MeasurementReport_list l, int i) {
	cereal_QcomGnss_MeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_MeasurementReport(s, p);
}
void cereal_set_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport *s, cereal_QcomGnss_MeasurementReport_list l, int i) {
	cereal_QcomGnss_MeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_MeasurementReport(s, p);
}

cereal_QcomGnss_MeasurementReport_SV_ptr cereal_new_QcomGnss_MeasurementReport_SV(struct capn_segment *s) {
	cereal_QcomGnss_MeasurementReport_SV_ptr p;
	p.p = capn_new_struct(s, 64, 1);
	return p;
}
cereal_QcomGnss_MeasurementReport_SV_list cereal_new_QcomGnss_MeasurementReport_SV_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_MeasurementReport_SV_list p;
	p.p = capn_new_list(s, len, 64, 1);
	return p;
}
void cereal_read_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV *s, cereal_QcomGnss_MeasurementReport_SV_ptr p) {
	capn_resolve(&p.p);
	s->svId = capn_read8(p.p, 0);
	s->observationState = (enum cereal_QcomGnss_MeasurementReport_SV_SVObservationState)(int) capn_read16(p.p, 2);
	s->observations = capn_read8(p.p, 4);
	s->goodObservations = capn_read8(p.p, 5);
	s->gpsParityErrorCount = capn_read16(p.p, 6);
	s->glonassFrequencyIndex = (int8_t) ((int8_t)capn_read8(p.p, 1));
	s->glonassHemmingErrorCount = capn_read8(p.p, 8);
	s->filterStages = capn_read8(p.p, 9);
	s->carrierNoise = capn_read16(p.p, 10);
	s->latency = (int16_t) ((int16_t)capn_read16(p.p, 12));
	s->predetectIntegration = capn_read8(p.p, 14);
	s->postdetections = capn_read16(p.p, 16);
	s->unfilteredMeasurementIntegral = capn_read32(p.p, 20);
	s->unfilteredMeasurementFraction = capn_to_f32(capn_read32(p.p, 24));
	s->unfilteredTimeUncertainty = capn_to_f32(capn_read32(p.p, 28));
	s->unfilteredSpeed = capn_to_f32(capn_read32(p.p, 32));
	s->unfilteredSpeedUncertainty = capn_to_f32(capn_read32(p.p, 36));
	s->measurementStatus.p = capn_getp(p.p, 0, 0);
	s->multipathEstimate = capn_read32(p.p, 40);
	s->azimuth = capn_to_f32(capn_read32(p.p, 44));
	s->elevation = capn_to_f32(capn_read32(p.p, 48));
	s->carrierPhaseCyclesIntegral = (int32_t) ((int32_t)capn_read32(p.p, 52));
	s->carrierPhaseCyclesFraction = capn_read16(p.p, 18);
	s->fineSpeed = capn_to_f32(capn_read32(p.p, 56));
	s->fineSpeedUncertainty = capn_to_f32(capn_read32(p.p, 60));
	s->cycleSlipCount = capn_read8(p.p, 15);
}
void cereal_write_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV *s, cereal_QcomGnss_MeasurementReport_SV_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->svId);
	capn_write16(p.p, 2, (uint16_t) (s->observationState));
	capn_write8(p.p, 4, s->observations);
	capn_write8(p.p, 5, s->goodObservations);
	capn_write16(p.p, 6, s->gpsParityErrorCount);
	capn_write8(p.p, 1, (uint8_t) (s->glonassFrequencyIndex));
	capn_write8(p.p, 8, s->glonassHemmingErrorCount);
	capn_write8(p.p, 9, s->filterStages);
	capn_write16(p.p, 10, s->carrierNoise);
	capn_write16(p.p, 12, (uint16_t) (s->latency));
	capn_write8(p.p, 14, s->predetectIntegration);
	capn_write16(p.p, 16, s->postdetections);
	capn_write32(p.p, 20, s->unfilteredMeasurementIntegral);
	capn_write32(p.p, 24, capn_from_f32(s->unfilteredMeasurementFraction));
	capn_write32(p.p, 28, capn_from_f32(s->unfilteredTimeUncertainty));
	capn_write32(p.p, 32, capn_from_f32(s->unfilteredSpeed));
	capn_write32(p.p, 36, capn_from_f32(s->unfilteredSpeedUncertainty));
	capn_setp(p.p, 0, s->measurementStatus.p);
	capn_write32(p.p, 40, s->multipathEstimate);
	capn_write32(p.p, 44, capn_from_f32(s->azimuth));
	capn_write32(p.p, 48, capn_from_f32(s->elevation));
	capn_write32(p.p, 52, (uint32_t) (s->carrierPhaseCyclesIntegral));
	capn_write16(p.p, 18, s->carrierPhaseCyclesFraction);
	capn_write32(p.p, 56, capn_from_f32(s->fineSpeed));
	capn_write32(p.p, 60, capn_from_f32(s->fineSpeedUncertainty));
	capn_write8(p.p, 15, s->cycleSlipCount);
}
void cereal_get_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV *s, cereal_QcomGnss_MeasurementReport_SV_list l, int i) {
	cereal_QcomGnss_MeasurementReport_SV_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_MeasurementReport_SV(s, p);
}
void cereal_set_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV *s, cereal_QcomGnss_MeasurementReport_SV_list l, int i) {
	cereal_QcomGnss_MeasurementReport_SV_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_MeasurementReport_SV(s, p);
}

cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr cereal_new_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct capn_segment *s) {
	cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list cereal_new_QcomGnss_MeasurementReport_SV_MeasurementStatus_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus *s, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr p) {
	capn_resolve(&p.p);
	s->subMillisecondIsValid = (capn_read8(p.p, 0) & 1) != 0;
	s->subBitTimeIsKnown = (capn_read8(p.p, 0) & 2) != 0;
	s->satelliteTimeIsKnown = (capn_read8(p.p, 0) & 4) != 0;
	s->bitEdgeConfirmedFromSignal = (capn_read8(p.p, 0) & 8) != 0;
	s->measuredVelocity = (capn_read8(p.p, 0) & 16) != 0;
	s->fineOrCoarseVelocity = (capn_read8(p.p, 0) & 32) != 0;
	s->lockPointValid = (capn_read8(p.p, 0) & 64) != 0;
	s->lockPointPositive = (capn_read8(p.p, 0) & 128) != 0;
	s->lastUpdateFromDifference = (capn_read8(p.p, 1) & 1) != 0;
	s->lastUpdateFromVelocityDifference = (capn_read8(p.p, 1) & 2) != 0;
	s->strongIndicationOfCrossCorelation = (capn_read8(p.p, 1) & 4) != 0;
	s->tentativeMeasurement = (capn_read8(p.p, 1) & 8) != 0;
	s->measurementNotUsable = (capn_read8(p.p, 1) & 16) != 0;
	s->sirCheckIsNeeded = (capn_read8(p.p, 1) & 32) != 0;
	s->probationMode = (capn_read8(p.p, 1) & 64) != 0;
	s->glonassMeanderBitEdgeValid = (capn_read8(p.p, 1) & 128) != 0;
	s->glonassTimeMarkValid = (capn_read8(p.p, 2) & 1) != 0;
	s->gpsRoundRobinRxDiversity = (capn_read8(p.p, 2) & 2) != 0;
	s->gpsRxDiversity = (capn_read8(p.p, 2) & 4) != 0;
	s->gpsLowBandwidthRxDiversityCombined = (capn_read8(p.p, 2) & 8) != 0;
	s->gpsHighBandwidthNu4 = (capn_read8(p.p, 2) & 16) != 0;
	s->gpsHighBandwidthNu8 = (capn_read8(p.p, 2) & 32) != 0;
	s->gpsHighBandwidthUniform = (capn_read8(p.p, 2) & 64) != 0;
	s->gpsMultipathIndicator = (capn_read8(p.p, 2) & 128) != 0;
	s->imdJammingIndicator = (capn_read8(p.p, 3) & 1) != 0;
	s->lteB13TxJammingIndicator = (capn_read8(p.p, 3) & 2) != 0;
	s->freshMeasurementIndicator = (capn_read8(p.p, 3) & 4) != 0;
	s->multipathEstimateIsValid = (capn_read8(p.p, 3) & 8) != 0;
	s->directionIsValid = (capn_read8(p.p, 3) & 16) != 0;
}
void cereal_write_QcomGnss_MeasurementReport_SV_MeasurementStatus(const struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus *s, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->subMillisecondIsValid != 0);
	capn_write1(p.p, 1, s->subBitTimeIsKnown != 0);
	capn_write1(p.p, 2, s->satelliteTimeIsKnown != 0);
	capn_write1(p.p, 3, s->bitEdgeConfirmedFromSignal != 0);
	capn_write1(p.p, 4, s->measuredVelocity != 0);
	capn_write1(p.p, 5, s->fineOrCoarseVelocity != 0);
	capn_write1(p.p, 6, s->lockPointValid != 0);
	capn_write1(p.p, 7, s->lockPointPositive != 0);
	capn_write1(p.p, 8, s->lastUpdateFromDifference != 0);
	capn_write1(p.p, 9, s->lastUpdateFromVelocityDifference != 0);
	capn_write1(p.p, 10, s->strongIndicationOfCrossCorelation != 0);
	capn_write1(p.p, 11, s->tentativeMeasurement != 0);
	capn_write1(p.p, 12, s->measurementNotUsable != 0);
	capn_write1(p.p, 13, s->sirCheckIsNeeded != 0);
	capn_write1(p.p, 14, s->probationMode != 0);
	capn_write1(p.p, 15, s->glonassMeanderBitEdgeValid != 0);
	capn_write1(p.p, 16, s->glonassTimeMarkValid != 0);
	capn_write1(p.p, 17, s->gpsRoundRobinRxDiversity != 0);
	capn_write1(p.p, 18, s->gpsRxDiversity != 0);
	capn_write1(p.p, 19, s->gpsLowBandwidthRxDiversityCombined != 0);
	capn_write1(p.p, 20, s->gpsHighBandwidthNu4 != 0);
	capn_write1(p.p, 21, s->gpsHighBandwidthNu8 != 0);
	capn_write1(p.p, 22, s->gpsHighBandwidthUniform != 0);
	capn_write1(p.p, 23, s->gpsMultipathIndicator != 0);
	capn_write1(p.p, 24, s->imdJammingIndicator != 0);
	capn_write1(p.p, 25, s->lteB13TxJammingIndicator != 0);
	capn_write1(p.p, 26, s->freshMeasurementIndicator != 0);
	capn_write1(p.p, 27, s->multipathEstimateIsValid != 0);
	capn_write1(p.p, 28, s->directionIsValid != 0);
}
void cereal_get_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus *s, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list l, int i) {
	cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_MeasurementReport_SV_MeasurementStatus(s, p);
}
void cereal_set_QcomGnss_MeasurementReport_SV_MeasurementStatus(const struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus *s, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list l, int i) {
	cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_MeasurementReport_SV_MeasurementStatus(s, p);
}

cereal_QcomGnss_ClockReport_ptr cereal_new_QcomGnss_ClockReport(struct capn_segment *s) {
	cereal_QcomGnss_ClockReport_ptr p;
	p.p = capn_new_struct(s, 144, 0);
	return p;
}
cereal_QcomGnss_ClockReport_list cereal_new_QcomGnss_ClockReport_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_ClockReport_list p;
	p.p = capn_new_list(s, len, 144, 0);
	return p;
}
void cereal_read_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport *s, cereal_QcomGnss_ClockReport_ptr p) {
	capn_resolve(&p.p);
	s->hasFCount = (capn_read8(p.p, 0) & 1) != 0;
	s->fCount = capn_read32(p.p, 4);
	s->hasGpsWeekNumber = (capn_read8(p.p, 0) & 2) != 0;
	s->gpsWeekNumber = capn_read16(p.p, 2);
	s->hasGpsMilliseconds = (capn_read8(p.p, 0) & 4) != 0;
	s->gpsMilliseconds = capn_read32(p.p, 8);
	s->gpsTimeBias = capn_to_f32(capn_read32(p.p, 12));
	s->gpsClockTimeUncertainty = capn_to_f32(capn_read32(p.p, 16));
	s->gpsClockSource = capn_read8(p.p, 1);
	s->hasGlonassYear = (capn_read8(p.p, 0) & 8) != 0;
	s->glonassYear = capn_read8(p.p, 20);
	s->hasGlonassDay = (capn_read8(p.p, 0) & 16) != 0;
	s->glonassDay = capn_read16(p.p, 22);
	s->hasGlonassMilliseconds = (capn_read8(p.p, 0) & 32) != 0;
	s->glonassMilliseconds = capn_read32(p.p, 24);
	s->glonassTimeBias = capn_to_f32(capn_read32(p.p, 28));
	s->glonassClockTimeUncertainty = capn_to_f32(capn_read32(p.p, 32));
	s->glonassClockSource = capn_read8(p.p, 21);
	s->bdsWeek = capn_read16(p.p, 36);
	s->bdsMilliseconds = capn_read32(p.p, 40);
	s->bdsTimeBias = capn_to_f32(capn_read32(p.p, 44));
	s->bdsClockTimeUncertainty = capn_to_f32(capn_read32(p.p, 48));
	s->bdsClockSource = capn_read8(p.p, 38);
	s->galWeek = capn_read16(p.p, 52);
	s->galMilliseconds = capn_read32(p.p, 56);
	s->galTimeBias = capn_to_f32(capn_read32(p.p, 60));
	s->galClockTimeUncertainty = capn_to_f32(capn_read32(p.p, 64));
	s->galClockSource = capn_read8(p.p, 39);
	s->clockFrequencyBias = capn_to_f32(capn_read32(p.p, 68));
	s->clockFrequencyUncertainty = capn_to_f32(capn_read32(p.p, 72));
	s->frequencySource = capn_read8(p.p, 54);
	s->gpsLeapSeconds = capn_read8(p.p, 55);
	s->gpsLeapSecondsUncertainty = capn_read8(p.p, 76);
	s->gpsLeapSecondsSource = capn_read8(p.p, 77);
	s->gpsToGlonassTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 80));
	s->gpsToGlonassTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 84));
	s->gpsToBdsTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 88));
	s->gpsToBdsTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 92));
	s->bdsToGloTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 96));
	s->bdsToGloTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 100));
	s->gpsToGalTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 104));
	s->gpsToGalTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 108));
	s->galToGloTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 112));
	s->galToGloTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 116));
	s->galToBdsTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 120));
	s->galToBdsTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 124));
	s->hasRtcTime = (capn_read8(p.p, 0) & 64) != 0;
	s->systemRtcTime = capn_read32(p.p, 128);
	s->fCountOffset = capn_read32(p.p, 132);
	s->lpmRtcCount = capn_read32(p.p, 136);
	s->clockResets = capn_read32(p.p, 140);
}
void cereal_write_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport *s, cereal_QcomGnss_ClockReport_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->hasFCount != 0);
	capn_write32(p.p, 4, s->fCount);
	capn_write1(p.p, 1, s->hasGpsWeekNumber != 0);
	capn_write16(p.p, 2, s->gpsWeekNumber);
	capn_write1(p.p, 2, s->hasGpsMilliseconds != 0);
	capn_write32(p.p, 8, s->gpsMilliseconds);
	capn_write32(p.p, 12, capn_from_f32(s->gpsTimeBias));
	capn_write32(p.p, 16, capn_from_f32(s->gpsClockTimeUncertainty));
	capn_write8(p.p, 1, s->gpsClockSource);
	capn_write1(p.p, 3, s->hasGlonassYear != 0);
	capn_write8(p.p, 20, s->glonassYear);
	capn_write1(p.p, 4, s->hasGlonassDay != 0);
	capn_write16(p.p, 22, s->glonassDay);
	capn_write1(p.p, 5, s->hasGlonassMilliseconds != 0);
	capn_write32(p.p, 24, s->glonassMilliseconds);
	capn_write32(p.p, 28, capn_from_f32(s->glonassTimeBias));
	capn_write32(p.p, 32, capn_from_f32(s->glonassClockTimeUncertainty));
	capn_write8(p.p, 21, s->glonassClockSource);
	capn_write16(p.p, 36, s->bdsWeek);
	capn_write32(p.p, 40, s->bdsMilliseconds);
	capn_write32(p.p, 44, capn_from_f32(s->bdsTimeBias));
	capn_write32(p.p, 48, capn_from_f32(s->bdsClockTimeUncertainty));
	capn_write8(p.p, 38, s->bdsClockSource);
	capn_write16(p.p, 52, s->galWeek);
	capn_write32(p.p, 56, s->galMilliseconds);
	capn_write32(p.p, 60, capn_from_f32(s->galTimeBias));
	capn_write32(p.p, 64, capn_from_f32(s->galClockTimeUncertainty));
	capn_write8(p.p, 39, s->galClockSource);
	capn_write32(p.p, 68, capn_from_f32(s->clockFrequencyBias));
	capn_write32(p.p, 72, capn_from_f32(s->clockFrequencyUncertainty));
	capn_write8(p.p, 54, s->frequencySource);
	capn_write8(p.p, 55, s->gpsLeapSeconds);
	capn_write8(p.p, 76, s->gpsLeapSecondsUncertainty);
	capn_write8(p.p, 77, s->gpsLeapSecondsSource);
	capn_write32(p.p, 80, capn_from_f32(s->gpsToGlonassTimeBiasMilliseconds));
	capn_write32(p.p, 84, capn_from_f32(s->gpsToGlonassTimeBiasMillisecondsUncertainty));
	capn_write32(p.p, 88, capn_from_f32(s->gpsToBdsTimeBiasMilliseconds));
	capn_write32(p.p, 92, capn_from_f32(s->gpsToBdsTimeBiasMillisecondsUncertainty));
	capn_write32(p.p, 96, capn_from_f32(s->bdsToGloTimeBiasMilliseconds));
	capn_write32(p.p, 100, capn_from_f32(s->bdsToGloTimeBiasMillisecondsUncertainty));
	capn_write32(p.p, 104, capn_from_f32(s->gpsToGalTimeBiasMilliseconds));
	capn_write32(p.p, 108, capn_from_f32(s->gpsToGalTimeBiasMillisecondsUncertainty));
	capn_write32(p.p, 112, capn_from_f32(s->galToGloTimeBiasMilliseconds));
	capn_write32(p.p, 116, capn_from_f32(s->galToGloTimeBiasMillisecondsUncertainty));
	capn_write32(p.p, 120, capn_from_f32(s->galToBdsTimeBiasMilliseconds));
	capn_write32(p.p, 124, capn_from_f32(s->galToBdsTimeBiasMillisecondsUncertainty));
	capn_write1(p.p, 6, s->hasRtcTime != 0);
	capn_write32(p.p, 128, s->systemRtcTime);
	capn_write32(p.p, 132, s->fCountOffset);
	capn_write32(p.p, 136, s->lpmRtcCount);
	capn_write32(p.p, 140, s->clockResets);
}
void cereal_get_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport *s, cereal_QcomGnss_ClockReport_list l, int i) {
	cereal_QcomGnss_ClockReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_ClockReport(s, p);
}
void cereal_set_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport *s, cereal_QcomGnss_ClockReport_list l, int i) {
	cereal_QcomGnss_ClockReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_ClockReport(s, p);
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
	case cereal_Event_gpsLocation:
	case cereal_Event_carState:
	case cereal_Event_carControl:
	case cereal_Event_plan:
	case cereal_Event_liveLocation:
	case cereal_Event_ethernetData:
	case cereal_Event_navUpdate:
	case cereal_Event_cellInfo:
	case cereal_Event_wifiScan:
	case cereal_Event_androidGnss:
	case cereal_Event_qcomGnss:
		s->qcomGnss.p = capn_getp(p.p, 0, 0);
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
	case cereal_Event_gpsLocation:
	case cereal_Event_carState:
	case cereal_Event_carControl:
	case cereal_Event_plan:
	case cereal_Event_liveLocation:
	case cereal_Event_ethernetData:
	case cereal_Event_navUpdate:
	case cereal_Event_cellInfo:
	case cereal_Event_wifiScan:
	case cereal_Event_androidGnss:
	case cereal_Event_qcomGnss:
		capn_setp(p.p, 0, s->qcomGnss.p);
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
