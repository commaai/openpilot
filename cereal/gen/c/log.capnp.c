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
	p.p = capn_new_struct(s, 8, 15);
	return p;
}
cereal_InitData_list cereal_new_InitData_list(struct capn_segment *s, int len) {
	cereal_InitData_list p;
	p.p = capn_new_list(s, len, 8, 15);
	return p;
}
void cereal_read_InitData(struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	s->kernelArgs = capn_getp(p.p, 0, 0);
	s->kernelVersion = capn_get_text(p.p, 12, capn_val0);
	s->gctx = capn_get_text(p.p, 1, capn_val0);
	s->dongleId = capn_get_text(p.p, 2, capn_val0);
	s->deviceType = (enum cereal_InitData_DeviceType)(int) capn_read16(p.p, 0);
	s->version = capn_get_text(p.p, 3, capn_val0);
	s->gitCommit = capn_get_text(p.p, 8, capn_val0);
	s->gitBranch = capn_get_text(p.p, 9, capn_val0);
	s->gitRemote = capn_get_text(p.p, 10, capn_val0);
	s->androidBuildInfo.p = capn_getp(p.p, 4, 0);
	s->androidSensors.p = capn_getp(p.p, 5, 0);
	s->androidProperties.p = capn_getp(p.p, 13, 0);
	s->chffrAndroidExtra.p = capn_getp(p.p, 6, 0);
	s->iosBuildInfo.p = capn_getp(p.p, 11, 0);
	s->pandaInfo.p = capn_getp(p.p, 7, 0);
	s->dirty = (capn_read8(p.p, 2) & 1) != 0;
	s->passive = (capn_read8(p.p, 2) & 2) != 0;
	s->params.p = capn_getp(p.p, 14, 0);
}
void cereal_write_InitData(const struct cereal_InitData *s, cereal_InitData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->kernelArgs);
	capn_set_text(p.p, 12, s->kernelVersion);
	capn_set_text(p.p, 1, s->gctx);
	capn_set_text(p.p, 2, s->dongleId);
	capn_write16(p.p, 0, (uint16_t) (s->deviceType));
	capn_set_text(p.p, 3, s->version);
	capn_set_text(p.p, 8, s->gitCommit);
	capn_set_text(p.p, 9, s->gitBranch);
	capn_set_text(p.p, 10, s->gitRemote);
	capn_setp(p.p, 4, s->androidBuildInfo.p);
	capn_setp(p.p, 5, s->androidSensors.p);
	capn_setp(p.p, 13, s->androidProperties.p);
	capn_setp(p.p, 6, s->chffrAndroidExtra.p);
	capn_setp(p.p, 11, s->iosBuildInfo.p);
	capn_setp(p.p, 7, s->pandaInfo.p);
	capn_write1(p.p, 16, s->dirty != 0);
	capn_write1(p.p, 17, s->passive != 0);
	capn_setp(p.p, 14, s->params.p);
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

cereal_InitData_IosBuildInfo_ptr cereal_new_InitData_IosBuildInfo(struct capn_segment *s) {
	cereal_InitData_IosBuildInfo_ptr p;
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_InitData_IosBuildInfo_list cereal_new_InitData_IosBuildInfo_list(struct capn_segment *s, int len) {
	cereal_InitData_IosBuildInfo_list p;
	p.p = capn_new_list(s, len, 8, 3);
	return p;
}
void cereal_read_InitData_IosBuildInfo(struct cereal_InitData_IosBuildInfo *s, cereal_InitData_IosBuildInfo_ptr p) {
	capn_resolve(&p.p);
	s->appVersion = capn_get_text(p.p, 0, capn_val0);
	s->appBuild = capn_read32(p.p, 0);
	s->osVersion = capn_get_text(p.p, 1, capn_val0);
	s->deviceModel = capn_get_text(p.p, 2, capn_val0);
}
void cereal_write_InitData_IosBuildInfo(const struct cereal_InitData_IosBuildInfo *s, cereal_InitData_IosBuildInfo_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->appVersion);
	capn_write32(p.p, 0, s->appBuild);
	capn_set_text(p.p, 1, s->osVersion);
	capn_set_text(p.p, 2, s->deviceModel);
}
void cereal_get_InitData_IosBuildInfo(struct cereal_InitData_IosBuildInfo *s, cereal_InitData_IosBuildInfo_list l, int i) {
	cereal_InitData_IosBuildInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData_IosBuildInfo(s, p);
}
void cereal_set_InitData_IosBuildInfo(const struct cereal_InitData_IosBuildInfo *s, cereal_InitData_IosBuildInfo_list l, int i) {
	cereal_InitData_IosBuildInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData_IosBuildInfo(s, p);
}

cereal_InitData_PandaInfo_ptr cereal_new_InitData_PandaInfo(struct capn_segment *s) {
	cereal_InitData_PandaInfo_ptr p;
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_InitData_PandaInfo_list cereal_new_InitData_PandaInfo_list(struct capn_segment *s, int len) {
	cereal_InitData_PandaInfo_list p;
	p.p = capn_new_list(s, len, 8, 3);
	return p;
}
void cereal_read_InitData_PandaInfo(struct cereal_InitData_PandaInfo *s, cereal_InitData_PandaInfo_ptr p) {
	capn_resolve(&p.p);
	s->hasPanda = (capn_read8(p.p, 0) & 1) != 0;
	s->dongleId = capn_get_text(p.p, 0, capn_val0);
	s->stVersion = capn_get_text(p.p, 1, capn_val0);
	s->espVersion = capn_get_text(p.p, 2, capn_val0);
}
void cereal_write_InitData_PandaInfo(const struct cereal_InitData_PandaInfo *s, cereal_InitData_PandaInfo_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->hasPanda != 0);
	capn_set_text(p.p, 0, s->dongleId);
	capn_set_text(p.p, 1, s->stVersion);
	capn_set_text(p.p, 2, s->espVersion);
}
void cereal_get_InitData_PandaInfo(struct cereal_InitData_PandaInfo *s, cereal_InitData_PandaInfo_list l, int i) {
	cereal_InitData_PandaInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_InitData_PandaInfo(s, p);
}
void cereal_set_InitData_PandaInfo(const struct cereal_InitData_PandaInfo *s, cereal_InitData_PandaInfo_list l, int i) {
	cereal_InitData_PandaInfo_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_InitData_PandaInfo(s, p);
}

cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment *s) {
	cereal_FrameData_ptr p;
	p.p = capn_new_struct(s, 64, 3);
	return p;
}
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment *s, int len) {
	cereal_FrameData_list p;
	p.p = capn_new_list(s, len, 64, 3);
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
	s->lensPos = (int32_t) ((int32_t)capn_read32(p.p, 40));
	s->lensSag = capn_to_f32(capn_read32(p.p, 44));
	s->lensErr = capn_to_f32(capn_read32(p.p, 48));
	s->lensTruePos = capn_to_f32(capn_read32(p.p, 52));
	s->image = capn_get_data(p.p, 0);
	s->gainFrac = capn_to_f32(capn_read32(p.p, 56));
	s->frameType = (enum cereal_FrameData_FrameType)(int) capn_read16(p.p, 28);
	s->timestampSof = capn_read64(p.p, 32);
	s->transform.p = capn_getp(p.p, 2, 0);
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
	capn_write32(p.p, 40, (uint32_t) (s->lensPos));
	capn_write32(p.p, 44, capn_from_f32(s->lensSag));
	capn_write32(p.p, 48, capn_from_f32(s->lensErr));
	capn_write32(p.p, 52, capn_from_f32(s->lensTruePos));
	capn_setp(p.p, 0, s->image.p);
	capn_write32(p.p, 56, capn_from_f32(s->gainFrac));
	capn_write16(p.p, 28, (uint16_t) (s->frameType));
	capn_write64(p.p, 32, s->timestampSof);
	capn_setp(p.p, 2, s->transform.p);
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
	s->displayRotation = (int8_t) ((int8_t)capn_read8(p.p, 4));
}
void cereal_write_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult *s, cereal_FrameData_AndroidCaptureResult_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->sensitivity));
	capn_write64(p.p, 8, (uint64_t) (s->frameDuration));
	capn_write64(p.p, 16, (uint64_t) (s->exposureTime));
	capn_write64(p.p, 24, s->rollingShutterSkew);
	capn_setp(p.p, 0, s->colorCorrectionTransform.p);
	capn_setp(p.p, 1, s->colorCorrectionGains.p);
	capn_write8(p.p, 4, (uint8_t) (s->displayRotation));
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

cereal_Thumbnail_ptr cereal_new_Thumbnail(struct capn_segment *s) {
	cereal_Thumbnail_ptr p;
	p.p = capn_new_struct(s, 16, 1);
	return p;
}
cereal_Thumbnail_list cereal_new_Thumbnail_list(struct capn_segment *s, int len) {
	cereal_Thumbnail_list p;
	p.p = capn_new_list(s, len, 16, 1);
	return p;
}
void cereal_read_Thumbnail(struct cereal_Thumbnail *s, cereal_Thumbnail_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->timestampEof = capn_read64(p.p, 8);
	s->thumbnail = capn_get_data(p.p, 0);
}
void cereal_write_Thumbnail(const struct cereal_Thumbnail *s, cereal_Thumbnail_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write64(p.p, 8, s->timestampEof);
	capn_setp(p.p, 0, s->thumbnail.p);
}
void cereal_get_Thumbnail(struct cereal_Thumbnail *s, cereal_Thumbnail_list l, int i) {
	cereal_Thumbnail_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Thumbnail(s, p);
}
void cereal_set_Thumbnail(const struct cereal_Thumbnail *s, cereal_Thumbnail_list l, int i) {
	cereal_Thumbnail_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Thumbnail(s, p);
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
	p.p = capn_new_struct(s, 32, 1);
	return p;
}
cereal_SensorEventData_list cereal_new_SensorEventData_list(struct capn_segment *s, int len) {
	cereal_SensorEventData_list p;
	p.p = capn_new_list(s, len, 32, 1);
	return p;
}
void cereal_read_SensorEventData(struct cereal_SensorEventData *s, cereal_SensorEventData_ptr p) {
	capn_resolve(&p.p);
	s->version = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->sensor = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->type = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->timestamp = (int64_t) ((int64_t)(capn_read64(p.p, 16)));
	s->uncalibratedDEPRECATED = (capn_read8(p.p, 24) & 1) != 0;
	s->which = (enum cereal_SensorEventData_which)(int) capn_read16(p.p, 12);
	switch (s->which) {
	case cereal_SensorEventData_proximity:
	case cereal_SensorEventData_light:
		s->light = capn_to_f32(capn_read32(p.p, 28));
		break;
	case cereal_SensorEventData_acceleration:
	case cereal_SensorEventData_magnetic:
	case cereal_SensorEventData_orientation:
	case cereal_SensorEventData_gyro:
	case cereal_SensorEventData_pressure:
	case cereal_SensorEventData_magneticUncalibrated:
	case cereal_SensorEventData_gyroUncalibrated:
		s->gyroUncalibrated.p = capn_getp(p.p, 0, 0);
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
	capn_write1(p.p, 192, s->uncalibratedDEPRECATED != 0);
	capn_write16(p.p, 12, s->which);
	switch (s->which) {
	case cereal_SensorEventData_proximity:
	case cereal_SensorEventData_light:
		capn_write32(p.p, 28, capn_from_f32(s->light));
		break;
	case cereal_SensorEventData_acceleration:
	case cereal_SensorEventData_magnetic:
	case cereal_SensorEventData_orientation:
	case cereal_SensorEventData_gyro:
	case cereal_SensorEventData_pressure:
	case cereal_SensorEventData_magneticUncalibrated:
	case cereal_SensorEventData_gyroUncalibrated:
		capn_setp(p.p, 0, s->gyroUncalibrated.p);
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
	p.p = capn_new_struct(s, 64, 1);
	return p;
}
cereal_GpsLocationData_list cereal_new_GpsLocationData_list(struct capn_segment *s, int len) {
	cereal_GpsLocationData_list p;
	p.p = capn_new_list(s, len, 64, 1);
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
	s->vNED.p = capn_getp(p.p, 0, 0);
	s->verticalAccuracy = capn_to_f32(capn_read32(p.p, 48));
	s->bearingAccuracy = capn_to_f32(capn_read32(p.p, 52));
	s->speedAccuracy = capn_to_f32(capn_read32(p.p, 56));
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
	capn_setp(p.p, 0, s->vNED.p);
	capn_write32(p.p, 48, capn_from_f32(s->verticalAccuracy));
	capn_write32(p.p, 52, capn_from_f32(s->bearingAccuracy));
	capn_write32(p.p, 56, capn_from_f32(s->speedAccuracy));
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
	s->src = capn_read8(p.p, 6);
}
void cereal_write_CanData(const struct cereal_CanData *s, cereal_CanData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->address);
	capn_write16(p.p, 4, s->busTime);
	capn_setp(p.p, 0, s->dat.p);
	capn_write8(p.p, 6, s->src);
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
	p.p = capn_new_struct(s, 56, 1);
	return p;
}
cereal_ThermalData_list cereal_new_ThermalData_list(struct capn_segment *s, int len) {
	cereal_ThermalData_list p;
	p.p = capn_new_list(s, len, 56, 1);
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
	s->pa0 = capn_read16(p.p, 46);
	s->freeSpace = capn_to_f32(capn_read32(p.p, 16));
	s->batteryPercent = (int16_t) ((int16_t)capn_read16(p.p, 20));
	s->batteryStatus = capn_get_text(p.p, 0, capn_val0);
	s->batteryCurrent = (int32_t) ((int32_t)capn_read32(p.p, 28));
	s->batteryVoltage = (int32_t) ((int32_t)capn_read32(p.p, 40));
	s->usbOnline = (capn_read8(p.p, 24) & 2) != 0;
	s->networkType = (enum cereal_ThermalData_NetworkType)(int) capn_read16(p.p, 48);
	s->offroadPowerUsage = capn_read32(p.p, 52);
	s->fanSpeed = capn_read16(p.p, 22);
	s->started = (capn_read8(p.p, 24) & 1) != 0;
	s->startedTs = capn_read64(p.p, 32);
	s->thermalStatus = (enum cereal_ThermalData_ThermalStatus)(int) capn_read16(p.p, 26);
	s->chargingError = (capn_read8(p.p, 24) & 4) != 0;
	s->chargingDisabled = (capn_read8(p.p, 24) & 8) != 0;
	s->memUsedPercent = (int8_t) ((int8_t)capn_read8(p.p, 25));
	s->cpuPerc = (int8_t) ((int8_t)capn_read8(p.p, 44));
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
	capn_write16(p.p, 46, s->pa0);
	capn_write32(p.p, 16, capn_from_f32(s->freeSpace));
	capn_write16(p.p, 20, (uint16_t) (s->batteryPercent));
	capn_set_text(p.p, 0, s->batteryStatus);
	capn_write32(p.p, 28, (uint32_t) (s->batteryCurrent));
	capn_write32(p.p, 40, (uint32_t) (s->batteryVoltage));
	capn_write1(p.p, 193, s->usbOnline != 0);
	capn_write16(p.p, 48, (uint16_t) (s->networkType));
	capn_write32(p.p, 52, s->offroadPowerUsage);
	capn_write16(p.p, 22, s->fanSpeed);
	capn_write1(p.p, 192, s->started != 0);
	capn_write64(p.p, 32, s->startedTs);
	capn_write16(p.p, 26, (uint16_t) (s->thermalStatus));
	capn_write1(p.p, 194, s->chargingError != 0);
	capn_write1(p.p, 195, s->chargingDisabled != 0);
	capn_write8(p.p, 25, (uint8_t) (s->memUsedPercent));
	capn_write8(p.p, 44, (uint8_t) (s->cpuPerc));
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
	p.p = capn_new_struct(s, 40, 1);
	return p;
}
cereal_HealthData_list cereal_new_HealthData_list(struct capn_segment *s, int len) {
	cereal_HealthData_list p;
	p.p = capn_new_list(s, len, 40, 1);
	return p;
}
void cereal_read_HealthData(struct cereal_HealthData *s, cereal_HealthData_ptr p) {
	capn_resolve(&p.p);
	s->voltage = capn_read32(p.p, 0);
	s->current = capn_read32(p.p, 4);
	s->ignitionLine = (capn_read8(p.p, 8) & 1) != 0;
	s->controlsAllowed = (capn_read8(p.p, 8) & 2) != 0;
	s->gasInterceptorDetected = (capn_read8(p.p, 8) & 4) != 0;
	s->startedSignalDetectedDeprecated = (capn_read8(p.p, 8) & 8) != 0;
	s->hasGps = (capn_read8(p.p, 8) & 16) != 0;
	s->canSendErrs = capn_read32(p.p, 12);
	s->canFwdErrs = capn_read32(p.p, 16);
	s->canRxErrs = capn_read32(p.p, 36);
	s->gmlanSendErrs = capn_read32(p.p, 20);
	s->hwType = (enum cereal_HealthData_HwType)(int) capn_read16(p.p, 10);
	s->fanSpeedRpm = capn_read16(p.p, 24);
	s->usbPowerMode = (enum cereal_HealthData_UsbPowerMode)(int) capn_read16(p.p, 26);
	s->ignitionCan = (capn_read8(p.p, 8) & 32) != 0;
	s->safetyModel = (enum cereal_CarParams_SafetyModel)(int) capn_read16(p.p, 28);
	s->faultStatus = (enum cereal_HealthData_FaultStatus)(int) capn_read16(p.p, 30);
	s->powerSaveEnabled = (capn_read8(p.p, 8) & 64) != 0;
	s->uptime = capn_read32(p.p, 32);
	s->faults.p = capn_getp(p.p, 0, 0);
}
void cereal_write_HealthData(const struct cereal_HealthData *s, cereal_HealthData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->voltage);
	capn_write32(p.p, 4, s->current);
	capn_write1(p.p, 64, s->ignitionLine != 0);
	capn_write1(p.p, 65, s->controlsAllowed != 0);
	capn_write1(p.p, 66, s->gasInterceptorDetected != 0);
	capn_write1(p.p, 67, s->startedSignalDetectedDeprecated != 0);
	capn_write1(p.p, 68, s->hasGps != 0);
	capn_write32(p.p, 12, s->canSendErrs);
	capn_write32(p.p, 16, s->canFwdErrs);
	capn_write32(p.p, 36, s->canRxErrs);
	capn_write32(p.p, 20, s->gmlanSendErrs);
	capn_write16(p.p, 10, (uint16_t) (s->hwType));
	capn_write16(p.p, 24, s->fanSpeedRpm);
	capn_write16(p.p, 26, (uint16_t) (s->usbPowerMode));
	capn_write1(p.p, 69, s->ignitionCan != 0);
	capn_write16(p.p, 28, (uint16_t) (s->safetyModel));
	capn_write16(p.p, 30, (uint16_t) (s->faultStatus));
	capn_write1(p.p, 70, s->powerSaveEnabled != 0);
	capn_write32(p.p, 32, s->uptime);
	capn_setp(p.p, 0, s->faults.p);
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

cereal_RadarState_ptr cereal_new_RadarState(struct capn_segment *s) {
	cereal_RadarState_ptr p;
	p.p = capn_new_struct(s, 40, 5);
	return p;
}
cereal_RadarState_list cereal_new_RadarState_list(struct capn_segment *s, int len) {
	cereal_RadarState_list p;
	p.p = capn_new_list(s, len, 40, 5);
	return p;
}
void cereal_read_RadarState(struct cereal_RadarState *s, cereal_RadarState_ptr p) {
	capn_resolve(&p.p);
	s->canMonoTimes.p = capn_getp(p.p, 3, 0);
	s->mdMonoTime = capn_read64(p.p, 16);
	s->ftMonoTimeDEPRECATED = capn_read64(p.p, 24);
	s->controlsStateMonoTime = capn_read64(p.p, 32);
	s->radarErrors.p = capn_getp(p.p, 4, 0);
	s->warpMatrixDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->angleOffsetDEPRECATED = capn_to_f32(capn_read32(p.p, 0));
	s->calStatusDEPRECATED = (int8_t) ((int8_t)capn_read8(p.p, 4));
	s->calCycleDEPRECATED = (int32_t) ((int32_t)capn_read32(p.p, 12));
	s->calPercDEPRECATED = (int8_t) ((int8_t)capn_read8(p.p, 5));
	s->leadOne.p = capn_getp(p.p, 1, 0);
	s->leadTwo.p = capn_getp(p.p, 2, 0);
	s->cumLagMs = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_RadarState(const struct cereal_RadarState *s, cereal_RadarState_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 3, s->canMonoTimes.p);
	capn_write64(p.p, 16, s->mdMonoTime);
	capn_write64(p.p, 24, s->ftMonoTimeDEPRECATED);
	capn_write64(p.p, 32, s->controlsStateMonoTime);
	capn_setp(p.p, 4, s->radarErrors.p);
	capn_setp(p.p, 0, s->warpMatrixDEPRECATED.p);
	capn_write32(p.p, 0, capn_from_f32(s->angleOffsetDEPRECATED));
	capn_write8(p.p, 4, (uint8_t) (s->calStatusDEPRECATED));
	capn_write32(p.p, 12, (uint32_t) (s->calCycleDEPRECATED));
	capn_write8(p.p, 5, (uint8_t) (s->calPercDEPRECATED));
	capn_setp(p.p, 1, s->leadOne.p);
	capn_setp(p.p, 2, s->leadTwo.p);
	capn_write32(p.p, 8, capn_from_f32(s->cumLagMs));
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

cereal_RadarState_LeadData_ptr cereal_new_RadarState_LeadData(struct capn_segment *s) {
	cereal_RadarState_LeadData_ptr p;
	p.p = capn_new_struct(s, 56, 0);
	return p;
}
cereal_RadarState_LeadData_list cereal_new_RadarState_LeadData_list(struct capn_segment *s, int len) {
	cereal_RadarState_LeadData_list p;
	p.p = capn_new_list(s, len, 56, 0);
	return p;
}
void cereal_read_RadarState_LeadData(struct cereal_RadarState_LeadData *s, cereal_RadarState_LeadData_ptr p) {
	capn_resolve(&p.p);
	s->dRel = capn_to_f32(capn_read32(p.p, 0));
	s->yRel = capn_to_f32(capn_read32(p.p, 4));
	s->vRel = capn_to_f32(capn_read32(p.p, 8));
	s->aRel = capn_to_f32(capn_read32(p.p, 12));
	s->vLead = capn_to_f32(capn_read32(p.p, 16));
	s->aLeadDEPRECATED = capn_to_f32(capn_read32(p.p, 20));
	s->dPath = capn_to_f32(capn_read32(p.p, 24));
	s->vLat = capn_to_f32(capn_read32(p.p, 28));
	s->vLeadK = capn_to_f32(capn_read32(p.p, 32));
	s->aLeadK = capn_to_f32(capn_read32(p.p, 36));
	s->fcw = (capn_read8(p.p, 40) & 1) != 0;
	s->status = (capn_read8(p.p, 40) & 2) != 0;
	s->aLeadTau = capn_to_f32(capn_read32(p.p, 44));
	s->modelProb = capn_to_f32(capn_read32(p.p, 48));
	s->radar = (capn_read8(p.p, 40) & 4) != 0;
}
void cereal_write_RadarState_LeadData(const struct cereal_RadarState_LeadData *s, cereal_RadarState_LeadData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->dRel));
	capn_write32(p.p, 4, capn_from_f32(s->yRel));
	capn_write32(p.p, 8, capn_from_f32(s->vRel));
	capn_write32(p.p, 12, capn_from_f32(s->aRel));
	capn_write32(p.p, 16, capn_from_f32(s->vLead));
	capn_write32(p.p, 20, capn_from_f32(s->aLeadDEPRECATED));
	capn_write32(p.p, 24, capn_from_f32(s->dPath));
	capn_write32(p.p, 28, capn_from_f32(s->vLat));
	capn_write32(p.p, 32, capn_from_f32(s->vLeadK));
	capn_write32(p.p, 36, capn_from_f32(s->aLeadK));
	capn_write1(p.p, 320, s->fcw != 0);
	capn_write1(p.p, 321, s->status != 0);
	capn_write32(p.p, 44, capn_from_f32(s->aLeadTau));
	capn_write32(p.p, 48, capn_from_f32(s->modelProb));
	capn_write1(p.p, 322, s->radar != 0);
}
void cereal_get_RadarState_LeadData(struct cereal_RadarState_LeadData *s, cereal_RadarState_LeadData_list l, int i) {
	cereal_RadarState_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_RadarState_LeadData(s, p);
}
void cereal_set_RadarState_LeadData(const struct cereal_RadarState_LeadData *s, cereal_RadarState_LeadData_list l, int i) {
	cereal_RadarState_LeadData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_RadarState_LeadData(s, p);
}

cereal_LiveCalibrationData_ptr cereal_new_LiveCalibrationData(struct capn_segment *s) {
	cereal_LiveCalibrationData_ptr p;
	p.p = capn_new_struct(s, 8, 5);
	return p;
}
cereal_LiveCalibrationData_list cereal_new_LiveCalibrationData_list(struct capn_segment *s, int len) {
	cereal_LiveCalibrationData_list p;
	p.p = capn_new_list(s, len, 8, 5);
	return p;
}
void cereal_read_LiveCalibrationData(struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_ptr p) {
	capn_resolve(&p.p);
	s->warpMatrix.p = capn_getp(p.p, 0, 0);
	s->warpMatrix2.p = capn_getp(p.p, 2, 0);
	s->warpMatrixBig.p = capn_getp(p.p, 3, 0);
	s->calStatus = (int8_t) ((int8_t)capn_read8(p.p, 0));
	s->calCycle = (int32_t) ((int32_t)capn_read32(p.p, 4));
	s->calPerc = (int8_t) ((int8_t)capn_read8(p.p, 1));
	s->extrinsicMatrix.p = capn_getp(p.p, 1, 0);
	s->rpyCalib.p = capn_getp(p.p, 4, 0);
}
void cereal_write_LiveCalibrationData(const struct cereal_LiveCalibrationData *s, cereal_LiveCalibrationData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->warpMatrix.p);
	capn_setp(p.p, 2, s->warpMatrix2.p);
	capn_setp(p.p, 3, s->warpMatrixBig.p);
	capn_write8(p.p, 0, (uint8_t) (s->calStatus));
	capn_write32(p.p, 4, (uint32_t) (s->calCycle));
	capn_write8(p.p, 1, (uint8_t) (s->calPerc));
	capn_setp(p.p, 1, s->extrinsicMatrix.p);
	capn_setp(p.p, 4, s->rpyCalib.p);
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

cereal_ControlsState_ptr cereal_new_ControlsState(struct capn_segment *s) {
	cereal_ControlsState_ptr p;
	p.p = capn_new_struct(s, 176, 6);
	return p;
}
cereal_ControlsState_list cereal_new_ControlsState_list(struct capn_segment *s, int len) {
	cereal_ControlsState_list p;
	p.p = capn_new_list(s, len, 176, 6);
	return p;
}
void cereal_read_ControlsState(struct cereal_ControlsState *s, cereal_ControlsState_ptr p) {
	capn_resolve(&p.p);
	s->canMonoTimeDEPRECATED = capn_read64(p.p, 64);
	s->canMonoTimes.p = capn_getp(p.p, 0, 0);
	s->radarStateMonoTimeDEPRECATED = capn_read64(p.p, 72);
	s->mdMonoTimeDEPRECATED = capn_read64(p.p, 80);
	s->planMonoTime = capn_read64(p.p, 104);
	s->pathPlanMonoTime = capn_read64(p.p, 160);
	s->state = (enum cereal_ControlsState_OpenpilotState)(int) capn_read16(p.p, 116);
	s->vEgo = capn_to_f32(capn_read32(p.p, 0));
	s->vEgoRaw = capn_to_f32(capn_read32(p.p, 120));
	s->aEgoDEPRECATED = capn_to_f32(capn_read32(p.p, 4));
	s->longControlState = (enum cereal_ControlsState_LongControlState)(int) capn_read16(p.p, 90);
	s->vPid = capn_to_f32(capn_read32(p.p, 8));
	s->vTargetLead = capn_to_f32(capn_read32(p.p, 12));
	s->upAccelCmd = capn_to_f32(capn_read32(p.p, 16));
	s->uiAccelCmd = capn_to_f32(capn_read32(p.p, 20));
	s->ufAccelCmd = capn_to_f32(capn_read32(p.p, 124));
	s->yActualDEPRECATED = capn_to_f32(capn_read32(p.p, 24));
	s->yDesDEPRECATED = capn_to_f32(capn_read32(p.p, 28));
	s->upSteerDEPRECATED = capn_to_f32(capn_read32(p.p, 32));
	s->uiSteerDEPRECATED = capn_to_f32(capn_read32(p.p, 36));
	s->ufSteerDEPRECATED = capn_to_f32(capn_read32(p.p, 128));
	s->aTargetMinDEPRECATED = capn_to_f32(capn_read32(p.p, 40));
	s->aTargetMaxDEPRECATED = capn_to_f32(capn_read32(p.p, 44));
	s->aTarget = capn_to_f32(capn_read32(p.p, 132));
	s->jerkFactor = capn_to_f32(capn_read32(p.p, 48));
	s->angleSteers = capn_to_f32(capn_read32(p.p, 52));
	s->angleSteersDes = capn_to_f32(capn_read32(p.p, 112));
	s->curvature = capn_to_f32(capn_read32(p.p, 136));
	s->hudLeadDEPRECATED = (int32_t) ((int32_t)capn_read32(p.p, 56));
	s->cumLagMs = capn_to_f32(capn_read32(p.p, 60));
	s->startMonoTime = capn_read64(p.p, 152);
	s->mapValid = (capn_read8(p.p, 89) & 1) != 0;
	s->forceDecel = (capn_read8(p.p, 89) & 2) != 0;
	s->enabled = (capn_read8(p.p, 88) & 1) != 0;
	s->active = (capn_read8(p.p, 88) & 8) != 0;
	s->steerOverride = (capn_read8(p.p, 88) & 2) != 0;
	s->vCruise = capn_to_f32(capn_read32(p.p, 92));
	s->rearViewCam = (capn_read8(p.p, 88) & 4) != 0;
	s->alertText1 = capn_get_text(p.p, 1, capn_val0);
	s->alertText2 = capn_get_text(p.p, 2, capn_val0);
	s->alertStatus = (enum cereal_ControlsState_AlertStatus)(int) capn_read16(p.p, 118);
	s->alertSize = (enum cereal_ControlsState_AlertSize)(int) capn_read16(p.p, 140);
	s->alertBlinkingRate = capn_to_f32(capn_read32(p.p, 144));
	s->alertType = capn_get_text(p.p, 3, capn_val0);
	s->alertSoundDEPRECATED = capn_get_text(p.p, 4, capn_val0);
	s->alertSound = (enum cereal_CarControl_HUDControl_AudibleAlert)(int) capn_read16(p.p, 168);
	s->awarenessStatus = capn_to_f32(capn_read32(p.p, 96));
	s->angleModelBiasDEPRECATED = capn_to_f32(capn_read32(p.p, 100));
	s->gpsPlannerActive = (capn_read8(p.p, 88) & 16) != 0;
	s->engageable = (capn_read8(p.p, 88) & 32) != 0;
	s->driverMonitoringOn = (capn_read8(p.p, 88) & 64) != 0;
	s->vCurvature = capn_to_f32(capn_read32(p.p, 148));
	s->decelForTurn = (capn_read8(p.p, 88) & 128) != 0;
	s->decelForModel = (capn_read8(p.p, 89) & 4) != 0;
	s->canErrorCounter = capn_read32(p.p, 172);
	s->lateralControlState_which = (enum cereal_ControlsState_lateralControlState_which)(int) capn_read16(p.p, 142);
	switch (s->lateralControlState_which) {
	case cereal_ControlsState_lateralControlState_indiState:
	case cereal_ControlsState_lateralControlState_pidState:
	case cereal_ControlsState_lateralControlState_lqrState:
		s->lateralControlState.lqrState.p = capn_getp(p.p, 5, 0);
		break;
	default:
		break;
	}
}
void cereal_write_ControlsState(const struct cereal_ControlsState *s, cereal_ControlsState_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 64, s->canMonoTimeDEPRECATED);
	capn_setp(p.p, 0, s->canMonoTimes.p);
	capn_write64(p.p, 72, s->radarStateMonoTimeDEPRECATED);
	capn_write64(p.p, 80, s->mdMonoTimeDEPRECATED);
	capn_write64(p.p, 104, s->planMonoTime);
	capn_write64(p.p, 160, s->pathPlanMonoTime);
	capn_write16(p.p, 116, (uint16_t) (s->state));
	capn_write32(p.p, 0, capn_from_f32(s->vEgo));
	capn_write32(p.p, 120, capn_from_f32(s->vEgoRaw));
	capn_write32(p.p, 4, capn_from_f32(s->aEgoDEPRECATED));
	capn_write16(p.p, 90, (uint16_t) (s->longControlState));
	capn_write32(p.p, 8, capn_from_f32(s->vPid));
	capn_write32(p.p, 12, capn_from_f32(s->vTargetLead));
	capn_write32(p.p, 16, capn_from_f32(s->upAccelCmd));
	capn_write32(p.p, 20, capn_from_f32(s->uiAccelCmd));
	capn_write32(p.p, 124, capn_from_f32(s->ufAccelCmd));
	capn_write32(p.p, 24, capn_from_f32(s->yActualDEPRECATED));
	capn_write32(p.p, 28, capn_from_f32(s->yDesDEPRECATED));
	capn_write32(p.p, 32, capn_from_f32(s->upSteerDEPRECATED));
	capn_write32(p.p, 36, capn_from_f32(s->uiSteerDEPRECATED));
	capn_write32(p.p, 128, capn_from_f32(s->ufSteerDEPRECATED));
	capn_write32(p.p, 40, capn_from_f32(s->aTargetMinDEPRECATED));
	capn_write32(p.p, 44, capn_from_f32(s->aTargetMaxDEPRECATED));
	capn_write32(p.p, 132, capn_from_f32(s->aTarget));
	capn_write32(p.p, 48, capn_from_f32(s->jerkFactor));
	capn_write32(p.p, 52, capn_from_f32(s->angleSteers));
	capn_write32(p.p, 112, capn_from_f32(s->angleSteersDes));
	capn_write32(p.p, 136, capn_from_f32(s->curvature));
	capn_write32(p.p, 56, (uint32_t) (s->hudLeadDEPRECATED));
	capn_write32(p.p, 60, capn_from_f32(s->cumLagMs));
	capn_write64(p.p, 152, s->startMonoTime);
	capn_write1(p.p, 712, s->mapValid != 0);
	capn_write1(p.p, 713, s->forceDecel != 0);
	capn_write1(p.p, 704, s->enabled != 0);
	capn_write1(p.p, 707, s->active != 0);
	capn_write1(p.p, 705, s->steerOverride != 0);
	capn_write32(p.p, 92, capn_from_f32(s->vCruise));
	capn_write1(p.p, 706, s->rearViewCam != 0);
	capn_set_text(p.p, 1, s->alertText1);
	capn_set_text(p.p, 2, s->alertText2);
	capn_write16(p.p, 118, (uint16_t) (s->alertStatus));
	capn_write16(p.p, 140, (uint16_t) (s->alertSize));
	capn_write32(p.p, 144, capn_from_f32(s->alertBlinkingRate));
	capn_set_text(p.p, 3, s->alertType);
	capn_set_text(p.p, 4, s->alertSoundDEPRECATED);
	capn_write16(p.p, 168, (uint16_t) (s->alertSound));
	capn_write32(p.p, 96, capn_from_f32(s->awarenessStatus));
	capn_write32(p.p, 100, capn_from_f32(s->angleModelBiasDEPRECATED));
	capn_write1(p.p, 708, s->gpsPlannerActive != 0);
	capn_write1(p.p, 709, s->engageable != 0);
	capn_write1(p.p, 710, s->driverMonitoringOn != 0);
	capn_write32(p.p, 148, capn_from_f32(s->vCurvature));
	capn_write1(p.p, 711, s->decelForTurn != 0);
	capn_write1(p.p, 714, s->decelForModel != 0);
	capn_write32(p.p, 172, s->canErrorCounter);
	capn_write16(p.p, 142, s->lateralControlState_which);
	switch (s->lateralControlState_which) {
	case cereal_ControlsState_lateralControlState_indiState:
	case cereal_ControlsState_lateralControlState_pidState:
	case cereal_ControlsState_lateralControlState_lqrState:
		capn_setp(p.p, 5, s->lateralControlState.lqrState.p);
		break;
	default:
		break;
	}
}
void cereal_get_ControlsState(struct cereal_ControlsState *s, cereal_ControlsState_list l, int i) {
	cereal_ControlsState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ControlsState(s, p);
}
void cereal_set_ControlsState(const struct cereal_ControlsState *s, cereal_ControlsState_list l, int i) {
	cereal_ControlsState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ControlsState(s, p);
}

cereal_ControlsState_LateralINDIState_ptr cereal_new_ControlsState_LateralINDIState(struct capn_segment *s) {
	cereal_ControlsState_LateralINDIState_ptr p;
	p.p = capn_new_struct(s, 40, 0);
	return p;
}
cereal_ControlsState_LateralINDIState_list cereal_new_ControlsState_LateralINDIState_list(struct capn_segment *s, int len) {
	cereal_ControlsState_LateralINDIState_list p;
	p.p = capn_new_list(s, len, 40, 0);
	return p;
}
void cereal_read_ControlsState_LateralINDIState(struct cereal_ControlsState_LateralINDIState *s, cereal_ControlsState_LateralINDIState_ptr p) {
	capn_resolve(&p.p);
	s->active = (capn_read8(p.p, 0) & 1) != 0;
	s->steerAngle = capn_to_f32(capn_read32(p.p, 4));
	s->steerRate = capn_to_f32(capn_read32(p.p, 8));
	s->steerAccel = capn_to_f32(capn_read32(p.p, 12));
	s->rateSetPoint = capn_to_f32(capn_read32(p.p, 16));
	s->accelSetPoint = capn_to_f32(capn_read32(p.p, 20));
	s->accelError = capn_to_f32(capn_read32(p.p, 24));
	s->delayedOutput = capn_to_f32(capn_read32(p.p, 28));
	s->delta = capn_to_f32(capn_read32(p.p, 32));
	s->output = capn_to_f32(capn_read32(p.p, 36));
	s->saturated = (capn_read8(p.p, 0) & 2) != 0;
}
void cereal_write_ControlsState_LateralINDIState(const struct cereal_ControlsState_LateralINDIState *s, cereal_ControlsState_LateralINDIState_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->active != 0);
	capn_write32(p.p, 4, capn_from_f32(s->steerAngle));
	capn_write32(p.p, 8, capn_from_f32(s->steerRate));
	capn_write32(p.p, 12, capn_from_f32(s->steerAccel));
	capn_write32(p.p, 16, capn_from_f32(s->rateSetPoint));
	capn_write32(p.p, 20, capn_from_f32(s->accelSetPoint));
	capn_write32(p.p, 24, capn_from_f32(s->accelError));
	capn_write32(p.p, 28, capn_from_f32(s->delayedOutput));
	capn_write32(p.p, 32, capn_from_f32(s->delta));
	capn_write32(p.p, 36, capn_from_f32(s->output));
	capn_write1(p.p, 1, s->saturated != 0);
}
void cereal_get_ControlsState_LateralINDIState(struct cereal_ControlsState_LateralINDIState *s, cereal_ControlsState_LateralINDIState_list l, int i) {
	cereal_ControlsState_LateralINDIState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ControlsState_LateralINDIState(s, p);
}
void cereal_set_ControlsState_LateralINDIState(const struct cereal_ControlsState_LateralINDIState *s, cereal_ControlsState_LateralINDIState_list l, int i) {
	cereal_ControlsState_LateralINDIState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ControlsState_LateralINDIState(s, p);
}

cereal_ControlsState_LateralPIDState_ptr cereal_new_ControlsState_LateralPIDState(struct capn_segment *s) {
	cereal_ControlsState_LateralPIDState_ptr p;
	p.p = capn_new_struct(s, 32, 0);
	return p;
}
cereal_ControlsState_LateralPIDState_list cereal_new_ControlsState_LateralPIDState_list(struct capn_segment *s, int len) {
	cereal_ControlsState_LateralPIDState_list p;
	p.p = capn_new_list(s, len, 32, 0);
	return p;
}
void cereal_read_ControlsState_LateralPIDState(struct cereal_ControlsState_LateralPIDState *s, cereal_ControlsState_LateralPIDState_ptr p) {
	capn_resolve(&p.p);
	s->active = (capn_read8(p.p, 0) & 1) != 0;
	s->steerAngle = capn_to_f32(capn_read32(p.p, 4));
	s->steerRate = capn_to_f32(capn_read32(p.p, 8));
	s->angleError = capn_to_f32(capn_read32(p.p, 12));
	s->p = capn_to_f32(capn_read32(p.p, 16));
	s->i = capn_to_f32(capn_read32(p.p, 20));
	s->f = capn_to_f32(capn_read32(p.p, 24));
	s->output = capn_to_f32(capn_read32(p.p, 28));
	s->saturated = (capn_read8(p.p, 0) & 2) != 0;
}
void cereal_write_ControlsState_LateralPIDState(const struct cereal_ControlsState_LateralPIDState *s, cereal_ControlsState_LateralPIDState_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->active != 0);
	capn_write32(p.p, 4, capn_from_f32(s->steerAngle));
	capn_write32(p.p, 8, capn_from_f32(s->steerRate));
	capn_write32(p.p, 12, capn_from_f32(s->angleError));
	capn_write32(p.p, 16, capn_from_f32(s->p));
	capn_write32(p.p, 20, capn_from_f32(s->i));
	capn_write32(p.p, 24, capn_from_f32(s->f));
	capn_write32(p.p, 28, capn_from_f32(s->output));
	capn_write1(p.p, 1, s->saturated != 0);
}
void cereal_get_ControlsState_LateralPIDState(struct cereal_ControlsState_LateralPIDState *s, cereal_ControlsState_LateralPIDState_list l, int i) {
	cereal_ControlsState_LateralPIDState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ControlsState_LateralPIDState(s, p);
}
void cereal_set_ControlsState_LateralPIDState(const struct cereal_ControlsState_LateralPIDState *s, cereal_ControlsState_LateralPIDState_list l, int i) {
	cereal_ControlsState_LateralPIDState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ControlsState_LateralPIDState(s, p);
}

cereal_ControlsState_LateralLQRState_ptr cereal_new_ControlsState_LateralLQRState(struct capn_segment *s) {
	cereal_ControlsState_LateralLQRState_ptr p;
	p.p = capn_new_struct(s, 24, 0);
	return p;
}
cereal_ControlsState_LateralLQRState_list cereal_new_ControlsState_LateralLQRState_list(struct capn_segment *s, int len) {
	cereal_ControlsState_LateralLQRState_list p;
	p.p = capn_new_list(s, len, 24, 0);
	return p;
}
void cereal_read_ControlsState_LateralLQRState(struct cereal_ControlsState_LateralLQRState *s, cereal_ControlsState_LateralLQRState_ptr p) {
	capn_resolve(&p.p);
	s->active = (capn_read8(p.p, 0) & 1) != 0;
	s->steerAngle = capn_to_f32(capn_read32(p.p, 4));
	s->i = capn_to_f32(capn_read32(p.p, 8));
	s->output = capn_to_f32(capn_read32(p.p, 12));
	s->lqrOutput = capn_to_f32(capn_read32(p.p, 16));
	s->saturated = (capn_read8(p.p, 0) & 2) != 0;
}
void cereal_write_ControlsState_LateralLQRState(const struct cereal_ControlsState_LateralLQRState *s, cereal_ControlsState_LateralLQRState_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->active != 0);
	capn_write32(p.p, 4, capn_from_f32(s->steerAngle));
	capn_write32(p.p, 8, capn_from_f32(s->i));
	capn_write32(p.p, 12, capn_from_f32(s->output));
	capn_write32(p.p, 16, capn_from_f32(s->lqrOutput));
	capn_write1(p.p, 1, s->saturated != 0);
}
void cereal_get_ControlsState_LateralLQRState(struct cereal_ControlsState_LateralLQRState *s, cereal_ControlsState_LateralLQRState_list l, int i) {
	cereal_ControlsState_LateralLQRState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ControlsState_LateralLQRState(s, p);
}
void cereal_set_ControlsState_LateralLQRState(const struct cereal_ControlsState_LateralLQRState *s, cereal_ControlsState_LateralLQRState_list l, int i) {
	cereal_ControlsState_LateralLQRState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ControlsState_LateralLQRState(s, p);
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
	p.p = capn_new_struct(s, 16, 10);
	return p;
}
cereal_ModelData_list cereal_new_ModelData_list(struct capn_segment *s, int len) {
	cereal_ModelData_list p;
	p.p = capn_new_list(s, len, 16, 10);
	return p;
}
void cereal_read_ModelData(struct cereal_ModelData *s, cereal_ModelData_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->timestampEof = capn_read64(p.p, 8);
	s->path.p = capn_getp(p.p, 0, 0);
	s->leftLane.p = capn_getp(p.p, 1, 0);
	s->rightLane.p = capn_getp(p.p, 2, 0);
	s->lead.p = capn_getp(p.p, 3, 0);
	s->freePath.p = capn_getp(p.p, 5, 0);
	s->settings.p = capn_getp(p.p, 4, 0);
	s->leadFuture.p = capn_getp(p.p, 6, 0);
	s->speed.p = capn_getp(p.p, 7, 0);
	s->meta.p = capn_getp(p.p, 8, 0);
	s->longitudinal.p = capn_getp(p.p, 9, 0);
}
void cereal_write_ModelData(const struct cereal_ModelData *s, cereal_ModelData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write64(p.p, 8, s->timestampEof);
	capn_setp(p.p, 0, s->path.p);
	capn_setp(p.p, 1, s->leftLane.p);
	capn_setp(p.p, 2, s->rightLane.p);
	capn_setp(p.p, 3, s->lead.p);
	capn_setp(p.p, 5, s->freePath.p);
	capn_setp(p.p, 4, s->settings.p);
	capn_setp(p.p, 6, s->leadFuture.p);
	capn_setp(p.p, 7, s->speed.p);
	capn_setp(p.p, 8, s->meta.p);
	capn_setp(p.p, 9, s->longitudinal.p);
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
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_ModelData_PathData_list cereal_new_ModelData_PathData_list(struct capn_segment *s, int len) {
	cereal_ModelData_PathData_list p;
	p.p = capn_new_list(s, len, 8, 3);
	return p;
}
void cereal_read_ModelData_PathData(struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_ptr p) {
	capn_resolve(&p.p);
	s->points.p = capn_getp(p.p, 0, 0);
	s->prob = capn_to_f32(capn_read32(p.p, 0));
	s->std = capn_to_f32(capn_read32(p.p, 4));
	s->stds.p = capn_getp(p.p, 1, 0);
	s->poly.p = capn_getp(p.p, 2, 0);
}
void cereal_write_ModelData_PathData(const struct cereal_ModelData_PathData *s, cereal_ModelData_PathData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->points.p);
	capn_write32(p.p, 0, capn_from_f32(s->prob));
	capn_write32(p.p, 4, capn_from_f32(s->std));
	capn_setp(p.p, 1, s->stds.p);
	capn_setp(p.p, 2, s->poly.p);
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
	p.p = capn_new_struct(s, 40, 0);
	return p;
}
cereal_ModelData_LeadData_list cereal_new_ModelData_LeadData_list(struct capn_segment *s, int len) {
	cereal_ModelData_LeadData_list p;
	p.p = capn_new_list(s, len, 40, 0);
	return p;
}
void cereal_read_ModelData_LeadData(struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_ptr p) {
	capn_resolve(&p.p);
	s->dist = capn_to_f32(capn_read32(p.p, 0));
	s->prob = capn_to_f32(capn_read32(p.p, 4));
	s->std = capn_to_f32(capn_read32(p.p, 8));
	s->relVel = capn_to_f32(capn_read32(p.p, 12));
	s->relVelStd = capn_to_f32(capn_read32(p.p, 16));
	s->relY = capn_to_f32(capn_read32(p.p, 20));
	s->relYStd = capn_to_f32(capn_read32(p.p, 24));
	s->relA = capn_to_f32(capn_read32(p.p, 28));
	s->relAStd = capn_to_f32(capn_read32(p.p, 32));
}
void cereal_write_ModelData_LeadData(const struct cereal_ModelData_LeadData *s, cereal_ModelData_LeadData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->dist));
	capn_write32(p.p, 4, capn_from_f32(s->prob));
	capn_write32(p.p, 8, capn_from_f32(s->std));
	capn_write32(p.p, 12, capn_from_f32(s->relVel));
	capn_write32(p.p, 16, capn_from_f32(s->relVelStd));
	capn_write32(p.p, 20, capn_from_f32(s->relY));
	capn_write32(p.p, 24, capn_from_f32(s->relYStd));
	capn_write32(p.p, 28, capn_from_f32(s->relA));
	capn_write32(p.p, 32, capn_from_f32(s->relAStd));
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
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_ModelData_ModelSettings_list cereal_new_ModelData_ModelSettings_list(struct capn_segment *s, int len) {
	cereal_ModelData_ModelSettings_list p;
	p.p = capn_new_list(s, len, 8, 3);
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
	s->inputTransform.p = capn_getp(p.p, 2, 0);
}
void cereal_write_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings *s, cereal_ModelData_ModelSettings_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->bigBoxX);
	capn_write16(p.p, 2, s->bigBoxY);
	capn_write16(p.p, 4, s->bigBoxWidth);
	capn_write16(p.p, 6, s->bigBoxHeight);
	capn_setp(p.p, 0, s->boxProjection.p);
	capn_setp(p.p, 1, s->yuvCorrection.p);
	capn_setp(p.p, 2, s->inputTransform.p);
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

cereal_ModelData_MetaData_ptr cereal_new_ModelData_MetaData(struct capn_segment *s) {
	cereal_ModelData_MetaData_ptr p;
	p.p = capn_new_struct(s, 16, 1);
	return p;
}
cereal_ModelData_MetaData_list cereal_new_ModelData_MetaData_list(struct capn_segment *s, int len) {
	cereal_ModelData_MetaData_list p;
	p.p = capn_new_list(s, len, 16, 1);
	return p;
}
void cereal_read_ModelData_MetaData(struct cereal_ModelData_MetaData *s, cereal_ModelData_MetaData_ptr p) {
	capn_resolve(&p.p);
	s->engagedProb = capn_to_f32(capn_read32(p.p, 0));
	s->desirePrediction.p = capn_getp(p.p, 0, 0);
	s->brakeDisengageProb = capn_to_f32(capn_read32(p.p, 4));
	s->gasDisengageProb = capn_to_f32(capn_read32(p.p, 8));
	s->steerOverrideProb = capn_to_f32(capn_read32(p.p, 12));
}
void cereal_write_ModelData_MetaData(const struct cereal_ModelData_MetaData *s, cereal_ModelData_MetaData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->engagedProb));
	capn_setp(p.p, 0, s->desirePrediction.p);
	capn_write32(p.p, 4, capn_from_f32(s->brakeDisengageProb));
	capn_write32(p.p, 8, capn_from_f32(s->gasDisengageProb));
	capn_write32(p.p, 12, capn_from_f32(s->steerOverrideProb));
}
void cereal_get_ModelData_MetaData(struct cereal_ModelData_MetaData *s, cereal_ModelData_MetaData_list l, int i) {
	cereal_ModelData_MetaData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData_MetaData(s, p);
}
void cereal_set_ModelData_MetaData(const struct cereal_ModelData_MetaData *s, cereal_ModelData_MetaData_list l, int i) {
	cereal_ModelData_MetaData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData_MetaData(s, p);
}

cereal_ModelData_LongitudinalData_ptr cereal_new_ModelData_LongitudinalData(struct capn_segment *s) {
	cereal_ModelData_LongitudinalData_ptr p;
	p.p = capn_new_struct(s, 0, 2);
	return p;
}
cereal_ModelData_LongitudinalData_list cereal_new_ModelData_LongitudinalData_list(struct capn_segment *s, int len) {
	cereal_ModelData_LongitudinalData_list p;
	p.p = capn_new_list(s, len, 0, 2);
	return p;
}
void cereal_read_ModelData_LongitudinalData(struct cereal_ModelData_LongitudinalData *s, cereal_ModelData_LongitudinalData_ptr p) {
	capn_resolve(&p.p);
	s->speeds.p = capn_getp(p.p, 0, 0);
	s->accelerations.p = capn_getp(p.p, 1, 0);
}
void cereal_write_ModelData_LongitudinalData(const struct cereal_ModelData_LongitudinalData *s, cereal_ModelData_LongitudinalData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->speeds.p);
	capn_setp(p.p, 1, s->accelerations.p);
}
void cereal_get_ModelData_LongitudinalData(struct cereal_ModelData_LongitudinalData *s, cereal_ModelData_LongitudinalData_list l, int i) {
	cereal_ModelData_LongitudinalData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ModelData_LongitudinalData(s, p);
}
void cereal_set_ModelData_LongitudinalData(const struct cereal_ModelData_LongitudinalData *s, cereal_ModelData_LongitudinalData_list l, int i) {
	cereal_ModelData_LongitudinalData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ModelData_LongitudinalData(s, p);
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
	p.p = capn_new_struct(s, 80, 3);
	return p;
}
cereal_Plan_list cereal_new_Plan_list(struct capn_segment *s, int len) {
	cereal_Plan_list p;
	p.p = capn_new_list(s, len, 80, 3);
	return p;
}
void cereal_read_Plan(struct cereal_Plan *s, cereal_Plan_ptr p) {
	capn_resolve(&p.p);
	s->mdMonoTime = capn_read64(p.p, 24);
	s->radarStateMonoTime = capn_read64(p.p, 32);
	s->commIssue = (capn_read8(p.p, 1) & 8) != 0;
	s->eventsDEPRECATED.p = capn_getp(p.p, 2, 0);
	s->lateralValidDEPRECATED = (capn_read8(p.p, 0) & 1) != 0;
	s->dPolyDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->laneWidthDEPRECATED = capn_to_f32(capn_read32(p.p, 20));
	s->longitudinalValidDEPRECATED = (capn_read8(p.p, 0) & 2) != 0;
	s->vCruise = capn_to_f32(capn_read32(p.p, 44));
	s->aCruise = capn_to_f32(capn_read32(p.p, 48));
	s->vTarget = capn_to_f32(capn_read32(p.p, 4));
	s->vTargetFuture = capn_to_f32(capn_read32(p.p, 40));
	s->vMax = capn_to_f32(capn_read32(p.p, 56));
	s->aTargetMinDEPRECATED = capn_to_f32(capn_read32(p.p, 8));
	s->aTargetMaxDEPRECATED = capn_to_f32(capn_read32(p.p, 12));
	s->aTarget = capn_to_f32(capn_read32(p.p, 52));
	s->vStart = capn_to_f32(capn_read32(p.p, 64));
	s->aStart = capn_to_f32(capn_read32(p.p, 68));
	s->jerkFactor = capn_to_f32(capn_read32(p.p, 16));
	s->hasLead = (capn_read8(p.p, 0) & 4) != 0;
	s->hasLeftLaneDEPRECATED = (capn_read8(p.p, 0) & 64) != 0;
	s->hasRightLaneDEPRECATED = (capn_read8(p.p, 0) & 128) != 0;
	s->fcw = (capn_read8(p.p, 0) & 8) != 0;
	s->longitudinalPlanSource = (enum cereal_Plan_LongitudinalPlanSource)(int) capn_read16(p.p, 2);
	s->gpsTrajectory.p = capn_getp(p.p, 1, 0);
	s->gpsPlannerActive = (capn_read8(p.p, 0) & 16) != 0;
	s->vCurvature = capn_to_f32(capn_read32(p.p, 60));
	s->decelForTurn = (capn_read8(p.p, 0) & 32) != 0;
	s->mapValid = (capn_read8(p.p, 1) & 1) != 0;
	s->radarValid = (capn_read8(p.p, 1) & 2) != 0;
	s->radarCanError = (capn_read8(p.p, 1) & 4) != 0;
	s->processingDelay = capn_to_f32(capn_read32(p.p, 72));
}
void cereal_write_Plan(const struct cereal_Plan *s, cereal_Plan_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 24, s->mdMonoTime);
	capn_write64(p.p, 32, s->radarStateMonoTime);
	capn_write1(p.p, 11, s->commIssue != 0);
	capn_setp(p.p, 2, s->eventsDEPRECATED.p);
	capn_write1(p.p, 0, s->lateralValidDEPRECATED != 0);
	capn_setp(p.p, 0, s->dPolyDEPRECATED.p);
	capn_write32(p.p, 20, capn_from_f32(s->laneWidthDEPRECATED));
	capn_write1(p.p, 1, s->longitudinalValidDEPRECATED != 0);
	capn_write32(p.p, 44, capn_from_f32(s->vCruise));
	capn_write32(p.p, 48, capn_from_f32(s->aCruise));
	capn_write32(p.p, 4, capn_from_f32(s->vTarget));
	capn_write32(p.p, 40, capn_from_f32(s->vTargetFuture));
	capn_write32(p.p, 56, capn_from_f32(s->vMax));
	capn_write32(p.p, 8, capn_from_f32(s->aTargetMinDEPRECATED));
	capn_write32(p.p, 12, capn_from_f32(s->aTargetMaxDEPRECATED));
	capn_write32(p.p, 52, capn_from_f32(s->aTarget));
	capn_write32(p.p, 64, capn_from_f32(s->vStart));
	capn_write32(p.p, 68, capn_from_f32(s->aStart));
	capn_write32(p.p, 16, capn_from_f32(s->jerkFactor));
	capn_write1(p.p, 2, s->hasLead != 0);
	capn_write1(p.p, 6, s->hasLeftLaneDEPRECATED != 0);
	capn_write1(p.p, 7, s->hasRightLaneDEPRECATED != 0);
	capn_write1(p.p, 3, s->fcw != 0);
	capn_write16(p.p, 2, (uint16_t) (s->longitudinalPlanSource));
	capn_setp(p.p, 1, s->gpsTrajectory.p);
	capn_write1(p.p, 4, s->gpsPlannerActive != 0);
	capn_write32(p.p, 60, capn_from_f32(s->vCurvature));
	capn_write1(p.p, 5, s->decelForTurn != 0);
	capn_write1(p.p, 8, s->mapValid != 0);
	capn_write1(p.p, 9, s->radarValid != 0);
	capn_write1(p.p, 10, s->radarCanError != 0);
	capn_write32(p.p, 72, capn_from_f32(s->processingDelay));
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

cereal_Plan_GpsTrajectory_ptr cereal_new_Plan_GpsTrajectory(struct capn_segment *s) {
	cereal_Plan_GpsTrajectory_ptr p;
	p.p = capn_new_struct(s, 0, 2);
	return p;
}
cereal_Plan_GpsTrajectory_list cereal_new_Plan_GpsTrajectory_list(struct capn_segment *s, int len) {
	cereal_Plan_GpsTrajectory_list p;
	p.p = capn_new_list(s, len, 0, 2);
	return p;
}
void cereal_read_Plan_GpsTrajectory(struct cereal_Plan_GpsTrajectory *s, cereal_Plan_GpsTrajectory_ptr p) {
	capn_resolve(&p.p);
	s->x.p = capn_getp(p.p, 0, 0);
	s->y.p = capn_getp(p.p, 1, 0);
}
void cereal_write_Plan_GpsTrajectory(const struct cereal_Plan_GpsTrajectory *s, cereal_Plan_GpsTrajectory_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->x.p);
	capn_setp(p.p, 1, s->y.p);
}
void cereal_get_Plan_GpsTrajectory(struct cereal_Plan_GpsTrajectory *s, cereal_Plan_GpsTrajectory_list l, int i) {
	cereal_Plan_GpsTrajectory_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Plan_GpsTrajectory(s, p);
}
void cereal_set_Plan_GpsTrajectory(const struct cereal_Plan_GpsTrajectory *s, cereal_Plan_GpsTrajectory_list l, int i) {
	cereal_Plan_GpsTrajectory_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Plan_GpsTrajectory(s, p);
}

cereal_PathPlan_ptr cereal_new_PathPlan(struct capn_segment *s) {
	cereal_PathPlan_ptr p;
	p.p = capn_new_struct(s, 40, 4);
	return p;
}
cereal_PathPlan_list cereal_new_PathPlan_list(struct capn_segment *s, int len) {
	cereal_PathPlan_list p;
	p.p = capn_new_list(s, len, 40, 4);
	return p;
}
void cereal_read_PathPlan(struct cereal_PathPlan *s, cereal_PathPlan_ptr p) {
	capn_resolve(&p.p);
	s->laneWidth = capn_to_f32(capn_read32(p.p, 0));
	s->dPoly.p = capn_getp(p.p, 0, 0);
	s->cPoly.p = capn_getp(p.p, 1, 0);
	s->cProb = capn_to_f32(capn_read32(p.p, 4));
	s->lPoly.p = capn_getp(p.p, 2, 0);
	s->lProb = capn_to_f32(capn_read32(p.p, 8));
	s->rPoly.p = capn_getp(p.p, 3, 0);
	s->rProb = capn_to_f32(capn_read32(p.p, 12));
	s->angleSteers = capn_to_f32(capn_read32(p.p, 16));
	s->rateSteers = capn_to_f32(capn_read32(p.p, 28));
	s->mpcSolutionValid = (capn_read8(p.p, 20) & 1) != 0;
	s->paramsValid = (capn_read8(p.p, 20) & 2) != 0;
	s->modelValidDEPRECATED = (capn_read8(p.p, 20) & 4) != 0;
	s->angleOffset = capn_to_f32(capn_read32(p.p, 24));
	s->sensorValid = (capn_read8(p.p, 20) & 8) != 0;
	s->commIssue = (capn_read8(p.p, 20) & 16) != 0;
	s->posenetValid = (capn_read8(p.p, 20) & 32) != 0;
	s->desire = (enum cereal_PathPlan_Desire)(int) capn_read16(p.p, 22);
	s->laneChangeState = (enum cereal_PathPlan_LaneChangeState)(int) capn_read16(p.p, 32);
	s->laneChangeDirection = (enum cereal_PathPlan_LaneChangeDirection)(int) capn_read16(p.p, 34);
}
void cereal_write_PathPlan(const struct cereal_PathPlan *s, cereal_PathPlan_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->laneWidth));
	capn_setp(p.p, 0, s->dPoly.p);
	capn_setp(p.p, 1, s->cPoly.p);
	capn_write32(p.p, 4, capn_from_f32(s->cProb));
	capn_setp(p.p, 2, s->lPoly.p);
	capn_write32(p.p, 8, capn_from_f32(s->lProb));
	capn_setp(p.p, 3, s->rPoly.p);
	capn_write32(p.p, 12, capn_from_f32(s->rProb));
	capn_write32(p.p, 16, capn_from_f32(s->angleSteers));
	capn_write32(p.p, 28, capn_from_f32(s->rateSteers));
	capn_write1(p.p, 160, s->mpcSolutionValid != 0);
	capn_write1(p.p, 161, s->paramsValid != 0);
	capn_write1(p.p, 162, s->modelValidDEPRECATED != 0);
	capn_write32(p.p, 24, capn_from_f32(s->angleOffset));
	capn_write1(p.p, 163, s->sensorValid != 0);
	capn_write1(p.p, 164, s->commIssue != 0);
	capn_write1(p.p, 165, s->posenetValid != 0);
	capn_write16(p.p, 22, (uint16_t) (s->desire));
	capn_write16(p.p, 32, (uint16_t) (s->laneChangeState));
	capn_write16(p.p, 34, (uint16_t) (s->laneChangeDirection));
}
void cereal_get_PathPlan(struct cereal_PathPlan *s, cereal_PathPlan_list l, int i) {
	cereal_PathPlan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_PathPlan(s, p);
}
void cereal_set_PathPlan(const struct cereal_PathPlan *s, cereal_PathPlan_list l, int i) {
	cereal_PathPlan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_PathPlan(s, p);
}

cereal_LiveLocationData_ptr cereal_new_LiveLocationData(struct capn_segment *s) {
	cereal_LiveLocationData_ptr p;
	p.p = capn_new_struct(s, 80, 7);
	return p;
}
cereal_LiveLocationData_list cereal_new_LiveLocationData_list(struct capn_segment *s, int len) {
	cereal_LiveLocationData_list p;
	p.p = capn_new_list(s, len, 80, 7);
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
	s->source = (enum cereal_LiveLocationData_SensorSource)(int) capn_read16(p.p, 2);
	s->fixMonoTime = capn_read64(p.p, 48);
	s->gpsWeek = (int32_t) ((int32_t)capn_read32(p.p, 56));
	s->timeOfWeek = capn_to_f64(capn_read64(p.p, 64));
	s->positionECEF.p = capn_getp(p.p, 4, 0);
	s->poseQuatECEF.p = capn_getp(p.p, 5, 0);
	s->pitchCalibration = capn_to_f32(capn_read32(p.p, 60));
	s->yawCalibration = capn_to_f32(capn_read32(p.p, 72));
	s->imuFrame.p = capn_getp(p.p, 6, 0);
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
	capn_write16(p.p, 2, (uint16_t) (s->source));
	capn_write64(p.p, 48, s->fixMonoTime);
	capn_write32(p.p, 56, (uint32_t) (s->gpsWeek));
	capn_write64(p.p, 64, capn_from_f64(s->timeOfWeek));
	capn_setp(p.p, 4, s->positionECEF.p);
	capn_setp(p.p, 5, s->poseQuatECEF.p);
	capn_write32(p.p, 60, capn_from_f32(s->pitchCalibration));
	capn_write32(p.p, 72, capn_from_f32(s->yawCalibration));
	capn_setp(p.p, 6, s->imuFrame.p);
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

cereal_NavStatus_ptr cereal_new_NavStatus(struct capn_segment *s) {
	cereal_NavStatus_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_NavStatus_list cereal_new_NavStatus_list(struct capn_segment *s, int len) {
	cereal_NavStatus_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_NavStatus(struct cereal_NavStatus *s, cereal_NavStatus_ptr p) {
	capn_resolve(&p.p);
	s->isNavigating = (capn_read8(p.p, 0) & 1) != 0;
	s->currentAddress.p = capn_getp(p.p, 0, 0);
}
void cereal_write_NavStatus(const struct cereal_NavStatus *s, cereal_NavStatus_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->isNavigating != 0);
	capn_setp(p.p, 0, s->currentAddress.p);
}
void cereal_get_NavStatus(struct cereal_NavStatus *s, cereal_NavStatus_list l, int i) {
	cereal_NavStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_NavStatus(s, p);
}
void cereal_set_NavStatus(const struct cereal_NavStatus *s, cereal_NavStatus_list l, int i) {
	cereal_NavStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_NavStatus(s, p);
}

cereal_NavStatus_Address_ptr cereal_new_NavStatus_Address(struct capn_segment *s) {
	cereal_NavStatus_Address_ptr p;
	p.p = capn_new_struct(s, 16, 7);
	return p;
}
cereal_NavStatus_Address_list cereal_new_NavStatus_Address_list(struct capn_segment *s, int len) {
	cereal_NavStatus_Address_list p;
	p.p = capn_new_list(s, len, 16, 7);
	return p;
}
void cereal_read_NavStatus_Address(struct cereal_NavStatus_Address *s, cereal_NavStatus_Address_ptr p) {
	capn_resolve(&p.p);
	s->title = capn_get_text(p.p, 0, capn_val0);
	s->lat = capn_to_f64(capn_read64(p.p, 0));
	s->lng = capn_to_f64(capn_read64(p.p, 8));
	s->house = capn_get_text(p.p, 1, capn_val0);
	s->address = capn_get_text(p.p, 2, capn_val0);
	s->street = capn_get_text(p.p, 3, capn_val0);
	s->city = capn_get_text(p.p, 4, capn_val0);
	s->state = capn_get_text(p.p, 5, capn_val0);
	s->country = capn_get_text(p.p, 6, capn_val0);
}
void cereal_write_NavStatus_Address(const struct cereal_NavStatus_Address *s, cereal_NavStatus_Address_ptr p) {
	capn_resolve(&p.p);
	capn_set_text(p.p, 0, s->title);
	capn_write64(p.p, 0, capn_from_f64(s->lat));
	capn_write64(p.p, 8, capn_from_f64(s->lng));
	capn_set_text(p.p, 1, s->house);
	capn_set_text(p.p, 2, s->address);
	capn_set_text(p.p, 3, s->street);
	capn_set_text(p.p, 4, s->city);
	capn_set_text(p.p, 5, s->state);
	capn_set_text(p.p, 6, s->country);
}
void cereal_get_NavStatus_Address(struct cereal_NavStatus_Address *s, cereal_NavStatus_Address_list l, int i) {
	cereal_NavStatus_Address_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_NavStatus_Address(s, p);
}
void cereal_set_NavStatus_Address(const struct cereal_NavStatus_Address *s, cereal_NavStatus_Address_list l, int i) {
	cereal_NavStatus_Address_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_NavStatus_Address(s, p);
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
	case cereal_QcomGnss_rawLog:
		s->rawLog = capn_get_data(p.p, 0);
		break;
	case cereal_QcomGnss_measurementReport:
	case cereal_QcomGnss_clockReport:
	case cereal_QcomGnss_drMeasurementReport:
	case cereal_QcomGnss_drSvPoly:
		s->drSvPoly.p = capn_getp(p.p, 0, 0);
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
	case cereal_QcomGnss_rawLog:
		capn_setp(p.p, 0, s->rawLog.p);
		break;
	case cereal_QcomGnss_measurementReport:
	case cereal_QcomGnss_clockReport:
	case cereal_QcomGnss_drMeasurementReport:
	case cereal_QcomGnss_drSvPoly:
		capn_setp(p.p, 0, s->drSvPoly.p);
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

cereal_QcomGnss_MeasurementStatus_ptr cereal_new_QcomGnss_MeasurementStatus(struct capn_segment *s) {
	cereal_QcomGnss_MeasurementStatus_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_QcomGnss_MeasurementStatus_list cereal_new_QcomGnss_MeasurementStatus_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_MeasurementStatus_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_QcomGnss_MeasurementStatus(struct cereal_QcomGnss_MeasurementStatus *s, cereal_QcomGnss_MeasurementStatus_ptr p) {
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
	s->multipathIndicator = (capn_read8(p.p, 2) & 128) != 0;
	s->imdJammingIndicator = (capn_read8(p.p, 3) & 1) != 0;
	s->lteB13TxJammingIndicator = (capn_read8(p.p, 3) & 2) != 0;
	s->freshMeasurementIndicator = (capn_read8(p.p, 3) & 4) != 0;
	s->multipathEstimateIsValid = (capn_read8(p.p, 3) & 8) != 0;
	s->directionIsValid = (capn_read8(p.p, 3) & 16) != 0;
}
void cereal_write_QcomGnss_MeasurementStatus(const struct cereal_QcomGnss_MeasurementStatus *s, cereal_QcomGnss_MeasurementStatus_ptr p) {
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
	capn_write1(p.p, 23, s->multipathIndicator != 0);
	capn_write1(p.p, 24, s->imdJammingIndicator != 0);
	capn_write1(p.p, 25, s->lteB13TxJammingIndicator != 0);
	capn_write1(p.p, 26, s->freshMeasurementIndicator != 0);
	capn_write1(p.p, 27, s->multipathEstimateIsValid != 0);
	capn_write1(p.p, 28, s->directionIsValid != 0);
}
void cereal_get_QcomGnss_MeasurementStatus(struct cereal_QcomGnss_MeasurementStatus *s, cereal_QcomGnss_MeasurementStatus_list l, int i) {
	cereal_QcomGnss_MeasurementStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_MeasurementStatus(s, p);
}
void cereal_set_QcomGnss_MeasurementStatus(const struct cereal_QcomGnss_MeasurementStatus *s, cereal_QcomGnss_MeasurementStatus_list l, int i) {
	cereal_QcomGnss_MeasurementStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_MeasurementStatus(s, p);
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
	s->source = (enum cereal_QcomGnss_MeasurementSource)(int) capn_read16(p.p, 0);
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
	s->observationState = (enum cereal_QcomGnss_SVObservationState)(int) capn_read16(p.p, 2);
	s->observations = capn_read8(p.p, 4);
	s->goodObservations = capn_read8(p.p, 5);
	s->gpsParityErrorCount = capn_read16(p.p, 6);
	s->glonassFrequencyIndex = (int8_t) ((int8_t)capn_read8(p.p, 1));
	s->glonassHemmingErrorCount = capn_read8(p.p, 8);
	s->filterStages = capn_read8(p.p, 9);
	s->carrierNoise = capn_read16(p.p, 10);
	s->latency = (int16_t) ((int16_t)capn_read16(p.p, 12));
	s->predetectInterval = capn_read8(p.p, 14);
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
	capn_write8(p.p, 14, s->predetectInterval);
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
	s->hasGpsWeek = (capn_read8(p.p, 0) & 2) != 0;
	s->gpsWeek = capn_read16(p.p, 2);
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
	capn_write1(p.p, 1, s->hasGpsWeek != 0);
	capn_write16(p.p, 2, s->gpsWeek);
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

cereal_QcomGnss_DrMeasurementReport_ptr cereal_new_QcomGnss_DrMeasurementReport(struct capn_segment *s) {
	cereal_QcomGnss_DrMeasurementReport_ptr p;
	p.p = capn_new_struct(s, 80, 1);
	return p;
}
cereal_QcomGnss_DrMeasurementReport_list cereal_new_QcomGnss_DrMeasurementReport_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_DrMeasurementReport_list p;
	p.p = capn_new_list(s, len, 80, 1);
	return p;
}
void cereal_read_QcomGnss_DrMeasurementReport(struct cereal_QcomGnss_DrMeasurementReport *s, cereal_QcomGnss_DrMeasurementReport_ptr p) {
	capn_resolve(&p.p);
	s->reason = capn_read8(p.p, 0);
	s->seqNum = capn_read8(p.p, 1);
	s->seqMax = capn_read8(p.p, 2);
	s->rfLoss = capn_read16(p.p, 4);
	s->systemRtcValid = (capn_read8(p.p, 3) & 1) != 0;
	s->fCount = capn_read32(p.p, 8);
	s->clockResets = capn_read32(p.p, 12);
	s->systemRtcTime = capn_read64(p.p, 16);
	s->gpsLeapSeconds = capn_read8(p.p, 6);
	s->gpsLeapSecondsUncertainty = capn_read8(p.p, 7);
	s->gpsToGlonassTimeBiasMilliseconds = capn_to_f32(capn_read32(p.p, 24));
	s->gpsToGlonassTimeBiasMillisecondsUncertainty = capn_to_f32(capn_read32(p.p, 28));
	s->gpsWeek = capn_read16(p.p, 32);
	s->gpsMilliseconds = capn_read32(p.p, 36);
	s->gpsTimeBiasMs = capn_read32(p.p, 40);
	s->gpsClockTimeUncertaintyMs = capn_read32(p.p, 44);
	s->gpsClockSource = capn_read8(p.p, 34);
	s->glonassClockSource = capn_read8(p.p, 35);
	s->glonassYear = capn_read8(p.p, 48);
	s->glonassDay = capn_read16(p.p, 50);
	s->glonassMilliseconds = capn_read32(p.p, 52);
	s->glonassTimeBias = capn_to_f32(capn_read32(p.p, 56));
	s->glonassClockTimeUncertainty = capn_to_f32(capn_read32(p.p, 60));
	s->clockFrequencyBias = capn_to_f32(capn_read32(p.p, 64));
	s->clockFrequencyUncertainty = capn_to_f32(capn_read32(p.p, 68));
	s->frequencySource = capn_read8(p.p, 49);
	s->source = (enum cereal_QcomGnss_MeasurementSource)(int) capn_read16(p.p, 72);
	s->sv.p = capn_getp(p.p, 0, 0);
}
void cereal_write_QcomGnss_DrMeasurementReport(const struct cereal_QcomGnss_DrMeasurementReport *s, cereal_QcomGnss_DrMeasurementReport_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->reason);
	capn_write8(p.p, 1, s->seqNum);
	capn_write8(p.p, 2, s->seqMax);
	capn_write16(p.p, 4, s->rfLoss);
	capn_write1(p.p, 24, s->systemRtcValid != 0);
	capn_write32(p.p, 8, s->fCount);
	capn_write32(p.p, 12, s->clockResets);
	capn_write64(p.p, 16, s->systemRtcTime);
	capn_write8(p.p, 6, s->gpsLeapSeconds);
	capn_write8(p.p, 7, s->gpsLeapSecondsUncertainty);
	capn_write32(p.p, 24, capn_from_f32(s->gpsToGlonassTimeBiasMilliseconds));
	capn_write32(p.p, 28, capn_from_f32(s->gpsToGlonassTimeBiasMillisecondsUncertainty));
	capn_write16(p.p, 32, s->gpsWeek);
	capn_write32(p.p, 36, s->gpsMilliseconds);
	capn_write32(p.p, 40, s->gpsTimeBiasMs);
	capn_write32(p.p, 44, s->gpsClockTimeUncertaintyMs);
	capn_write8(p.p, 34, s->gpsClockSource);
	capn_write8(p.p, 35, s->glonassClockSource);
	capn_write8(p.p, 48, s->glonassYear);
	capn_write16(p.p, 50, s->glonassDay);
	capn_write32(p.p, 52, s->glonassMilliseconds);
	capn_write32(p.p, 56, capn_from_f32(s->glonassTimeBias));
	capn_write32(p.p, 60, capn_from_f32(s->glonassClockTimeUncertainty));
	capn_write32(p.p, 64, capn_from_f32(s->clockFrequencyBias));
	capn_write32(p.p, 68, capn_from_f32(s->clockFrequencyUncertainty));
	capn_write8(p.p, 49, s->frequencySource);
	capn_write16(p.p, 72, (uint16_t) (s->source));
	capn_setp(p.p, 0, s->sv.p);
}
void cereal_get_QcomGnss_DrMeasurementReport(struct cereal_QcomGnss_DrMeasurementReport *s, cereal_QcomGnss_DrMeasurementReport_list l, int i) {
	cereal_QcomGnss_DrMeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_DrMeasurementReport(s, p);
}
void cereal_set_QcomGnss_DrMeasurementReport(const struct cereal_QcomGnss_DrMeasurementReport *s, cereal_QcomGnss_DrMeasurementReport_list l, int i) {
	cereal_QcomGnss_DrMeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_DrMeasurementReport(s, p);
}

cereal_QcomGnss_DrMeasurementReport_SV_ptr cereal_new_QcomGnss_DrMeasurementReport_SV(struct capn_segment *s) {
	cereal_QcomGnss_DrMeasurementReport_SV_ptr p;
	p.p = capn_new_struct(s, 96, 1);
	return p;
}
cereal_QcomGnss_DrMeasurementReport_SV_list cereal_new_QcomGnss_DrMeasurementReport_SV_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_DrMeasurementReport_SV_list p;
	p.p = capn_new_list(s, len, 96, 1);
	return p;
}
void cereal_read_QcomGnss_DrMeasurementReport_SV(struct cereal_QcomGnss_DrMeasurementReport_SV *s, cereal_QcomGnss_DrMeasurementReport_SV_ptr p) {
	capn_resolve(&p.p);
	s->svId = capn_read8(p.p, 0);
	s->glonassFrequencyIndex = (int8_t) ((int8_t)capn_read8(p.p, 1));
	s->observationState = (enum cereal_QcomGnss_SVObservationState)(int) capn_read16(p.p, 2);
	s->observations = capn_read8(p.p, 4);
	s->goodObservations = capn_read8(p.p, 5);
	s->filterStages = capn_read8(p.p, 6);
	s->predetectInterval = capn_read8(p.p, 7);
	s->cycleSlipCount = capn_read8(p.p, 8);
	s->postdetections = capn_read16(p.p, 10);
	s->measurementStatus.p = capn_getp(p.p, 0, 0);
	s->carrierNoise = capn_read16(p.p, 12);
	s->rfLoss = capn_read16(p.p, 14);
	s->latency = (int16_t) ((int16_t)capn_read16(p.p, 16));
	s->filteredMeasurementFraction = capn_to_f32(capn_read32(p.p, 20));
	s->filteredMeasurementIntegral = capn_read32(p.p, 24);
	s->filteredTimeUncertainty = capn_to_f32(capn_read32(p.p, 28));
	s->filteredSpeed = capn_to_f32(capn_read32(p.p, 32));
	s->filteredSpeedUncertainty = capn_to_f32(capn_read32(p.p, 36));
	s->unfilteredMeasurementFraction = capn_to_f32(capn_read32(p.p, 40));
	s->unfilteredMeasurementIntegral = capn_read32(p.p, 44);
	s->unfilteredTimeUncertainty = capn_to_f32(capn_read32(p.p, 48));
	s->unfilteredSpeed = capn_to_f32(capn_read32(p.p, 52));
	s->unfilteredSpeedUncertainty = capn_to_f32(capn_read32(p.p, 56));
	s->multipathEstimate = capn_read32(p.p, 60);
	s->azimuth = capn_to_f32(capn_read32(p.p, 64));
	s->elevation = capn_to_f32(capn_read32(p.p, 68));
	s->dopplerAcceleration = capn_to_f32(capn_read32(p.p, 72));
	s->fineSpeed = capn_to_f32(capn_read32(p.p, 76));
	s->fineSpeedUncertainty = capn_to_f32(capn_read32(p.p, 80));
	s->carrierPhase = capn_to_f64(capn_read64(p.p, 88));
	s->fCount = capn_read32(p.p, 84);
	s->parityErrorCount = capn_read16(p.p, 18);
	s->goodParity = (capn_read8(p.p, 9) & 1) != 0;
}
void cereal_write_QcomGnss_DrMeasurementReport_SV(const struct cereal_QcomGnss_DrMeasurementReport_SV *s, cereal_QcomGnss_DrMeasurementReport_SV_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->svId);
	capn_write8(p.p, 1, (uint8_t) (s->glonassFrequencyIndex));
	capn_write16(p.p, 2, (uint16_t) (s->observationState));
	capn_write8(p.p, 4, s->observations);
	capn_write8(p.p, 5, s->goodObservations);
	capn_write8(p.p, 6, s->filterStages);
	capn_write8(p.p, 7, s->predetectInterval);
	capn_write8(p.p, 8, s->cycleSlipCount);
	capn_write16(p.p, 10, s->postdetections);
	capn_setp(p.p, 0, s->measurementStatus.p);
	capn_write16(p.p, 12, s->carrierNoise);
	capn_write16(p.p, 14, s->rfLoss);
	capn_write16(p.p, 16, (uint16_t) (s->latency));
	capn_write32(p.p, 20, capn_from_f32(s->filteredMeasurementFraction));
	capn_write32(p.p, 24, s->filteredMeasurementIntegral);
	capn_write32(p.p, 28, capn_from_f32(s->filteredTimeUncertainty));
	capn_write32(p.p, 32, capn_from_f32(s->filteredSpeed));
	capn_write32(p.p, 36, capn_from_f32(s->filteredSpeedUncertainty));
	capn_write32(p.p, 40, capn_from_f32(s->unfilteredMeasurementFraction));
	capn_write32(p.p, 44, s->unfilteredMeasurementIntegral);
	capn_write32(p.p, 48, capn_from_f32(s->unfilteredTimeUncertainty));
	capn_write32(p.p, 52, capn_from_f32(s->unfilteredSpeed));
	capn_write32(p.p, 56, capn_from_f32(s->unfilteredSpeedUncertainty));
	capn_write32(p.p, 60, s->multipathEstimate);
	capn_write32(p.p, 64, capn_from_f32(s->azimuth));
	capn_write32(p.p, 68, capn_from_f32(s->elevation));
	capn_write32(p.p, 72, capn_from_f32(s->dopplerAcceleration));
	capn_write32(p.p, 76, capn_from_f32(s->fineSpeed));
	capn_write32(p.p, 80, capn_from_f32(s->fineSpeedUncertainty));
	capn_write64(p.p, 88, capn_from_f64(s->carrierPhase));
	capn_write32(p.p, 84, s->fCount);
	capn_write16(p.p, 18, s->parityErrorCount);
	capn_write1(p.p, 72, s->goodParity != 0);
}
void cereal_get_QcomGnss_DrMeasurementReport_SV(struct cereal_QcomGnss_DrMeasurementReport_SV *s, cereal_QcomGnss_DrMeasurementReport_SV_list l, int i) {
	cereal_QcomGnss_DrMeasurementReport_SV_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_DrMeasurementReport_SV(s, p);
}
void cereal_set_QcomGnss_DrMeasurementReport_SV(const struct cereal_QcomGnss_DrMeasurementReport_SV *s, cereal_QcomGnss_DrMeasurementReport_SV_list l, int i) {
	cereal_QcomGnss_DrMeasurementReport_SV_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_DrMeasurementReport_SV(s, p);
}

cereal_QcomGnss_DrSvPolyReport_ptr cereal_new_QcomGnss_DrSvPolyReport(struct capn_segment *s) {
	cereal_QcomGnss_DrSvPolyReport_ptr p;
	p.p = capn_new_struct(s, 56, 4);
	return p;
}
cereal_QcomGnss_DrSvPolyReport_list cereal_new_QcomGnss_DrSvPolyReport_list(struct capn_segment *s, int len) {
	cereal_QcomGnss_DrSvPolyReport_list p;
	p.p = capn_new_list(s, len, 56, 4);
	return p;
}
void cereal_read_QcomGnss_DrSvPolyReport(struct cereal_QcomGnss_DrSvPolyReport *s, cereal_QcomGnss_DrSvPolyReport_ptr p) {
	capn_resolve(&p.p);
	s->svId = capn_read16(p.p, 0);
	s->frequencyIndex = (int8_t) ((int8_t)capn_read8(p.p, 2));
	s->hasPosition = (capn_read8(p.p, 3) & 1) != 0;
	s->hasIono = (capn_read8(p.p, 3) & 2) != 0;
	s->hasTropo = (capn_read8(p.p, 3) & 4) != 0;
	s->hasElevation = (capn_read8(p.p, 3) & 8) != 0;
	s->polyFromXtra = (capn_read8(p.p, 3) & 16) != 0;
	s->hasSbasIono = (capn_read8(p.p, 3) & 32) != 0;
	s->iode = capn_read16(p.p, 4);
	s->t0 = capn_to_f64(capn_read64(p.p, 8));
	s->xyz0.p = capn_getp(p.p, 0, 0);
	s->xyzN.p = capn_getp(p.p, 1, 0);
	s->other.p = capn_getp(p.p, 2, 0);
	s->positionUncertainty = capn_to_f32(capn_read32(p.p, 16));
	s->ionoDelay = capn_to_f32(capn_read32(p.p, 20));
	s->ionoDot = capn_to_f32(capn_read32(p.p, 24));
	s->sbasIonoDelay = capn_to_f32(capn_read32(p.p, 28));
	s->sbasIonoDot = capn_to_f32(capn_read32(p.p, 32));
	s->tropoDelay = capn_to_f32(capn_read32(p.p, 36));
	s->elevation = capn_to_f32(capn_read32(p.p, 40));
	s->elevationDot = capn_to_f32(capn_read32(p.p, 44));
	s->elevationUncertainty = capn_to_f32(capn_read32(p.p, 48));
	s->velocityCoeff.p = capn_getp(p.p, 3, 0);
}
void cereal_write_QcomGnss_DrSvPolyReport(const struct cereal_QcomGnss_DrSvPolyReport *s, cereal_QcomGnss_DrSvPolyReport_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->svId);
	capn_write8(p.p, 2, (uint8_t) (s->frequencyIndex));
	capn_write1(p.p, 24, s->hasPosition != 0);
	capn_write1(p.p, 25, s->hasIono != 0);
	capn_write1(p.p, 26, s->hasTropo != 0);
	capn_write1(p.p, 27, s->hasElevation != 0);
	capn_write1(p.p, 28, s->polyFromXtra != 0);
	capn_write1(p.p, 29, s->hasSbasIono != 0);
	capn_write16(p.p, 4, s->iode);
	capn_write64(p.p, 8, capn_from_f64(s->t0));
	capn_setp(p.p, 0, s->xyz0.p);
	capn_setp(p.p, 1, s->xyzN.p);
	capn_setp(p.p, 2, s->other.p);
	capn_write32(p.p, 16, capn_from_f32(s->positionUncertainty));
	capn_write32(p.p, 20, capn_from_f32(s->ionoDelay));
	capn_write32(p.p, 24, capn_from_f32(s->ionoDot));
	capn_write32(p.p, 28, capn_from_f32(s->sbasIonoDelay));
	capn_write32(p.p, 32, capn_from_f32(s->sbasIonoDot));
	capn_write32(p.p, 36, capn_from_f32(s->tropoDelay));
	capn_write32(p.p, 40, capn_from_f32(s->elevation));
	capn_write32(p.p, 44, capn_from_f32(s->elevationDot));
	capn_write32(p.p, 48, capn_from_f32(s->elevationUncertainty));
	capn_setp(p.p, 3, s->velocityCoeff.p);
}
void cereal_get_QcomGnss_DrSvPolyReport(struct cereal_QcomGnss_DrSvPolyReport *s, cereal_QcomGnss_DrSvPolyReport_list l, int i) {
	cereal_QcomGnss_DrSvPolyReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_QcomGnss_DrSvPolyReport(s, p);
}
void cereal_set_QcomGnss_DrSvPolyReport(const struct cereal_QcomGnss_DrSvPolyReport *s, cereal_QcomGnss_DrSvPolyReport_list l, int i) {
	cereal_QcomGnss_DrSvPolyReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_QcomGnss_DrSvPolyReport(s, p);
}

cereal_LidarPts_ptr cereal_new_LidarPts(struct capn_segment *s) {
	cereal_LidarPts_ptr p;
	p.p = capn_new_struct(s, 8, 4);
	return p;
}
cereal_LidarPts_list cereal_new_LidarPts_list(struct capn_segment *s, int len) {
	cereal_LidarPts_list p;
	p.p = capn_new_list(s, len, 8, 4);
	return p;
}
void cereal_read_LidarPts(struct cereal_LidarPts *s, cereal_LidarPts_ptr p) {
	capn_resolve(&p.p);
	s->r.p = capn_getp(p.p, 0, 0);
	s->theta.p = capn_getp(p.p, 1, 0);
	s->reflect.p = capn_getp(p.p, 2, 0);
	s->idx = capn_read64(p.p, 0);
	s->pkt = capn_get_data(p.p, 3);
}
void cereal_write_LidarPts(const struct cereal_LidarPts *s, cereal_LidarPts_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->r.p);
	capn_setp(p.p, 1, s->theta.p);
	capn_setp(p.p, 2, s->reflect.p);
	capn_write64(p.p, 0, s->idx);
	capn_setp(p.p, 3, s->pkt.p);
}
void cereal_get_LidarPts(struct cereal_LidarPts *s, cereal_LidarPts_list l, int i) {
	cereal_LidarPts_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LidarPts(s, p);
}
void cereal_set_LidarPts(const struct cereal_LidarPts *s, cereal_LidarPts_list l, int i) {
	cereal_LidarPts_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LidarPts(s, p);
}

cereal_ProcLog_ptr cereal_new_ProcLog(struct capn_segment *s) {
	cereal_ProcLog_ptr p;
	p.p = capn_new_struct(s, 0, 3);
	return p;
}
cereal_ProcLog_list cereal_new_ProcLog_list(struct capn_segment *s, int len) {
	cereal_ProcLog_list p;
	p.p = capn_new_list(s, len, 0, 3);
	return p;
}
void cereal_read_ProcLog(struct cereal_ProcLog *s, cereal_ProcLog_ptr p) {
	capn_resolve(&p.p);
	s->cpuTimes.p = capn_getp(p.p, 0, 0);
	s->mem.p = capn_getp(p.p, 1, 0);
	s->procs.p = capn_getp(p.p, 2, 0);
}
void cereal_write_ProcLog(const struct cereal_ProcLog *s, cereal_ProcLog_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->cpuTimes.p);
	capn_setp(p.p, 1, s->mem.p);
	capn_setp(p.p, 2, s->procs.p);
}
void cereal_get_ProcLog(struct cereal_ProcLog *s, cereal_ProcLog_list l, int i) {
	cereal_ProcLog_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ProcLog(s, p);
}
void cereal_set_ProcLog(const struct cereal_ProcLog *s, cereal_ProcLog_list l, int i) {
	cereal_ProcLog_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ProcLog(s, p);
}

cereal_ProcLog_Process_ptr cereal_new_ProcLog_Process(struct capn_segment *s) {
	cereal_ProcLog_Process_ptr p;
	p.p = capn_new_struct(s, 72, 3);
	return p;
}
cereal_ProcLog_Process_list cereal_new_ProcLog_Process_list(struct capn_segment *s, int len) {
	cereal_ProcLog_Process_list p;
	p.p = capn_new_list(s, len, 72, 3);
	return p;
}
void cereal_read_ProcLog_Process(struct cereal_ProcLog_Process *s, cereal_ProcLog_Process_ptr p) {
	capn_resolve(&p.p);
	s->pid = (int32_t) ((int32_t)capn_read32(p.p, 0));
	s->name = capn_get_text(p.p, 0, capn_val0);
	s->state = capn_read8(p.p, 4);
	s->ppid = (int32_t) ((int32_t)capn_read32(p.p, 8));
	s->cpuUser = capn_to_f32(capn_read32(p.p, 12));
	s->cpuSystem = capn_to_f32(capn_read32(p.p, 16));
	s->cpuChildrenUser = capn_to_f32(capn_read32(p.p, 20));
	s->cpuChildrenSystem = capn_to_f32(capn_read32(p.p, 24));
	s->priority = (int64_t) ((int64_t)(capn_read64(p.p, 32)));
	s->nice = (int32_t) ((int32_t)capn_read32(p.p, 28));
	s->numThreads = (int32_t) ((int32_t)capn_read32(p.p, 40));
	s->startTime = capn_to_f64(capn_read64(p.p, 48));
	s->memVms = capn_read64(p.p, 56);
	s->memRss = capn_read64(p.p, 64);
	s->processor = (int32_t) ((int32_t)capn_read32(p.p, 44));
	s->cmdline = capn_getp(p.p, 1, 0);
	s->exe = capn_get_text(p.p, 2, capn_val0);
}
void cereal_write_ProcLog_Process(const struct cereal_ProcLog_Process *s, cereal_ProcLog_Process_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, (uint32_t) (s->pid));
	capn_set_text(p.p, 0, s->name);
	capn_write8(p.p, 4, s->state);
	capn_write32(p.p, 8, (uint32_t) (s->ppid));
	capn_write32(p.p, 12, capn_from_f32(s->cpuUser));
	capn_write32(p.p, 16, capn_from_f32(s->cpuSystem));
	capn_write32(p.p, 20, capn_from_f32(s->cpuChildrenUser));
	capn_write32(p.p, 24, capn_from_f32(s->cpuChildrenSystem));
	capn_write64(p.p, 32, (uint64_t) (s->priority));
	capn_write32(p.p, 28, (uint32_t) (s->nice));
	capn_write32(p.p, 40, (uint32_t) (s->numThreads));
	capn_write64(p.p, 48, capn_from_f64(s->startTime));
	capn_write64(p.p, 56, s->memVms);
	capn_write64(p.p, 64, s->memRss);
	capn_write32(p.p, 44, (uint32_t) (s->processor));
	capn_setp(p.p, 1, s->cmdline);
	capn_set_text(p.p, 2, s->exe);
}
void cereal_get_ProcLog_Process(struct cereal_ProcLog_Process *s, cereal_ProcLog_Process_list l, int i) {
	cereal_ProcLog_Process_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ProcLog_Process(s, p);
}
void cereal_set_ProcLog_Process(const struct cereal_ProcLog_Process *s, cereal_ProcLog_Process_list l, int i) {
	cereal_ProcLog_Process_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ProcLog_Process(s, p);
}

cereal_ProcLog_CPUTimes_ptr cereal_new_ProcLog_CPUTimes(struct capn_segment *s) {
	cereal_ProcLog_CPUTimes_ptr p;
	p.p = capn_new_struct(s, 40, 0);
	return p;
}
cereal_ProcLog_CPUTimes_list cereal_new_ProcLog_CPUTimes_list(struct capn_segment *s, int len) {
	cereal_ProcLog_CPUTimes_list p;
	p.p = capn_new_list(s, len, 40, 0);
	return p;
}
void cereal_read_ProcLog_CPUTimes(struct cereal_ProcLog_CPUTimes *s, cereal_ProcLog_CPUTimes_ptr p) {
	capn_resolve(&p.p);
	s->cpuNum = (int64_t) ((int64_t)(capn_read64(p.p, 0)));
	s->user = capn_to_f32(capn_read32(p.p, 8));
	s->nice = capn_to_f32(capn_read32(p.p, 12));
	s->system = capn_to_f32(capn_read32(p.p, 16));
	s->idle = capn_to_f32(capn_read32(p.p, 20));
	s->iowait = capn_to_f32(capn_read32(p.p, 24));
	s->irq = capn_to_f32(capn_read32(p.p, 28));
	s->softirq = capn_to_f32(capn_read32(p.p, 32));
}
void cereal_write_ProcLog_CPUTimes(const struct cereal_ProcLog_CPUTimes *s, cereal_ProcLog_CPUTimes_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, (uint64_t) (s->cpuNum));
	capn_write32(p.p, 8, capn_from_f32(s->user));
	capn_write32(p.p, 12, capn_from_f32(s->nice));
	capn_write32(p.p, 16, capn_from_f32(s->system));
	capn_write32(p.p, 20, capn_from_f32(s->idle));
	capn_write32(p.p, 24, capn_from_f32(s->iowait));
	capn_write32(p.p, 28, capn_from_f32(s->irq));
	capn_write32(p.p, 32, capn_from_f32(s->softirq));
}
void cereal_get_ProcLog_CPUTimes(struct cereal_ProcLog_CPUTimes *s, cereal_ProcLog_CPUTimes_list l, int i) {
	cereal_ProcLog_CPUTimes_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ProcLog_CPUTimes(s, p);
}
void cereal_set_ProcLog_CPUTimes(const struct cereal_ProcLog_CPUTimes *s, cereal_ProcLog_CPUTimes_list l, int i) {
	cereal_ProcLog_CPUTimes_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ProcLog_CPUTimes(s, p);
}

cereal_ProcLog_Mem_ptr cereal_new_ProcLog_Mem(struct capn_segment *s) {
	cereal_ProcLog_Mem_ptr p;
	p.p = capn_new_struct(s, 64, 0);
	return p;
}
cereal_ProcLog_Mem_list cereal_new_ProcLog_Mem_list(struct capn_segment *s, int len) {
	cereal_ProcLog_Mem_list p;
	p.p = capn_new_list(s, len, 64, 0);
	return p;
}
void cereal_read_ProcLog_Mem(struct cereal_ProcLog_Mem *s, cereal_ProcLog_Mem_ptr p) {
	capn_resolve(&p.p);
	s->total = capn_read64(p.p, 0);
	s->free = capn_read64(p.p, 8);
	s->available = capn_read64(p.p, 16);
	s->buffers = capn_read64(p.p, 24);
	s->cached = capn_read64(p.p, 32);
	s->active = capn_read64(p.p, 40);
	s->inactive = capn_read64(p.p, 48);
	s->shared = capn_read64(p.p, 56);
}
void cereal_write_ProcLog_Mem(const struct cereal_ProcLog_Mem *s, cereal_ProcLog_Mem_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->total);
	capn_write64(p.p, 8, s->free);
	capn_write64(p.p, 16, s->available);
	capn_write64(p.p, 24, s->buffers);
	capn_write64(p.p, 32, s->cached);
	capn_write64(p.p, 40, s->active);
	capn_write64(p.p, 48, s->inactive);
	capn_write64(p.p, 56, s->shared);
}
void cereal_get_ProcLog_Mem(struct cereal_ProcLog_Mem *s, cereal_ProcLog_Mem_list l, int i) {
	cereal_ProcLog_Mem_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ProcLog_Mem(s, p);
}
void cereal_set_ProcLog_Mem(const struct cereal_ProcLog_Mem *s, cereal_ProcLog_Mem_list l, int i) {
	cereal_ProcLog_Mem_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ProcLog_Mem(s, p);
}

cereal_UbloxGnss_ptr cereal_new_UbloxGnss(struct capn_segment *s) {
	cereal_UbloxGnss_ptr p;
	p.p = capn_new_struct(s, 8, 1);
	return p;
}
cereal_UbloxGnss_list cereal_new_UbloxGnss_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_list p;
	p.p = capn_new_list(s, len, 8, 1);
	return p;
}
void cereal_read_UbloxGnss(struct cereal_UbloxGnss *s, cereal_UbloxGnss_ptr p) {
	capn_resolve(&p.p);
	s->which = (enum cereal_UbloxGnss_which)(int) capn_read16(p.p, 0);
	switch (s->which) {
	case cereal_UbloxGnss_measurementReport:
	case cereal_UbloxGnss_ephemeris:
	case cereal_UbloxGnss_ionoData:
	case cereal_UbloxGnss_hwStatus:
		s->hwStatus.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_UbloxGnss(const struct cereal_UbloxGnss *s, cereal_UbloxGnss_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->which);
	switch (s->which) {
	case cereal_UbloxGnss_measurementReport:
	case cereal_UbloxGnss_ephemeris:
	case cereal_UbloxGnss_ionoData:
	case cereal_UbloxGnss_hwStatus:
		capn_setp(p.p, 0, s->hwStatus.p);
		break;
	default:
		break;
	}
}
void cereal_get_UbloxGnss(struct cereal_UbloxGnss *s, cereal_UbloxGnss_list l, int i) {
	cereal_UbloxGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss(s, p);
}
void cereal_set_UbloxGnss(const struct cereal_UbloxGnss *s, cereal_UbloxGnss_list l, int i) {
	cereal_UbloxGnss_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss(s, p);
}

cereal_UbloxGnss_MeasurementReport_ptr cereal_new_UbloxGnss_MeasurementReport(struct capn_segment *s) {
	cereal_UbloxGnss_MeasurementReport_ptr p;
	p.p = capn_new_struct(s, 16, 2);
	return p;
}
cereal_UbloxGnss_MeasurementReport_list cereal_new_UbloxGnss_MeasurementReport_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_MeasurementReport_list p;
	p.p = capn_new_list(s, len, 16, 2);
	return p;
}
void cereal_read_UbloxGnss_MeasurementReport(struct cereal_UbloxGnss_MeasurementReport *s, cereal_UbloxGnss_MeasurementReport_ptr p) {
	capn_resolve(&p.p);
	s->rcvTow = capn_to_f64(capn_read64(p.p, 0));
	s->gpsWeek = capn_read16(p.p, 8);
	s->leapSeconds = capn_read16(p.p, 10);
	s->receiverStatus.p = capn_getp(p.p, 0, 0);
	s->numMeas = capn_read8(p.p, 12);
	s->measurements.p = capn_getp(p.p, 1, 0);
}
void cereal_write_UbloxGnss_MeasurementReport(const struct cereal_UbloxGnss_MeasurementReport *s, cereal_UbloxGnss_MeasurementReport_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, capn_from_f64(s->rcvTow));
	capn_write16(p.p, 8, s->gpsWeek);
	capn_write16(p.p, 10, s->leapSeconds);
	capn_setp(p.p, 0, s->receiverStatus.p);
	capn_write8(p.p, 12, s->numMeas);
	capn_setp(p.p, 1, s->measurements.p);
}
void cereal_get_UbloxGnss_MeasurementReport(struct cereal_UbloxGnss_MeasurementReport *s, cereal_UbloxGnss_MeasurementReport_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_MeasurementReport(s, p);
}
void cereal_set_UbloxGnss_MeasurementReport(const struct cereal_UbloxGnss_MeasurementReport *s, cereal_UbloxGnss_MeasurementReport_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_MeasurementReport(s, p);
}

cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr cereal_new_UbloxGnss_MeasurementReport_ReceiverStatus(struct capn_segment *s) {
	cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list cereal_new_UbloxGnss_MeasurementReport_ReceiverStatus_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_UbloxGnss_MeasurementReport_ReceiverStatus(struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus *s, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr p) {
	capn_resolve(&p.p);
	s->leapSecValid = (capn_read8(p.p, 0) & 1) != 0;
	s->clkReset = (capn_read8(p.p, 0) & 2) != 0;
}
void cereal_write_UbloxGnss_MeasurementReport_ReceiverStatus(const struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus *s, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->leapSecValid != 0);
	capn_write1(p.p, 1, s->clkReset != 0);
}
void cereal_get_UbloxGnss_MeasurementReport_ReceiverStatus(struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus *s, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_MeasurementReport_ReceiverStatus(s, p);
}
void cereal_set_UbloxGnss_MeasurementReport_ReceiverStatus(const struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus *s, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_MeasurementReport_ReceiverStatus(s, p);
}

cereal_UbloxGnss_MeasurementReport_Measurement_ptr cereal_new_UbloxGnss_MeasurementReport_Measurement(struct capn_segment *s) {
	cereal_UbloxGnss_MeasurementReport_Measurement_ptr p;
	p.p = capn_new_struct(s, 40, 1);
	return p;
}
cereal_UbloxGnss_MeasurementReport_Measurement_list cereal_new_UbloxGnss_MeasurementReport_Measurement_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_MeasurementReport_Measurement_list p;
	p.p = capn_new_list(s, len, 40, 1);
	return p;
}
void cereal_read_UbloxGnss_MeasurementReport_Measurement(struct cereal_UbloxGnss_MeasurementReport_Measurement *s, cereal_UbloxGnss_MeasurementReport_Measurement_ptr p) {
	capn_resolve(&p.p);
	s->svId = capn_read8(p.p, 0);
	s->trackingStatus.p = capn_getp(p.p, 0, 0);
	s->pseudorange = capn_to_f64(capn_read64(p.p, 8));
	s->carrierCycles = capn_to_f64(capn_read64(p.p, 16));
	s->doppler = capn_to_f32(capn_read32(p.p, 4));
	s->gnssId = capn_read8(p.p, 1);
	s->glonassFrequencyIndex = capn_read8(p.p, 2);
	s->locktime = capn_read16(p.p, 24);
	s->cno = capn_read8(p.p, 3);
	s->pseudorangeStdev = capn_to_f32(capn_read32(p.p, 28));
	s->carrierPhaseStdev = capn_to_f32(capn_read32(p.p, 32));
	s->dopplerStdev = capn_to_f32(capn_read32(p.p, 36));
	s->sigId = capn_read8(p.p, 26);
}
void cereal_write_UbloxGnss_MeasurementReport_Measurement(const struct cereal_UbloxGnss_MeasurementReport_Measurement *s, cereal_UbloxGnss_MeasurementReport_Measurement_ptr p) {
	capn_resolve(&p.p);
	capn_write8(p.p, 0, s->svId);
	capn_setp(p.p, 0, s->trackingStatus.p);
	capn_write64(p.p, 8, capn_from_f64(s->pseudorange));
	capn_write64(p.p, 16, capn_from_f64(s->carrierCycles));
	capn_write32(p.p, 4, capn_from_f32(s->doppler));
	capn_write8(p.p, 1, s->gnssId);
	capn_write8(p.p, 2, s->glonassFrequencyIndex);
	capn_write16(p.p, 24, s->locktime);
	capn_write8(p.p, 3, s->cno);
	capn_write32(p.p, 28, capn_from_f32(s->pseudorangeStdev));
	capn_write32(p.p, 32, capn_from_f32(s->carrierPhaseStdev));
	capn_write32(p.p, 36, capn_from_f32(s->dopplerStdev));
	capn_write8(p.p, 26, s->sigId);
}
void cereal_get_UbloxGnss_MeasurementReport_Measurement(struct cereal_UbloxGnss_MeasurementReport_Measurement *s, cereal_UbloxGnss_MeasurementReport_Measurement_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_Measurement_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_MeasurementReport_Measurement(s, p);
}
void cereal_set_UbloxGnss_MeasurementReport_Measurement(const struct cereal_UbloxGnss_MeasurementReport_Measurement *s, cereal_UbloxGnss_MeasurementReport_Measurement_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_Measurement_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_MeasurementReport_Measurement(s, p);
}

cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr cereal_new_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct capn_segment *s) {
	cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list cereal_new_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus *s, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr p) {
	capn_resolve(&p.p);
	s->pseudorangeValid = (capn_read8(p.p, 0) & 1) != 0;
	s->carrierPhaseValid = (capn_read8(p.p, 0) & 2) != 0;
	s->halfCycleValid = (capn_read8(p.p, 0) & 4) != 0;
	s->halfCycleSubtracted = (capn_read8(p.p, 0) & 8) != 0;
}
void cereal_write_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(const struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus *s, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->pseudorangeValid != 0);
	capn_write1(p.p, 1, s->carrierPhaseValid != 0);
	capn_write1(p.p, 2, s->halfCycleValid != 0);
	capn_write1(p.p, 3, s->halfCycleSubtracted != 0);
}
void cereal_get_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus *s, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(s, p);
}
void cereal_set_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(const struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus *s, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list l, int i) {
	cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(s, p);
}

cereal_UbloxGnss_Ephemeris_ptr cereal_new_UbloxGnss_Ephemeris(struct capn_segment *s) {
	cereal_UbloxGnss_Ephemeris_ptr p;
	p.p = capn_new_struct(s, 264, 2);
	return p;
}
cereal_UbloxGnss_Ephemeris_list cereal_new_UbloxGnss_Ephemeris_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_Ephemeris_list p;
	p.p = capn_new_list(s, len, 264, 2);
	return p;
}
void cereal_read_UbloxGnss_Ephemeris(struct cereal_UbloxGnss_Ephemeris *s, cereal_UbloxGnss_Ephemeris_ptr p) {
	capn_resolve(&p.p);
	s->svId = capn_read16(p.p, 0);
	s->year = capn_read16(p.p, 2);
	s->month = capn_read16(p.p, 4);
	s->day = capn_read16(p.p, 6);
	s->hour = capn_read16(p.p, 8);
	s->minute = capn_read16(p.p, 10);
	s->second = capn_to_f32(capn_read32(p.p, 12));
	s->af0 = capn_to_f64(capn_read64(p.p, 16));
	s->af1 = capn_to_f64(capn_read64(p.p, 24));
	s->af2 = capn_to_f64(capn_read64(p.p, 32));
	s->iode = capn_to_f64(capn_read64(p.p, 40));
	s->crs = capn_to_f64(capn_read64(p.p, 48));
	s->deltaN = capn_to_f64(capn_read64(p.p, 56));
	s->m0 = capn_to_f64(capn_read64(p.p, 64));
	s->cuc = capn_to_f64(capn_read64(p.p, 72));
	s->ecc = capn_to_f64(capn_read64(p.p, 80));
	s->cus = capn_to_f64(capn_read64(p.p, 88));
	s->a = capn_to_f64(capn_read64(p.p, 96));
	s->toe = capn_to_f64(capn_read64(p.p, 104));
	s->cic = capn_to_f64(capn_read64(p.p, 112));
	s->omega0 = capn_to_f64(capn_read64(p.p, 120));
	s->cis = capn_to_f64(capn_read64(p.p, 128));
	s->i0 = capn_to_f64(capn_read64(p.p, 136));
	s->crc = capn_to_f64(capn_read64(p.p, 144));
	s->omega = capn_to_f64(capn_read64(p.p, 152));
	s->omegaDot = capn_to_f64(capn_read64(p.p, 160));
	s->iDot = capn_to_f64(capn_read64(p.p, 168));
	s->codesL2 = capn_to_f64(capn_read64(p.p, 176));
	s->gpsWeek = capn_to_f64(capn_read64(p.p, 184));
	s->l2 = capn_to_f64(capn_read64(p.p, 192));
	s->svAcc = capn_to_f64(capn_read64(p.p, 200));
	s->svHealth = capn_to_f64(capn_read64(p.p, 208));
	s->tgd = capn_to_f64(capn_read64(p.p, 216));
	s->iodc = capn_to_f64(capn_read64(p.p, 224));
	s->transmissionTime = capn_to_f64(capn_read64(p.p, 232));
	s->fitInterval = capn_to_f64(capn_read64(p.p, 240));
	s->toc = capn_to_f64(capn_read64(p.p, 248));
	s->ionoCoeffsValid = (capn_read8(p.p, 256) & 1) != 0;
	s->ionoAlpha.p = capn_getp(p.p, 0, 0);
	s->ionoBeta.p = capn_getp(p.p, 1, 0);
}
void cereal_write_UbloxGnss_Ephemeris(const struct cereal_UbloxGnss_Ephemeris *s, cereal_UbloxGnss_Ephemeris_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->svId);
	capn_write16(p.p, 2, s->year);
	capn_write16(p.p, 4, s->month);
	capn_write16(p.p, 6, s->day);
	capn_write16(p.p, 8, s->hour);
	capn_write16(p.p, 10, s->minute);
	capn_write32(p.p, 12, capn_from_f32(s->second));
	capn_write64(p.p, 16, capn_from_f64(s->af0));
	capn_write64(p.p, 24, capn_from_f64(s->af1));
	capn_write64(p.p, 32, capn_from_f64(s->af2));
	capn_write64(p.p, 40, capn_from_f64(s->iode));
	capn_write64(p.p, 48, capn_from_f64(s->crs));
	capn_write64(p.p, 56, capn_from_f64(s->deltaN));
	capn_write64(p.p, 64, capn_from_f64(s->m0));
	capn_write64(p.p, 72, capn_from_f64(s->cuc));
	capn_write64(p.p, 80, capn_from_f64(s->ecc));
	capn_write64(p.p, 88, capn_from_f64(s->cus));
	capn_write64(p.p, 96, capn_from_f64(s->a));
	capn_write64(p.p, 104, capn_from_f64(s->toe));
	capn_write64(p.p, 112, capn_from_f64(s->cic));
	capn_write64(p.p, 120, capn_from_f64(s->omega0));
	capn_write64(p.p, 128, capn_from_f64(s->cis));
	capn_write64(p.p, 136, capn_from_f64(s->i0));
	capn_write64(p.p, 144, capn_from_f64(s->crc));
	capn_write64(p.p, 152, capn_from_f64(s->omega));
	capn_write64(p.p, 160, capn_from_f64(s->omegaDot));
	capn_write64(p.p, 168, capn_from_f64(s->iDot));
	capn_write64(p.p, 176, capn_from_f64(s->codesL2));
	capn_write64(p.p, 184, capn_from_f64(s->gpsWeek));
	capn_write64(p.p, 192, capn_from_f64(s->l2));
	capn_write64(p.p, 200, capn_from_f64(s->svAcc));
	capn_write64(p.p, 208, capn_from_f64(s->svHealth));
	capn_write64(p.p, 216, capn_from_f64(s->tgd));
	capn_write64(p.p, 224, capn_from_f64(s->iodc));
	capn_write64(p.p, 232, capn_from_f64(s->transmissionTime));
	capn_write64(p.p, 240, capn_from_f64(s->fitInterval));
	capn_write64(p.p, 248, capn_from_f64(s->toc));
	capn_write1(p.p, 2048, s->ionoCoeffsValid != 0);
	capn_setp(p.p, 0, s->ionoAlpha.p);
	capn_setp(p.p, 1, s->ionoBeta.p);
}
void cereal_get_UbloxGnss_Ephemeris(struct cereal_UbloxGnss_Ephemeris *s, cereal_UbloxGnss_Ephemeris_list l, int i) {
	cereal_UbloxGnss_Ephemeris_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_Ephemeris(s, p);
}
void cereal_set_UbloxGnss_Ephemeris(const struct cereal_UbloxGnss_Ephemeris *s, cereal_UbloxGnss_Ephemeris_list l, int i) {
	cereal_UbloxGnss_Ephemeris_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_Ephemeris(s, p);
}

cereal_UbloxGnss_IonoData_ptr cereal_new_UbloxGnss_IonoData(struct capn_segment *s) {
	cereal_UbloxGnss_IonoData_ptr p;
	p.p = capn_new_struct(s, 24, 2);
	return p;
}
cereal_UbloxGnss_IonoData_list cereal_new_UbloxGnss_IonoData_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_IonoData_list p;
	p.p = capn_new_list(s, len, 24, 2);
	return p;
}
void cereal_read_UbloxGnss_IonoData(struct cereal_UbloxGnss_IonoData *s, cereal_UbloxGnss_IonoData_ptr p) {
	capn_resolve(&p.p);
	s->svHealth = capn_read32(p.p, 0);
	s->tow = capn_to_f64(capn_read64(p.p, 8));
	s->gpsWeek = capn_to_f64(capn_read64(p.p, 16));
	s->ionoAlpha.p = capn_getp(p.p, 0, 0);
	s->ionoBeta.p = capn_getp(p.p, 1, 0);
	s->healthValid = (capn_read8(p.p, 4) & 1) != 0;
	s->ionoCoeffsValid = (capn_read8(p.p, 4) & 2) != 0;
}
void cereal_write_UbloxGnss_IonoData(const struct cereal_UbloxGnss_IonoData *s, cereal_UbloxGnss_IonoData_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->svHealth);
	capn_write64(p.p, 8, capn_from_f64(s->tow));
	capn_write64(p.p, 16, capn_from_f64(s->gpsWeek));
	capn_setp(p.p, 0, s->ionoAlpha.p);
	capn_setp(p.p, 1, s->ionoBeta.p);
	capn_write1(p.p, 32, s->healthValid != 0);
	capn_write1(p.p, 33, s->ionoCoeffsValid != 0);
}
void cereal_get_UbloxGnss_IonoData(struct cereal_UbloxGnss_IonoData *s, cereal_UbloxGnss_IonoData_list l, int i) {
	cereal_UbloxGnss_IonoData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_IonoData(s, p);
}
void cereal_set_UbloxGnss_IonoData(const struct cereal_UbloxGnss_IonoData *s, cereal_UbloxGnss_IonoData_list l, int i) {
	cereal_UbloxGnss_IonoData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_IonoData(s, p);
}

cereal_UbloxGnss_HwStatus_ptr cereal_new_UbloxGnss_HwStatus(struct capn_segment *s) {
	cereal_UbloxGnss_HwStatus_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_UbloxGnss_HwStatus_list cereal_new_UbloxGnss_HwStatus_list(struct capn_segment *s, int len) {
	cereal_UbloxGnss_HwStatus_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_UbloxGnss_HwStatus(struct cereal_UbloxGnss_HwStatus *s, cereal_UbloxGnss_HwStatus_ptr p) {
	capn_resolve(&p.p);
	s->noisePerMS = capn_read16(p.p, 0);
	s->agcCnt = capn_read16(p.p, 2);
	s->aStatus = (enum cereal_UbloxGnss_HwStatus_AntennaSupervisorState)(int) capn_read16(p.p, 4);
	s->aPower = (enum cereal_UbloxGnss_HwStatus_AntennaPowerStatus)(int) capn_read16(p.p, 6);
	s->jamInd = capn_read8(p.p, 8);
}
void cereal_write_UbloxGnss_HwStatus(const struct cereal_UbloxGnss_HwStatus *s, cereal_UbloxGnss_HwStatus_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, s->noisePerMS);
	capn_write16(p.p, 2, s->agcCnt);
	capn_write16(p.p, 4, (uint16_t) (s->aStatus));
	capn_write16(p.p, 6, (uint16_t) (s->aPower));
	capn_write8(p.p, 8, s->jamInd);
}
void cereal_get_UbloxGnss_HwStatus(struct cereal_UbloxGnss_HwStatus *s, cereal_UbloxGnss_HwStatus_list l, int i) {
	cereal_UbloxGnss_HwStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UbloxGnss_HwStatus(s, p);
}
void cereal_set_UbloxGnss_HwStatus(const struct cereal_UbloxGnss_HwStatus *s, cereal_UbloxGnss_HwStatus_list l, int i) {
	cereal_UbloxGnss_HwStatus_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UbloxGnss_HwStatus(s, p);
}

cereal_Clocks_ptr cereal_new_Clocks(struct capn_segment *s) {
	cereal_Clocks_ptr p;
	p.p = capn_new_struct(s, 40, 0);
	return p;
}
cereal_Clocks_list cereal_new_Clocks_list(struct capn_segment *s, int len) {
	cereal_Clocks_list p;
	p.p = capn_new_list(s, len, 40, 0);
	return p;
}
void cereal_read_Clocks(struct cereal_Clocks *s, cereal_Clocks_ptr p) {
	capn_resolve(&p.p);
	s->bootTimeNanos = capn_read64(p.p, 0);
	s->monotonicNanos = capn_read64(p.p, 8);
	s->monotonicRawNanos = capn_read64(p.p, 16);
	s->wallTimeNanos = capn_read64(p.p, 24);
	s->modemUptimeMillis = capn_read64(p.p, 32);
}
void cereal_write_Clocks(const struct cereal_Clocks *s, cereal_Clocks_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->bootTimeNanos);
	capn_write64(p.p, 8, s->monotonicNanos);
	capn_write64(p.p, 16, s->monotonicRawNanos);
	capn_write64(p.p, 24, s->wallTimeNanos);
	capn_write64(p.p, 32, s->modemUptimeMillis);
}
void cereal_get_Clocks(struct cereal_Clocks *s, cereal_Clocks_list l, int i) {
	cereal_Clocks_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Clocks(s, p);
}
void cereal_set_Clocks(const struct cereal_Clocks *s, cereal_Clocks_list l, int i) {
	cereal_Clocks_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Clocks(s, p);
}

cereal_LiveMpcData_ptr cereal_new_LiveMpcData(struct capn_segment *s) {
	cereal_LiveMpcData_ptr p;
	p.p = capn_new_struct(s, 24, 4);
	return p;
}
cereal_LiveMpcData_list cereal_new_LiveMpcData_list(struct capn_segment *s, int len) {
	cereal_LiveMpcData_list p;
	p.p = capn_new_list(s, len, 24, 4);
	return p;
}
void cereal_read_LiveMpcData(struct cereal_LiveMpcData *s, cereal_LiveMpcData_ptr p) {
	capn_resolve(&p.p);
	s->x.p = capn_getp(p.p, 0, 0);
	s->y.p = capn_getp(p.p, 1, 0);
	s->psi.p = capn_getp(p.p, 2, 0);
	s->delta.p = capn_getp(p.p, 3, 0);
	s->qpIterations = capn_read32(p.p, 0);
	s->calculationTime = capn_read64(p.p, 8);
	s->cost = capn_to_f64(capn_read64(p.p, 16));
}
void cereal_write_LiveMpcData(const struct cereal_LiveMpcData *s, cereal_LiveMpcData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->x.p);
	capn_setp(p.p, 1, s->y.p);
	capn_setp(p.p, 2, s->psi.p);
	capn_setp(p.p, 3, s->delta.p);
	capn_write32(p.p, 0, s->qpIterations);
	capn_write64(p.p, 8, s->calculationTime);
	capn_write64(p.p, 16, capn_from_f64(s->cost));
}
void cereal_get_LiveMpcData(struct cereal_LiveMpcData *s, cereal_LiveMpcData_list l, int i) {
	cereal_LiveMpcData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveMpcData(s, p);
}
void cereal_set_LiveMpcData(const struct cereal_LiveMpcData *s, cereal_LiveMpcData_list l, int i) {
	cereal_LiveMpcData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveMpcData(s, p);
}

cereal_LiveLongitudinalMpcData_ptr cereal_new_LiveLongitudinalMpcData(struct capn_segment *s) {
	cereal_LiveLongitudinalMpcData_ptr p;
	p.p = capn_new_struct(s, 32, 6);
	return p;
}
cereal_LiveLongitudinalMpcData_list cereal_new_LiveLongitudinalMpcData_list(struct capn_segment *s, int len) {
	cereal_LiveLongitudinalMpcData_list p;
	p.p = capn_new_list(s, len, 32, 6);
	return p;
}
void cereal_read_LiveLongitudinalMpcData(struct cereal_LiveLongitudinalMpcData *s, cereal_LiveLongitudinalMpcData_ptr p) {
	capn_resolve(&p.p);
	s->xEgo.p = capn_getp(p.p, 0, 0);
	s->vEgo.p = capn_getp(p.p, 1, 0);
	s->aEgo.p = capn_getp(p.p, 2, 0);
	s->xLead.p = capn_getp(p.p, 3, 0);
	s->vLead.p = capn_getp(p.p, 4, 0);
	s->aLead.p = capn_getp(p.p, 5, 0);
	s->aLeadTau = capn_to_f32(capn_read32(p.p, 0));
	s->qpIterations = capn_read32(p.p, 4);
	s->mpcId = capn_read32(p.p, 8);
	s->calculationTime = capn_read64(p.p, 16);
	s->cost = capn_to_f64(capn_read64(p.p, 24));
}
void cereal_write_LiveLongitudinalMpcData(const struct cereal_LiveLongitudinalMpcData *s, cereal_LiveLongitudinalMpcData_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->xEgo.p);
	capn_setp(p.p, 1, s->vEgo.p);
	capn_setp(p.p, 2, s->aEgo.p);
	capn_setp(p.p, 3, s->xLead.p);
	capn_setp(p.p, 4, s->vLead.p);
	capn_setp(p.p, 5, s->aLead.p);
	capn_write32(p.p, 0, capn_from_f32(s->aLeadTau));
	capn_write32(p.p, 4, s->qpIterations);
	capn_write32(p.p, 8, s->mpcId);
	capn_write64(p.p, 16, s->calculationTime);
	capn_write64(p.p, 24, capn_from_f64(s->cost));
}
void cereal_get_LiveLongitudinalMpcData(struct cereal_LiveLongitudinalMpcData *s, cereal_LiveLongitudinalMpcData_list l, int i) {
	cereal_LiveLongitudinalMpcData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveLongitudinalMpcData(s, p);
}
void cereal_set_LiveLongitudinalMpcData(const struct cereal_LiveLongitudinalMpcData *s, cereal_LiveLongitudinalMpcData_list l, int i) {
	cereal_LiveLongitudinalMpcData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveLongitudinalMpcData(s, p);
}

cereal_ECEFPointDEPRECATED_ptr cereal_new_ECEFPointDEPRECATED(struct capn_segment *s) {
	cereal_ECEFPointDEPRECATED_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_ECEFPointDEPRECATED_list cereal_new_ECEFPointDEPRECATED_list(struct capn_segment *s, int len) {
	cereal_ECEFPointDEPRECATED_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_ECEFPointDEPRECATED(struct cereal_ECEFPointDEPRECATED *s, cereal_ECEFPointDEPRECATED_ptr p) {
	capn_resolve(&p.p);
	s->x = capn_to_f32(capn_read32(p.p, 0));
	s->y = capn_to_f32(capn_read32(p.p, 4));
	s->z = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_ECEFPointDEPRECATED(const struct cereal_ECEFPointDEPRECATED *s, cereal_ECEFPointDEPRECATED_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, capn_from_f32(s->x));
	capn_write32(p.p, 4, capn_from_f32(s->y));
	capn_write32(p.p, 8, capn_from_f32(s->z));
}
void cereal_get_ECEFPointDEPRECATED(struct cereal_ECEFPointDEPRECATED *s, cereal_ECEFPointDEPRECATED_list l, int i) {
	cereal_ECEFPointDEPRECATED_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ECEFPointDEPRECATED(s, p);
}
void cereal_set_ECEFPointDEPRECATED(const struct cereal_ECEFPointDEPRECATED *s, cereal_ECEFPointDEPRECATED_list l, int i) {
	cereal_ECEFPointDEPRECATED_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ECEFPointDEPRECATED(s, p);
}

cereal_ECEFPoint_ptr cereal_new_ECEFPoint(struct capn_segment *s) {
	cereal_ECEFPoint_ptr p;
	p.p = capn_new_struct(s, 24, 0);
	return p;
}
cereal_ECEFPoint_list cereal_new_ECEFPoint_list(struct capn_segment *s, int len) {
	cereal_ECEFPoint_list p;
	p.p = capn_new_list(s, len, 24, 0);
	return p;
}
void cereal_read_ECEFPoint(struct cereal_ECEFPoint *s, cereal_ECEFPoint_ptr p) {
	capn_resolve(&p.p);
	s->x = capn_to_f64(capn_read64(p.p, 0));
	s->y = capn_to_f64(capn_read64(p.p, 8));
	s->z = capn_to_f64(capn_read64(p.p, 16));
}
void cereal_write_ECEFPoint(const struct cereal_ECEFPoint *s, cereal_ECEFPoint_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, capn_from_f64(s->x));
	capn_write64(p.p, 8, capn_from_f64(s->y));
	capn_write64(p.p, 16, capn_from_f64(s->z));
}
void cereal_get_ECEFPoint(struct cereal_ECEFPoint *s, cereal_ECEFPoint_list l, int i) {
	cereal_ECEFPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_ECEFPoint(s, p);
}
void cereal_set_ECEFPoint(const struct cereal_ECEFPoint *s, cereal_ECEFPoint_list l, int i) {
	cereal_ECEFPoint_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_ECEFPoint(s, p);
}

cereal_GPSPlannerPoints_ptr cereal_new_GPSPlannerPoints(struct capn_segment *s) {
	cereal_GPSPlannerPoints_ptr p;
	p.p = capn_new_struct(s, 16, 5);
	return p;
}
cereal_GPSPlannerPoints_list cereal_new_GPSPlannerPoints_list(struct capn_segment *s, int len) {
	cereal_GPSPlannerPoints_list p;
	p.p = capn_new_list(s, len, 16, 5);
	return p;
}
void cereal_read_GPSPlannerPoints(struct cereal_GPSPlannerPoints *s, cereal_GPSPlannerPoints_ptr p) {
	capn_resolve(&p.p);
	s->curPosDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->pointsDEPRECATED.p = capn_getp(p.p, 1, 0);
	s->curPos.p = capn_getp(p.p, 3, 0);
	s->points.p = capn_getp(p.p, 4, 0);
	s->valid = (capn_read8(p.p, 0) & 1) != 0;
	s->trackName = capn_get_text(p.p, 2, capn_val0);
	s->speedLimit = capn_to_f32(capn_read32(p.p, 4));
	s->accelTarget = capn_to_f32(capn_read32(p.p, 8));
}
void cereal_write_GPSPlannerPoints(const struct cereal_GPSPlannerPoints *s, cereal_GPSPlannerPoints_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->curPosDEPRECATED.p);
	capn_setp(p.p, 1, s->pointsDEPRECATED.p);
	capn_setp(p.p, 3, s->curPos.p);
	capn_setp(p.p, 4, s->points.p);
	capn_write1(p.p, 0, s->valid != 0);
	capn_set_text(p.p, 2, s->trackName);
	capn_write32(p.p, 4, capn_from_f32(s->speedLimit));
	capn_write32(p.p, 8, capn_from_f32(s->accelTarget));
}
void cereal_get_GPSPlannerPoints(struct cereal_GPSPlannerPoints *s, cereal_GPSPlannerPoints_list l, int i) {
	cereal_GPSPlannerPoints_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_GPSPlannerPoints(s, p);
}
void cereal_set_GPSPlannerPoints(const struct cereal_GPSPlannerPoints *s, cereal_GPSPlannerPoints_list l, int i) {
	cereal_GPSPlannerPoints_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_GPSPlannerPoints(s, p);
}

cereal_GPSPlannerPlan_ptr cereal_new_GPSPlannerPlan(struct capn_segment *s) {
	cereal_GPSPlannerPlan_ptr p;
	p.p = capn_new_struct(s, 16, 4);
	return p;
}
cereal_GPSPlannerPlan_list cereal_new_GPSPlannerPlan_list(struct capn_segment *s, int len) {
	cereal_GPSPlannerPlan_list p;
	p.p = capn_new_list(s, len, 16, 4);
	return p;
}
void cereal_read_GPSPlannerPlan(struct cereal_GPSPlannerPlan *s, cereal_GPSPlannerPlan_ptr p) {
	capn_resolve(&p.p);
	s->valid = (capn_read8(p.p, 0) & 1) != 0;
	s->poly.p = capn_getp(p.p, 0, 0);
	s->trackName = capn_get_text(p.p, 1, capn_val0);
	s->speed = capn_to_f32(capn_read32(p.p, 4));
	s->acceleration = capn_to_f32(capn_read32(p.p, 8));
	s->pointsDEPRECATED.p = capn_getp(p.p, 2, 0);
	s->points.p = capn_getp(p.p, 3, 0);
	s->xLookahead = capn_to_f32(capn_read32(p.p, 12));
}
void cereal_write_GPSPlannerPlan(const struct cereal_GPSPlannerPlan *s, cereal_GPSPlannerPlan_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->valid != 0);
	capn_setp(p.p, 0, s->poly.p);
	capn_set_text(p.p, 1, s->trackName);
	capn_write32(p.p, 4, capn_from_f32(s->speed));
	capn_write32(p.p, 8, capn_from_f32(s->acceleration));
	capn_setp(p.p, 2, s->pointsDEPRECATED.p);
	capn_setp(p.p, 3, s->points.p);
	capn_write32(p.p, 12, capn_from_f32(s->xLookahead));
}
void cereal_get_GPSPlannerPlan(struct cereal_GPSPlannerPlan *s, cereal_GPSPlannerPlan_list l, int i) {
	cereal_GPSPlannerPlan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_GPSPlannerPlan(s, p);
}
void cereal_set_GPSPlannerPlan(const struct cereal_GPSPlannerPlan *s, cereal_GPSPlannerPlan_list l, int i) {
	cereal_GPSPlannerPlan_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_GPSPlannerPlan(s, p);
}

cereal_TrafficEvent_ptr cereal_new_TrafficEvent(struct capn_segment *s) {
	cereal_TrafficEvent_ptr p;
	p.p = capn_new_struct(s, 16, 0);
	return p;
}
cereal_TrafficEvent_list cereal_new_TrafficEvent_list(struct capn_segment *s, int len) {
	cereal_TrafficEvent_list p;
	p.p = capn_new_list(s, len, 16, 0);
	return p;
}
void cereal_read_TrafficEvent(struct cereal_TrafficEvent *s, cereal_TrafficEvent_ptr p) {
	capn_resolve(&p.p);
	s->type = (enum cereal_TrafficEvent_Type)(int) capn_read16(p.p, 0);
	s->distance = capn_to_f32(capn_read32(p.p, 4));
	s->action = (enum cereal_TrafficEvent_Action)(int) capn_read16(p.p, 2);
	s->resuming = (capn_read8(p.p, 8) & 1) != 0;
}
void cereal_write_TrafficEvent(const struct cereal_TrafficEvent *s, cereal_TrafficEvent_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, (uint16_t) (s->type));
	capn_write32(p.p, 4, capn_from_f32(s->distance));
	capn_write16(p.p, 2, (uint16_t) (s->action));
	capn_write1(p.p, 64, s->resuming != 0);
}
void cereal_get_TrafficEvent(struct cereal_TrafficEvent *s, cereal_TrafficEvent_list l, int i) {
	cereal_TrafficEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_TrafficEvent(s, p);
}
void cereal_set_TrafficEvent(const struct cereal_TrafficEvent *s, cereal_TrafficEvent_list l, int i) {
	cereal_TrafficEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_TrafficEvent(s, p);
}

cereal_OrbslamCorrection_ptr cereal_new_OrbslamCorrection(struct capn_segment *s) {
	cereal_OrbslamCorrection_ptr p;
	p.p = capn_new_struct(s, 16, 4);
	return p;
}
cereal_OrbslamCorrection_list cereal_new_OrbslamCorrection_list(struct capn_segment *s, int len) {
	cereal_OrbslamCorrection_list p;
	p.p = capn_new_list(s, len, 16, 4);
	return p;
}
void cereal_read_OrbslamCorrection(struct cereal_OrbslamCorrection *s, cereal_OrbslamCorrection_ptr p) {
	capn_resolve(&p.p);
	s->correctionMonoTime = capn_read64(p.p, 0);
	s->prePositionECEF.p = capn_getp(p.p, 0, 0);
	s->postPositionECEF.p = capn_getp(p.p, 1, 0);
	s->prePoseQuatECEF.p = capn_getp(p.p, 2, 0);
	s->postPoseQuatECEF.p = capn_getp(p.p, 3, 0);
	s->numInliers = capn_read32(p.p, 8);
}
void cereal_write_OrbslamCorrection(const struct cereal_OrbslamCorrection *s, cereal_OrbslamCorrection_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->correctionMonoTime);
	capn_setp(p.p, 0, s->prePositionECEF.p);
	capn_setp(p.p, 1, s->postPositionECEF.p);
	capn_setp(p.p, 2, s->prePoseQuatECEF.p);
	capn_setp(p.p, 3, s->postPoseQuatECEF.p);
	capn_write32(p.p, 8, s->numInliers);
}
void cereal_get_OrbslamCorrection(struct cereal_OrbslamCorrection *s, cereal_OrbslamCorrection_list l, int i) {
	cereal_OrbslamCorrection_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbslamCorrection(s, p);
}
void cereal_set_OrbslamCorrection(const struct cereal_OrbslamCorrection *s, cereal_OrbslamCorrection_list l, int i) {
	cereal_OrbslamCorrection_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbslamCorrection(s, p);
}

cereal_OrbObservation_ptr cereal_new_OrbObservation(struct capn_segment *s) {
	cereal_OrbObservation_ptr p;
	p.p = capn_new_struct(s, 16, 2);
	return p;
}
cereal_OrbObservation_list cereal_new_OrbObservation_list(struct capn_segment *s, int len) {
	cereal_OrbObservation_list p;
	p.p = capn_new_list(s, len, 16, 2);
	return p;
}
void cereal_read_OrbObservation(struct cereal_OrbObservation *s, cereal_OrbObservation_ptr p) {
	capn_resolve(&p.p);
	s->observationMonoTime = capn_read64(p.p, 0);
	s->normalizedCoordinates.p = capn_getp(p.p, 0, 0);
	s->locationECEF.p = capn_getp(p.p, 1, 0);
	s->matchDistance = capn_read32(p.p, 8);
}
void cereal_write_OrbObservation(const struct cereal_OrbObservation *s, cereal_OrbObservation_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->observationMonoTime);
	capn_setp(p.p, 0, s->normalizedCoordinates.p);
	capn_setp(p.p, 1, s->locationECEF.p);
	capn_write32(p.p, 8, s->matchDistance);
}
void cereal_get_OrbObservation(struct cereal_OrbObservation *s, cereal_OrbObservation_list l, int i) {
	cereal_OrbObservation_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbObservation(s, p);
}
void cereal_set_OrbObservation(const struct cereal_OrbObservation *s, cereal_OrbObservation_list l, int i) {
	cereal_OrbObservation_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbObservation(s, p);
}

cereal_UiNavigationEvent_ptr cereal_new_UiNavigationEvent(struct capn_segment *s) {
	cereal_UiNavigationEvent_ptr p;
	p.p = capn_new_struct(s, 8, 2);
	return p;
}
cereal_UiNavigationEvent_list cereal_new_UiNavigationEvent_list(struct capn_segment *s, int len) {
	cereal_UiNavigationEvent_list p;
	p.p = capn_new_list(s, len, 8, 2);
	return p;
}
void cereal_read_UiNavigationEvent(struct cereal_UiNavigationEvent *s, cereal_UiNavigationEvent_ptr p) {
	capn_resolve(&p.p);
	s->type = (enum cereal_UiNavigationEvent_Type)(int) capn_read16(p.p, 0);
	s->status = (enum cereal_UiNavigationEvent_Status)(int) capn_read16(p.p, 2);
	s->distanceTo = capn_to_f32(capn_read32(p.p, 4));
	s->endRoadPointDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->endRoadPoint.p = capn_getp(p.p, 1, 0);
}
void cereal_write_UiNavigationEvent(const struct cereal_UiNavigationEvent *s, cereal_UiNavigationEvent_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, (uint16_t) (s->type));
	capn_write16(p.p, 2, (uint16_t) (s->status));
	capn_write32(p.p, 4, capn_from_f32(s->distanceTo));
	capn_setp(p.p, 0, s->endRoadPointDEPRECATED.p);
	capn_setp(p.p, 1, s->endRoadPoint.p);
}
void cereal_get_UiNavigationEvent(struct cereal_UiNavigationEvent *s, cereal_UiNavigationEvent_list l, int i) {
	cereal_UiNavigationEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UiNavigationEvent(s, p);
}
void cereal_set_UiNavigationEvent(const struct cereal_UiNavigationEvent *s, cereal_UiNavigationEvent_list l, int i) {
	cereal_UiNavigationEvent_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UiNavigationEvent(s, p);
}

cereal_UiLayoutState_ptr cereal_new_UiLayoutState(struct capn_segment *s) {
	cereal_UiLayoutState_ptr p;
	p.p = capn_new_struct(s, 8, 0);
	return p;
}
cereal_UiLayoutState_list cereal_new_UiLayoutState_list(struct capn_segment *s, int len) {
	cereal_UiLayoutState_list p;
	p.p = capn_new_list(s, len, 8, 0);
	return p;
}
void cereal_read_UiLayoutState(struct cereal_UiLayoutState *s, cereal_UiLayoutState_ptr p) {
	capn_resolve(&p.p);
	s->activeApp = (enum cereal_UiLayoutState_App)(int) capn_read16(p.p, 0);
	s->sidebarCollapsed = (capn_read8(p.p, 2) & 1) != 0;
	s->mapEnabled = (capn_read8(p.p, 2) & 2) != 0;
}
void cereal_write_UiLayoutState(const struct cereal_UiLayoutState *s, cereal_UiLayoutState_ptr p) {
	capn_resolve(&p.p);
	capn_write16(p.p, 0, (uint16_t) (s->activeApp));
	capn_write1(p.p, 16, s->sidebarCollapsed != 0);
	capn_write1(p.p, 17, s->mapEnabled != 0);
}
void cereal_get_UiLayoutState(struct cereal_UiLayoutState *s, cereal_UiLayoutState_list l, int i) {
	cereal_UiLayoutState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_UiLayoutState(s, p);
}
void cereal_set_UiLayoutState(const struct cereal_UiLayoutState *s, cereal_UiLayoutState_list l, int i) {
	cereal_UiLayoutState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_UiLayoutState(s, p);
}

cereal_Joystick_ptr cereal_new_Joystick(struct capn_segment *s) {
	cereal_Joystick_ptr p;
	p.p = capn_new_struct(s, 0, 2);
	return p;
}
cereal_Joystick_list cereal_new_Joystick_list(struct capn_segment *s, int len) {
	cereal_Joystick_list p;
	p.p = capn_new_list(s, len, 0, 2);
	return p;
}
void cereal_read_Joystick(struct cereal_Joystick *s, cereal_Joystick_ptr p) {
	capn_resolve(&p.p);
	s->axes.p = capn_getp(p.p, 0, 0);
	s->buttons.p = capn_getp(p.p, 1, 0);
}
void cereal_write_Joystick(const struct cereal_Joystick *s, cereal_Joystick_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->axes.p);
	capn_setp(p.p, 1, s->buttons.p);
}
void cereal_get_Joystick(struct cereal_Joystick *s, cereal_Joystick_list l, int i) {
	cereal_Joystick_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Joystick(s, p);
}
void cereal_set_Joystick(const struct cereal_Joystick *s, cereal_Joystick_list l, int i) {
	cereal_Joystick_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Joystick(s, p);
}

cereal_OrbOdometry_ptr cereal_new_OrbOdometry(struct capn_segment *s) {
	cereal_OrbOdometry_ptr p;
	p.p = capn_new_struct(s, 32, 2);
	return p;
}
cereal_OrbOdometry_list cereal_new_OrbOdometry_list(struct capn_segment *s, int len) {
	cereal_OrbOdometry_list p;
	p.p = capn_new_list(s, len, 32, 2);
	return p;
}
void cereal_read_OrbOdometry(struct cereal_OrbOdometry *s, cereal_OrbOdometry_ptr p) {
	capn_resolve(&p.p);
	s->startMonoTime = capn_read64(p.p, 0);
	s->endMonoTime = capn_read64(p.p, 8);
	s->f.p = capn_getp(p.p, 0, 0);
	s->err = capn_to_f64(capn_read64(p.p, 16));
	s->inliers = (int32_t) ((int32_t)capn_read32(p.p, 24));
	s->matches.p = capn_getp(p.p, 1, 0);
}
void cereal_write_OrbOdometry(const struct cereal_OrbOdometry *s, cereal_OrbOdometry_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->startMonoTime);
	capn_write64(p.p, 8, s->endMonoTime);
	capn_setp(p.p, 0, s->f.p);
	capn_write64(p.p, 16, capn_from_f64(s->err));
	capn_write32(p.p, 24, (uint32_t) (s->inliers));
	capn_setp(p.p, 1, s->matches.p);
}
void cereal_get_OrbOdometry(struct cereal_OrbOdometry *s, cereal_OrbOdometry_list l, int i) {
	cereal_OrbOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbOdometry(s, p);
}
void cereal_set_OrbOdometry(const struct cereal_OrbOdometry *s, cereal_OrbOdometry_list l, int i) {
	cereal_OrbOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbOdometry(s, p);
}

cereal_OrbFeatures_ptr cereal_new_OrbFeatures(struct capn_segment *s) {
	cereal_OrbFeatures_ptr p;
	p.p = capn_new_struct(s, 16, 5);
	return p;
}
cereal_OrbFeatures_list cereal_new_OrbFeatures_list(struct capn_segment *s, int len) {
	cereal_OrbFeatures_list p;
	p.p = capn_new_list(s, len, 16, 5);
	return p;
}
void cereal_read_OrbFeatures(struct cereal_OrbFeatures *s, cereal_OrbFeatures_ptr p) {
	capn_resolve(&p.p);
	s->timestampEof = capn_read64(p.p, 0);
	s->xs.p = capn_getp(p.p, 0, 0);
	s->ys.p = capn_getp(p.p, 1, 0);
	s->descriptors = capn_get_data(p.p, 2);
	s->octaves.p = capn_getp(p.p, 3, 0);
	s->timestampLastEof = capn_read64(p.p, 8);
	s->matches.p = capn_getp(p.p, 4, 0);
}
void cereal_write_OrbFeatures(const struct cereal_OrbFeatures *s, cereal_OrbFeatures_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->timestampEof);
	capn_setp(p.p, 0, s->xs.p);
	capn_setp(p.p, 1, s->ys.p);
	capn_setp(p.p, 2, s->descriptors.p);
	capn_setp(p.p, 3, s->octaves.p);
	capn_write64(p.p, 8, s->timestampLastEof);
	capn_setp(p.p, 4, s->matches.p);
}
void cereal_get_OrbFeatures(struct cereal_OrbFeatures *s, cereal_OrbFeatures_list l, int i) {
	cereal_OrbFeatures_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbFeatures(s, p);
}
void cereal_set_OrbFeatures(const struct cereal_OrbFeatures *s, cereal_OrbFeatures_list l, int i) {
	cereal_OrbFeatures_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbFeatures(s, p);
}

cereal_OrbFeaturesSummary_ptr cereal_new_OrbFeaturesSummary(struct capn_segment *s) {
	cereal_OrbFeaturesSummary_ptr p;
	p.p = capn_new_struct(s, 32, 0);
	return p;
}
cereal_OrbFeaturesSummary_list cereal_new_OrbFeaturesSummary_list(struct capn_segment *s, int len) {
	cereal_OrbFeaturesSummary_list p;
	p.p = capn_new_list(s, len, 32, 0);
	return p;
}
void cereal_read_OrbFeaturesSummary(struct cereal_OrbFeaturesSummary *s, cereal_OrbFeaturesSummary_ptr p) {
	capn_resolve(&p.p);
	s->timestampEof = capn_read64(p.p, 0);
	s->timestampLastEof = capn_read64(p.p, 8);
	s->featureCount = capn_read16(p.p, 16);
	s->matchCount = capn_read16(p.p, 18);
	s->computeNs = capn_read64(p.p, 24);
}
void cereal_write_OrbFeaturesSummary(const struct cereal_OrbFeaturesSummary *s, cereal_OrbFeaturesSummary_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->timestampEof);
	capn_write64(p.p, 8, s->timestampLastEof);
	capn_write16(p.p, 16, s->featureCount);
	capn_write16(p.p, 18, s->matchCount);
	capn_write64(p.p, 24, s->computeNs);
}
void cereal_get_OrbFeaturesSummary(struct cereal_OrbFeaturesSummary *s, cereal_OrbFeaturesSummary_list l, int i) {
	cereal_OrbFeaturesSummary_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbFeaturesSummary(s, p);
}
void cereal_set_OrbFeaturesSummary(const struct cereal_OrbFeaturesSummary *s, cereal_OrbFeaturesSummary_list l, int i) {
	cereal_OrbFeaturesSummary_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbFeaturesSummary(s, p);
}

cereal_OrbKeyFrame_ptr cereal_new_OrbKeyFrame(struct capn_segment *s) {
	cereal_OrbKeyFrame_ptr p;
	p.p = capn_new_struct(s, 8, 3);
	return p;
}
cereal_OrbKeyFrame_list cereal_new_OrbKeyFrame_list(struct capn_segment *s, int len) {
	cereal_OrbKeyFrame_list p;
	p.p = capn_new_list(s, len, 8, 3);
	return p;
}
void cereal_read_OrbKeyFrame(struct cereal_OrbKeyFrame *s, cereal_OrbKeyFrame_ptr p) {
	capn_resolve(&p.p);
	s->id = capn_read64(p.p, 0);
	s->pos.p = capn_getp(p.p, 0, 0);
	s->dpos.p = capn_getp(p.p, 1, 0);
	s->descriptors = capn_get_data(p.p, 2);
}
void cereal_write_OrbKeyFrame(const struct cereal_OrbKeyFrame *s, cereal_OrbKeyFrame_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->id);
	capn_setp(p.p, 0, s->pos.p);
	capn_setp(p.p, 1, s->dpos.p);
	capn_setp(p.p, 2, s->descriptors.p);
}
void cereal_get_OrbKeyFrame(struct cereal_OrbKeyFrame *s, cereal_OrbKeyFrame_list l, int i) {
	cereal_OrbKeyFrame_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_OrbKeyFrame(s, p);
}
void cereal_set_OrbKeyFrame(const struct cereal_OrbKeyFrame *s, cereal_OrbKeyFrame_list l, int i) {
	cereal_OrbKeyFrame_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_OrbKeyFrame(s, p);
}

cereal_DriverState_ptr cereal_new_DriverState(struct capn_segment *s) {
	cereal_DriverState_ptr p;
	p.p = capn_new_struct(s, 32, 5);
	return p;
}
cereal_DriverState_list cereal_new_DriverState_list(struct capn_segment *s, int len) {
	cereal_DriverState_list p;
	p.p = capn_new_list(s, len, 32, 5);
	return p;
}
void cereal_read_DriverState(struct cereal_DriverState *s, cereal_DriverState_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->descriptorDEPRECATED.p = capn_getp(p.p, 0, 0);
	s->stdDEPRECATED = capn_to_f32(capn_read32(p.p, 4));
	s->faceOrientation.p = capn_getp(p.p, 1, 0);
	s->facePosition.p = capn_getp(p.p, 2, 0);
	s->faceProb = capn_to_f32(capn_read32(p.p, 8));
	s->leftEyeProb = capn_to_f32(capn_read32(p.p, 12));
	s->rightEyeProb = capn_to_f32(capn_read32(p.p, 16));
	s->leftBlinkProb = capn_to_f32(capn_read32(p.p, 20));
	s->rightBlinkProb = capn_to_f32(capn_read32(p.p, 24));
	s->irPwrDEPRECATED = capn_to_f32(capn_read32(p.p, 28));
	s->faceOrientationStd.p = capn_getp(p.p, 3, 0);
	s->facePositionStd.p = capn_getp(p.p, 4, 0);
}
void cereal_write_DriverState(const struct cereal_DriverState *s, cereal_DriverState_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_setp(p.p, 0, s->descriptorDEPRECATED.p);
	capn_write32(p.p, 4, capn_from_f32(s->stdDEPRECATED));
	capn_setp(p.p, 1, s->faceOrientation.p);
	capn_setp(p.p, 2, s->facePosition.p);
	capn_write32(p.p, 8, capn_from_f32(s->faceProb));
	capn_write32(p.p, 12, capn_from_f32(s->leftEyeProb));
	capn_write32(p.p, 16, capn_from_f32(s->rightEyeProb));
	capn_write32(p.p, 20, capn_from_f32(s->leftBlinkProb));
	capn_write32(p.p, 24, capn_from_f32(s->rightBlinkProb));
	capn_write32(p.p, 28, capn_from_f32(s->irPwrDEPRECATED));
	capn_setp(p.p, 3, s->faceOrientationStd.p);
	capn_setp(p.p, 4, s->facePositionStd.p);
}
void cereal_get_DriverState(struct cereal_DriverState *s, cereal_DriverState_list l, int i) {
	cereal_DriverState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_DriverState(s, p);
}
void cereal_set_DriverState(const struct cereal_DriverState *s, cereal_DriverState_list l, int i) {
	cereal_DriverState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_DriverState(s, p);
}

cereal_DMonitoringState_ptr cereal_new_DMonitoringState(struct capn_segment *s) {
	cereal_DMonitoringState_ptr p;
	p.p = capn_new_struct(s, 40, 1);
	return p;
}
cereal_DMonitoringState_list cereal_new_DMonitoringState_list(struct capn_segment *s, int len) {
	cereal_DMonitoringState_list p;
	p.p = capn_new_list(s, len, 40, 1);
	return p;
}
void cereal_read_DMonitoringState(struct cereal_DMonitoringState *s, cereal_DMonitoringState_ptr p) {
	capn_resolve(&p.p);
	s->events.p = capn_getp(p.p, 0, 0);
	s->faceDetected = (capn_read8(p.p, 0) & 1) != 0;
	s->isDistracted = (capn_read8(p.p, 0) & 2) != 0;
	s->awarenessStatus = capn_to_f32(capn_read32(p.p, 4));
	s->isRHD = (capn_read8(p.p, 0) & 4) != 0;
	s->rhdChecked = (capn_read8(p.p, 0) & 8) != 0;
	s->posePitchOffset = capn_to_f32(capn_read32(p.p, 8));
	s->posePitchValidCount = capn_read32(p.p, 12);
	s->poseYawOffset = capn_to_f32(capn_read32(p.p, 16));
	s->poseYawValidCount = capn_read32(p.p, 20);
	s->stepChange = capn_to_f32(capn_read32(p.p, 24));
	s->awarenessActive = capn_to_f32(capn_read32(p.p, 28));
	s->awarenessPassive = capn_to_f32(capn_read32(p.p, 32));
	s->isLowStd = (capn_read8(p.p, 0) & 16) != 0;
	s->hiStdCount = capn_read32(p.p, 36);
}
void cereal_write_DMonitoringState(const struct cereal_DMonitoringState *s, cereal_DMonitoringState_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->events.p);
	capn_write1(p.p, 0, s->faceDetected != 0);
	capn_write1(p.p, 1, s->isDistracted != 0);
	capn_write32(p.p, 4, capn_from_f32(s->awarenessStatus));
	capn_write1(p.p, 2, s->isRHD != 0);
	capn_write1(p.p, 3, s->rhdChecked != 0);
	capn_write32(p.p, 8, capn_from_f32(s->posePitchOffset));
	capn_write32(p.p, 12, s->posePitchValidCount);
	capn_write32(p.p, 16, capn_from_f32(s->poseYawOffset));
	capn_write32(p.p, 20, s->poseYawValidCount);
	capn_write32(p.p, 24, capn_from_f32(s->stepChange));
	capn_write32(p.p, 28, capn_from_f32(s->awarenessActive));
	capn_write32(p.p, 32, capn_from_f32(s->awarenessPassive));
	capn_write1(p.p, 4, s->isLowStd != 0);
	capn_write32(p.p, 36, s->hiStdCount);
}
void cereal_get_DMonitoringState(struct cereal_DMonitoringState *s, cereal_DMonitoringState_list l, int i) {
	cereal_DMonitoringState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_DMonitoringState(s, p);
}
void cereal_set_DMonitoringState(const struct cereal_DMonitoringState *s, cereal_DMonitoringState_list l, int i) {
	cereal_DMonitoringState_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_DMonitoringState(s, p);
}

cereal_Boot_ptr cereal_new_Boot(struct capn_segment *s) {
	cereal_Boot_ptr p;
	p.p = capn_new_struct(s, 8, 2);
	return p;
}
cereal_Boot_list cereal_new_Boot_list(struct capn_segment *s, int len) {
	cereal_Boot_list p;
	p.p = capn_new_list(s, len, 8, 2);
	return p;
}
void cereal_read_Boot(struct cereal_Boot *s, cereal_Boot_ptr p) {
	capn_resolve(&p.p);
	s->wallTimeNanos = capn_read64(p.p, 0);
	s->lastKmsg = capn_get_data(p.p, 0);
	s->lastPmsg = capn_get_data(p.p, 1);
}
void cereal_write_Boot(const struct cereal_Boot *s, cereal_Boot_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->wallTimeNanos);
	capn_setp(p.p, 0, s->lastKmsg.p);
	capn_setp(p.p, 1, s->lastPmsg.p);
}
void cereal_get_Boot(struct cereal_Boot *s, cereal_Boot_list l, int i) {
	cereal_Boot_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_Boot(s, p);
}
void cereal_set_Boot(const struct cereal_Boot *s, cereal_Boot_list l, int i) {
	cereal_Boot_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_Boot(s, p);
}

cereal_LiveParametersData_ptr cereal_new_LiveParametersData(struct capn_segment *s) {
	cereal_LiveParametersData_ptr p;
	p.p = capn_new_struct(s, 32, 0);
	return p;
}
cereal_LiveParametersData_list cereal_new_LiveParametersData_list(struct capn_segment *s, int len) {
	cereal_LiveParametersData_list p;
	p.p = capn_new_list(s, len, 32, 0);
	return p;
}
void cereal_read_LiveParametersData(struct cereal_LiveParametersData *s, cereal_LiveParametersData_ptr p) {
	capn_resolve(&p.p);
	s->valid = (capn_read8(p.p, 0) & 1) != 0;
	s->gyroBias = capn_to_f32(capn_read32(p.p, 4));
	s->angleOffset = capn_to_f32(capn_read32(p.p, 8));
	s->angleOffsetAverage = capn_to_f32(capn_read32(p.p, 12));
	s->stiffnessFactor = capn_to_f32(capn_read32(p.p, 16));
	s->steerRatio = capn_to_f32(capn_read32(p.p, 20));
	s->sensorValid = (capn_read8(p.p, 0) & 2) != 0;
	s->yawRate = capn_to_f32(capn_read32(p.p, 24));
	s->posenetSpeed = capn_to_f32(capn_read32(p.p, 28));
	s->posenetValid = (capn_read8(p.p, 0) & 4) != 0;
}
void cereal_write_LiveParametersData(const struct cereal_LiveParametersData *s, cereal_LiveParametersData_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->valid != 0);
	capn_write32(p.p, 4, capn_from_f32(s->gyroBias));
	capn_write32(p.p, 8, capn_from_f32(s->angleOffset));
	capn_write32(p.p, 12, capn_from_f32(s->angleOffsetAverage));
	capn_write32(p.p, 16, capn_from_f32(s->stiffnessFactor));
	capn_write32(p.p, 20, capn_from_f32(s->steerRatio));
	capn_write1(p.p, 1, s->sensorValid != 0);
	capn_write32(p.p, 24, capn_from_f32(s->yawRate));
	capn_write32(p.p, 28, capn_from_f32(s->posenetSpeed));
	capn_write1(p.p, 2, s->posenetValid != 0);
}
void cereal_get_LiveParametersData(struct cereal_LiveParametersData *s, cereal_LiveParametersData_list l, int i) {
	cereal_LiveParametersData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveParametersData(s, p);
}
void cereal_set_LiveParametersData(const struct cereal_LiveParametersData *s, cereal_LiveParametersData_list l, int i) {
	cereal_LiveParametersData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveParametersData(s, p);
}

cereal_LiveMapData_ptr cereal_new_LiveMapData(struct capn_segment *s) {
	cereal_LiveMapData_ptr p;
	p.p = capn_new_struct(s, 40, 5);
	return p;
}
cereal_LiveMapData_list cereal_new_LiveMapData_list(struct capn_segment *s, int len) {
	cereal_LiveMapData_list p;
	p.p = capn_new_list(s, len, 40, 5);
	return p;
}
void cereal_read_LiveMapData(struct cereal_LiveMapData *s, cereal_LiveMapData_ptr p) {
	capn_resolve(&p.p);
	s->speedLimitValid = (capn_read8(p.p, 0) & 1) != 0;
	s->speedLimit = capn_to_f32(capn_read32(p.p, 4));
	s->speedAdvisoryValid = (capn_read8(p.p, 0) & 8) != 0;
	s->speedAdvisory = capn_to_f32(capn_read32(p.p, 24));
	s->speedLimitAheadValid = (capn_read8(p.p, 0) & 16) != 0;
	s->speedLimitAhead = capn_to_f32(capn_read32(p.p, 28));
	s->speedLimitAheadDistance = capn_to_f32(capn_read32(p.p, 32));
	s->curvatureValid = (capn_read8(p.p, 0) & 2) != 0;
	s->curvature = capn_to_f32(capn_read32(p.p, 8));
	s->wayId = capn_read64(p.p, 16);
	s->roadX.p = capn_getp(p.p, 0, 0);
	s->roadY.p = capn_getp(p.p, 1, 0);
	s->lastGps.p = capn_getp(p.p, 2, 0);
	s->roadCurvatureX.p = capn_getp(p.p, 3, 0);
	s->roadCurvature.p = capn_getp(p.p, 4, 0);
	s->distToTurn = capn_to_f32(capn_read32(p.p, 12));
	s->mapValid = (capn_read8(p.p, 0) & 4) != 0;
}
void cereal_write_LiveMapData(const struct cereal_LiveMapData *s, cereal_LiveMapData_ptr p) {
	capn_resolve(&p.p);
	capn_write1(p.p, 0, s->speedLimitValid != 0);
	capn_write32(p.p, 4, capn_from_f32(s->speedLimit));
	capn_write1(p.p, 3, s->speedAdvisoryValid != 0);
	capn_write32(p.p, 24, capn_from_f32(s->speedAdvisory));
	capn_write1(p.p, 4, s->speedLimitAheadValid != 0);
	capn_write32(p.p, 28, capn_from_f32(s->speedLimitAhead));
	capn_write32(p.p, 32, capn_from_f32(s->speedLimitAheadDistance));
	capn_write1(p.p, 1, s->curvatureValid != 0);
	capn_write32(p.p, 8, capn_from_f32(s->curvature));
	capn_write64(p.p, 16, s->wayId);
	capn_setp(p.p, 0, s->roadX.p);
	capn_setp(p.p, 1, s->roadY.p);
	capn_setp(p.p, 2, s->lastGps.p);
	capn_setp(p.p, 3, s->roadCurvatureX.p);
	capn_setp(p.p, 4, s->roadCurvature.p);
	capn_write32(p.p, 12, capn_from_f32(s->distToTurn));
	capn_write1(p.p, 2, s->mapValid != 0);
}
void cereal_get_LiveMapData(struct cereal_LiveMapData *s, cereal_LiveMapData_list l, int i) {
	cereal_LiveMapData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_LiveMapData(s, p);
}
void cereal_set_LiveMapData(const struct cereal_LiveMapData *s, cereal_LiveMapData_list l, int i) {
	cereal_LiveMapData_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_LiveMapData(s, p);
}

cereal_CameraOdometry_ptr cereal_new_CameraOdometry(struct capn_segment *s) {
	cereal_CameraOdometry_ptr p;
	p.p = capn_new_struct(s, 16, 4);
	return p;
}
cereal_CameraOdometry_list cereal_new_CameraOdometry_list(struct capn_segment *s, int len) {
	cereal_CameraOdometry_list p;
	p.p = capn_new_list(s, len, 16, 4);
	return p;
}
void cereal_read_CameraOdometry(struct cereal_CameraOdometry *s, cereal_CameraOdometry_ptr p) {
	capn_resolve(&p.p);
	s->frameId = capn_read32(p.p, 0);
	s->timestampEof = capn_read64(p.p, 8);
	s->trans.p = capn_getp(p.p, 0, 0);
	s->rot.p = capn_getp(p.p, 1, 0);
	s->transStd.p = capn_getp(p.p, 2, 0);
	s->rotStd.p = capn_getp(p.p, 3, 0);
}
void cereal_write_CameraOdometry(const struct cereal_CameraOdometry *s, cereal_CameraOdometry_ptr p) {
	capn_resolve(&p.p);
	capn_write32(p.p, 0, s->frameId);
	capn_write64(p.p, 8, s->timestampEof);
	capn_setp(p.p, 0, s->trans.p);
	capn_setp(p.p, 1, s->rot.p);
	capn_setp(p.p, 2, s->transStd.p);
	capn_setp(p.p, 3, s->rotStd.p);
}
void cereal_get_CameraOdometry(struct cereal_CameraOdometry *s, cereal_CameraOdometry_list l, int i) {
	cereal_CameraOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_CameraOdometry(s, p);
}
void cereal_set_CameraOdometry(const struct cereal_CameraOdometry *s, cereal_CameraOdometry_list l, int i) {
	cereal_CameraOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_CameraOdometry(s, p);
}

cereal_KalmanOdometry_ptr cereal_new_KalmanOdometry(struct capn_segment *s) {
	cereal_KalmanOdometry_ptr p;
	p.p = capn_new_struct(s, 0, 4);
	return p;
}
cereal_KalmanOdometry_list cereal_new_KalmanOdometry_list(struct capn_segment *s, int len) {
	cereal_KalmanOdometry_list p;
	p.p = capn_new_list(s, len, 0, 4);
	return p;
}
void cereal_read_KalmanOdometry(struct cereal_KalmanOdometry *s, cereal_KalmanOdometry_ptr p) {
	capn_resolve(&p.p);
	s->trans.p = capn_getp(p.p, 0, 0);
	s->rot.p = capn_getp(p.p, 1, 0);
	s->transStd.p = capn_getp(p.p, 2, 0);
	s->rotStd.p = capn_getp(p.p, 3, 0);
}
void cereal_write_KalmanOdometry(const struct cereal_KalmanOdometry *s, cereal_KalmanOdometry_ptr p) {
	capn_resolve(&p.p);
	capn_setp(p.p, 0, s->trans.p);
	capn_setp(p.p, 1, s->rot.p);
	capn_setp(p.p, 2, s->transStd.p);
	capn_setp(p.p, 3, s->rotStd.p);
}
void cereal_get_KalmanOdometry(struct cereal_KalmanOdometry *s, cereal_KalmanOdometry_list l, int i) {
	cereal_KalmanOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_read_KalmanOdometry(s, p);
}
void cereal_set_KalmanOdometry(const struct cereal_KalmanOdometry *s, cereal_KalmanOdometry_list l, int i) {
	cereal_KalmanOdometry_ptr p;
	p.p = capn_getp(l.p, i, 0);
	cereal_write_KalmanOdometry(s, p);
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
	s->valid = (capn_read8(p.p, 10) & 1) != 1;
	s->which = (enum cereal_Event_which)(int) capn_read16(p.p, 8);
	switch (s->which) {
	case cereal_Event_logMessage:
		s->logMessage = capn_get_text(p.p, 0, capn_val0);
		break;
	case cereal_Event_ubloxRaw:
	case cereal_Event_applanixRaw:
		s->applanixRaw = capn_get_data(p.p, 0);
		break;
	case cereal_Event_initData:
	case cereal_Event_frame:
	case cereal_Event_gpsNMEA:
	case cereal_Event_sensorEventDEPRECATED:
	case cereal_Event_can:
	case cereal_Event_thermal:
	case cereal_Event_controlsState:
	case cereal_Event_liveEventDEPRECATED:
	case cereal_Event_model:
	case cereal_Event_features:
	case cereal_Event_sensorEvents:
	case cereal_Event_health:
	case cereal_Event_radarState:
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
	case cereal_Event_lidarPts:
	case cereal_Event_procLog:
	case cereal_Event_ubloxGnss:
	case cereal_Event_clocks:
	case cereal_Event_liveMpc:
	case cereal_Event_liveLongitudinalMpc:
	case cereal_Event_navStatus:
	case cereal_Event_gpsPlannerPoints:
	case cereal_Event_gpsPlannerPlan:
	case cereal_Event_trafficEvents:
	case cereal_Event_liveLocationTiming:
	case cereal_Event_orbslamCorrectionDEPRECATED:
	case cereal_Event_liveLocationCorrected:
	case cereal_Event_orbObservation:
	case cereal_Event_gpsLocationExternal:
	case cereal_Event_location:
	case cereal_Event_uiNavigationEvent:
	case cereal_Event_liveLocationKalman:
	case cereal_Event_testJoystick:
	case cereal_Event_orbOdometry:
	case cereal_Event_orbFeatures:
	case cereal_Event_applanixLocation:
	case cereal_Event_orbKeyFrame:
	case cereal_Event_uiLayoutState:
	case cereal_Event_orbFeaturesSummary:
	case cereal_Event_driverState:
	case cereal_Event_boot:
	case cereal_Event_liveParameters:
	case cereal_Event_liveMapData:
	case cereal_Event_cameraOdometry:
	case cereal_Event_pathPlan:
	case cereal_Event_kalmanOdometry:
	case cereal_Event_thumbnail:
	case cereal_Event_carEvents:
	case cereal_Event_carParams:
	case cereal_Event_frontFrame:
	case cereal_Event_dMonitoringState:
		s->dMonitoringState.p = capn_getp(p.p, 0, 0);
		break;
	default:
		break;
	}
}
void cereal_write_Event(const struct cereal_Event *s, cereal_Event_ptr p) {
	capn_resolve(&p.p);
	capn_write64(p.p, 0, s->logMonoTime);
	capn_write1(p.p, 80, s->valid != 1);
	capn_write16(p.p, 8, s->which);
	switch (s->which) {
	case cereal_Event_logMessage:
		capn_set_text(p.p, 0, s->logMessage);
		break;
	case cereal_Event_ubloxRaw:
	case cereal_Event_applanixRaw:
		capn_setp(p.p, 0, s->applanixRaw.p);
		break;
	case cereal_Event_initData:
	case cereal_Event_frame:
	case cereal_Event_gpsNMEA:
	case cereal_Event_sensorEventDEPRECATED:
	case cereal_Event_can:
	case cereal_Event_thermal:
	case cereal_Event_controlsState:
	case cereal_Event_liveEventDEPRECATED:
	case cereal_Event_model:
	case cereal_Event_features:
	case cereal_Event_sensorEvents:
	case cereal_Event_health:
	case cereal_Event_radarState:
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
	case cereal_Event_lidarPts:
	case cereal_Event_procLog:
	case cereal_Event_ubloxGnss:
	case cereal_Event_clocks:
	case cereal_Event_liveMpc:
	case cereal_Event_liveLongitudinalMpc:
	case cereal_Event_navStatus:
	case cereal_Event_gpsPlannerPoints:
	case cereal_Event_gpsPlannerPlan:
	case cereal_Event_trafficEvents:
	case cereal_Event_liveLocationTiming:
	case cereal_Event_orbslamCorrectionDEPRECATED:
	case cereal_Event_liveLocationCorrected:
	case cereal_Event_orbObservation:
	case cereal_Event_gpsLocationExternal:
	case cereal_Event_location:
	case cereal_Event_uiNavigationEvent:
	case cereal_Event_liveLocationKalman:
	case cereal_Event_testJoystick:
	case cereal_Event_orbOdometry:
	case cereal_Event_orbFeatures:
	case cereal_Event_applanixLocation:
	case cereal_Event_orbKeyFrame:
	case cereal_Event_uiLayoutState:
	case cereal_Event_orbFeaturesSummary:
	case cereal_Event_driverState:
	case cereal_Event_boot:
	case cereal_Event_liveParameters:
	case cereal_Event_liveMapData:
	case cereal_Event_cameraOdometry:
	case cereal_Event_pathPlan:
	case cereal_Event_kalmanOdometry:
	case cereal_Event_thumbnail:
	case cereal_Event_carEvents:
	case cereal_Event_carParams:
	case cereal_Event_frontFrame:
	case cereal_Event_dMonitoringState:
		capn_setp(p.p, 0, s->dMonitoringState.p);
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
