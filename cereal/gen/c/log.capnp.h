#ifndef CAPN_F3B1F17E25A4285B
#define CAPN_F3B1F17E25A4285B
/* AUTO GENERATED - DO NOT EDIT */
#include <capnp_c.h>

#if CAPN_VERSION != 1
#error "version mismatch between capnp_c.h and generated code"
#endif

#include "c++.capnp.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cereal_InitData;
struct cereal_FrameData;
struct cereal_GPSNMEAData;
struct cereal_SensorEventData;
struct cereal_SensorEventData_SensorVec;
struct cereal_CanData;
struct cereal_ThermalData;
struct cereal_HealthData;
struct cereal_LiveUI;
struct cereal_Live20Data;
struct cereal_Live20Data_LeadData;
struct cereal_LiveCalibrationData;
struct cereal_LiveTracks;
struct cereal_Live100Data;
struct cereal_LiveEventData;
struct cereal_ModelData;
struct cereal_ModelData_PathData;
struct cereal_ModelData_LeadData;
struct cereal_ModelData_ModelSettings;
struct cereal_CalibrationFeatures;
struct cereal_EncodeIndex;
struct cereal_AndroidLogEntry;
struct cereal_LogRotate;
struct cereal_Event;

typedef struct {capn_ptr p;} cereal_InitData_ptr;
typedef struct {capn_ptr p;} cereal_FrameData_ptr;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_ptr;
typedef struct {capn_ptr p;} cereal_CanData_ptr;
typedef struct {capn_ptr p;} cereal_ThermalData_ptr;
typedef struct {capn_ptr p;} cereal_HealthData_ptr;
typedef struct {capn_ptr p;} cereal_LiveUI_ptr;
typedef struct {capn_ptr p;} cereal_Live20Data_ptr;
typedef struct {capn_ptr p;} cereal_Live20Data_LeadData_ptr;
typedef struct {capn_ptr p;} cereal_LiveCalibrationData_ptr;
typedef struct {capn_ptr p;} cereal_LiveTracks_ptr;
typedef struct {capn_ptr p;} cereal_Live100Data_ptr;
typedef struct {capn_ptr p;} cereal_LiveEventData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_PathData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_LeadData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_ModelSettings_ptr;
typedef struct {capn_ptr p;} cereal_CalibrationFeatures_ptr;
typedef struct {capn_ptr p;} cereal_EncodeIndex_ptr;
typedef struct {capn_ptr p;} cereal_AndroidLogEntry_ptr;
typedef struct {capn_ptr p;} cereal_LogRotate_ptr;
typedef struct {capn_ptr p;} cereal_Event_ptr;

typedef struct {capn_ptr p;} cereal_InitData_list;
typedef struct {capn_ptr p;} cereal_FrameData_list;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_list;
typedef struct {capn_ptr p;} cereal_CanData_list;
typedef struct {capn_ptr p;} cereal_ThermalData_list;
typedef struct {capn_ptr p;} cereal_HealthData_list;
typedef struct {capn_ptr p;} cereal_LiveUI_list;
typedef struct {capn_ptr p;} cereal_Live20Data_list;
typedef struct {capn_ptr p;} cereal_Live20Data_LeadData_list;
typedef struct {capn_ptr p;} cereal_LiveCalibrationData_list;
typedef struct {capn_ptr p;} cereal_LiveTracks_list;
typedef struct {capn_ptr p;} cereal_Live100Data_list;
typedef struct {capn_ptr p;} cereal_LiveEventData_list;
typedef struct {capn_ptr p;} cereal_ModelData_list;
typedef struct {capn_ptr p;} cereal_ModelData_PathData_list;
typedef struct {capn_ptr p;} cereal_ModelData_LeadData_list;
typedef struct {capn_ptr p;} cereal_ModelData_ModelSettings_list;
typedef struct {capn_ptr p;} cereal_CalibrationFeatures_list;
typedef struct {capn_ptr p;} cereal_EncodeIndex_list;
typedef struct {capn_ptr p;} cereal_AndroidLogEntry_list;
typedef struct {capn_ptr p;} cereal_LogRotate_list;
typedef struct {capn_ptr p;} cereal_Event_list;

enum cereal_EncodeIndex_Type {
	cereal_EncodeIndex_Type_bigBoxLossless = 0,
	cereal_EncodeIndex_Type_fullHEVC = 1,
	cereal_EncodeIndex_Type_bigBoxHEVC = 2
};
extern int32_t cereal_logVersion;

struct cereal_InitData {
	capn_ptr kernelArgs;
	capn_text gctx;
	capn_text dongleId;
};

static const size_t cereal_InitData_word_count = 0;

static const size_t cereal_InitData_pointer_count = 3;

static const size_t cereal_InitData_struct_bytes_count = 24;

struct cereal_FrameData {
	uint32_t frameId;
	uint32_t encodeId;
	uint64_t timestampEof;
	int32_t frameLength;
	int32_t integLines;
	int32_t globalGain;
	capn_data image;
};

static const size_t cereal_FrameData_word_count = 4;

static const size_t cereal_FrameData_pointer_count = 1;

static const size_t cereal_FrameData_struct_bytes_count = 40;

struct cereal_GPSNMEAData {
	int64_t timestamp;
	uint64_t localWallTime;
	capn_text nmea;
};

static const size_t cereal_GPSNMEAData_word_count = 2;

static const size_t cereal_GPSNMEAData_pointer_count = 1;

static const size_t cereal_GPSNMEAData_struct_bytes_count = 24;
enum cereal_SensorEventData_which {
	cereal_SensorEventData_acceleration = 0,
	cereal_SensorEventData_magnetic = 1,
	cereal_SensorEventData_orientation = 2,
	cereal_SensorEventData_gyro = 3
};

struct cereal_SensorEventData {
	int32_t version;
	int32_t sensor;
	int32_t type;
	int64_t timestamp;
	enum cereal_SensorEventData_which which;
	union {
		cereal_SensorEventData_SensorVec_ptr acceleration;
		cereal_SensorEventData_SensorVec_ptr magnetic;
		cereal_SensorEventData_SensorVec_ptr orientation;
		cereal_SensorEventData_SensorVec_ptr gyro;
	};
};

static const size_t cereal_SensorEventData_word_count = 3;

static const size_t cereal_SensorEventData_pointer_count = 1;

static const size_t cereal_SensorEventData_struct_bytes_count = 32;

struct cereal_SensorEventData_SensorVec {
	capn_list32 v;
	int8_t status;
};

static const size_t cereal_SensorEventData_SensorVec_word_count = 1;

static const size_t cereal_SensorEventData_SensorVec_pointer_count = 1;

static const size_t cereal_SensorEventData_SensorVec_struct_bytes_count = 16;

struct cereal_CanData {
	uint32_t address;
	uint16_t busTime;
	capn_data dat;
	int8_t src;
};

static const size_t cereal_CanData_word_count = 1;

static const size_t cereal_CanData_pointer_count = 1;

static const size_t cereal_CanData_struct_bytes_count = 16;

struct cereal_ThermalData {
	uint16_t cpu0;
	uint16_t cpu1;
	uint16_t cpu2;
	uint16_t cpu3;
	uint16_t mem;
	uint16_t gpu;
	uint32_t bat;
};

static const size_t cereal_ThermalData_word_count = 2;

static const size_t cereal_ThermalData_pointer_count = 0;

static const size_t cereal_ThermalData_struct_bytes_count = 16;

struct cereal_HealthData {
	uint32_t voltage;
	uint32_t current;
	unsigned started : 1;
	unsigned controlsAllowed : 1;
	unsigned gasInterceptorDetected : 1;
};

static const size_t cereal_HealthData_word_count = 2;

static const size_t cereal_HealthData_pointer_count = 0;

static const size_t cereal_HealthData_struct_bytes_count = 16;

struct cereal_LiveUI {
	unsigned rearViewCam : 1;
	capn_text alertText1;
	capn_text alertText2;
	float awarenessStatus;
};

static const size_t cereal_LiveUI_word_count = 1;

static const size_t cereal_LiveUI_pointer_count = 2;

static const size_t cereal_LiveUI_struct_bytes_count = 24;

struct cereal_Live20Data {
	capn_list64 canMonoTimes;
	uint64_t mdMonoTime;
	uint64_t ftMonoTime;
	capn_list32 warpMatrix;
	float angleOffset;
	int8_t calStatus;
	int32_t calCycle;
	int8_t calPerc;
	cereal_Live20Data_LeadData_ptr leadOne;
	cereal_Live20Data_LeadData_ptr leadTwo;
	float cumLagMs;
};

static const size_t cereal_Live20Data_word_count = 4;

static const size_t cereal_Live20Data_pointer_count = 4;

static const size_t cereal_Live20Data_struct_bytes_count = 64;

struct cereal_Live20Data_LeadData {
	float dRel;
	float yRel;
	float vRel;
	float aRel;
	float vLead;
	float aLead;
	float dPath;
	float vLat;
	float vLeadK;
	float aLeadK;
	unsigned fcw : 1;
	unsigned status : 1;
};

static const size_t cereal_Live20Data_LeadData_word_count = 6;

static const size_t cereal_Live20Data_LeadData_pointer_count = 0;

static const size_t cereal_Live20Data_LeadData_struct_bytes_count = 48;

struct cereal_LiveCalibrationData {
	capn_list32 warpMatrix;
	int8_t calStatus;
	int32_t calCycle;
	int8_t calPerc;
};

static const size_t cereal_LiveCalibrationData_word_count = 1;

static const size_t cereal_LiveCalibrationData_pointer_count = 1;

static const size_t cereal_LiveCalibrationData_struct_bytes_count = 16;

struct cereal_LiveTracks {
	int32_t trackId;
	float dRel;
	float yRel;
	float vRel;
	float aRel;
	float timeStamp;
	float status;
	float currentTime;
	unsigned stationary : 1;
	unsigned oncoming : 1;
};

static const size_t cereal_LiveTracks_word_count = 5;

static const size_t cereal_LiveTracks_pointer_count = 0;

static const size_t cereal_LiveTracks_struct_bytes_count = 40;

struct cereal_Live100Data {
	uint64_t canMonoTime;
	capn_list64 canMonoTimes;
	uint64_t l20MonoTime;
	uint64_t mdMonoTime;
	float vEgo;
	float aEgo;
	float vPid;
	float vTargetLead;
	float upAccelCmd;
	float uiAccelCmd;
	float yActual;
	float yDes;
	float upSteer;
	float uiSteer;
	float aTargetMin;
	float aTargetMax;
	float jerkFactor;
	float angleSteers;
	int32_t hudLead;
	float cumLagMs;
	unsigned enabled : 1;
	unsigned steerOverride : 1;
	float vCruise;
	unsigned rearViewCam : 1;
	capn_text alertText1;
	capn_text alertText2;
	float awarenessStatus;
};

static const size_t cereal_Live100Data_word_count = 13;

static const size_t cereal_Live100Data_pointer_count = 3;

static const size_t cereal_Live100Data_struct_bytes_count = 128;

struct cereal_LiveEventData {
	capn_text name;
	int32_t value;
};

static const size_t cereal_LiveEventData_word_count = 1;

static const size_t cereal_LiveEventData_pointer_count = 1;

static const size_t cereal_LiveEventData_struct_bytes_count = 16;

struct cereal_ModelData {
	uint32_t frameId;
	cereal_ModelData_PathData_ptr path;
	cereal_ModelData_PathData_ptr leftLane;
	cereal_ModelData_PathData_ptr rightLane;
	cereal_ModelData_LeadData_ptr lead;
	cereal_ModelData_ModelSettings_ptr settings;
};

static const size_t cereal_ModelData_word_count = 1;

static const size_t cereal_ModelData_pointer_count = 5;

static const size_t cereal_ModelData_struct_bytes_count = 48;

struct cereal_ModelData_PathData {
	capn_list32 points;
	float prob;
	float std;
};

static const size_t cereal_ModelData_PathData_word_count = 1;

static const size_t cereal_ModelData_PathData_pointer_count = 1;

static const size_t cereal_ModelData_PathData_struct_bytes_count = 16;

struct cereal_ModelData_LeadData {
	float dist;
	float prob;
	float std;
};

static const size_t cereal_ModelData_LeadData_word_count = 2;

static const size_t cereal_ModelData_LeadData_pointer_count = 0;

static const size_t cereal_ModelData_LeadData_struct_bytes_count = 16;

struct cereal_ModelData_ModelSettings {
	uint16_t bigBoxX;
	uint16_t bigBoxY;
	uint16_t bigBoxWidth;
	uint16_t bigBoxHeight;
	capn_list32 boxProjection;
	capn_list32 yuvCorrection;
};

static const size_t cereal_ModelData_ModelSettings_word_count = 1;

static const size_t cereal_ModelData_ModelSettings_pointer_count = 2;

static const size_t cereal_ModelData_ModelSettings_struct_bytes_count = 24;

struct cereal_CalibrationFeatures {
	uint32_t frameId;
	capn_list32 p0;
	capn_list32 p1;
	capn_list8 status;
};

static const size_t cereal_CalibrationFeatures_word_count = 1;

static const size_t cereal_CalibrationFeatures_pointer_count = 3;

static const size_t cereal_CalibrationFeatures_struct_bytes_count = 32;

struct cereal_EncodeIndex {
	uint32_t frameId;
	enum cereal_EncodeIndex_Type type;
	uint32_t encodeId;
	int32_t segmentNum;
	uint32_t segmentId;
};

static const size_t cereal_EncodeIndex_word_count = 3;

static const size_t cereal_EncodeIndex_pointer_count = 0;

static const size_t cereal_EncodeIndex_struct_bytes_count = 24;

struct cereal_AndroidLogEntry {
	uint8_t id;
	uint64_t ts;
	uint8_t priority;
	int32_t pid;
	int32_t tid;
	capn_text tag;
	capn_text message;
};

static const size_t cereal_AndroidLogEntry_word_count = 3;

static const size_t cereal_AndroidLogEntry_pointer_count = 2;

static const size_t cereal_AndroidLogEntry_struct_bytes_count = 40;

struct cereal_LogRotate {
	int32_t segmentNum;
	capn_text path;
};

static const size_t cereal_LogRotate_word_count = 1;

static const size_t cereal_LogRotate_pointer_count = 1;

static const size_t cereal_LogRotate_struct_bytes_count = 16;
enum cereal_Event_which {
	cereal_Event_initData = 0,
	cereal_Event_frame = 1,
	cereal_Event_gpsNMEA = 2,
	cereal_Event_sensorEventDEPRECATED = 3,
	cereal_Event_can = 4,
	cereal_Event_thermal = 5,
	cereal_Event_live100 = 6,
	cereal_Event_liveEventDEPRECATED = 7,
	cereal_Event_model = 8,
	cereal_Event_features = 9,
	cereal_Event_sensorEvents = 10,
	cereal_Event_health = 11,
	cereal_Event_live20 = 12,
	cereal_Event_liveUIDEPRECATED = 13,
	cereal_Event_encodeIdx = 14,
	cereal_Event_liveTracks = 15,
	cereal_Event_sendcan = 16,
	cereal_Event_logMessage = 17,
	cereal_Event_liveCalibration = 18,
	cereal_Event_androidLogEntry = 19
};

struct cereal_Event {
	uint64_t logMonoTime;
	enum cereal_Event_which which;
	union {
		cereal_InitData_ptr initData;
		cereal_FrameData_ptr frame;
		cereal_GPSNMEAData_ptr gpsNMEA;
		cereal_SensorEventData_ptr sensorEventDEPRECATED;
		cereal_CanData_list can;
		cereal_ThermalData_ptr thermal;
		cereal_Live100Data_ptr live100;
		cereal_LiveEventData_list liveEventDEPRECATED;
		cereal_ModelData_ptr model;
		cereal_CalibrationFeatures_ptr features;
		cereal_SensorEventData_list sensorEvents;
		cereal_HealthData_ptr health;
		cereal_Live20Data_ptr live20;
		cereal_LiveUI_ptr liveUIDEPRECATED;
		cereal_EncodeIndex_ptr encodeIdx;
		cereal_LiveTracks_list liveTracks;
		cereal_CanData_list sendcan;
		capn_text logMessage;
		cereal_LiveCalibrationData_ptr liveCalibration;
		cereal_AndroidLogEntry_ptr androidLogEntry;
	};
};

static const size_t cereal_Event_word_count = 2;

static const size_t cereal_Event_pointer_count = 1;

static const size_t cereal_Event_struct_bytes_count = 24;

cereal_InitData_ptr cereal_new_InitData(struct capn_segment*);
cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment*);
cereal_GPSNMEAData_ptr cereal_new_GPSNMEAData(struct capn_segment*);
cereal_SensorEventData_ptr cereal_new_SensorEventData(struct capn_segment*);
cereal_SensorEventData_SensorVec_ptr cereal_new_SensorEventData_SensorVec(struct capn_segment*);
cereal_CanData_ptr cereal_new_CanData(struct capn_segment*);
cereal_ThermalData_ptr cereal_new_ThermalData(struct capn_segment*);
cereal_HealthData_ptr cereal_new_HealthData(struct capn_segment*);
cereal_LiveUI_ptr cereal_new_LiveUI(struct capn_segment*);
cereal_Live20Data_ptr cereal_new_Live20Data(struct capn_segment*);
cereal_Live20Data_LeadData_ptr cereal_new_Live20Data_LeadData(struct capn_segment*);
cereal_LiveCalibrationData_ptr cereal_new_LiveCalibrationData(struct capn_segment*);
cereal_LiveTracks_ptr cereal_new_LiveTracks(struct capn_segment*);
cereal_Live100Data_ptr cereal_new_Live100Data(struct capn_segment*);
cereal_LiveEventData_ptr cereal_new_LiveEventData(struct capn_segment*);
cereal_ModelData_ptr cereal_new_ModelData(struct capn_segment*);
cereal_ModelData_PathData_ptr cereal_new_ModelData_PathData(struct capn_segment*);
cereal_ModelData_LeadData_ptr cereal_new_ModelData_LeadData(struct capn_segment*);
cereal_ModelData_ModelSettings_ptr cereal_new_ModelData_ModelSettings(struct capn_segment*);
cereal_CalibrationFeatures_ptr cereal_new_CalibrationFeatures(struct capn_segment*);
cereal_EncodeIndex_ptr cereal_new_EncodeIndex(struct capn_segment*);
cereal_AndroidLogEntry_ptr cereal_new_AndroidLogEntry(struct capn_segment*);
cereal_LogRotate_ptr cereal_new_LogRotate(struct capn_segment*);
cereal_Event_ptr cereal_new_Event(struct capn_segment*);

cereal_InitData_list cereal_new_InitData_list(struct capn_segment*, int len);
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment*, int len);
cereal_GPSNMEAData_list cereal_new_GPSNMEAData_list(struct capn_segment*, int len);
cereal_SensorEventData_list cereal_new_SensorEventData_list(struct capn_segment*, int len);
cereal_SensorEventData_SensorVec_list cereal_new_SensorEventData_SensorVec_list(struct capn_segment*, int len);
cereal_CanData_list cereal_new_CanData_list(struct capn_segment*, int len);
cereal_ThermalData_list cereal_new_ThermalData_list(struct capn_segment*, int len);
cereal_HealthData_list cereal_new_HealthData_list(struct capn_segment*, int len);
cereal_LiveUI_list cereal_new_LiveUI_list(struct capn_segment*, int len);
cereal_Live20Data_list cereal_new_Live20Data_list(struct capn_segment*, int len);
cereal_Live20Data_LeadData_list cereal_new_Live20Data_LeadData_list(struct capn_segment*, int len);
cereal_LiveCalibrationData_list cereal_new_LiveCalibrationData_list(struct capn_segment*, int len);
cereal_LiveTracks_list cereal_new_LiveTracks_list(struct capn_segment*, int len);
cereal_Live100Data_list cereal_new_Live100Data_list(struct capn_segment*, int len);
cereal_LiveEventData_list cereal_new_LiveEventData_list(struct capn_segment*, int len);
cereal_ModelData_list cereal_new_ModelData_list(struct capn_segment*, int len);
cereal_ModelData_PathData_list cereal_new_ModelData_PathData_list(struct capn_segment*, int len);
cereal_ModelData_LeadData_list cereal_new_ModelData_LeadData_list(struct capn_segment*, int len);
cereal_ModelData_ModelSettings_list cereal_new_ModelData_ModelSettings_list(struct capn_segment*, int len);
cereal_CalibrationFeatures_list cereal_new_CalibrationFeatures_list(struct capn_segment*, int len);
cereal_EncodeIndex_list cereal_new_EncodeIndex_list(struct capn_segment*, int len);
cereal_AndroidLogEntry_list cereal_new_AndroidLogEntry_list(struct capn_segment*, int len);
cereal_LogRotate_list cereal_new_LogRotate_list(struct capn_segment*, int len);
cereal_Event_list cereal_new_Event_list(struct capn_segment*, int len);

void cereal_read_InitData(struct cereal_InitData*, cereal_InitData_ptr);
void cereal_read_FrameData(struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_read_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_read_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_read_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_read_CanData(struct cereal_CanData*, cereal_CanData_ptr);
void cereal_read_ThermalData(struct cereal_ThermalData*, cereal_ThermalData_ptr);
void cereal_read_HealthData(struct cereal_HealthData*, cereal_HealthData_ptr);
void cereal_read_LiveUI(struct cereal_LiveUI*, cereal_LiveUI_ptr);
void cereal_read_Live20Data(struct cereal_Live20Data*, cereal_Live20Data_ptr);
void cereal_read_Live20Data_LeadData(struct cereal_Live20Data_LeadData*, cereal_Live20Data_LeadData_ptr);
void cereal_read_LiveCalibrationData(struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_ptr);
void cereal_read_LiveTracks(struct cereal_LiveTracks*, cereal_LiveTracks_ptr);
void cereal_read_Live100Data(struct cereal_Live100Data*, cereal_Live100Data_ptr);
void cereal_read_LiveEventData(struct cereal_LiveEventData*, cereal_LiveEventData_ptr);
void cereal_read_ModelData(struct cereal_ModelData*, cereal_ModelData_ptr);
void cereal_read_ModelData_PathData(struct cereal_ModelData_PathData*, cereal_ModelData_PathData_ptr);
void cereal_read_ModelData_LeadData(struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_ptr);
void cereal_read_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_ptr);
void cereal_read_CalibrationFeatures(struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_ptr);
void cereal_read_EncodeIndex(struct cereal_EncodeIndex*, cereal_EncodeIndex_ptr);
void cereal_read_AndroidLogEntry(struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_ptr);
void cereal_read_LogRotate(struct cereal_LogRotate*, cereal_LogRotate_ptr);
void cereal_read_Event(struct cereal_Event*, cereal_Event_ptr);

void cereal_write_InitData(const struct cereal_InitData*, cereal_InitData_ptr);
void cereal_write_FrameData(const struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_write_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_write_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_write_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_write_CanData(const struct cereal_CanData*, cereal_CanData_ptr);
void cereal_write_ThermalData(const struct cereal_ThermalData*, cereal_ThermalData_ptr);
void cereal_write_HealthData(const struct cereal_HealthData*, cereal_HealthData_ptr);
void cereal_write_LiveUI(const struct cereal_LiveUI*, cereal_LiveUI_ptr);
void cereal_write_Live20Data(const struct cereal_Live20Data*, cereal_Live20Data_ptr);
void cereal_write_Live20Data_LeadData(const struct cereal_Live20Data_LeadData*, cereal_Live20Data_LeadData_ptr);
void cereal_write_LiveCalibrationData(const struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_ptr);
void cereal_write_LiveTracks(const struct cereal_LiveTracks*, cereal_LiveTracks_ptr);
void cereal_write_Live100Data(const struct cereal_Live100Data*, cereal_Live100Data_ptr);
void cereal_write_LiveEventData(const struct cereal_LiveEventData*, cereal_LiveEventData_ptr);
void cereal_write_ModelData(const struct cereal_ModelData*, cereal_ModelData_ptr);
void cereal_write_ModelData_PathData(const struct cereal_ModelData_PathData*, cereal_ModelData_PathData_ptr);
void cereal_write_ModelData_LeadData(const struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_ptr);
void cereal_write_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_ptr);
void cereal_write_CalibrationFeatures(const struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_ptr);
void cereal_write_EncodeIndex(const struct cereal_EncodeIndex*, cereal_EncodeIndex_ptr);
void cereal_write_AndroidLogEntry(const struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_ptr);
void cereal_write_LogRotate(const struct cereal_LogRotate*, cereal_LogRotate_ptr);
void cereal_write_Event(const struct cereal_Event*, cereal_Event_ptr);

void cereal_get_InitData(struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_get_FrameData(struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_get_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_get_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_get_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_get_CanData(struct cereal_CanData*, cereal_CanData_list, int i);
void cereal_get_ThermalData(struct cereal_ThermalData*, cereal_ThermalData_list, int i);
void cereal_get_HealthData(struct cereal_HealthData*, cereal_HealthData_list, int i);
void cereal_get_LiveUI(struct cereal_LiveUI*, cereal_LiveUI_list, int i);
void cereal_get_Live20Data(struct cereal_Live20Data*, cereal_Live20Data_list, int i);
void cereal_get_Live20Data_LeadData(struct cereal_Live20Data_LeadData*, cereal_Live20Data_LeadData_list, int i);
void cereal_get_LiveCalibrationData(struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_list, int i);
void cereal_get_LiveTracks(struct cereal_LiveTracks*, cereal_LiveTracks_list, int i);
void cereal_get_Live100Data(struct cereal_Live100Data*, cereal_Live100Data_list, int i);
void cereal_get_LiveEventData(struct cereal_LiveEventData*, cereal_LiveEventData_list, int i);
void cereal_get_ModelData(struct cereal_ModelData*, cereal_ModelData_list, int i);
void cereal_get_ModelData_PathData(struct cereal_ModelData_PathData*, cereal_ModelData_PathData_list, int i);
void cereal_get_ModelData_LeadData(struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_list, int i);
void cereal_get_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_list, int i);
void cereal_get_CalibrationFeatures(struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_list, int i);
void cereal_get_EncodeIndex(struct cereal_EncodeIndex*, cereal_EncodeIndex_list, int i);
void cereal_get_AndroidLogEntry(struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_list, int i);
void cereal_get_LogRotate(struct cereal_LogRotate*, cereal_LogRotate_list, int i);
void cereal_get_Event(struct cereal_Event*, cereal_Event_list, int i);

void cereal_set_InitData(const struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_set_FrameData(const struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_set_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_set_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_set_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_set_CanData(const struct cereal_CanData*, cereal_CanData_list, int i);
void cereal_set_ThermalData(const struct cereal_ThermalData*, cereal_ThermalData_list, int i);
void cereal_set_HealthData(const struct cereal_HealthData*, cereal_HealthData_list, int i);
void cereal_set_LiveUI(const struct cereal_LiveUI*, cereal_LiveUI_list, int i);
void cereal_set_Live20Data(const struct cereal_Live20Data*, cereal_Live20Data_list, int i);
void cereal_set_Live20Data_LeadData(const struct cereal_Live20Data_LeadData*, cereal_Live20Data_LeadData_list, int i);
void cereal_set_LiveCalibrationData(const struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_list, int i);
void cereal_set_LiveTracks(const struct cereal_LiveTracks*, cereal_LiveTracks_list, int i);
void cereal_set_Live100Data(const struct cereal_Live100Data*, cereal_Live100Data_list, int i);
void cereal_set_LiveEventData(const struct cereal_LiveEventData*, cereal_LiveEventData_list, int i);
void cereal_set_ModelData(const struct cereal_ModelData*, cereal_ModelData_list, int i);
void cereal_set_ModelData_PathData(const struct cereal_ModelData_PathData*, cereal_ModelData_PathData_list, int i);
void cereal_set_ModelData_LeadData(const struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_list, int i);
void cereal_set_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_list, int i);
void cereal_set_CalibrationFeatures(const struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_list, int i);
void cereal_set_EncodeIndex(const struct cereal_EncodeIndex*, cereal_EncodeIndex_list, int i);
void cereal_set_AndroidLogEntry(const struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_list, int i);
void cereal_set_LogRotate(const struct cereal_LogRotate*, cereal_LogRotate_list, int i);
void cereal_set_Event(const struct cereal_Event*, cereal_Event_list, int i);

#ifdef __cplusplus
}
#endif
#endif
