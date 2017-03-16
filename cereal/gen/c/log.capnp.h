#ifndef CAPN_F3B1F17E25A4285B
#define CAPN_F3B1F17E25A4285B
/* AUTO GENERATED - DO NOT EDIT */
#include <capnp_c.h>

#if CAPN_VERSION != 1
#error "version mismatch between capnp_c.h and generated code"
#endif

#include "c++.capnp.h"
#include "car.capnp.h"
#include "java.capnp.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cereal_Map;
struct cereal_Map_Entry;
struct cereal_InitData;
struct cereal_InitData_AndroidBuildInfo;
struct cereal_InitData_AndroidSensor;
struct cereal_InitData_ChffrAndroidExtra;
struct cereal_FrameData;
struct cereal_FrameData_AndroidCaptureResult;
struct cereal_GPSNMEAData;
struct cereal_SensorEventData;
struct cereal_SensorEventData_SensorVec;
struct cereal_GpsLocationData;
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
struct cereal_Plan;
struct cereal_LiveLocationData;
struct cereal_LiveLocationData_Accuracy;
struct cereal_EthernetPacket;
struct cereal_NavUpdate;
struct cereal_NavUpdate_LatLng;
struct cereal_NavUpdate_Segment;
struct cereal_CellInfo;
struct cereal_WifiScan;
struct cereal_AndroidGnss;
struct cereal_AndroidGnss_Measurements;
struct cereal_AndroidGnss_Measurements_Clock;
struct cereal_AndroidGnss_Measurements_Measurement;
struct cereal_AndroidGnss_NavigationMessage;
struct cereal_QcomGnss;
struct cereal_QcomGnss_MeasurementReport;
struct cereal_QcomGnss_MeasurementReport_SV;
struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus;
struct cereal_QcomGnss_ClockReport;
struct cereal_Event;

typedef struct {capn_ptr p;} cereal_Map_ptr;
typedef struct {capn_ptr p;} cereal_Map_Entry_ptr;
typedef struct {capn_ptr p;} cereal_InitData_ptr;
typedef struct {capn_ptr p;} cereal_InitData_AndroidBuildInfo_ptr;
typedef struct {capn_ptr p;} cereal_InitData_AndroidSensor_ptr;
typedef struct {capn_ptr p;} cereal_InitData_ChffrAndroidExtra_ptr;
typedef struct {capn_ptr p;} cereal_FrameData_ptr;
typedef struct {capn_ptr p;} cereal_FrameData_AndroidCaptureResult_ptr;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_ptr;
typedef struct {capn_ptr p;} cereal_GpsLocationData_ptr;
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
typedef struct {capn_ptr p;} cereal_Plan_ptr;
typedef struct {capn_ptr p;} cereal_LiveLocationData_ptr;
typedef struct {capn_ptr p;} cereal_LiveLocationData_Accuracy_ptr;
typedef struct {capn_ptr p;} cereal_EthernetPacket_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_LatLng_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_Segment_ptr;
typedef struct {capn_ptr p;} cereal_CellInfo_ptr;
typedef struct {capn_ptr p;} cereal_WifiScan_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Clock_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Measurement_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_NavigationMessage_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_ClockReport_ptr;
typedef struct {capn_ptr p;} cereal_Event_ptr;

typedef struct {capn_ptr p;} cereal_Map_list;
typedef struct {capn_ptr p;} cereal_Map_Entry_list;
typedef struct {capn_ptr p;} cereal_InitData_list;
typedef struct {capn_ptr p;} cereal_InitData_AndroidBuildInfo_list;
typedef struct {capn_ptr p;} cereal_InitData_AndroidSensor_list;
typedef struct {capn_ptr p;} cereal_InitData_ChffrAndroidExtra_list;
typedef struct {capn_ptr p;} cereal_FrameData_list;
typedef struct {capn_ptr p;} cereal_FrameData_AndroidCaptureResult_list;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_list;
typedef struct {capn_ptr p;} cereal_GpsLocationData_list;
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
typedef struct {capn_ptr p;} cereal_Plan_list;
typedef struct {capn_ptr p;} cereal_LiveLocationData_list;
typedef struct {capn_ptr p;} cereal_LiveLocationData_Accuracy_list;
typedef struct {capn_ptr p;} cereal_EthernetPacket_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_LatLng_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_Segment_list;
typedef struct {capn_ptr p;} cereal_CellInfo_list;
typedef struct {capn_ptr p;} cereal_WifiScan_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Clock_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Measurement_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_NavigationMessage_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_ClockReport_list;
typedef struct {capn_ptr p;} cereal_Event_list;

enum cereal_InitData_DeviceType {
	cereal_InitData_DeviceType_unknown = 0,
	cereal_InitData_DeviceType_neo = 1,
	cereal_InitData_DeviceType_chffrAndroid = 2
};

enum cereal_FrameData_FrameType {
	cereal_FrameData_FrameType_unknown = 0,
	cereal_FrameData_FrameType_neo = 1,
	cereal_FrameData_FrameType_chffrAndroid = 2
};

enum cereal_SensorEventData_SensorSource {
	cereal_SensorEventData_SensorSource_android = 0,
	cereal_SensorEventData_SensorSource_iOS = 1,
	cereal_SensorEventData_SensorSource_fiber = 2,
	cereal_SensorEventData_SensorSource_velodyne = 3
};

enum cereal_GpsLocationData_SensorSource {
	cereal_GpsLocationData_SensorSource_android = 0,
	cereal_GpsLocationData_SensorSource_iOS = 1,
	cereal_GpsLocationData_SensorSource_car = 2,
	cereal_GpsLocationData_SensorSource_velodyne = 3,
	cereal_GpsLocationData_SensorSource_fusion = 4,
	cereal_GpsLocationData_SensorSource_external = 5
};

enum cereal_EncodeIndex_Type {
	cereal_EncodeIndex_Type_bigBoxLossless = 0,
	cereal_EncodeIndex_Type_fullHEVC = 1,
	cereal_EncodeIndex_Type_bigBoxHEVC = 2,
	cereal_EncodeIndex_Type_chffrAndroidH264 = 3
};

enum cereal_NavUpdate_Segment_Instruction {
	cereal_NavUpdate_Segment_Instruction_turnLeft = 0,
	cereal_NavUpdate_Segment_Instruction_turnRight = 1,
	cereal_NavUpdate_Segment_Instruction_keepLeft = 2,
	cereal_NavUpdate_Segment_Instruction_keepRight = 3,
	cereal_NavUpdate_Segment_Instruction_straight = 4,
	cereal_NavUpdate_Segment_Instruction_roundaboutExitNumber = 5,
	cereal_NavUpdate_Segment_Instruction_roundaboutExit = 6,
	cereal_NavUpdate_Segment_Instruction_roundaboutTurnLeft = 7,
	cereal_NavUpdate_Segment_Instruction_unkn8 = 8,
	cereal_NavUpdate_Segment_Instruction_roundaboutStraight = 9,
	cereal_NavUpdate_Segment_Instruction_unkn10 = 10,
	cereal_NavUpdate_Segment_Instruction_roundaboutTurnRight = 11,
	cereal_NavUpdate_Segment_Instruction_unkn12 = 12,
	cereal_NavUpdate_Segment_Instruction_roundaboutUturn = 13,
	cereal_NavUpdate_Segment_Instruction_unkn14 = 14,
	cereal_NavUpdate_Segment_Instruction_arrive = 15,
	cereal_NavUpdate_Segment_Instruction_exitLeft = 16,
	cereal_NavUpdate_Segment_Instruction_exitRight = 17,
	cereal_NavUpdate_Segment_Instruction_unkn18 = 18,
	cereal_NavUpdate_Segment_Instruction_uturn = 19
};

enum cereal_WifiScan_ChannelWidth {
	cereal_WifiScan_ChannelWidth_w20Mhz = 0,
	cereal_WifiScan_ChannelWidth_w40Mhz = 1,
	cereal_WifiScan_ChannelWidth_w80Mhz = 2,
	cereal_WifiScan_ChannelWidth_w160Mhz = 3,
	cereal_WifiScan_ChannelWidth_w80Plus80Mhz = 4
};

enum cereal_AndroidGnss_Measurements_Measurement_Constellation {
	cereal_AndroidGnss_Measurements_Measurement_Constellation_unknown = 0,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_gps = 1,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_sbas = 2,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_glonass = 3,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_qzss = 4,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_beidou = 5,
	cereal_AndroidGnss_Measurements_Measurement_Constellation_galileo = 6
};

enum cereal_AndroidGnss_Measurements_Measurement_State {
	cereal_AndroidGnss_Measurements_Measurement_State_unknown = 0,
	cereal_AndroidGnss_Measurements_Measurement_State_codeLock = 1,
	cereal_AndroidGnss_Measurements_Measurement_State_bitSync = 2,
	cereal_AndroidGnss_Measurements_Measurement_State_subframeSync = 3,
	cereal_AndroidGnss_Measurements_Measurement_State_towDecoded = 4,
	cereal_AndroidGnss_Measurements_Measurement_State_msecAmbiguous = 5,
	cereal_AndroidGnss_Measurements_Measurement_State_symbolSync = 6,
	cereal_AndroidGnss_Measurements_Measurement_State_gloStringSync = 7,
	cereal_AndroidGnss_Measurements_Measurement_State_gloTodDecoded = 8,
	cereal_AndroidGnss_Measurements_Measurement_State_bdsD2BitSync = 9,
	cereal_AndroidGnss_Measurements_Measurement_State_bdsD2SubframeSync = 10,
	cereal_AndroidGnss_Measurements_Measurement_State_galE1bcCodeLock = 11,
	cereal_AndroidGnss_Measurements_Measurement_State_galE1c2ndCodeLock = 12,
	cereal_AndroidGnss_Measurements_Measurement_State_galE1bPageSync = 13,
	cereal_AndroidGnss_Measurements_Measurement_State_sbasSync = 14
};

enum cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator {
	cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator_unknown = 0,
	cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator_detected = 1,
	cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator_notDetected = 2
};

enum cereal_AndroidGnss_NavigationMessage_Status {
	cereal_AndroidGnss_NavigationMessage_Status_unknown = 0,
	cereal_AndroidGnss_NavigationMessage_Status_parityPassed = 1,
	cereal_AndroidGnss_NavigationMessage_Status_parityRebuilt = 2
};

enum cereal_QcomGnss_MeasurementReport_Source {
	cereal_QcomGnss_MeasurementReport_Source_gps = 0,
	cereal_QcomGnss_MeasurementReport_Source_glonass = 1
};

enum cereal_QcomGnss_MeasurementReport_SV_SVObservationState {
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_idle = 0,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_search = 1,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_searchVerify = 2,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_bitEdge = 3,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_trackVerify = 4,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_track = 5,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_restart = 6,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_dpo = 7,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_glo10msBe = 8,
	cereal_QcomGnss_MeasurementReport_SV_SVObservationState_glo10msAt = 9
};
extern int32_t cereal_logVersion;

struct cereal_Map {
	cereal_Map_Entry_list entries;
};

static const size_t cereal_Map_word_count = 0;

static const size_t cereal_Map_pointer_count = 1;

static const size_t cereal_Map_struct_bytes_count = 8;

struct cereal_Map_Entry {
	capn_ptr key;
	capn_ptr value;
};

static const size_t cereal_Map_Entry_word_count = 0;

static const size_t cereal_Map_Entry_pointer_count = 2;

static const size_t cereal_Map_Entry_struct_bytes_count = 16;

struct cereal_InitData {
	capn_ptr kernelArgs;
	capn_text gctx;
	capn_text dongleId;
	enum cereal_InitData_DeviceType deviceType;
	capn_text version;
	cereal_InitData_AndroidBuildInfo_ptr androidBuildInfo;
	cereal_InitData_AndroidSensor_list androidSensors;
	cereal_InitData_ChffrAndroidExtra_ptr chffrAndroidExtra;
};

static const size_t cereal_InitData_word_count = 1;

static const size_t cereal_InitData_pointer_count = 7;

static const size_t cereal_InitData_struct_bytes_count = 64;

struct cereal_InitData_AndroidBuildInfo {
	capn_text board;
	capn_text bootloader;
	capn_text brand;
	capn_text device;
	capn_text display;
	capn_text fingerprint;
	capn_text hardware;
	capn_text host;
	capn_text id;
	capn_text manufacturer;
	capn_text model;
	capn_text product;
	capn_text radioVersion;
	capn_text serial;
	capn_ptr supportedAbis;
	capn_text tags;
	int64_t time;
	capn_text type;
	capn_text user;
	capn_text versionCodename;
	capn_text versionRelease;
	int32_t versionSdk;
	capn_text versionSecurityPatch;
};

static const size_t cereal_InitData_AndroidBuildInfo_word_count = 2;

static const size_t cereal_InitData_AndroidBuildInfo_pointer_count = 21;

static const size_t cereal_InitData_AndroidBuildInfo_struct_bytes_count = 184;

struct cereal_InitData_AndroidSensor {
	int32_t id;
	capn_text name;
	capn_text vendor;
	int32_t version;
	int32_t handle;
	int32_t type;
	float maxRange;
	float resolution;
	float power;
	int32_t minDelay;
	uint32_t fifoReservedEventCount;
	uint32_t fifoMaxEventCount;
	capn_text stringType;
	int32_t maxDelay;
};

static const size_t cereal_InitData_AndroidSensor_word_count = 6;

static const size_t cereal_InitData_AndroidSensor_pointer_count = 3;

static const size_t cereal_InitData_AndroidSensor_struct_bytes_count = 72;

struct cereal_InitData_ChffrAndroidExtra {
	cereal_Map_ptr allCameraCharacteristics;
};

static const size_t cereal_InitData_ChffrAndroidExtra_word_count = 0;

static const size_t cereal_InitData_ChffrAndroidExtra_pointer_count = 1;

static const size_t cereal_InitData_ChffrAndroidExtra_struct_bytes_count = 8;

struct cereal_FrameData {
	uint32_t frameId;
	uint32_t encodeId;
	uint64_t timestampEof;
	int32_t frameLength;
	int32_t integLines;
	int32_t globalGain;
	capn_data image;
	enum cereal_FrameData_FrameType frameType;
	uint64_t timestampSof;
	cereal_FrameData_AndroidCaptureResult_ptr androidCaptureResult;
};

static const size_t cereal_FrameData_word_count = 5;

static const size_t cereal_FrameData_pointer_count = 2;

static const size_t cereal_FrameData_struct_bytes_count = 56;

struct cereal_FrameData_AndroidCaptureResult {
	int32_t sensitivity;
	int64_t frameDuration;
	int64_t exposureTime;
	uint64_t rollingShutterSkew;
	capn_list32 colorCorrectionTransform;
	capn_list32 colorCorrectionGains;
};

static const size_t cereal_FrameData_AndroidCaptureResult_word_count = 4;

static const size_t cereal_FrameData_AndroidCaptureResult_pointer_count = 2;

static const size_t cereal_FrameData_AndroidCaptureResult_struct_bytes_count = 48;

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
	enum cereal_SensorEventData_SensorSource source;
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

struct cereal_GpsLocationData {
	uint16_t flags;
	double latitude;
	double longitude;
	double altitude;
	float speed;
	float bearing;
	float accuracy;
	int64_t timestamp;
	enum cereal_GpsLocationData_SensorSource source;
};

static const size_t cereal_GpsLocationData_word_count = 6;

static const size_t cereal_GpsLocationData_pointer_count = 0;

static const size_t cereal_GpsLocationData_struct_bytes_count = 48;

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
	float freeSpace;
	int16_t batteryPercent;
	capn_text batteryStatus;
};

static const size_t cereal_ThermalData_word_count = 3;

static const size_t cereal_ThermalData_pointer_count = 1;

static const size_t cereal_ThermalData_struct_bytes_count = 32;

struct cereal_HealthData {
	uint32_t voltage;
	uint32_t current;
	unsigned started : 1;
	unsigned controlsAllowed : 1;
	unsigned gasInterceptorDetected : 1;
	unsigned startedSignalDetected : 1;
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
	capn_list32 warpMatrixDEPRECATED;
	float angleOffsetDEPRECATED;
	int8_t calStatusDEPRECATED;
	int32_t calCycleDEPRECATED;
	int8_t calPercDEPRECATED;
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
	float aEgoDEPRECATED;
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
	int32_t hudLeadDEPRECATED;
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
	uint32_t segmentIdEncode;
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

struct cereal_Plan {
	unsigned lateralValid : 1;
	capn_list32 dPoly;
	unsigned longitudalValid : 1;
	float vTarget;
	float aTargetMin;
	float aTargetMax;
	float jerkFactor;
};

static const size_t cereal_Plan_word_count = 3;

static const size_t cereal_Plan_pointer_count = 1;

static const size_t cereal_Plan_struct_bytes_count = 32;

struct cereal_LiveLocationData {
	uint8_t status;
	double lat;
	double lon;
	float alt;
	float speed;
	capn_list32 vNED;
	float roll;
	float pitch;
	float heading;
	float wanderAngle;
	float trackAngle;
	capn_list32 gyro;
	capn_list32 accel;
	cereal_LiveLocationData_Accuracy_ptr accuracy;
};

static const size_t cereal_LiveLocationData_word_count = 6;

static const size_t cereal_LiveLocationData_pointer_count = 4;

static const size_t cereal_LiveLocationData_struct_bytes_count = 80;

struct cereal_LiveLocationData_Accuracy {
	capn_list32 pNEDError;
	capn_list32 vNEDError;
	float rollError;
	float pitchError;
	float headingError;
	float ellipsoidSemiMajorError;
	float ellipsoidSemiMinorError;
	float ellipsoidOrientationError;
};

static const size_t cereal_LiveLocationData_Accuracy_word_count = 3;

static const size_t cereal_LiveLocationData_Accuracy_pointer_count = 2;

static const size_t cereal_LiveLocationData_Accuracy_struct_bytes_count = 40;

struct cereal_EthernetPacket {
	capn_data pkt;
	float ts;
};

static const size_t cereal_EthernetPacket_word_count = 1;

static const size_t cereal_EthernetPacket_pointer_count = 1;

static const size_t cereal_EthernetPacket_struct_bytes_count = 16;

struct cereal_NavUpdate {
	unsigned isNavigating : 1;
	int32_t curSegment;
	cereal_NavUpdate_Segment_list segments;
};

static const size_t cereal_NavUpdate_word_count = 1;

static const size_t cereal_NavUpdate_pointer_count = 1;

static const size_t cereal_NavUpdate_struct_bytes_count = 16;

struct cereal_NavUpdate_LatLng {
	double lat;
	double lng;
};

static const size_t cereal_NavUpdate_LatLng_word_count = 2;

static const size_t cereal_NavUpdate_LatLng_pointer_count = 0;

static const size_t cereal_NavUpdate_LatLng_struct_bytes_count = 16;

struct cereal_NavUpdate_Segment {
	cereal_NavUpdate_LatLng_ptr from;
	cereal_NavUpdate_LatLng_ptr to;
	int32_t updateTime;
	int32_t distance;
	int32_t crossTime;
	int32_t exitNo;
	enum cereal_NavUpdate_Segment_Instruction instruction;
	cereal_NavUpdate_LatLng_list parts;
};

static const size_t cereal_NavUpdate_Segment_word_count = 3;

static const size_t cereal_NavUpdate_Segment_pointer_count = 3;

static const size_t cereal_NavUpdate_Segment_struct_bytes_count = 48;

struct cereal_CellInfo {
	uint64_t timestamp;
	capn_text repr;
};

static const size_t cereal_CellInfo_word_count = 1;

static const size_t cereal_CellInfo_pointer_count = 1;

static const size_t cereal_CellInfo_struct_bytes_count = 16;

struct cereal_WifiScan {
	capn_text bssid;
	capn_text ssid;
	capn_text capabilities;
	int32_t frequency;
	int32_t level;
	int64_t timestamp;
	int32_t centerFreq0;
	int32_t centerFreq1;
	enum cereal_WifiScan_ChannelWidth channelWidth;
	capn_text operatorFriendlyName;
	capn_text venueName;
	unsigned is80211mcResponder : 1;
	unsigned passpoint : 1;
	int32_t distanceCm;
	int32_t distanceSdCm;
};

static const size_t cereal_WifiScan_word_count = 5;

static const size_t cereal_WifiScan_pointer_count = 5;

static const size_t cereal_WifiScan_struct_bytes_count = 80;
enum cereal_AndroidGnss_which {
	cereal_AndroidGnss_measurements = 0,
	cereal_AndroidGnss_navigationMessage = 1
};

struct cereal_AndroidGnss {
	enum cereal_AndroidGnss_which which;
	union {
		cereal_AndroidGnss_Measurements_ptr measurements;
		cereal_AndroidGnss_NavigationMessage_ptr navigationMessage;
	};
};

static const size_t cereal_AndroidGnss_word_count = 1;

static const size_t cereal_AndroidGnss_pointer_count = 1;

static const size_t cereal_AndroidGnss_struct_bytes_count = 16;

struct cereal_AndroidGnss_Measurements {
	cereal_AndroidGnss_Measurements_Clock_ptr clock;
	cereal_AndroidGnss_Measurements_Measurement_list measurements;
};

static const size_t cereal_AndroidGnss_Measurements_word_count = 0;

static const size_t cereal_AndroidGnss_Measurements_pointer_count = 2;

static const size_t cereal_AndroidGnss_Measurements_struct_bytes_count = 16;

struct cereal_AndroidGnss_Measurements_Clock {
	int64_t timeNanos;
	int32_t hardwareClockDiscontinuityCount;
	unsigned hasTimeUncertaintyNanos : 1;
	double timeUncertaintyNanos;
	unsigned hasLeapSecond : 1;
	int32_t leapSecond;
	unsigned hasFullBiasNanos : 1;
	int64_t fullBiasNanos;
	unsigned hasBiasNanos : 1;
	double biasNanos;
	unsigned hasBiasUncertaintyNanos : 1;
	double biasUncertaintyNanos;
	unsigned hasDriftNanosPerSecond : 1;
	double driftNanosPerSecond;
	unsigned hasDriftUncertaintyNanosPerSecond : 1;
	double driftUncertaintyNanosPerSecond;
};

static const size_t cereal_AndroidGnss_Measurements_Clock_word_count = 9;

static const size_t cereal_AndroidGnss_Measurements_Clock_pointer_count = 0;

static const size_t cereal_AndroidGnss_Measurements_Clock_struct_bytes_count = 72;

struct cereal_AndroidGnss_Measurements_Measurement {
	int32_t svId;
	enum cereal_AndroidGnss_Measurements_Measurement_Constellation constellation;
	double timeOffsetNanos;
	int32_t state;
	int64_t receivedSvTimeNanos;
	int64_t receivedSvTimeUncertaintyNanos;
	double cn0DbHz;
	double pseudorangeRateMetersPerSecond;
	double pseudorangeRateUncertaintyMetersPerSecond;
	int32_t accumulatedDeltaRangeState;
	double accumulatedDeltaRangeMeters;
	double accumulatedDeltaRangeUncertaintyMeters;
	unsigned hasCarrierFrequencyHz : 1;
	float carrierFrequencyHz;
	unsigned hasCarrierCycles : 1;
	int64_t carrierCycles;
	unsigned hasCarrierPhase : 1;
	double carrierPhase;
	unsigned hasCarrierPhaseUncertainty : 1;
	double carrierPhaseUncertainty;
	unsigned hasSnrInDb : 1;
	double snrInDb;
	enum cereal_AndroidGnss_Measurements_Measurement_MultipathIndicator multipathIndicator;
};

static const size_t cereal_AndroidGnss_Measurements_Measurement_word_count = 15;

static const size_t cereal_AndroidGnss_Measurements_Measurement_pointer_count = 0;

static const size_t cereal_AndroidGnss_Measurements_Measurement_struct_bytes_count = 120;

struct cereal_AndroidGnss_NavigationMessage {
	int32_t type;
	int32_t svId;
	int32_t messageId;
	int32_t submessageId;
	capn_data data;
	enum cereal_AndroidGnss_NavigationMessage_Status status;
};

static const size_t cereal_AndroidGnss_NavigationMessage_word_count = 3;

static const size_t cereal_AndroidGnss_NavigationMessage_pointer_count = 1;

static const size_t cereal_AndroidGnss_NavigationMessage_struct_bytes_count = 32;
enum cereal_QcomGnss_which {
	cereal_QcomGnss_measurementReport = 0,
	cereal_QcomGnss_clockReport = 1
};

struct cereal_QcomGnss {
	uint64_t logTs;
	enum cereal_QcomGnss_which which;
	union {
		cereal_QcomGnss_MeasurementReport_ptr measurementReport;
		cereal_QcomGnss_ClockReport_ptr clockReport;
	};
};

static const size_t cereal_QcomGnss_word_count = 2;

static const size_t cereal_QcomGnss_pointer_count = 1;

static const size_t cereal_QcomGnss_struct_bytes_count = 24;

struct cereal_QcomGnss_MeasurementReport {
	enum cereal_QcomGnss_MeasurementReport_Source source;
	uint32_t fCount;
	uint16_t gpsWeek;
	uint8_t glonassCycleNumber;
	uint16_t glonassNumberOfDays;
	uint32_t milliseconds;
	float timeBias;
	float clockTimeUncertainty;
	float clockFrequencyBias;
	float clockFrequencyUncertainty;
	cereal_QcomGnss_MeasurementReport_SV_list sv;
};

static const size_t cereal_QcomGnss_MeasurementReport_word_count = 4;

static const size_t cereal_QcomGnss_MeasurementReport_pointer_count = 1;

static const size_t cereal_QcomGnss_MeasurementReport_struct_bytes_count = 40;

struct cereal_QcomGnss_MeasurementReport_SV {
	uint8_t svId;
	enum cereal_QcomGnss_MeasurementReport_SV_SVObservationState observationState;
	uint8_t observations;
	uint8_t goodObservations;
	uint16_t gpsParityErrorCount;
	int8_t glonassFrequencyIndex;
	uint8_t glonassHemmingErrorCount;
	uint8_t filterStages;
	uint16_t carrierNoise;
	int16_t latency;
	uint8_t predetectIntegration;
	uint16_t postdetections;
	uint32_t unfilteredMeasurementIntegral;
	float unfilteredMeasurementFraction;
	float unfilteredTimeUncertainty;
	float unfilteredSpeed;
	float unfilteredSpeedUncertainty;
	cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr measurementStatus;
	uint32_t multipathEstimate;
	float azimuth;
	float elevation;
	int32_t carrierPhaseCyclesIntegral;
	uint16_t carrierPhaseCyclesFraction;
	float fineSpeed;
	float fineSpeedUncertainty;
	uint8_t cycleSlipCount;
};

static const size_t cereal_QcomGnss_MeasurementReport_SV_word_count = 8;

static const size_t cereal_QcomGnss_MeasurementReport_SV_pointer_count = 1;

static const size_t cereal_QcomGnss_MeasurementReport_SV_struct_bytes_count = 72;

struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus {
	unsigned subMillisecondIsValid : 1;
	unsigned subBitTimeIsKnown : 1;
	unsigned satelliteTimeIsKnown : 1;
	unsigned bitEdgeConfirmedFromSignal : 1;
	unsigned measuredVelocity : 1;
	unsigned fineOrCoarseVelocity : 1;
	unsigned lockPointValid : 1;
	unsigned lockPointPositive : 1;
	unsigned lastUpdateFromDifference : 1;
	unsigned lastUpdateFromVelocityDifference : 1;
	unsigned strongIndicationOfCrossCorelation : 1;
	unsigned tentativeMeasurement : 1;
	unsigned measurementNotUsable : 1;
	unsigned sirCheckIsNeeded : 1;
	unsigned probationMode : 1;
	unsigned glonassMeanderBitEdgeValid : 1;
	unsigned glonassTimeMarkValid : 1;
	unsigned gpsRoundRobinRxDiversity : 1;
	unsigned gpsRxDiversity : 1;
	unsigned gpsLowBandwidthRxDiversityCombined : 1;
	unsigned gpsHighBandwidthNu4 : 1;
	unsigned gpsHighBandwidthNu8 : 1;
	unsigned gpsHighBandwidthUniform : 1;
	unsigned gpsMultipathIndicator : 1;
	unsigned imdJammingIndicator : 1;
	unsigned lteB13TxJammingIndicator : 1;
	unsigned freshMeasurementIndicator : 1;
	unsigned multipathEstimateIsValid : 1;
	unsigned directionIsValid : 1;
};

static const size_t cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_word_count = 1;

static const size_t cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_pointer_count = 0;

static const size_t cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_struct_bytes_count = 8;

struct cereal_QcomGnss_ClockReport {
	unsigned hasFCount : 1;
	uint32_t fCount;
	unsigned hasGpsWeekNumber : 1;
	uint16_t gpsWeekNumber;
	unsigned hasGpsMilliseconds : 1;
	uint32_t gpsMilliseconds;
	float gpsTimeBias;
	float gpsClockTimeUncertainty;
	uint8_t gpsClockSource;
	unsigned hasGlonassYear : 1;
	uint8_t glonassYear;
	unsigned hasGlonassDay : 1;
	uint16_t glonassDay;
	unsigned hasGlonassMilliseconds : 1;
	uint32_t glonassMilliseconds;
	float glonassTimeBias;
	float glonassClockTimeUncertainty;
	uint8_t glonassClockSource;
	uint16_t bdsWeek;
	uint32_t bdsMilliseconds;
	float bdsTimeBias;
	float bdsClockTimeUncertainty;
	uint8_t bdsClockSource;
	uint16_t galWeek;
	uint32_t galMilliseconds;
	float galTimeBias;
	float galClockTimeUncertainty;
	uint8_t galClockSource;
	float clockFrequencyBias;
	float clockFrequencyUncertainty;
	uint8_t frequencySource;
	uint8_t gpsLeapSeconds;
	uint8_t gpsLeapSecondsUncertainty;
	uint8_t gpsLeapSecondsSource;
	float gpsToGlonassTimeBiasMilliseconds;
	float gpsToGlonassTimeBiasMillisecondsUncertainty;
	float gpsToBdsTimeBiasMilliseconds;
	float gpsToBdsTimeBiasMillisecondsUncertainty;
	float bdsToGloTimeBiasMilliseconds;
	float bdsToGloTimeBiasMillisecondsUncertainty;
	float gpsToGalTimeBiasMilliseconds;
	float gpsToGalTimeBiasMillisecondsUncertainty;
	float galToGloTimeBiasMilliseconds;
	float galToGloTimeBiasMillisecondsUncertainty;
	float galToBdsTimeBiasMilliseconds;
	float galToBdsTimeBiasMillisecondsUncertainty;
	unsigned hasRtcTime : 1;
	uint32_t systemRtcTime;
	uint32_t fCountOffset;
	uint32_t lpmRtcCount;
	uint32_t clockResets;
};

static const size_t cereal_QcomGnss_ClockReport_word_count = 18;

static const size_t cereal_QcomGnss_ClockReport_pointer_count = 0;

static const size_t cereal_QcomGnss_ClockReport_struct_bytes_count = 144;
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
	cereal_Event_androidLogEntry = 19,
	cereal_Event_gpsLocation = 20,
	cereal_Event_carState = 21,
	cereal_Event_carControl = 22,
	cereal_Event_plan = 23,
	cereal_Event_liveLocation = 24,
	cereal_Event_ethernetData = 25,
	cereal_Event_navUpdate = 26,
	cereal_Event_cellInfo = 27,
	cereal_Event_wifiScan = 28,
	cereal_Event_androidGnss = 29,
	cereal_Event_qcomGnss = 30
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
		cereal_GpsLocationData_ptr gpsLocation;
		cereal_CarState_ptr carState;
		cereal_CarControl_ptr carControl;
		cereal_Plan_ptr plan;
		cereal_LiveLocationData_ptr liveLocation;
		cereal_EthernetPacket_list ethernetData;
		cereal_NavUpdate_ptr navUpdate;
		cereal_CellInfo_list cellInfo;
		cereal_WifiScan_list wifiScan;
		cereal_AndroidGnss_ptr androidGnss;
		cereal_QcomGnss_ptr qcomGnss;
	};
};

static const size_t cereal_Event_word_count = 2;

static const size_t cereal_Event_pointer_count = 1;

static const size_t cereal_Event_struct_bytes_count = 24;

cereal_Map_ptr cereal_new_Map(struct capn_segment*);
cereal_Map_Entry_ptr cereal_new_Map_Entry(struct capn_segment*);
cereal_InitData_ptr cereal_new_InitData(struct capn_segment*);
cereal_InitData_AndroidBuildInfo_ptr cereal_new_InitData_AndroidBuildInfo(struct capn_segment*);
cereal_InitData_AndroidSensor_ptr cereal_new_InitData_AndroidSensor(struct capn_segment*);
cereal_InitData_ChffrAndroidExtra_ptr cereal_new_InitData_ChffrAndroidExtra(struct capn_segment*);
cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment*);
cereal_FrameData_AndroidCaptureResult_ptr cereal_new_FrameData_AndroidCaptureResult(struct capn_segment*);
cereal_GPSNMEAData_ptr cereal_new_GPSNMEAData(struct capn_segment*);
cereal_SensorEventData_ptr cereal_new_SensorEventData(struct capn_segment*);
cereal_SensorEventData_SensorVec_ptr cereal_new_SensorEventData_SensorVec(struct capn_segment*);
cereal_GpsLocationData_ptr cereal_new_GpsLocationData(struct capn_segment*);
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
cereal_Plan_ptr cereal_new_Plan(struct capn_segment*);
cereal_LiveLocationData_ptr cereal_new_LiveLocationData(struct capn_segment*);
cereal_LiveLocationData_Accuracy_ptr cereal_new_LiveLocationData_Accuracy(struct capn_segment*);
cereal_EthernetPacket_ptr cereal_new_EthernetPacket(struct capn_segment*);
cereal_NavUpdate_ptr cereal_new_NavUpdate(struct capn_segment*);
cereal_NavUpdate_LatLng_ptr cereal_new_NavUpdate_LatLng(struct capn_segment*);
cereal_NavUpdate_Segment_ptr cereal_new_NavUpdate_Segment(struct capn_segment*);
cereal_CellInfo_ptr cereal_new_CellInfo(struct capn_segment*);
cereal_WifiScan_ptr cereal_new_WifiScan(struct capn_segment*);
cereal_AndroidGnss_ptr cereal_new_AndroidGnss(struct capn_segment*);
cereal_AndroidGnss_Measurements_ptr cereal_new_AndroidGnss_Measurements(struct capn_segment*);
cereal_AndroidGnss_Measurements_Clock_ptr cereal_new_AndroidGnss_Measurements_Clock(struct capn_segment*);
cereal_AndroidGnss_Measurements_Measurement_ptr cereal_new_AndroidGnss_Measurements_Measurement(struct capn_segment*);
cereal_AndroidGnss_NavigationMessage_ptr cereal_new_AndroidGnss_NavigationMessage(struct capn_segment*);
cereal_QcomGnss_ptr cereal_new_QcomGnss(struct capn_segment*);
cereal_QcomGnss_MeasurementReport_ptr cereal_new_QcomGnss_MeasurementReport(struct capn_segment*);
cereal_QcomGnss_MeasurementReport_SV_ptr cereal_new_QcomGnss_MeasurementReport_SV(struct capn_segment*);
cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr cereal_new_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct capn_segment*);
cereal_QcomGnss_ClockReport_ptr cereal_new_QcomGnss_ClockReport(struct capn_segment*);
cereal_Event_ptr cereal_new_Event(struct capn_segment*);

cereal_Map_list cereal_new_Map_list(struct capn_segment*, int len);
cereal_Map_Entry_list cereal_new_Map_Entry_list(struct capn_segment*, int len);
cereal_InitData_list cereal_new_InitData_list(struct capn_segment*, int len);
cereal_InitData_AndroidBuildInfo_list cereal_new_InitData_AndroidBuildInfo_list(struct capn_segment*, int len);
cereal_InitData_AndroidSensor_list cereal_new_InitData_AndroidSensor_list(struct capn_segment*, int len);
cereal_InitData_ChffrAndroidExtra_list cereal_new_InitData_ChffrAndroidExtra_list(struct capn_segment*, int len);
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment*, int len);
cereal_FrameData_AndroidCaptureResult_list cereal_new_FrameData_AndroidCaptureResult_list(struct capn_segment*, int len);
cereal_GPSNMEAData_list cereal_new_GPSNMEAData_list(struct capn_segment*, int len);
cereal_SensorEventData_list cereal_new_SensorEventData_list(struct capn_segment*, int len);
cereal_SensorEventData_SensorVec_list cereal_new_SensorEventData_SensorVec_list(struct capn_segment*, int len);
cereal_GpsLocationData_list cereal_new_GpsLocationData_list(struct capn_segment*, int len);
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
cereal_Plan_list cereal_new_Plan_list(struct capn_segment*, int len);
cereal_LiveLocationData_list cereal_new_LiveLocationData_list(struct capn_segment*, int len);
cereal_LiveLocationData_Accuracy_list cereal_new_LiveLocationData_Accuracy_list(struct capn_segment*, int len);
cereal_EthernetPacket_list cereal_new_EthernetPacket_list(struct capn_segment*, int len);
cereal_NavUpdate_list cereal_new_NavUpdate_list(struct capn_segment*, int len);
cereal_NavUpdate_LatLng_list cereal_new_NavUpdate_LatLng_list(struct capn_segment*, int len);
cereal_NavUpdate_Segment_list cereal_new_NavUpdate_Segment_list(struct capn_segment*, int len);
cereal_CellInfo_list cereal_new_CellInfo_list(struct capn_segment*, int len);
cereal_WifiScan_list cereal_new_WifiScan_list(struct capn_segment*, int len);
cereal_AndroidGnss_list cereal_new_AndroidGnss_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_list cereal_new_AndroidGnss_Measurements_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_Clock_list cereal_new_AndroidGnss_Measurements_Clock_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_Measurement_list cereal_new_AndroidGnss_Measurements_Measurement_list(struct capn_segment*, int len);
cereal_AndroidGnss_NavigationMessage_list cereal_new_AndroidGnss_NavigationMessage_list(struct capn_segment*, int len);
cereal_QcomGnss_list cereal_new_QcomGnss_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementReport_list cereal_new_QcomGnss_MeasurementReport_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementReport_SV_list cereal_new_QcomGnss_MeasurementReport_SV_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list cereal_new_QcomGnss_MeasurementReport_SV_MeasurementStatus_list(struct capn_segment*, int len);
cereal_QcomGnss_ClockReport_list cereal_new_QcomGnss_ClockReport_list(struct capn_segment*, int len);
cereal_Event_list cereal_new_Event_list(struct capn_segment*, int len);

void cereal_read_Map(struct cereal_Map*, cereal_Map_ptr);
void cereal_read_Map_Entry(struct cereal_Map_Entry*, cereal_Map_Entry_ptr);
void cereal_read_InitData(struct cereal_InitData*, cereal_InitData_ptr);
void cereal_read_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_ptr);
void cereal_read_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_ptr);
void cereal_read_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_ptr);
void cereal_read_FrameData(struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_read_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_ptr);
void cereal_read_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_read_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_read_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_read_GpsLocationData(struct cereal_GpsLocationData*, cereal_GpsLocationData_ptr);
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
void cereal_read_Plan(struct cereal_Plan*, cereal_Plan_ptr);
void cereal_read_LiveLocationData(struct cereal_LiveLocationData*, cereal_LiveLocationData_ptr);
void cereal_read_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_ptr);
void cereal_read_EthernetPacket(struct cereal_EthernetPacket*, cereal_EthernetPacket_ptr);
void cereal_read_NavUpdate(struct cereal_NavUpdate*, cereal_NavUpdate_ptr);
void cereal_read_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_ptr);
void cereal_read_NavUpdate_Segment(struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_ptr);
void cereal_read_CellInfo(struct cereal_CellInfo*, cereal_CellInfo_ptr);
void cereal_read_WifiScan(struct cereal_WifiScan*, cereal_WifiScan_ptr);
void cereal_read_AndroidGnss(struct cereal_AndroidGnss*, cereal_AndroidGnss_ptr);
void cereal_read_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_ptr);
void cereal_read_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_ptr);
void cereal_read_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_ptr);
void cereal_read_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_ptr);
void cereal_read_QcomGnss(struct cereal_QcomGnss*, cereal_QcomGnss_ptr);
void cereal_read_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_ptr);
void cereal_read_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_ptr);
void cereal_read_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus*, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr);
void cereal_read_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_ptr);
void cereal_read_Event(struct cereal_Event*, cereal_Event_ptr);

void cereal_write_Map(const struct cereal_Map*, cereal_Map_ptr);
void cereal_write_Map_Entry(const struct cereal_Map_Entry*, cereal_Map_Entry_ptr);
void cereal_write_InitData(const struct cereal_InitData*, cereal_InitData_ptr);
void cereal_write_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_ptr);
void cereal_write_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_ptr);
void cereal_write_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_ptr);
void cereal_write_FrameData(const struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_write_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_ptr);
void cereal_write_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_write_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_write_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_write_GpsLocationData(const struct cereal_GpsLocationData*, cereal_GpsLocationData_ptr);
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
void cereal_write_Plan(const struct cereal_Plan*, cereal_Plan_ptr);
void cereal_write_LiveLocationData(const struct cereal_LiveLocationData*, cereal_LiveLocationData_ptr);
void cereal_write_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_ptr);
void cereal_write_EthernetPacket(const struct cereal_EthernetPacket*, cereal_EthernetPacket_ptr);
void cereal_write_NavUpdate(const struct cereal_NavUpdate*, cereal_NavUpdate_ptr);
void cereal_write_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_ptr);
void cereal_write_NavUpdate_Segment(const struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_ptr);
void cereal_write_CellInfo(const struct cereal_CellInfo*, cereal_CellInfo_ptr);
void cereal_write_WifiScan(const struct cereal_WifiScan*, cereal_WifiScan_ptr);
void cereal_write_AndroidGnss(const struct cereal_AndroidGnss*, cereal_AndroidGnss_ptr);
void cereal_write_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_ptr);
void cereal_write_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_ptr);
void cereal_write_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_ptr);
void cereal_write_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_ptr);
void cereal_write_QcomGnss(const struct cereal_QcomGnss*, cereal_QcomGnss_ptr);
void cereal_write_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_ptr);
void cereal_write_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_ptr);
void cereal_write_QcomGnss_MeasurementReport_SV_MeasurementStatus(const struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus*, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_ptr);
void cereal_write_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_ptr);
void cereal_write_Event(const struct cereal_Event*, cereal_Event_ptr);

void cereal_get_Map(struct cereal_Map*, cereal_Map_list, int i);
void cereal_get_Map_Entry(struct cereal_Map_Entry*, cereal_Map_Entry_list, int i);
void cereal_get_InitData(struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_get_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_list, int i);
void cereal_get_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_list, int i);
void cereal_get_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_list, int i);
void cereal_get_FrameData(struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_get_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_list, int i);
void cereal_get_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_get_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_get_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_get_GpsLocationData(struct cereal_GpsLocationData*, cereal_GpsLocationData_list, int i);
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
void cereal_get_Plan(struct cereal_Plan*, cereal_Plan_list, int i);
void cereal_get_LiveLocationData(struct cereal_LiveLocationData*, cereal_LiveLocationData_list, int i);
void cereal_get_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_list, int i);
void cereal_get_EthernetPacket(struct cereal_EthernetPacket*, cereal_EthernetPacket_list, int i);
void cereal_get_NavUpdate(struct cereal_NavUpdate*, cereal_NavUpdate_list, int i);
void cereal_get_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_list, int i);
void cereal_get_NavUpdate_Segment(struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_list, int i);
void cereal_get_CellInfo(struct cereal_CellInfo*, cereal_CellInfo_list, int i);
void cereal_get_WifiScan(struct cereal_WifiScan*, cereal_WifiScan_list, int i);
void cereal_get_AndroidGnss(struct cereal_AndroidGnss*, cereal_AndroidGnss_list, int i);
void cereal_get_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_list, int i);
void cereal_get_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_list, int i);
void cereal_get_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_list, int i);
void cereal_get_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_list, int i);
void cereal_get_QcomGnss(struct cereal_QcomGnss*, cereal_QcomGnss_list, int i);
void cereal_get_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_list, int i);
void cereal_get_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_list, int i);
void cereal_get_QcomGnss_MeasurementReport_SV_MeasurementStatus(struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus*, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list, int i);
void cereal_get_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_list, int i);
void cereal_get_Event(struct cereal_Event*, cereal_Event_list, int i);

void cereal_set_Map(const struct cereal_Map*, cereal_Map_list, int i);
void cereal_set_Map_Entry(const struct cereal_Map_Entry*, cereal_Map_Entry_list, int i);
void cereal_set_InitData(const struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_set_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_list, int i);
void cereal_set_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_list, int i);
void cereal_set_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_list, int i);
void cereal_set_FrameData(const struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_set_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_list, int i);
void cereal_set_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_set_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_set_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_set_GpsLocationData(const struct cereal_GpsLocationData*, cereal_GpsLocationData_list, int i);
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
void cereal_set_Plan(const struct cereal_Plan*, cereal_Plan_list, int i);
void cereal_set_LiveLocationData(const struct cereal_LiveLocationData*, cereal_LiveLocationData_list, int i);
void cereal_set_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_list, int i);
void cereal_set_EthernetPacket(const struct cereal_EthernetPacket*, cereal_EthernetPacket_list, int i);
void cereal_set_NavUpdate(const struct cereal_NavUpdate*, cereal_NavUpdate_list, int i);
void cereal_set_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_list, int i);
void cereal_set_NavUpdate_Segment(const struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_list, int i);
void cereal_set_CellInfo(const struct cereal_CellInfo*, cereal_CellInfo_list, int i);
void cereal_set_WifiScan(const struct cereal_WifiScan*, cereal_WifiScan_list, int i);
void cereal_set_AndroidGnss(const struct cereal_AndroidGnss*, cereal_AndroidGnss_list, int i);
void cereal_set_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_list, int i);
void cereal_set_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_list, int i);
void cereal_set_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_list, int i);
void cereal_set_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_list, int i);
void cereal_set_QcomGnss(const struct cereal_QcomGnss*, cereal_QcomGnss_list, int i);
void cereal_set_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_list, int i);
void cereal_set_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_list, int i);
void cereal_set_QcomGnss_MeasurementReport_SV_MeasurementStatus(const struct cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus*, cereal_QcomGnss_MeasurementReport_SV_MeasurementStatus_list, int i);
void cereal_set_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_list, int i);
void cereal_set_Event(const struct cereal_Event*, cereal_Event_list, int i);

#ifdef __cplusplus
}
#endif
#endif
