#ifndef CAPN_8E2AF1E78AF8B8D
#define CAPN_8E2AF1E78AF8B8D
/* AUTO GENERATED - DO NOT EDIT */
#include <capnp_c.h>

#if CAPN_VERSION != 1
#error "version mismatch between capnp_c.h and generated code"
#endif

#include "./include/c++.capnp.h"
#include "./include/java.capnp.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cereal_CarEvent;
struct cereal_CarState;
struct cereal_CarState_WheelSpeeds;
struct cereal_CarState_CruiseState;
struct cereal_CarState_ButtonEvent;
struct cereal_RadarData;
struct cereal_RadarData_RadarPoint;
struct cereal_CarControl;
struct cereal_CarControl_Actuators;
struct cereal_CarControl_CruiseControl;
struct cereal_CarControl_HUDControl;
struct cereal_CarParams;
struct cereal_CarParams_LateralParams;
struct cereal_CarParams_LateralPIDTuning;
struct cereal_CarParams_LongitudinalPIDTuning;
struct cereal_CarParams_LateralINDITuning;
struct cereal_CarParams_LateralLQRTuning;
struct cereal_CarParams_CarFw;

typedef struct {capn_ptr p;} cereal_CarEvent_ptr;
typedef struct {capn_ptr p;} cereal_CarState_ptr;
typedef struct {capn_ptr p;} cereal_CarState_WheelSpeeds_ptr;
typedef struct {capn_ptr p;} cereal_CarState_CruiseState_ptr;
typedef struct {capn_ptr p;} cereal_CarState_ButtonEvent_ptr;
typedef struct {capn_ptr p;} cereal_RadarData_ptr;
typedef struct {capn_ptr p;} cereal_RadarData_RadarPoint_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_Actuators_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_CruiseControl_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_HUDControl_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_LateralParams_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_LateralPIDTuning_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_LongitudinalPIDTuning_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_LateralINDITuning_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_LateralLQRTuning_ptr;
typedef struct {capn_ptr p;} cereal_CarParams_CarFw_ptr;

typedef struct {capn_ptr p;} cereal_CarEvent_list;
typedef struct {capn_ptr p;} cereal_CarState_list;
typedef struct {capn_ptr p;} cereal_CarState_WheelSpeeds_list;
typedef struct {capn_ptr p;} cereal_CarState_CruiseState_list;
typedef struct {capn_ptr p;} cereal_CarState_ButtonEvent_list;
typedef struct {capn_ptr p;} cereal_RadarData_list;
typedef struct {capn_ptr p;} cereal_RadarData_RadarPoint_list;
typedef struct {capn_ptr p;} cereal_CarControl_list;
typedef struct {capn_ptr p;} cereal_CarControl_Actuators_list;
typedef struct {capn_ptr p;} cereal_CarControl_CruiseControl_list;
typedef struct {capn_ptr p;} cereal_CarControl_HUDControl_list;
typedef struct {capn_ptr p;} cereal_CarParams_list;
typedef struct {capn_ptr p;} cereal_CarParams_LateralParams_list;
typedef struct {capn_ptr p;} cereal_CarParams_LateralPIDTuning_list;
typedef struct {capn_ptr p;} cereal_CarParams_LongitudinalPIDTuning_list;
typedef struct {capn_ptr p;} cereal_CarParams_LateralINDITuning_list;
typedef struct {capn_ptr p;} cereal_CarParams_LateralLQRTuning_list;
typedef struct {capn_ptr p;} cereal_CarParams_CarFw_list;

enum cereal_CarEvent_EventName {
	cereal_CarEvent_EventName_canError = 0,
	cereal_CarEvent_EventName_steerUnavailable = 1,
	cereal_CarEvent_EventName_brakeUnavailable = 2,
	cereal_CarEvent_EventName_gasUnavailable = 3,
	cereal_CarEvent_EventName_wrongGear = 4,
	cereal_CarEvent_EventName_doorOpen = 5,
	cereal_CarEvent_EventName_seatbeltNotLatched = 6,
	cereal_CarEvent_EventName_espDisabled = 7,
	cereal_CarEvent_EventName_wrongCarMode = 8,
	cereal_CarEvent_EventName_steerTempUnavailable = 9,
	cereal_CarEvent_EventName_reverseGear = 10,
	cereal_CarEvent_EventName_buttonCancel = 11,
	cereal_CarEvent_EventName_buttonEnable = 12,
	cereal_CarEvent_EventName_pedalPressed = 13,
	cereal_CarEvent_EventName_cruiseDisabled = 14,
	cereal_CarEvent_EventName_radarCanError = 15,
	cereal_CarEvent_EventName_dataNeeded = 16,
	cereal_CarEvent_EventName_speedTooLow = 17,
	cereal_CarEvent_EventName_outOfSpace = 18,
	cereal_CarEvent_EventName_overheat = 19,
	cereal_CarEvent_EventName_calibrationIncomplete = 20,
	cereal_CarEvent_EventName_calibrationInvalid = 21,
	cereal_CarEvent_EventName_controlsMismatch = 22,
	cereal_CarEvent_EventName_pcmEnable = 23,
	cereal_CarEvent_EventName_pcmDisable = 24,
	cereal_CarEvent_EventName_noTarget = 25,
	cereal_CarEvent_EventName_radarFault = 26,
	cereal_CarEvent_EventName_modelCommIssueDEPRECATED = 27,
	cereal_CarEvent_EventName_brakeHold = 28,
	cereal_CarEvent_EventName_parkBrake = 29,
	cereal_CarEvent_EventName_manualRestart = 30,
	cereal_CarEvent_EventName_lowSpeedLockout = 31,
	cereal_CarEvent_EventName_plannerError = 32,
	cereal_CarEvent_EventName_ipasOverride = 33,
	cereal_CarEvent_EventName_debugAlert = 34,
	cereal_CarEvent_EventName_steerTempUnavailableMute = 35,
	cereal_CarEvent_EventName_resumeRequired = 36,
	cereal_CarEvent_EventName_preDriverDistracted = 37,
	cereal_CarEvent_EventName_promptDriverDistracted = 38,
	cereal_CarEvent_EventName_driverDistracted = 39,
	cereal_CarEvent_EventName_geofence = 40,
	cereal_CarEvent_EventName_driverMonitorOn = 41,
	cereal_CarEvent_EventName_driverMonitorOff = 42,
	cereal_CarEvent_EventName_preDriverUnresponsive = 43,
	cereal_CarEvent_EventName_promptDriverUnresponsive = 44,
	cereal_CarEvent_EventName_driverUnresponsive = 45,
	cereal_CarEvent_EventName_belowSteerSpeed = 46,
	cereal_CarEvent_EventName_calibrationProgress = 47,
	cereal_CarEvent_EventName_lowBattery = 48,
	cereal_CarEvent_EventName_invalidGiraffeHonda = 49,
	cereal_CarEvent_EventName_vehicleModelInvalid = 50,
	cereal_CarEvent_EventName_controlsFailed = 51,
	cereal_CarEvent_EventName_sensorDataInvalid = 52,
	cereal_CarEvent_EventName_commIssue = 53,
	cereal_CarEvent_EventName_tooDistracted = 54,
	cereal_CarEvent_EventName_posenetInvalid = 55,
	cereal_CarEvent_EventName_soundsUnavailable = 56,
	cereal_CarEvent_EventName_preLaneChangeLeft = 57,
	cereal_CarEvent_EventName_preLaneChangeRight = 58,
	cereal_CarEvent_EventName_laneChange = 59,
	cereal_CarEvent_EventName_invalidGiraffeToyota = 60,
	cereal_CarEvent_EventName_internetConnectivityNeeded = 61,
	cereal_CarEvent_EventName_communityFeatureDisallowed = 62,
	cereal_CarEvent_EventName_lowMemory = 63,
	cereal_CarEvent_EventName_stockAeb = 64,
	cereal_CarEvent_EventName_ldw = 65,
	cereal_CarEvent_EventName_carUnrecognized = 66,
	cereal_CarEvent_EventName_radarCommIssue = 67,
	cereal_CarEvent_EventName_driverMonitorLowAcc = 68
};

enum cereal_CarState_GearShifter {
	cereal_CarState_GearShifter_unknown = 0,
	cereal_CarState_GearShifter_park = 1,
	cereal_CarState_GearShifter_drive = 2,
	cereal_CarState_GearShifter_neutral = 3,
	cereal_CarState_GearShifter_reverse = 4,
	cereal_CarState_GearShifter_sport = 5,
	cereal_CarState_GearShifter_low = 6,
	cereal_CarState_GearShifter_brake = 7,
	cereal_CarState_GearShifter_eco = 8,
	cereal_CarState_GearShifter_manumatic = 9
};

enum cereal_CarState_ButtonEvent_Type {
	cereal_CarState_ButtonEvent_Type_unknown = 0,
	cereal_CarState_ButtonEvent_Type_leftBlinker = 1,
	cereal_CarState_ButtonEvent_Type_rightBlinker = 2,
	cereal_CarState_ButtonEvent_Type_accelCruise = 3,
	cereal_CarState_ButtonEvent_Type_decelCruise = 4,
	cereal_CarState_ButtonEvent_Type_cancel = 5,
	cereal_CarState_ButtonEvent_Type_altButton1 = 6,
	cereal_CarState_ButtonEvent_Type_altButton2 = 7,
	cereal_CarState_ButtonEvent_Type_altButton3 = 8,
	cereal_CarState_ButtonEvent_Type_setCruise = 9,
	cereal_CarState_ButtonEvent_Type_resumeCruise = 10,
	cereal_CarState_ButtonEvent_Type_gapAdjustCruise = 11
};

enum cereal_RadarData_Error {
	cereal_RadarData_Error_canError = 0,
	cereal_RadarData_Error_fault = 1,
	cereal_RadarData_Error_wrongConfig = 2
};

enum cereal_CarControl_HUDControl_VisualAlert {
	cereal_CarControl_HUDControl_VisualAlert_none = 0,
	cereal_CarControl_HUDControl_VisualAlert_fcw = 1,
	cereal_CarControl_HUDControl_VisualAlert_steerRequired = 2,
	cereal_CarControl_HUDControl_VisualAlert_brakePressed = 3,
	cereal_CarControl_HUDControl_VisualAlert_wrongGear = 4,
	cereal_CarControl_HUDControl_VisualAlert_seatbeltUnbuckled = 5,
	cereal_CarControl_HUDControl_VisualAlert_speedTooHigh = 6,
	cereal_CarControl_HUDControl_VisualAlert_ldw = 7
};

enum cereal_CarControl_HUDControl_AudibleAlert {
	cereal_CarControl_HUDControl_AudibleAlert_none = 0,
	cereal_CarControl_HUDControl_AudibleAlert_chimeEngage = 1,
	cereal_CarControl_HUDControl_AudibleAlert_chimeDisengage = 2,
	cereal_CarControl_HUDControl_AudibleAlert_chimeError = 3,
	cereal_CarControl_HUDControl_AudibleAlert_chimeWarning1 = 4,
	cereal_CarControl_HUDControl_AudibleAlert_chimeWarning2 = 5,
	cereal_CarControl_HUDControl_AudibleAlert_chimeWarningRepeat = 6,
	cereal_CarControl_HUDControl_AudibleAlert_chimePrompt = 7
};

enum cereal_CarParams_SafetyModel {
	cereal_CarParams_SafetyModel_silent = 0,
	cereal_CarParams_SafetyModel_hondaNidec = 1,
	cereal_CarParams_SafetyModel_toyota = 2,
	cereal_CarParams_SafetyModel_elm327 = 3,
	cereal_CarParams_SafetyModel_gm = 4,
	cereal_CarParams_SafetyModel_hondaBoschGiraffe = 5,
	cereal_CarParams_SafetyModel_ford = 6,
	cereal_CarParams_SafetyModel_cadillac = 7,
	cereal_CarParams_SafetyModel_hyundai = 8,
	cereal_CarParams_SafetyModel_chrysler = 9,
	cereal_CarParams_SafetyModel_tesla = 10,
	cereal_CarParams_SafetyModel_subaru = 11,
	cereal_CarParams_SafetyModel_gmPassive = 12,
	cereal_CarParams_SafetyModel_mazda = 13,
	cereal_CarParams_SafetyModel_nissan = 14,
	cereal_CarParams_SafetyModel_volkswagen = 15,
	cereal_CarParams_SafetyModel_toyotaIpas = 16,
	cereal_CarParams_SafetyModel_allOutput = 17,
	cereal_CarParams_SafetyModel_gmAscm = 18,
	cereal_CarParams_SafetyModel_noOutput = 19,
	cereal_CarParams_SafetyModel_hondaBoschHarness = 20,
	cereal_CarParams_SafetyModel_volkswagenPq = 21
};

enum cereal_CarParams_SteerControlType {
	cereal_CarParams_SteerControlType_torque = 0,
	cereal_CarParams_SteerControlType_angle = 1
};

enum cereal_CarParams_TransmissionType {
	cereal_CarParams_TransmissionType_unknown = 0,
	cereal_CarParams_TransmissionType_automatic = 1,
	cereal_CarParams_TransmissionType_manual = 2
};

enum cereal_CarParams_Ecu {
	cereal_CarParams_Ecu_eps = 0,
	cereal_CarParams_Ecu_esp = 1,
	cereal_CarParams_Ecu_fwdRadar = 2,
	cereal_CarParams_Ecu_fwdCamera = 3,
	cereal_CarParams_Ecu_engine = 4,
	cereal_CarParams_Ecu_unknown = 5,
	cereal_CarParams_Ecu_dsu = 6,
	cereal_CarParams_Ecu_apgs = 7
};

enum cereal_CarParams_FingerprintSource {
	cereal_CarParams_FingerprintSource_can = 0,
	cereal_CarParams_FingerprintSource_fw = 1,
	cereal_CarParams_FingerprintSource_fixed = 2
};

struct cereal_CarEvent {
	enum cereal_CarEvent_EventName name;
	unsigned enable : 1;
	unsigned noEntry : 1;
	unsigned warning : 1;
	unsigned userDisable : 1;
	unsigned softDisable : 1;
	unsigned immediateDisable : 1;
	unsigned preEnable : 1;
	unsigned permanent : 1;
};

static const size_t cereal_CarEvent_word_count = 1;

static const size_t cereal_CarEvent_pointer_count = 0;

static const size_t cereal_CarEvent_struct_bytes_count = 8;

struct cereal_CarState {
	capn_list16 errorsDEPRECATED;
	cereal_CarEvent_list events;
	float vEgo;
	float aEgo;
	float vEgoRaw;
	float yawRate;
	unsigned standstill : 1;
	cereal_CarState_WheelSpeeds_ptr wheelSpeeds;
	float gas;
	unsigned gasPressed : 1;
	float brake;
	unsigned brakePressed : 1;
	unsigned brakeLights : 1;
	float steeringAngle;
	float steeringRate;
	float steeringTorque;
	float steeringTorqueEps;
	unsigned steeringPressed : 1;
	unsigned steeringRateLimited : 1;
	unsigned stockAeb : 1;
	unsigned stockFcw : 1;
	cereal_CarState_CruiseState_ptr cruiseState;
	enum cereal_CarState_GearShifter gearShifter;
	cereal_CarState_ButtonEvent_list buttonEvents;
	unsigned leftBlinker : 1;
	unsigned rightBlinker : 1;
	unsigned genericToggle : 1;
	unsigned doorOpen : 1;
	unsigned seatbeltUnlatched : 1;
	unsigned canValid : 1;
	unsigned clutchPressed : 1;
	capn_list64 canMonoTimes;
};

static const size_t cereal_CarState_word_count = 6;

static const size_t cereal_CarState_pointer_count = 6;

static const size_t cereal_CarState_struct_bytes_count = 96;

struct cereal_CarState_WheelSpeeds {
	float fl;
	float fr;
	float rl;
	float rr;
};

static const size_t cereal_CarState_WheelSpeeds_word_count = 2;

static const size_t cereal_CarState_WheelSpeeds_pointer_count = 0;

static const size_t cereal_CarState_WheelSpeeds_struct_bytes_count = 16;

struct cereal_CarState_CruiseState {
	unsigned enabled : 1;
	float speed;
	unsigned available : 1;
	float speedOffset;
	unsigned standstill : 1;
};

static const size_t cereal_CarState_CruiseState_word_count = 2;

static const size_t cereal_CarState_CruiseState_pointer_count = 0;

static const size_t cereal_CarState_CruiseState_struct_bytes_count = 16;

struct cereal_CarState_ButtonEvent {
	unsigned pressed : 1;
	enum cereal_CarState_ButtonEvent_Type type;
};

static const size_t cereal_CarState_ButtonEvent_word_count = 1;

static const size_t cereal_CarState_ButtonEvent_pointer_count = 0;

static const size_t cereal_CarState_ButtonEvent_struct_bytes_count = 8;

struct cereal_RadarData {
	capn_list16 errors;
	cereal_RadarData_RadarPoint_list points;
	capn_list64 canMonoTimes;
};

static const size_t cereal_RadarData_word_count = 0;

static const size_t cereal_RadarData_pointer_count = 3;

static const size_t cereal_RadarData_struct_bytes_count = 24;

struct cereal_RadarData_RadarPoint {
	uint64_t trackId;
	float dRel;
	float yRel;
	float vRel;
	float aRel;
	float yvRel;
	unsigned measured : 1;
};

static const size_t cereal_RadarData_RadarPoint_word_count = 4;

static const size_t cereal_RadarData_RadarPoint_pointer_count = 0;

static const size_t cereal_RadarData_RadarPoint_struct_bytes_count = 32;

struct cereal_CarControl {
	unsigned enabled : 1;
	unsigned active : 1;
	float gasDEPRECATED;
	float brakeDEPRECATED;
	float steeringTorqueDEPRECATED;
	cereal_CarControl_Actuators_ptr actuators;
	cereal_CarControl_CruiseControl_ptr cruiseControl;
	cereal_CarControl_HUDControl_ptr hudControl;
};

static const size_t cereal_CarControl_word_count = 2;

static const size_t cereal_CarControl_pointer_count = 3;

static const size_t cereal_CarControl_struct_bytes_count = 40;

struct cereal_CarControl_Actuators {
	float gas;
	float brake;
	float steer;
	float steerAngle;
};

static const size_t cereal_CarControl_Actuators_word_count = 2;

static const size_t cereal_CarControl_Actuators_pointer_count = 0;

static const size_t cereal_CarControl_Actuators_struct_bytes_count = 16;

struct cereal_CarControl_CruiseControl {
	unsigned cancel : 1;
	unsigned override : 1;
	float speedOverride;
	float accelOverride;
};

static const size_t cereal_CarControl_CruiseControl_word_count = 2;

static const size_t cereal_CarControl_CruiseControl_pointer_count = 0;

static const size_t cereal_CarControl_CruiseControl_struct_bytes_count = 16;

struct cereal_CarControl_HUDControl {
	unsigned speedVisible : 1;
	float setSpeed;
	unsigned lanesVisible : 1;
	unsigned leadVisible : 1;
	enum cereal_CarControl_HUDControl_VisualAlert visualAlert;
	enum cereal_CarControl_HUDControl_AudibleAlert audibleAlert;
	unsigned rightLaneVisible : 1;
	unsigned leftLaneVisible : 1;
	unsigned rightLaneDepart : 1;
	unsigned leftLaneDepart : 1;
};

static const size_t cereal_CarControl_HUDControl_word_count = 2;

static const size_t cereal_CarControl_HUDControl_pointer_count = 0;

static const size_t cereal_CarControl_HUDControl_struct_bytes_count = 16;
enum cereal_CarParams_lateralTuning_which {
	cereal_CarParams_lateralTuning_pid = 0,
	cereal_CarParams_lateralTuning_indi = 1,
	cereal_CarParams_lateralTuning_lqr = 2
};

struct cereal_CarParams {
	capn_text carName;
	capn_text carFingerprint;
	unsigned enableGasInterceptor : 1;
	unsigned enableCruise : 1;
	unsigned enableCamera : 1;
	unsigned enableDsu : 1;
	unsigned enableApgs : 1;
	float minEnableSpeed;
	float minSteerSpeed;
	enum cereal_CarParams_SafetyModel safetyModel;
	enum cereal_CarParams_SafetyModel safetyModelPassive;
	int16_t safetyParam;
	capn_list32 steerMaxBP;
	capn_list32 steerMaxV;
	capn_list32 gasMaxBP;
	capn_list32 gasMaxV;
	capn_list32 brakeMaxBP;
	capn_list32 brakeMaxV;
	float mass;
	float wheelbase;
	float centerToFront;
	float steerRatio;
	float steerRatioRear;
	float rotationalInertia;
	float tireStiffnessFront;
	float tireStiffnessRear;
	cereal_CarParams_LongitudinalPIDTuning_ptr longitudinalTuning;
	cereal_CarParams_LateralParams_ptr lateralParams;
	enum cereal_CarParams_lateralTuning_which lateralTuning_which;
	union {
		cereal_CarParams_LateralPIDTuning_ptr pid;
		cereal_CarParams_LateralINDITuning_ptr indi;
		cereal_CarParams_LateralLQRTuning_ptr lqr;
	} lateralTuning;
	unsigned steerLimitAlert : 1;
	float steerLimitTimer;
	float vEgoStopping;
	unsigned directAccelControl : 1;
	unsigned stoppingControl : 1;
	float startAccel;
	float steerRateCost;
	enum cereal_CarParams_SteerControlType steerControlType;
	unsigned radarOffCan : 1;
	float steerActuatorDelay;
	unsigned openpilotLongitudinalControl : 1;
	capn_text carVin;
	unsigned isPandaBlack : 1;
	unsigned dashcamOnly : 1;
	enum cereal_CarParams_TransmissionType transmissionType;
	cereal_CarParams_CarFw_list carFw;
	float radarTimeStep;
	unsigned communityFeature : 1;
	enum cereal_CarParams_FingerprintSource fingerprintSource;
};

static const size_t cereal_CarParams_word_count = 10;

static const size_t cereal_CarParams_pointer_count = 13;

static const size_t cereal_CarParams_struct_bytes_count = 184;

struct cereal_CarParams_LateralParams {
	capn_list32 torqueBP;
	capn_list32 torqueV;
};

static const size_t cereal_CarParams_LateralParams_word_count = 0;

static const size_t cereal_CarParams_LateralParams_pointer_count = 2;

static const size_t cereal_CarParams_LateralParams_struct_bytes_count = 16;

struct cereal_CarParams_LateralPIDTuning {
	capn_list32 kpBP;
	capn_list32 kpV;
	capn_list32 kiBP;
	capn_list32 kiV;
	float kf;
};

static const size_t cereal_CarParams_LateralPIDTuning_word_count = 1;

static const size_t cereal_CarParams_LateralPIDTuning_pointer_count = 4;

static const size_t cereal_CarParams_LateralPIDTuning_struct_bytes_count = 40;

struct cereal_CarParams_LongitudinalPIDTuning {
	capn_list32 kpBP;
	capn_list32 kpV;
	capn_list32 kiBP;
	capn_list32 kiV;
	capn_list32 deadzoneBP;
	capn_list32 deadzoneV;
};

static const size_t cereal_CarParams_LongitudinalPIDTuning_word_count = 0;

static const size_t cereal_CarParams_LongitudinalPIDTuning_pointer_count = 6;

static const size_t cereal_CarParams_LongitudinalPIDTuning_struct_bytes_count = 48;

struct cereal_CarParams_LateralINDITuning {
	float outerLoopGain;
	float innerLoopGain;
	float timeConstant;
	float actuatorEffectiveness;
};

static const size_t cereal_CarParams_LateralINDITuning_word_count = 2;

static const size_t cereal_CarParams_LateralINDITuning_pointer_count = 0;

static const size_t cereal_CarParams_LateralINDITuning_struct_bytes_count = 16;

struct cereal_CarParams_LateralLQRTuning {
	float scale;
	float ki;
	float dcGain;
	capn_list32 a;
	capn_list32 b;
	capn_list32 c;
	capn_list32 k;
	capn_list32 l;
};

static const size_t cereal_CarParams_LateralLQRTuning_word_count = 2;

static const size_t cereal_CarParams_LateralLQRTuning_pointer_count = 5;

static const size_t cereal_CarParams_LateralLQRTuning_struct_bytes_count = 56;

struct cereal_CarParams_CarFw {
	enum cereal_CarParams_Ecu ecu;
	capn_data fwVersion;
	uint32_t address;
	uint8_t subAddress;
};

static const size_t cereal_CarParams_CarFw_word_count = 1;

static const size_t cereal_CarParams_CarFw_pointer_count = 1;

static const size_t cereal_CarParams_CarFw_struct_bytes_count = 16;

cereal_CarEvent_ptr cereal_new_CarEvent(struct capn_segment*);
cereal_CarState_ptr cereal_new_CarState(struct capn_segment*);
cereal_CarState_WheelSpeeds_ptr cereal_new_CarState_WheelSpeeds(struct capn_segment*);
cereal_CarState_CruiseState_ptr cereal_new_CarState_CruiseState(struct capn_segment*);
cereal_CarState_ButtonEvent_ptr cereal_new_CarState_ButtonEvent(struct capn_segment*);
cereal_RadarData_ptr cereal_new_RadarData(struct capn_segment*);
cereal_RadarData_RadarPoint_ptr cereal_new_RadarData_RadarPoint(struct capn_segment*);
cereal_CarControl_ptr cereal_new_CarControl(struct capn_segment*);
cereal_CarControl_Actuators_ptr cereal_new_CarControl_Actuators(struct capn_segment*);
cereal_CarControl_CruiseControl_ptr cereal_new_CarControl_CruiseControl(struct capn_segment*);
cereal_CarControl_HUDControl_ptr cereal_new_CarControl_HUDControl(struct capn_segment*);
cereal_CarParams_ptr cereal_new_CarParams(struct capn_segment*);
cereal_CarParams_LateralParams_ptr cereal_new_CarParams_LateralParams(struct capn_segment*);
cereal_CarParams_LateralPIDTuning_ptr cereal_new_CarParams_LateralPIDTuning(struct capn_segment*);
cereal_CarParams_LongitudinalPIDTuning_ptr cereal_new_CarParams_LongitudinalPIDTuning(struct capn_segment*);
cereal_CarParams_LateralINDITuning_ptr cereal_new_CarParams_LateralINDITuning(struct capn_segment*);
cereal_CarParams_LateralLQRTuning_ptr cereal_new_CarParams_LateralLQRTuning(struct capn_segment*);
cereal_CarParams_CarFw_ptr cereal_new_CarParams_CarFw(struct capn_segment*);

cereal_CarEvent_list cereal_new_CarEvent_list(struct capn_segment*, int len);
cereal_CarState_list cereal_new_CarState_list(struct capn_segment*, int len);
cereal_CarState_WheelSpeeds_list cereal_new_CarState_WheelSpeeds_list(struct capn_segment*, int len);
cereal_CarState_CruiseState_list cereal_new_CarState_CruiseState_list(struct capn_segment*, int len);
cereal_CarState_ButtonEvent_list cereal_new_CarState_ButtonEvent_list(struct capn_segment*, int len);
cereal_RadarData_list cereal_new_RadarData_list(struct capn_segment*, int len);
cereal_RadarData_RadarPoint_list cereal_new_RadarData_RadarPoint_list(struct capn_segment*, int len);
cereal_CarControl_list cereal_new_CarControl_list(struct capn_segment*, int len);
cereal_CarControl_Actuators_list cereal_new_CarControl_Actuators_list(struct capn_segment*, int len);
cereal_CarControl_CruiseControl_list cereal_new_CarControl_CruiseControl_list(struct capn_segment*, int len);
cereal_CarControl_HUDControl_list cereal_new_CarControl_HUDControl_list(struct capn_segment*, int len);
cereal_CarParams_list cereal_new_CarParams_list(struct capn_segment*, int len);
cereal_CarParams_LateralParams_list cereal_new_CarParams_LateralParams_list(struct capn_segment*, int len);
cereal_CarParams_LateralPIDTuning_list cereal_new_CarParams_LateralPIDTuning_list(struct capn_segment*, int len);
cereal_CarParams_LongitudinalPIDTuning_list cereal_new_CarParams_LongitudinalPIDTuning_list(struct capn_segment*, int len);
cereal_CarParams_LateralINDITuning_list cereal_new_CarParams_LateralINDITuning_list(struct capn_segment*, int len);
cereal_CarParams_LateralLQRTuning_list cereal_new_CarParams_LateralLQRTuning_list(struct capn_segment*, int len);
cereal_CarParams_CarFw_list cereal_new_CarParams_CarFw_list(struct capn_segment*, int len);

void cereal_read_CarEvent(struct cereal_CarEvent*, cereal_CarEvent_ptr);
void cereal_read_CarState(struct cereal_CarState*, cereal_CarState_ptr);
void cereal_read_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_ptr);
void cereal_read_CarState_CruiseState(struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_ptr);
void cereal_read_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_ptr);
void cereal_read_RadarData(struct cereal_RadarData*, cereal_RadarData_ptr);
void cereal_read_RadarData_RadarPoint(struct cereal_RadarData_RadarPoint*, cereal_RadarData_RadarPoint_ptr);
void cereal_read_CarControl(struct cereal_CarControl*, cereal_CarControl_ptr);
void cereal_read_CarControl_Actuators(struct cereal_CarControl_Actuators*, cereal_CarControl_Actuators_ptr);
void cereal_read_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_ptr);
void cereal_read_CarControl_HUDControl(struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_ptr);
void cereal_read_CarParams(struct cereal_CarParams*, cereal_CarParams_ptr);
void cereal_read_CarParams_LateralParams(struct cereal_CarParams_LateralParams*, cereal_CarParams_LateralParams_ptr);
void cereal_read_CarParams_LateralPIDTuning(struct cereal_CarParams_LateralPIDTuning*, cereal_CarParams_LateralPIDTuning_ptr);
void cereal_read_CarParams_LongitudinalPIDTuning(struct cereal_CarParams_LongitudinalPIDTuning*, cereal_CarParams_LongitudinalPIDTuning_ptr);
void cereal_read_CarParams_LateralINDITuning(struct cereal_CarParams_LateralINDITuning*, cereal_CarParams_LateralINDITuning_ptr);
void cereal_read_CarParams_LateralLQRTuning(struct cereal_CarParams_LateralLQRTuning*, cereal_CarParams_LateralLQRTuning_ptr);
void cereal_read_CarParams_CarFw(struct cereal_CarParams_CarFw*, cereal_CarParams_CarFw_ptr);

void cereal_write_CarEvent(const struct cereal_CarEvent*, cereal_CarEvent_ptr);
void cereal_write_CarState(const struct cereal_CarState*, cereal_CarState_ptr);
void cereal_write_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_ptr);
void cereal_write_CarState_CruiseState(const struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_ptr);
void cereal_write_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_ptr);
void cereal_write_RadarData(const struct cereal_RadarData*, cereal_RadarData_ptr);
void cereal_write_RadarData_RadarPoint(const struct cereal_RadarData_RadarPoint*, cereal_RadarData_RadarPoint_ptr);
void cereal_write_CarControl(const struct cereal_CarControl*, cereal_CarControl_ptr);
void cereal_write_CarControl_Actuators(const struct cereal_CarControl_Actuators*, cereal_CarControl_Actuators_ptr);
void cereal_write_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_ptr);
void cereal_write_CarControl_HUDControl(const struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_ptr);
void cereal_write_CarParams(const struct cereal_CarParams*, cereal_CarParams_ptr);
void cereal_write_CarParams_LateralParams(const struct cereal_CarParams_LateralParams*, cereal_CarParams_LateralParams_ptr);
void cereal_write_CarParams_LateralPIDTuning(const struct cereal_CarParams_LateralPIDTuning*, cereal_CarParams_LateralPIDTuning_ptr);
void cereal_write_CarParams_LongitudinalPIDTuning(const struct cereal_CarParams_LongitudinalPIDTuning*, cereal_CarParams_LongitudinalPIDTuning_ptr);
void cereal_write_CarParams_LateralINDITuning(const struct cereal_CarParams_LateralINDITuning*, cereal_CarParams_LateralINDITuning_ptr);
void cereal_write_CarParams_LateralLQRTuning(const struct cereal_CarParams_LateralLQRTuning*, cereal_CarParams_LateralLQRTuning_ptr);
void cereal_write_CarParams_CarFw(const struct cereal_CarParams_CarFw*, cereal_CarParams_CarFw_ptr);

void cereal_get_CarEvent(struct cereal_CarEvent*, cereal_CarEvent_list, int i);
void cereal_get_CarState(struct cereal_CarState*, cereal_CarState_list, int i);
void cereal_get_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_list, int i);
void cereal_get_CarState_CruiseState(struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_list, int i);
void cereal_get_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_list, int i);
void cereal_get_RadarData(struct cereal_RadarData*, cereal_RadarData_list, int i);
void cereal_get_RadarData_RadarPoint(struct cereal_RadarData_RadarPoint*, cereal_RadarData_RadarPoint_list, int i);
void cereal_get_CarControl(struct cereal_CarControl*, cereal_CarControl_list, int i);
void cereal_get_CarControl_Actuators(struct cereal_CarControl_Actuators*, cereal_CarControl_Actuators_list, int i);
void cereal_get_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_list, int i);
void cereal_get_CarControl_HUDControl(struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_list, int i);
void cereal_get_CarParams(struct cereal_CarParams*, cereal_CarParams_list, int i);
void cereal_get_CarParams_LateralParams(struct cereal_CarParams_LateralParams*, cereal_CarParams_LateralParams_list, int i);
void cereal_get_CarParams_LateralPIDTuning(struct cereal_CarParams_LateralPIDTuning*, cereal_CarParams_LateralPIDTuning_list, int i);
void cereal_get_CarParams_LongitudinalPIDTuning(struct cereal_CarParams_LongitudinalPIDTuning*, cereal_CarParams_LongitudinalPIDTuning_list, int i);
void cereal_get_CarParams_LateralINDITuning(struct cereal_CarParams_LateralINDITuning*, cereal_CarParams_LateralINDITuning_list, int i);
void cereal_get_CarParams_LateralLQRTuning(struct cereal_CarParams_LateralLQRTuning*, cereal_CarParams_LateralLQRTuning_list, int i);
void cereal_get_CarParams_CarFw(struct cereal_CarParams_CarFw*, cereal_CarParams_CarFw_list, int i);

void cereal_set_CarEvent(const struct cereal_CarEvent*, cereal_CarEvent_list, int i);
void cereal_set_CarState(const struct cereal_CarState*, cereal_CarState_list, int i);
void cereal_set_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_list, int i);
void cereal_set_CarState_CruiseState(const struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_list, int i);
void cereal_set_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_list, int i);
void cereal_set_RadarData(const struct cereal_RadarData*, cereal_RadarData_list, int i);
void cereal_set_RadarData_RadarPoint(const struct cereal_RadarData_RadarPoint*, cereal_RadarData_RadarPoint_list, int i);
void cereal_set_CarControl(const struct cereal_CarControl*, cereal_CarControl_list, int i);
void cereal_set_CarControl_Actuators(const struct cereal_CarControl_Actuators*, cereal_CarControl_Actuators_list, int i);
void cereal_set_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_list, int i);
void cereal_set_CarControl_HUDControl(const struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_list, int i);
void cereal_set_CarParams(const struct cereal_CarParams*, cereal_CarParams_list, int i);
void cereal_set_CarParams_LateralParams(const struct cereal_CarParams_LateralParams*, cereal_CarParams_LateralParams_list, int i);
void cereal_set_CarParams_LateralPIDTuning(const struct cereal_CarParams_LateralPIDTuning*, cereal_CarParams_LateralPIDTuning_list, int i);
void cereal_set_CarParams_LongitudinalPIDTuning(const struct cereal_CarParams_LongitudinalPIDTuning*, cereal_CarParams_LongitudinalPIDTuning_list, int i);
void cereal_set_CarParams_LateralINDITuning(const struct cereal_CarParams_LateralINDITuning*, cereal_CarParams_LateralINDITuning_list, int i);
void cereal_set_CarParams_LateralLQRTuning(const struct cereal_CarParams_LateralLQRTuning*, cereal_CarParams_LateralLQRTuning_list, int i);
void cereal_set_CarParams_CarFw(const struct cereal_CarParams_CarFw*, cereal_CarParams_CarFw_list, int i);

#ifdef __cplusplus
}
#endif
#endif
