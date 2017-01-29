#ifndef CAPN_8E2AF1E78AF8B8D
#define CAPN_8E2AF1E78AF8B8D
/* AUTO GENERATED - DO NOT EDIT */
#include <capnp_c.h>

#if CAPN_VERSION != 1
#error "version mismatch between capnp_c.h and generated code"
#endif

#include "c++.capnp.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cereal_CarState;
struct cereal_CarState_WheelSpeeds;
struct cereal_CarState_CruiseState;
struct cereal_CarState_ButtonEvent;
struct cereal_RadarState;
struct cereal_RadarState_RadarPoint;
struct cereal_CarControl;
struct cereal_CarControl_CruiseControl;
struct cereal_CarControl_HUDControl;

typedef struct {capn_ptr p;} cereal_CarState_ptr;
typedef struct {capn_ptr p;} cereal_CarState_WheelSpeeds_ptr;
typedef struct {capn_ptr p;} cereal_CarState_CruiseState_ptr;
typedef struct {capn_ptr p;} cereal_CarState_ButtonEvent_ptr;
typedef struct {capn_ptr p;} cereal_RadarState_ptr;
typedef struct {capn_ptr p;} cereal_RadarState_RadarPoint_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_CruiseControl_ptr;
typedef struct {capn_ptr p;} cereal_CarControl_HUDControl_ptr;

typedef struct {capn_ptr p;} cereal_CarState_list;
typedef struct {capn_ptr p;} cereal_CarState_WheelSpeeds_list;
typedef struct {capn_ptr p;} cereal_CarState_CruiseState_list;
typedef struct {capn_ptr p;} cereal_CarState_ButtonEvent_list;
typedef struct {capn_ptr p;} cereal_RadarState_list;
typedef struct {capn_ptr p;} cereal_RadarState_RadarPoint_list;
typedef struct {capn_ptr p;} cereal_CarControl_list;
typedef struct {capn_ptr p;} cereal_CarControl_CruiseControl_list;
typedef struct {capn_ptr p;} cereal_CarControl_HUDControl_list;

enum cereal_CarState_Error {
	cereal_CarState_Error_commIssue = 0,
	cereal_CarState_Error_steerUnavailable = 1,
	cereal_CarState_Error_brakeUnavailable = 2,
	cereal_CarState_Error_gasUnavailable = 3,
	cereal_CarState_Error_wrongGear = 4,
	cereal_CarState_Error_doorOpen = 5,
	cereal_CarState_Error_seatbeltNotLatched = 6,
	cereal_CarState_Error_espDisabled = 7,
	cereal_CarState_Error_wrongCarMode = 8,
	cereal_CarState_Error_steerTemporarilyUnavailable = 9,
	cereal_CarState_Error_reverseGear = 10
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
	cereal_CarState_ButtonEvent_Type_altButton3 = 8
};

enum cereal_RadarState_Error {
	cereal_RadarState_Error_notValid = 0
};

enum cereal_CarControl_HUDControl_VisualAlert {
	cereal_CarControl_HUDControl_VisualAlert_none = 0,
	cereal_CarControl_HUDControl_VisualAlert_fcw = 1,
	cereal_CarControl_HUDControl_VisualAlert_steerRequired = 2,
	cereal_CarControl_HUDControl_VisualAlert_brakePressed = 3,
	cereal_CarControl_HUDControl_VisualAlert_wrongGear = 4,
	cereal_CarControl_HUDControl_VisualAlert_seatbeltUnbuckled = 5,
	cereal_CarControl_HUDControl_VisualAlert_speedTooHigh = 6
};

enum cereal_CarControl_HUDControl_AudibleAlert {
	cereal_CarControl_HUDControl_AudibleAlert_none = 0,
	cereal_CarControl_HUDControl_AudibleAlert_beepSingle = 1,
	cereal_CarControl_HUDControl_AudibleAlert_beepTriple = 2,
	cereal_CarControl_HUDControl_AudibleAlert_beepRepeated = 3,
	cereal_CarControl_HUDControl_AudibleAlert_chimeSingle = 4,
	cereal_CarControl_HUDControl_AudibleAlert_chimeDouble = 5,
	cereal_CarControl_HUDControl_AudibleAlert_chimeRepeated = 6,
	cereal_CarControl_HUDControl_AudibleAlert_chimeContinuous = 7
};

struct cereal_CarState {
	capn_list16 errors;
	float vEgo;
	cereal_CarState_WheelSpeeds_ptr wheelSpeeds;
	float gas;
	unsigned gasPressed : 1;
	float brake;
	unsigned brakePressed : 1;
	float steeringAngle;
	float steeringTorque;
	unsigned steeringPressed : 1;
	cereal_CarState_CruiseState_ptr cruiseState;
	cereal_CarState_ButtonEvent_list buttonEvents;
	capn_list64 canMonoTimes;
};

static const size_t cereal_CarState_word_count = 3;

static const size_t cereal_CarState_pointer_count = 5;

static const size_t cereal_CarState_struct_bytes_count = 64;

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
};

static const size_t cereal_CarState_CruiseState_word_count = 1;

static const size_t cereal_CarState_CruiseState_pointer_count = 0;

static const size_t cereal_CarState_CruiseState_struct_bytes_count = 8;

struct cereal_CarState_ButtonEvent {
	unsigned pressed : 1;
	enum cereal_CarState_ButtonEvent_Type type;
};

static const size_t cereal_CarState_ButtonEvent_word_count = 1;

static const size_t cereal_CarState_ButtonEvent_pointer_count = 0;

static const size_t cereal_CarState_ButtonEvent_struct_bytes_count = 8;

struct cereal_RadarState {
	capn_list16 errors;
	cereal_RadarState_RadarPoint_list points;
	capn_list64 canMonoTimes;
};

static const size_t cereal_RadarState_word_count = 0;

static const size_t cereal_RadarState_pointer_count = 3;

static const size_t cereal_RadarState_struct_bytes_count = 24;

struct cereal_RadarState_RadarPoint {
	uint64_t trackId;
	float dRel;
	float yRel;
	float vRel;
	float aRel;
	float yvRel;
};

static const size_t cereal_RadarState_RadarPoint_word_count = 4;

static const size_t cereal_RadarState_RadarPoint_pointer_count = 0;

static const size_t cereal_RadarState_RadarPoint_struct_bytes_count = 32;

struct cereal_CarControl {
	unsigned enabled : 1;
	float gas;
	float brake;
	float steeringTorque;
	cereal_CarControl_CruiseControl_ptr cruiseControl;
	cereal_CarControl_HUDControl_ptr hudControl;
};

static const size_t cereal_CarControl_word_count = 2;

static const size_t cereal_CarControl_pointer_count = 2;

static const size_t cereal_CarControl_struct_bytes_count = 32;

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
};

static const size_t cereal_CarControl_HUDControl_word_count = 2;

static const size_t cereal_CarControl_HUDControl_pointer_count = 0;

static const size_t cereal_CarControl_HUDControl_struct_bytes_count = 16;

cereal_CarState_ptr cereal_new_CarState(struct capn_segment*);
cereal_CarState_WheelSpeeds_ptr cereal_new_CarState_WheelSpeeds(struct capn_segment*);
cereal_CarState_CruiseState_ptr cereal_new_CarState_CruiseState(struct capn_segment*);
cereal_CarState_ButtonEvent_ptr cereal_new_CarState_ButtonEvent(struct capn_segment*);
cereal_RadarState_ptr cereal_new_RadarState(struct capn_segment*);
cereal_RadarState_RadarPoint_ptr cereal_new_RadarState_RadarPoint(struct capn_segment*);
cereal_CarControl_ptr cereal_new_CarControl(struct capn_segment*);
cereal_CarControl_CruiseControl_ptr cereal_new_CarControl_CruiseControl(struct capn_segment*);
cereal_CarControl_HUDControl_ptr cereal_new_CarControl_HUDControl(struct capn_segment*);

cereal_CarState_list cereal_new_CarState_list(struct capn_segment*, int len);
cereal_CarState_WheelSpeeds_list cereal_new_CarState_WheelSpeeds_list(struct capn_segment*, int len);
cereal_CarState_CruiseState_list cereal_new_CarState_CruiseState_list(struct capn_segment*, int len);
cereal_CarState_ButtonEvent_list cereal_new_CarState_ButtonEvent_list(struct capn_segment*, int len);
cereal_RadarState_list cereal_new_RadarState_list(struct capn_segment*, int len);
cereal_RadarState_RadarPoint_list cereal_new_RadarState_RadarPoint_list(struct capn_segment*, int len);
cereal_CarControl_list cereal_new_CarControl_list(struct capn_segment*, int len);
cereal_CarControl_CruiseControl_list cereal_new_CarControl_CruiseControl_list(struct capn_segment*, int len);
cereal_CarControl_HUDControl_list cereal_new_CarControl_HUDControl_list(struct capn_segment*, int len);

void cereal_read_CarState(struct cereal_CarState*, cereal_CarState_ptr);
void cereal_read_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_ptr);
void cereal_read_CarState_CruiseState(struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_ptr);
void cereal_read_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_ptr);
void cereal_read_RadarState(struct cereal_RadarState*, cereal_RadarState_ptr);
void cereal_read_RadarState_RadarPoint(struct cereal_RadarState_RadarPoint*, cereal_RadarState_RadarPoint_ptr);
void cereal_read_CarControl(struct cereal_CarControl*, cereal_CarControl_ptr);
void cereal_read_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_ptr);
void cereal_read_CarControl_HUDControl(struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_ptr);

void cereal_write_CarState(const struct cereal_CarState*, cereal_CarState_ptr);
void cereal_write_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_ptr);
void cereal_write_CarState_CruiseState(const struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_ptr);
void cereal_write_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_ptr);
void cereal_write_RadarState(const struct cereal_RadarState*, cereal_RadarState_ptr);
void cereal_write_RadarState_RadarPoint(const struct cereal_RadarState_RadarPoint*, cereal_RadarState_RadarPoint_ptr);
void cereal_write_CarControl(const struct cereal_CarControl*, cereal_CarControl_ptr);
void cereal_write_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_ptr);
void cereal_write_CarControl_HUDControl(const struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_ptr);

void cereal_get_CarState(struct cereal_CarState*, cereal_CarState_list, int i);
void cereal_get_CarState_WheelSpeeds(struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_list, int i);
void cereal_get_CarState_CruiseState(struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_list, int i);
void cereal_get_CarState_ButtonEvent(struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_list, int i);
void cereal_get_RadarState(struct cereal_RadarState*, cereal_RadarState_list, int i);
void cereal_get_RadarState_RadarPoint(struct cereal_RadarState_RadarPoint*, cereal_RadarState_RadarPoint_list, int i);
void cereal_get_CarControl(struct cereal_CarControl*, cereal_CarControl_list, int i);
void cereal_get_CarControl_CruiseControl(struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_list, int i);
void cereal_get_CarControl_HUDControl(struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_list, int i);

void cereal_set_CarState(const struct cereal_CarState*, cereal_CarState_list, int i);
void cereal_set_CarState_WheelSpeeds(const struct cereal_CarState_WheelSpeeds*, cereal_CarState_WheelSpeeds_list, int i);
void cereal_set_CarState_CruiseState(const struct cereal_CarState_CruiseState*, cereal_CarState_CruiseState_list, int i);
void cereal_set_CarState_ButtonEvent(const struct cereal_CarState_ButtonEvent*, cereal_CarState_ButtonEvent_list, int i);
void cereal_set_RadarState(const struct cereal_RadarState*, cereal_RadarState_list, int i);
void cereal_set_RadarState_RadarPoint(const struct cereal_RadarState_RadarPoint*, cereal_RadarState_RadarPoint_list, int i);
void cereal_set_CarControl(const struct cereal_CarControl*, cereal_CarControl_list, int i);
void cereal_set_CarControl_CruiseControl(const struct cereal_CarControl_CruiseControl*, cereal_CarControl_CruiseControl_list, int i);
void cereal_set_CarControl_HUDControl(const struct cereal_CarControl_HUDControl*, cereal_CarControl_HUDControl_list, int i);

#ifdef __cplusplus
}
#endif
#endif
