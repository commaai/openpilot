#ifndef CAPN_F3B1F17E25A4285B
#define CAPN_F3B1F17E25A4285B
/* AUTO GENERATED - DO NOT EDIT */
#include <capnp_c.h>

#if CAPN_VERSION != 1
#error "version mismatch between capnp_c.h and generated code"
#endif

#include "./include/c++.capnp.h"
#include "./include/java.capnp.h"
#include "car.capnp.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cereal_Map;
struct cereal_Map_Entry;
struct cereal_InitData;
struct cereal_InitData_AndroidBuildInfo;
struct cereal_InitData_AndroidSensor;
struct cereal_InitData_ChffrAndroidExtra;
struct cereal_InitData_IosBuildInfo;
struct cereal_InitData_PandaInfo;
struct cereal_FrameData;
struct cereal_FrameData_AndroidCaptureResult;
struct cereal_Thumbnail;
struct cereal_GPSNMEAData;
struct cereal_SensorEventData;
struct cereal_SensorEventData_SensorVec;
struct cereal_GpsLocationData;
struct cereal_CanData;
struct cereal_ThermalData;
struct cereal_HealthData;
struct cereal_LiveUI;
struct cereal_RadarState;
struct cereal_RadarState_LeadData;
struct cereal_LiveCalibrationData;
struct cereal_LiveTracks;
struct cereal_ControlsState;
struct cereal_ControlsState_LateralINDIState;
struct cereal_ControlsState_LateralPIDState;
struct cereal_ControlsState_LateralLQRState;
struct cereal_LiveEventData;
struct cereal_ModelData;
struct cereal_ModelData_PathData;
struct cereal_ModelData_LeadData;
struct cereal_ModelData_ModelSettings;
struct cereal_ModelData_MetaData;
struct cereal_ModelData_LongitudinalData;
struct cereal_CalibrationFeatures;
struct cereal_EncodeIndex;
struct cereal_AndroidLogEntry;
struct cereal_LogRotate;
struct cereal_Plan;
struct cereal_Plan_GpsTrajectory;
struct cereal_PathPlan;
struct cereal_LiveLocationData;
struct cereal_LiveLocationData_Accuracy;
struct cereal_EthernetPacket;
struct cereal_NavUpdate;
struct cereal_NavUpdate_LatLng;
struct cereal_NavUpdate_Segment;
struct cereal_NavStatus;
struct cereal_NavStatus_Address;
struct cereal_CellInfo;
struct cereal_WifiScan;
struct cereal_AndroidGnss;
struct cereal_AndroidGnss_Measurements;
struct cereal_AndroidGnss_Measurements_Clock;
struct cereal_AndroidGnss_Measurements_Measurement;
struct cereal_AndroidGnss_NavigationMessage;
struct cereal_QcomGnss;
struct cereal_QcomGnss_MeasurementStatus;
struct cereal_QcomGnss_MeasurementReport;
struct cereal_QcomGnss_MeasurementReport_SV;
struct cereal_QcomGnss_ClockReport;
struct cereal_QcomGnss_DrMeasurementReport;
struct cereal_QcomGnss_DrMeasurementReport_SV;
struct cereal_QcomGnss_DrSvPolyReport;
struct cereal_LidarPts;
struct cereal_ProcLog;
struct cereal_ProcLog_Process;
struct cereal_ProcLog_CPUTimes;
struct cereal_ProcLog_Mem;
struct cereal_UbloxGnss;
struct cereal_UbloxGnss_MeasurementReport;
struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus;
struct cereal_UbloxGnss_MeasurementReport_Measurement;
struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus;
struct cereal_UbloxGnss_Ephemeris;
struct cereal_UbloxGnss_IonoData;
struct cereal_UbloxGnss_HwStatus;
struct cereal_Clocks;
struct cereal_LiveMpcData;
struct cereal_LiveLongitudinalMpcData;
struct cereal_ECEFPointDEPRECATED;
struct cereal_ECEFPoint;
struct cereal_GPSPlannerPoints;
struct cereal_GPSPlannerPlan;
struct cereal_TrafficEvent;
struct cereal_OrbslamCorrection;
struct cereal_OrbObservation;
struct cereal_UiNavigationEvent;
struct cereal_UiLayoutState;
struct cereal_Joystick;
struct cereal_OrbOdometry;
struct cereal_OrbFeatures;
struct cereal_OrbFeaturesSummary;
struct cereal_OrbKeyFrame;
struct cereal_DriverState;
struct cereal_DMonitoringState;
struct cereal_Boot;
struct cereal_LiveParametersData;
struct cereal_LiveMapData;
struct cereal_CameraOdometry;
struct cereal_KalmanOdometry;
struct cereal_Event;

typedef struct {capn_ptr p;} cereal_Map_ptr;
typedef struct {capn_ptr p;} cereal_Map_Entry_ptr;
typedef struct {capn_ptr p;} cereal_InitData_ptr;
typedef struct {capn_ptr p;} cereal_InitData_AndroidBuildInfo_ptr;
typedef struct {capn_ptr p;} cereal_InitData_AndroidSensor_ptr;
typedef struct {capn_ptr p;} cereal_InitData_ChffrAndroidExtra_ptr;
typedef struct {capn_ptr p;} cereal_InitData_IosBuildInfo_ptr;
typedef struct {capn_ptr p;} cereal_InitData_PandaInfo_ptr;
typedef struct {capn_ptr p;} cereal_FrameData_ptr;
typedef struct {capn_ptr p;} cereal_FrameData_AndroidCaptureResult_ptr;
typedef struct {capn_ptr p;} cereal_Thumbnail_ptr;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_ptr;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_ptr;
typedef struct {capn_ptr p;} cereal_GpsLocationData_ptr;
typedef struct {capn_ptr p;} cereal_CanData_ptr;
typedef struct {capn_ptr p;} cereal_ThermalData_ptr;
typedef struct {capn_ptr p;} cereal_HealthData_ptr;
typedef struct {capn_ptr p;} cereal_LiveUI_ptr;
typedef struct {capn_ptr p;} cereal_RadarState_ptr;
typedef struct {capn_ptr p;} cereal_RadarState_LeadData_ptr;
typedef struct {capn_ptr p;} cereal_LiveCalibrationData_ptr;
typedef struct {capn_ptr p;} cereal_LiveTracks_ptr;
typedef struct {capn_ptr p;} cereal_ControlsState_ptr;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralINDIState_ptr;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralPIDState_ptr;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralLQRState_ptr;
typedef struct {capn_ptr p;} cereal_LiveEventData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_PathData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_LeadData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_ModelSettings_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_MetaData_ptr;
typedef struct {capn_ptr p;} cereal_ModelData_LongitudinalData_ptr;
typedef struct {capn_ptr p;} cereal_CalibrationFeatures_ptr;
typedef struct {capn_ptr p;} cereal_EncodeIndex_ptr;
typedef struct {capn_ptr p;} cereal_AndroidLogEntry_ptr;
typedef struct {capn_ptr p;} cereal_LogRotate_ptr;
typedef struct {capn_ptr p;} cereal_Plan_ptr;
typedef struct {capn_ptr p;} cereal_Plan_GpsTrajectory_ptr;
typedef struct {capn_ptr p;} cereal_PathPlan_ptr;
typedef struct {capn_ptr p;} cereal_LiveLocationData_ptr;
typedef struct {capn_ptr p;} cereal_LiveLocationData_Accuracy_ptr;
typedef struct {capn_ptr p;} cereal_EthernetPacket_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_LatLng_ptr;
typedef struct {capn_ptr p;} cereal_NavUpdate_Segment_ptr;
typedef struct {capn_ptr p;} cereal_NavStatus_ptr;
typedef struct {capn_ptr p;} cereal_NavStatus_Address_ptr;
typedef struct {capn_ptr p;} cereal_CellInfo_ptr;
typedef struct {capn_ptr p;} cereal_WifiScan_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Clock_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Measurement_ptr;
typedef struct {capn_ptr p;} cereal_AndroidGnss_NavigationMessage_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementStatus_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_ClockReport_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrMeasurementReport_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrMeasurementReport_SV_ptr;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrSvPolyReport_ptr;
typedef struct {capn_ptr p;} cereal_LidarPts_ptr;
typedef struct {capn_ptr p;} cereal_ProcLog_ptr;
typedef struct {capn_ptr p;} cereal_ProcLog_Process_ptr;
typedef struct {capn_ptr p;} cereal_ProcLog_CPUTimes_ptr;
typedef struct {capn_ptr p;} cereal_ProcLog_Mem_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_Measurement_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_Ephemeris_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_IonoData_ptr;
typedef struct {capn_ptr p;} cereal_UbloxGnss_HwStatus_ptr;
typedef struct {capn_ptr p;} cereal_Clocks_ptr;
typedef struct {capn_ptr p;} cereal_LiveMpcData_ptr;
typedef struct {capn_ptr p;} cereal_LiveLongitudinalMpcData_ptr;
typedef struct {capn_ptr p;} cereal_ECEFPointDEPRECATED_ptr;
typedef struct {capn_ptr p;} cereal_ECEFPoint_ptr;
typedef struct {capn_ptr p;} cereal_GPSPlannerPoints_ptr;
typedef struct {capn_ptr p;} cereal_GPSPlannerPlan_ptr;
typedef struct {capn_ptr p;} cereal_TrafficEvent_ptr;
typedef struct {capn_ptr p;} cereal_OrbslamCorrection_ptr;
typedef struct {capn_ptr p;} cereal_OrbObservation_ptr;
typedef struct {capn_ptr p;} cereal_UiNavigationEvent_ptr;
typedef struct {capn_ptr p;} cereal_UiLayoutState_ptr;
typedef struct {capn_ptr p;} cereal_Joystick_ptr;
typedef struct {capn_ptr p;} cereal_OrbOdometry_ptr;
typedef struct {capn_ptr p;} cereal_OrbFeatures_ptr;
typedef struct {capn_ptr p;} cereal_OrbFeaturesSummary_ptr;
typedef struct {capn_ptr p;} cereal_OrbKeyFrame_ptr;
typedef struct {capn_ptr p;} cereal_DriverState_ptr;
typedef struct {capn_ptr p;} cereal_DMonitoringState_ptr;
typedef struct {capn_ptr p;} cereal_Boot_ptr;
typedef struct {capn_ptr p;} cereal_LiveParametersData_ptr;
typedef struct {capn_ptr p;} cereal_LiveMapData_ptr;
typedef struct {capn_ptr p;} cereal_CameraOdometry_ptr;
typedef struct {capn_ptr p;} cereal_KalmanOdometry_ptr;
typedef struct {capn_ptr p;} cereal_Event_ptr;

typedef struct {capn_ptr p;} cereal_Map_list;
typedef struct {capn_ptr p;} cereal_Map_Entry_list;
typedef struct {capn_ptr p;} cereal_InitData_list;
typedef struct {capn_ptr p;} cereal_InitData_AndroidBuildInfo_list;
typedef struct {capn_ptr p;} cereal_InitData_AndroidSensor_list;
typedef struct {capn_ptr p;} cereal_InitData_ChffrAndroidExtra_list;
typedef struct {capn_ptr p;} cereal_InitData_IosBuildInfo_list;
typedef struct {capn_ptr p;} cereal_InitData_PandaInfo_list;
typedef struct {capn_ptr p;} cereal_FrameData_list;
typedef struct {capn_ptr p;} cereal_FrameData_AndroidCaptureResult_list;
typedef struct {capn_ptr p;} cereal_Thumbnail_list;
typedef struct {capn_ptr p;} cereal_GPSNMEAData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_list;
typedef struct {capn_ptr p;} cereal_SensorEventData_SensorVec_list;
typedef struct {capn_ptr p;} cereal_GpsLocationData_list;
typedef struct {capn_ptr p;} cereal_CanData_list;
typedef struct {capn_ptr p;} cereal_ThermalData_list;
typedef struct {capn_ptr p;} cereal_HealthData_list;
typedef struct {capn_ptr p;} cereal_LiveUI_list;
typedef struct {capn_ptr p;} cereal_RadarState_list;
typedef struct {capn_ptr p;} cereal_RadarState_LeadData_list;
typedef struct {capn_ptr p;} cereal_LiveCalibrationData_list;
typedef struct {capn_ptr p;} cereal_LiveTracks_list;
typedef struct {capn_ptr p;} cereal_ControlsState_list;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralINDIState_list;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralPIDState_list;
typedef struct {capn_ptr p;} cereal_ControlsState_LateralLQRState_list;
typedef struct {capn_ptr p;} cereal_LiveEventData_list;
typedef struct {capn_ptr p;} cereal_ModelData_list;
typedef struct {capn_ptr p;} cereal_ModelData_PathData_list;
typedef struct {capn_ptr p;} cereal_ModelData_LeadData_list;
typedef struct {capn_ptr p;} cereal_ModelData_ModelSettings_list;
typedef struct {capn_ptr p;} cereal_ModelData_MetaData_list;
typedef struct {capn_ptr p;} cereal_ModelData_LongitudinalData_list;
typedef struct {capn_ptr p;} cereal_CalibrationFeatures_list;
typedef struct {capn_ptr p;} cereal_EncodeIndex_list;
typedef struct {capn_ptr p;} cereal_AndroidLogEntry_list;
typedef struct {capn_ptr p;} cereal_LogRotate_list;
typedef struct {capn_ptr p;} cereal_Plan_list;
typedef struct {capn_ptr p;} cereal_Plan_GpsTrajectory_list;
typedef struct {capn_ptr p;} cereal_PathPlan_list;
typedef struct {capn_ptr p;} cereal_LiveLocationData_list;
typedef struct {capn_ptr p;} cereal_LiveLocationData_Accuracy_list;
typedef struct {capn_ptr p;} cereal_EthernetPacket_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_LatLng_list;
typedef struct {capn_ptr p;} cereal_NavUpdate_Segment_list;
typedef struct {capn_ptr p;} cereal_NavStatus_list;
typedef struct {capn_ptr p;} cereal_NavStatus_Address_list;
typedef struct {capn_ptr p;} cereal_CellInfo_list;
typedef struct {capn_ptr p;} cereal_WifiScan_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Clock_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_Measurements_Measurement_list;
typedef struct {capn_ptr p;} cereal_AndroidGnss_NavigationMessage_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementStatus_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_MeasurementReport_SV_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_ClockReport_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrMeasurementReport_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrMeasurementReport_SV_list;
typedef struct {capn_ptr p;} cereal_QcomGnss_DrSvPolyReport_list;
typedef struct {capn_ptr p;} cereal_LidarPts_list;
typedef struct {capn_ptr p;} cereal_ProcLog_list;
typedef struct {capn_ptr p;} cereal_ProcLog_Process_list;
typedef struct {capn_ptr p;} cereal_ProcLog_CPUTimes_list;
typedef struct {capn_ptr p;} cereal_ProcLog_Mem_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_Measurement_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_Ephemeris_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_IonoData_list;
typedef struct {capn_ptr p;} cereal_UbloxGnss_HwStatus_list;
typedef struct {capn_ptr p;} cereal_Clocks_list;
typedef struct {capn_ptr p;} cereal_LiveMpcData_list;
typedef struct {capn_ptr p;} cereal_LiveLongitudinalMpcData_list;
typedef struct {capn_ptr p;} cereal_ECEFPointDEPRECATED_list;
typedef struct {capn_ptr p;} cereal_ECEFPoint_list;
typedef struct {capn_ptr p;} cereal_GPSPlannerPoints_list;
typedef struct {capn_ptr p;} cereal_GPSPlannerPlan_list;
typedef struct {capn_ptr p;} cereal_TrafficEvent_list;
typedef struct {capn_ptr p;} cereal_OrbslamCorrection_list;
typedef struct {capn_ptr p;} cereal_OrbObservation_list;
typedef struct {capn_ptr p;} cereal_UiNavigationEvent_list;
typedef struct {capn_ptr p;} cereal_UiLayoutState_list;
typedef struct {capn_ptr p;} cereal_Joystick_list;
typedef struct {capn_ptr p;} cereal_OrbOdometry_list;
typedef struct {capn_ptr p;} cereal_OrbFeatures_list;
typedef struct {capn_ptr p;} cereal_OrbFeaturesSummary_list;
typedef struct {capn_ptr p;} cereal_OrbKeyFrame_list;
typedef struct {capn_ptr p;} cereal_DriverState_list;
typedef struct {capn_ptr p;} cereal_DMonitoringState_list;
typedef struct {capn_ptr p;} cereal_Boot_list;
typedef struct {capn_ptr p;} cereal_LiveParametersData_list;
typedef struct {capn_ptr p;} cereal_LiveMapData_list;
typedef struct {capn_ptr p;} cereal_CameraOdometry_list;
typedef struct {capn_ptr p;} cereal_KalmanOdometry_list;
typedef struct {capn_ptr p;} cereal_Event_list;

enum cereal_InitData_DeviceType {
	cereal_InitData_DeviceType_unknown = 0,
	cereal_InitData_DeviceType_neo = 1,
	cereal_InitData_DeviceType_chffrAndroid = 2,
	cereal_InitData_DeviceType_chffrIos = 3
};

enum cereal_FrameData_FrameType {
	cereal_FrameData_FrameType_unknown = 0,
	cereal_FrameData_FrameType_neo = 1,
	cereal_FrameData_FrameType_chffrAndroid = 2,
	cereal_FrameData_FrameType_front = 3
};

enum cereal_SensorEventData_SensorSource {
	cereal_SensorEventData_SensorSource_android = 0,
	cereal_SensorEventData_SensorSource_iOS = 1,
	cereal_SensorEventData_SensorSource_fiber = 2,
	cereal_SensorEventData_SensorSource_velodyne = 3,
	cereal_SensorEventData_SensorSource_bno055 = 4,
	cereal_SensorEventData_SensorSource_lsm6ds3 = 5,
	cereal_SensorEventData_SensorSource_bmp280 = 6,
	cereal_SensorEventData_SensorSource_mmc3416x = 7
};

enum cereal_GpsLocationData_SensorSource {
	cereal_GpsLocationData_SensorSource_android = 0,
	cereal_GpsLocationData_SensorSource_iOS = 1,
	cereal_GpsLocationData_SensorSource_car = 2,
	cereal_GpsLocationData_SensorSource_velodyne = 3,
	cereal_GpsLocationData_SensorSource_fusion = 4,
	cereal_GpsLocationData_SensorSource_external = 5,
	cereal_GpsLocationData_SensorSource_ublox = 6,
	cereal_GpsLocationData_SensorSource_trimble = 7
};

enum cereal_ThermalData_ThermalStatus {
	cereal_ThermalData_ThermalStatus_green = 0,
	cereal_ThermalData_ThermalStatus_yellow = 1,
	cereal_ThermalData_ThermalStatus_red = 2,
	cereal_ThermalData_ThermalStatus_danger = 3
};

enum cereal_ThermalData_NetworkType {
	cereal_ThermalData_NetworkType_none = 0,
	cereal_ThermalData_NetworkType_wifi = 1,
	cereal_ThermalData_NetworkType_cell2G = 2,
	cereal_ThermalData_NetworkType_cell3G = 3,
	cereal_ThermalData_NetworkType_cell4G = 4,
	cereal_ThermalData_NetworkType_cell5G = 5
};

enum cereal_HealthData_FaultStatus {
	cereal_HealthData_FaultStatus_none = 0,
	cereal_HealthData_FaultStatus_faultTemp = 1,
	cereal_HealthData_FaultStatus_faultPerm = 2
};

enum cereal_HealthData_FaultType {
	cereal_HealthData_FaultType_relayMalfunction = 0
};

enum cereal_HealthData_HwType {
	cereal_HealthData_HwType_unknown = 0,
	cereal_HealthData_HwType_whitePanda = 1,
	cereal_HealthData_HwType_greyPanda = 2,
	cereal_HealthData_HwType_blackPanda = 3,
	cereal_HealthData_HwType_pedal = 4,
	cereal_HealthData_HwType_uno = 5
};

enum cereal_HealthData_UsbPowerMode {
	cereal_HealthData_UsbPowerMode_none = 0,
	cereal_HealthData_UsbPowerMode_client = 1,
	cereal_HealthData_UsbPowerMode_cdp = 2,
	cereal_HealthData_UsbPowerMode_dcp = 3
};

enum cereal_ControlsState_OpenpilotState {
	cereal_ControlsState_OpenpilotState_disabled = 0,
	cereal_ControlsState_OpenpilotState_preEnabled = 1,
	cereal_ControlsState_OpenpilotState_enabled = 2,
	cereal_ControlsState_OpenpilotState_softDisabling = 3
};

enum cereal_ControlsState_LongControlState {
	cereal_ControlsState_LongControlState_off = 0,
	cereal_ControlsState_LongControlState_pid = 1,
	cereal_ControlsState_LongControlState_stopping = 2,
	cereal_ControlsState_LongControlState_starting = 3
};

enum cereal_ControlsState_AlertStatus {
	cereal_ControlsState_AlertStatus_normal = 0,
	cereal_ControlsState_AlertStatus_userPrompt = 1,
	cereal_ControlsState_AlertStatus_critical = 2
};

enum cereal_ControlsState_AlertSize {
	cereal_ControlsState_AlertSize_none = 0,
	cereal_ControlsState_AlertSize_small = 1,
	cereal_ControlsState_AlertSize_mid = 2,
	cereal_ControlsState_AlertSize_full = 3
};

enum cereal_EncodeIndex_Type {
	cereal_EncodeIndex_Type_bigBoxLossless = 0,
	cereal_EncodeIndex_Type_fullHEVC = 1,
	cereal_EncodeIndex_Type_bigBoxHEVC = 2,
	cereal_EncodeIndex_Type_chffrAndroidH264 = 3,
	cereal_EncodeIndex_Type_fullLosslessClip = 4,
	cereal_EncodeIndex_Type_front = 5
};

enum cereal_Plan_LongitudinalPlanSource {
	cereal_Plan_LongitudinalPlanSource_cruise = 0,
	cereal_Plan_LongitudinalPlanSource_mpc1 = 1,
	cereal_Plan_LongitudinalPlanSource_mpc2 = 2,
	cereal_Plan_LongitudinalPlanSource_mpc3 = 3,
	cereal_Plan_LongitudinalPlanSource_model = 4
};

enum cereal_PathPlan_Desire {
	cereal_PathPlan_Desire_none = 0,
	cereal_PathPlan_Desire_turnLeft = 1,
	cereal_PathPlan_Desire_turnRight = 2,
	cereal_PathPlan_Desire_laneChangeLeft = 3,
	cereal_PathPlan_Desire_laneChangeRight = 4,
	cereal_PathPlan_Desire_keepLeft = 5,
	cereal_PathPlan_Desire_keepRight = 6
};

enum cereal_PathPlan_LaneChangeState {
	cereal_PathPlan_LaneChangeState_off = 0,
	cereal_PathPlan_LaneChangeState_preLaneChange = 1,
	cereal_PathPlan_LaneChangeState_laneChangeStarting = 2,
	cereal_PathPlan_LaneChangeState_laneChangeFinishing = 3
};

enum cereal_PathPlan_LaneChangeDirection {
	cereal_PathPlan_LaneChangeDirection_none = 0,
	cereal_PathPlan_LaneChangeDirection_left = 1,
	cereal_PathPlan_LaneChangeDirection_right = 2
};

enum cereal_LiveLocationData_SensorSource {
	cereal_LiveLocationData_SensorSource_applanix = 0,
	cereal_LiveLocationData_SensorSource_kalman = 1,
	cereal_LiveLocationData_SensorSource_orbslam = 2,
	cereal_LiveLocationData_SensorSource_timing = 3,
	cereal_LiveLocationData_SensorSource_dummy = 4
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

enum cereal_QcomGnss_MeasurementSource {
	cereal_QcomGnss_MeasurementSource_gps = 0,
	cereal_QcomGnss_MeasurementSource_glonass = 1,
	cereal_QcomGnss_MeasurementSource_beidou = 2
};

enum cereal_QcomGnss_SVObservationState {
	cereal_QcomGnss_SVObservationState_idle = 0,
	cereal_QcomGnss_SVObservationState_search = 1,
	cereal_QcomGnss_SVObservationState_searchVerify = 2,
	cereal_QcomGnss_SVObservationState_bitEdge = 3,
	cereal_QcomGnss_SVObservationState_trackVerify = 4,
	cereal_QcomGnss_SVObservationState_track = 5,
	cereal_QcomGnss_SVObservationState_restart = 6,
	cereal_QcomGnss_SVObservationState_dpo = 7,
	cereal_QcomGnss_SVObservationState_glo10msBe = 8,
	cereal_QcomGnss_SVObservationState_glo10msAt = 9
};

enum cereal_UbloxGnss_HwStatus_AntennaSupervisorState {
	cereal_UbloxGnss_HwStatus_AntennaSupervisorState_init = 0,
	cereal_UbloxGnss_HwStatus_AntennaSupervisorState_dontknow = 1,
	cereal_UbloxGnss_HwStatus_AntennaSupervisorState_ok = 2,
	cereal_UbloxGnss_HwStatus_AntennaSupervisorState_short = 3,
	cereal_UbloxGnss_HwStatus_AntennaSupervisorState_open = 4
};

enum cereal_UbloxGnss_HwStatus_AntennaPowerStatus {
	cereal_UbloxGnss_HwStatus_AntennaPowerStatus_off = 0,
	cereal_UbloxGnss_HwStatus_AntennaPowerStatus_on = 1,
	cereal_UbloxGnss_HwStatus_AntennaPowerStatus_dontknow = 2
};

enum cereal_TrafficEvent_Type {
	cereal_TrafficEvent_Type_stopSign = 0,
	cereal_TrafficEvent_Type_lightRed = 1,
	cereal_TrafficEvent_Type_lightYellow = 2,
	cereal_TrafficEvent_Type_lightGreen = 3,
	cereal_TrafficEvent_Type_stopLight = 4
};

enum cereal_TrafficEvent_Action {
	cereal_TrafficEvent_Action_none = 0,
	cereal_TrafficEvent_Action_yield = 1,
	cereal_TrafficEvent_Action_stop = 2,
	cereal_TrafficEvent_Action_resumeReady = 3
};

enum cereal_UiNavigationEvent_Type {
	cereal_UiNavigationEvent_Type_none = 0,
	cereal_UiNavigationEvent_Type_laneChangeLeft = 1,
	cereal_UiNavigationEvent_Type_laneChangeRight = 2,
	cereal_UiNavigationEvent_Type_mergeLeft = 3,
	cereal_UiNavigationEvent_Type_mergeRight = 4,
	cereal_UiNavigationEvent_Type_turnLeft = 5,
	cereal_UiNavigationEvent_Type_turnRight = 6
};

enum cereal_UiNavigationEvent_Status {
	cereal_UiNavigationEvent_Status_none = 0,
	cereal_UiNavigationEvent_Status_passive = 1,
	cereal_UiNavigationEvent_Status_approaching = 2,
	cereal_UiNavigationEvent_Status_active = 3
};

enum cereal_UiLayoutState_App {
	cereal_UiLayoutState_App_home = 0,
	cereal_UiLayoutState_App_music = 1,
	cereal_UiLayoutState_App_nav = 2,
	cereal_UiLayoutState_App_settings = 3
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
	capn_text kernelVersion;
	capn_text gctx;
	capn_text dongleId;
	enum cereal_InitData_DeviceType deviceType;
	capn_text version;
	capn_text gitCommit;
	capn_text gitBranch;
	capn_text gitRemote;
	cereal_InitData_AndroidBuildInfo_ptr androidBuildInfo;
	cereal_InitData_AndroidSensor_list androidSensors;
	cereal_Map_ptr androidProperties;
	cereal_InitData_ChffrAndroidExtra_ptr chffrAndroidExtra;
	cereal_InitData_IosBuildInfo_ptr iosBuildInfo;
	cereal_InitData_PandaInfo_ptr pandaInfo;
	unsigned dirty : 1;
	unsigned passive : 1;
	cereal_Map_ptr params;
};

static const size_t cereal_InitData_word_count = 1;

static const size_t cereal_InitData_pointer_count = 15;

static const size_t cereal_InitData_struct_bytes_count = 128;

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

struct cereal_InitData_IosBuildInfo {
	capn_text appVersion;
	uint32_t appBuild;
	capn_text osVersion;
	capn_text deviceModel;
};

static const size_t cereal_InitData_IosBuildInfo_word_count = 1;

static const size_t cereal_InitData_IosBuildInfo_pointer_count = 3;

static const size_t cereal_InitData_IosBuildInfo_struct_bytes_count = 32;

struct cereal_InitData_PandaInfo {
	unsigned hasPanda : 1;
	capn_text dongleId;
	capn_text stVersion;
	capn_text espVersion;
};

static const size_t cereal_InitData_PandaInfo_word_count = 1;

static const size_t cereal_InitData_PandaInfo_pointer_count = 3;

static const size_t cereal_InitData_PandaInfo_struct_bytes_count = 32;

struct cereal_FrameData {
	uint32_t frameId;
	uint32_t encodeId;
	uint64_t timestampEof;
	int32_t frameLength;
	int32_t integLines;
	int32_t globalGain;
	int32_t lensPos;
	float lensSag;
	float lensErr;
	float lensTruePos;
	capn_data image;
	float gainFrac;
	enum cereal_FrameData_FrameType frameType;
	uint64_t timestampSof;
	capn_list32 transform;
	cereal_FrameData_AndroidCaptureResult_ptr androidCaptureResult;
};

static const size_t cereal_FrameData_word_count = 8;

static const size_t cereal_FrameData_pointer_count = 3;

static const size_t cereal_FrameData_struct_bytes_count = 88;

struct cereal_FrameData_AndroidCaptureResult {
	int32_t sensitivity;
	int64_t frameDuration;
	int64_t exposureTime;
	uint64_t rollingShutterSkew;
	capn_list32 colorCorrectionTransform;
	capn_list32 colorCorrectionGains;
	int8_t displayRotation;
};

static const size_t cereal_FrameData_AndroidCaptureResult_word_count = 4;

static const size_t cereal_FrameData_AndroidCaptureResult_pointer_count = 2;

static const size_t cereal_FrameData_AndroidCaptureResult_struct_bytes_count = 48;

struct cereal_Thumbnail {
	uint32_t frameId;
	uint64_t timestampEof;
	capn_data thumbnail;
};

static const size_t cereal_Thumbnail_word_count = 2;

static const size_t cereal_Thumbnail_pointer_count = 1;

static const size_t cereal_Thumbnail_struct_bytes_count = 24;

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
	cereal_SensorEventData_gyro = 3,
	cereal_SensorEventData_pressure = 4,
	cereal_SensorEventData_magneticUncalibrated = 5,
	cereal_SensorEventData_gyroUncalibrated = 6,
	cereal_SensorEventData_proximity = 7,
	cereal_SensorEventData_light = 8
};

struct cereal_SensorEventData {
	int32_t version;
	int32_t sensor;
	int32_t type;
	int64_t timestamp;
	unsigned uncalibratedDEPRECATED : 1;
	enum cereal_SensorEventData_which which;
	union {
		cereal_SensorEventData_SensorVec_ptr acceleration;
		cereal_SensorEventData_SensorVec_ptr magnetic;
		cereal_SensorEventData_SensorVec_ptr orientation;
		cereal_SensorEventData_SensorVec_ptr gyro;
		cereal_SensorEventData_SensorVec_ptr pressure;
		cereal_SensorEventData_SensorVec_ptr magneticUncalibrated;
		cereal_SensorEventData_SensorVec_ptr gyroUncalibrated;
		float proximity;
		float light;
	};
	enum cereal_SensorEventData_SensorSource source;
};

static const size_t cereal_SensorEventData_word_count = 4;

static const size_t cereal_SensorEventData_pointer_count = 1;

static const size_t cereal_SensorEventData_struct_bytes_count = 40;

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
	capn_list32 vNED;
	float verticalAccuracy;
	float bearingAccuracy;
	float speedAccuracy;
};

static const size_t cereal_GpsLocationData_word_count = 8;

static const size_t cereal_GpsLocationData_pointer_count = 1;

static const size_t cereal_GpsLocationData_struct_bytes_count = 72;

struct cereal_CanData {
	uint32_t address;
	uint16_t busTime;
	capn_data dat;
	uint8_t src;
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
	uint16_t pa0;
	float freeSpace;
	int16_t batteryPercent;
	capn_text batteryStatus;
	int32_t batteryCurrent;
	int32_t batteryVoltage;
	unsigned usbOnline : 1;
	enum cereal_ThermalData_NetworkType networkType;
	uint32_t offroadPowerUsage;
	uint16_t fanSpeed;
	unsigned started : 1;
	uint64_t startedTs;
	enum cereal_ThermalData_ThermalStatus thermalStatus;
	unsigned chargingError : 1;
	unsigned chargingDisabled : 1;
	int8_t memUsedPercent;
	int8_t cpuPerc;
};

static const size_t cereal_ThermalData_word_count = 7;

static const size_t cereal_ThermalData_pointer_count = 1;

static const size_t cereal_ThermalData_struct_bytes_count = 64;

struct cereal_HealthData {
	uint32_t voltage;
	uint32_t current;
	unsigned ignitionLine : 1;
	unsigned controlsAllowed : 1;
	unsigned gasInterceptorDetected : 1;
	unsigned startedSignalDetectedDeprecated : 1;
	unsigned hasGps : 1;
	uint32_t canSendErrs;
	uint32_t canFwdErrs;
	uint32_t canRxErrs;
	uint32_t gmlanSendErrs;
	enum cereal_HealthData_HwType hwType;
	uint16_t fanSpeedRpm;
	enum cereal_HealthData_UsbPowerMode usbPowerMode;
	unsigned ignitionCan : 1;
	enum cereal_CarParams_SafetyModel safetyModel;
	enum cereal_HealthData_FaultStatus faultStatus;
	unsigned powerSaveEnabled : 1;
	uint32_t uptime;
	capn_list16 faults;
};

static const size_t cereal_HealthData_word_count = 5;

static const size_t cereal_HealthData_pointer_count = 1;

static const size_t cereal_HealthData_struct_bytes_count = 48;

struct cereal_LiveUI {
	unsigned rearViewCam : 1;
	capn_text alertText1;
	capn_text alertText2;
	float awarenessStatus;
};

static const size_t cereal_LiveUI_word_count = 1;

static const size_t cereal_LiveUI_pointer_count = 2;

static const size_t cereal_LiveUI_struct_bytes_count = 24;

struct cereal_RadarState {
	capn_list64 canMonoTimes;
	uint64_t mdMonoTime;
	uint64_t ftMonoTimeDEPRECATED;
	uint64_t controlsStateMonoTime;
	capn_list16 radarErrors;
	capn_list32 warpMatrixDEPRECATED;
	float angleOffsetDEPRECATED;
	int8_t calStatusDEPRECATED;
	int32_t calCycleDEPRECATED;
	int8_t calPercDEPRECATED;
	cereal_RadarState_LeadData_ptr leadOne;
	cereal_RadarState_LeadData_ptr leadTwo;
	float cumLagMs;
};

static const size_t cereal_RadarState_word_count = 5;

static const size_t cereal_RadarState_pointer_count = 5;

static const size_t cereal_RadarState_struct_bytes_count = 80;

struct cereal_RadarState_LeadData {
	float dRel;
	float yRel;
	float vRel;
	float aRel;
	float vLead;
	float aLeadDEPRECATED;
	float dPath;
	float vLat;
	float vLeadK;
	float aLeadK;
	unsigned fcw : 1;
	unsigned status : 1;
	float aLeadTau;
	float modelProb;
	unsigned radar : 1;
};

static const size_t cereal_RadarState_LeadData_word_count = 7;

static const size_t cereal_RadarState_LeadData_pointer_count = 0;

static const size_t cereal_RadarState_LeadData_struct_bytes_count = 56;

struct cereal_LiveCalibrationData {
	capn_list32 warpMatrix;
	capn_list32 warpMatrix2;
	capn_list32 warpMatrixBig;
	int8_t calStatus;
	int32_t calCycle;
	int8_t calPerc;
	capn_list32 extrinsicMatrix;
	capn_list32 rpyCalib;
};

static const size_t cereal_LiveCalibrationData_word_count = 1;

static const size_t cereal_LiveCalibrationData_pointer_count = 5;

static const size_t cereal_LiveCalibrationData_struct_bytes_count = 48;

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
enum cereal_ControlsState_lateralControlState_which {
	cereal_ControlsState_lateralControlState_indiState = 0,
	cereal_ControlsState_lateralControlState_pidState = 1,
	cereal_ControlsState_lateralControlState_lqrState = 2
};

struct cereal_ControlsState {
	uint64_t canMonoTimeDEPRECATED;
	capn_list64 canMonoTimes;
	uint64_t radarStateMonoTimeDEPRECATED;
	uint64_t mdMonoTimeDEPRECATED;
	uint64_t planMonoTime;
	uint64_t pathPlanMonoTime;
	enum cereal_ControlsState_OpenpilotState state;
	float vEgo;
	float vEgoRaw;
	float aEgoDEPRECATED;
	enum cereal_ControlsState_LongControlState longControlState;
	float vPid;
	float vTargetLead;
	float upAccelCmd;
	float uiAccelCmd;
	float ufAccelCmd;
	float yActualDEPRECATED;
	float yDesDEPRECATED;
	float upSteerDEPRECATED;
	float uiSteerDEPRECATED;
	float ufSteerDEPRECATED;
	float aTargetMinDEPRECATED;
	float aTargetMaxDEPRECATED;
	float aTarget;
	float jerkFactor;
	float angleSteers;
	float angleSteersDes;
	float curvature;
	int32_t hudLeadDEPRECATED;
	float cumLagMs;
	uint64_t startMonoTime;
	unsigned mapValid : 1;
	unsigned forceDecel : 1;
	unsigned enabled : 1;
	unsigned active : 1;
	unsigned steerOverride : 1;
	float vCruise;
	unsigned rearViewCam : 1;
	capn_text alertText1;
	capn_text alertText2;
	enum cereal_ControlsState_AlertStatus alertStatus;
	enum cereal_ControlsState_AlertSize alertSize;
	float alertBlinkingRate;
	capn_text alertType;
	capn_text alertSoundDEPRECATED;
	enum cereal_CarControl_HUDControl_AudibleAlert alertSound;
	float awarenessStatus;
	float angleModelBiasDEPRECATED;
	unsigned gpsPlannerActive : 1;
	unsigned engageable : 1;
	unsigned driverMonitoringOn : 1;
	float vCurvature;
	unsigned decelForTurn : 1;
	unsigned decelForModel : 1;
	uint32_t canErrorCounter;
	enum cereal_ControlsState_lateralControlState_which lateralControlState_which;
	union {
		cereal_ControlsState_LateralINDIState_ptr indiState;
		cereal_ControlsState_LateralPIDState_ptr pidState;
		cereal_ControlsState_LateralLQRState_ptr lqrState;
	} lateralControlState;
};

static const size_t cereal_ControlsState_word_count = 22;

static const size_t cereal_ControlsState_pointer_count = 6;

static const size_t cereal_ControlsState_struct_bytes_count = 224;

struct cereal_ControlsState_LateralINDIState {
	unsigned active : 1;
	float steerAngle;
	float steerRate;
	float steerAccel;
	float rateSetPoint;
	float accelSetPoint;
	float accelError;
	float delayedOutput;
	float delta;
	float output;
	unsigned saturated : 1;
};

static const size_t cereal_ControlsState_LateralINDIState_word_count = 5;

static const size_t cereal_ControlsState_LateralINDIState_pointer_count = 0;

static const size_t cereal_ControlsState_LateralINDIState_struct_bytes_count = 40;

struct cereal_ControlsState_LateralPIDState {
	unsigned active : 1;
	float steerAngle;
	float steerRate;
	float angleError;
	float p;
	float i;
	float f;
	float output;
	unsigned saturated : 1;
};

static const size_t cereal_ControlsState_LateralPIDState_word_count = 4;

static const size_t cereal_ControlsState_LateralPIDState_pointer_count = 0;

static const size_t cereal_ControlsState_LateralPIDState_struct_bytes_count = 32;

struct cereal_ControlsState_LateralLQRState {
	unsigned active : 1;
	float steerAngle;
	float i;
	float output;
	float lqrOutput;
	unsigned saturated : 1;
};

static const size_t cereal_ControlsState_LateralLQRState_word_count = 3;

static const size_t cereal_ControlsState_LateralLQRState_pointer_count = 0;

static const size_t cereal_ControlsState_LateralLQRState_struct_bytes_count = 24;

struct cereal_LiveEventData {
	capn_text name;
	int32_t value;
};

static const size_t cereal_LiveEventData_word_count = 1;

static const size_t cereal_LiveEventData_pointer_count = 1;

static const size_t cereal_LiveEventData_struct_bytes_count = 16;

struct cereal_ModelData {
	uint32_t frameId;
	uint64_t timestampEof;
	cereal_ModelData_PathData_ptr path;
	cereal_ModelData_PathData_ptr leftLane;
	cereal_ModelData_PathData_ptr rightLane;
	cereal_ModelData_LeadData_ptr lead;
	capn_list32 freePath;
	cereal_ModelData_ModelSettings_ptr settings;
	cereal_ModelData_LeadData_ptr leadFuture;
	capn_list32 speed;
	cereal_ModelData_MetaData_ptr meta;
	cereal_ModelData_LongitudinalData_ptr longitudinal;
};

static const size_t cereal_ModelData_word_count = 2;

static const size_t cereal_ModelData_pointer_count = 10;

static const size_t cereal_ModelData_struct_bytes_count = 96;

struct cereal_ModelData_PathData {
	capn_list32 points;
	float prob;
	float std;
	capn_list32 stds;
	capn_list32 poly;
};

static const size_t cereal_ModelData_PathData_word_count = 1;

static const size_t cereal_ModelData_PathData_pointer_count = 3;

static const size_t cereal_ModelData_PathData_struct_bytes_count = 32;

struct cereal_ModelData_LeadData {
	float dist;
	float prob;
	float std;
	float relVel;
	float relVelStd;
	float relY;
	float relYStd;
	float relA;
	float relAStd;
};

static const size_t cereal_ModelData_LeadData_word_count = 5;

static const size_t cereal_ModelData_LeadData_pointer_count = 0;

static const size_t cereal_ModelData_LeadData_struct_bytes_count = 40;

struct cereal_ModelData_ModelSettings {
	uint16_t bigBoxX;
	uint16_t bigBoxY;
	uint16_t bigBoxWidth;
	uint16_t bigBoxHeight;
	capn_list32 boxProjection;
	capn_list32 yuvCorrection;
	capn_list32 inputTransform;
};

static const size_t cereal_ModelData_ModelSettings_word_count = 1;

static const size_t cereal_ModelData_ModelSettings_pointer_count = 3;

static const size_t cereal_ModelData_ModelSettings_struct_bytes_count = 32;

struct cereal_ModelData_MetaData {
	float engagedProb;
	capn_list32 desirePrediction;
	float brakeDisengageProb;
	float gasDisengageProb;
	float steerOverrideProb;
};

static const size_t cereal_ModelData_MetaData_word_count = 2;

static const size_t cereal_ModelData_MetaData_pointer_count = 1;

static const size_t cereal_ModelData_MetaData_struct_bytes_count = 24;

struct cereal_ModelData_LongitudinalData {
	capn_list32 speeds;
	capn_list32 accelerations;
};

static const size_t cereal_ModelData_LongitudinalData_word_count = 0;

static const size_t cereal_ModelData_LongitudinalData_pointer_count = 2;

static const size_t cereal_ModelData_LongitudinalData_struct_bytes_count = 16;

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
	uint64_t mdMonoTime;
	uint64_t radarStateMonoTime;
	unsigned commIssue : 1;
	cereal_CarEvent_list eventsDEPRECATED;
	unsigned lateralValidDEPRECATED : 1;
	capn_list32 dPolyDEPRECATED;
	float laneWidthDEPRECATED;
	unsigned longitudinalValidDEPRECATED : 1;
	float vCruise;
	float aCruise;
	float vTarget;
	float vTargetFuture;
	float vMax;
	float aTargetMinDEPRECATED;
	float aTargetMaxDEPRECATED;
	float aTarget;
	float vStart;
	float aStart;
	float jerkFactor;
	unsigned hasLead : 1;
	unsigned hasLeftLaneDEPRECATED : 1;
	unsigned hasRightLaneDEPRECATED : 1;
	unsigned fcw : 1;
	enum cereal_Plan_LongitudinalPlanSource longitudinalPlanSource;
	cereal_Plan_GpsTrajectory_ptr gpsTrajectory;
	unsigned gpsPlannerActive : 1;
	float vCurvature;
	unsigned decelForTurn : 1;
	unsigned mapValid : 1;
	unsigned radarValid : 1;
	unsigned radarCanError : 1;
	float processingDelay;
};

static const size_t cereal_Plan_word_count = 10;

static const size_t cereal_Plan_pointer_count = 3;

static const size_t cereal_Plan_struct_bytes_count = 104;

struct cereal_Plan_GpsTrajectory {
	capn_list32 x;
	capn_list32 y;
};

static const size_t cereal_Plan_GpsTrajectory_word_count = 0;

static const size_t cereal_Plan_GpsTrajectory_pointer_count = 2;

static const size_t cereal_Plan_GpsTrajectory_struct_bytes_count = 16;

struct cereal_PathPlan {
	float laneWidth;
	capn_list32 dPoly;
	capn_list32 cPoly;
	float cProb;
	capn_list32 lPoly;
	float lProb;
	capn_list32 rPoly;
	float rProb;
	float angleSteers;
	float rateSteers;
	unsigned mpcSolutionValid : 1;
	unsigned paramsValid : 1;
	unsigned modelValidDEPRECATED : 1;
	float angleOffset;
	unsigned sensorValid : 1;
	unsigned commIssue : 1;
	unsigned posenetValid : 1;
	enum cereal_PathPlan_Desire desire;
	enum cereal_PathPlan_LaneChangeState laneChangeState;
	enum cereal_PathPlan_LaneChangeDirection laneChangeDirection;
};

static const size_t cereal_PathPlan_word_count = 5;

static const size_t cereal_PathPlan_pointer_count = 4;

static const size_t cereal_PathPlan_struct_bytes_count = 72;

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
	enum cereal_LiveLocationData_SensorSource source;
	uint64_t fixMonoTime;
	int32_t gpsWeek;
	double timeOfWeek;
	capn_list64 positionECEF;
	capn_list32 poseQuatECEF;
	float pitchCalibration;
	float yawCalibration;
	capn_list32 imuFrame;
};

static const size_t cereal_LiveLocationData_word_count = 10;

static const size_t cereal_LiveLocationData_pointer_count = 7;

static const size_t cereal_LiveLocationData_struct_bytes_count = 136;

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

struct cereal_NavStatus {
	unsigned isNavigating : 1;
	cereal_NavStatus_Address_ptr currentAddress;
};

static const size_t cereal_NavStatus_word_count = 1;

static const size_t cereal_NavStatus_pointer_count = 1;

static const size_t cereal_NavStatus_struct_bytes_count = 16;

struct cereal_NavStatus_Address {
	capn_text title;
	double lat;
	double lng;
	capn_text house;
	capn_text address;
	capn_text street;
	capn_text city;
	capn_text state;
	capn_text country;
};

static const size_t cereal_NavStatus_Address_word_count = 2;

static const size_t cereal_NavStatus_Address_pointer_count = 7;

static const size_t cereal_NavStatus_Address_struct_bytes_count = 72;

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
	cereal_QcomGnss_clockReport = 1,
	cereal_QcomGnss_drMeasurementReport = 2,
	cereal_QcomGnss_drSvPoly = 3,
	cereal_QcomGnss_rawLog = 4
};

struct cereal_QcomGnss {
	uint64_t logTs;
	enum cereal_QcomGnss_which which;
	union {
		cereal_QcomGnss_MeasurementReport_ptr measurementReport;
		cereal_QcomGnss_ClockReport_ptr clockReport;
		cereal_QcomGnss_DrMeasurementReport_ptr drMeasurementReport;
		cereal_QcomGnss_DrSvPolyReport_ptr drSvPoly;
		capn_data rawLog;
	};
};

static const size_t cereal_QcomGnss_word_count = 2;

static const size_t cereal_QcomGnss_pointer_count = 1;

static const size_t cereal_QcomGnss_struct_bytes_count = 24;

struct cereal_QcomGnss_MeasurementStatus {
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
	unsigned multipathIndicator : 1;
	unsigned imdJammingIndicator : 1;
	unsigned lteB13TxJammingIndicator : 1;
	unsigned freshMeasurementIndicator : 1;
	unsigned multipathEstimateIsValid : 1;
	unsigned directionIsValid : 1;
};

static const size_t cereal_QcomGnss_MeasurementStatus_word_count = 1;

static const size_t cereal_QcomGnss_MeasurementStatus_pointer_count = 0;

static const size_t cereal_QcomGnss_MeasurementStatus_struct_bytes_count = 8;

struct cereal_QcomGnss_MeasurementReport {
	enum cereal_QcomGnss_MeasurementSource source;
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
	enum cereal_QcomGnss_SVObservationState observationState;
	uint8_t observations;
	uint8_t goodObservations;
	uint16_t gpsParityErrorCount;
	int8_t glonassFrequencyIndex;
	uint8_t glonassHemmingErrorCount;
	uint8_t filterStages;
	uint16_t carrierNoise;
	int16_t latency;
	uint8_t predetectInterval;
	uint16_t postdetections;
	uint32_t unfilteredMeasurementIntegral;
	float unfilteredMeasurementFraction;
	float unfilteredTimeUncertainty;
	float unfilteredSpeed;
	float unfilteredSpeedUncertainty;
	cereal_QcomGnss_MeasurementStatus_ptr measurementStatus;
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

struct cereal_QcomGnss_ClockReport {
	unsigned hasFCount : 1;
	uint32_t fCount;
	unsigned hasGpsWeek : 1;
	uint16_t gpsWeek;
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

struct cereal_QcomGnss_DrMeasurementReport {
	uint8_t reason;
	uint8_t seqNum;
	uint8_t seqMax;
	uint16_t rfLoss;
	unsigned systemRtcValid : 1;
	uint32_t fCount;
	uint32_t clockResets;
	uint64_t systemRtcTime;
	uint8_t gpsLeapSeconds;
	uint8_t gpsLeapSecondsUncertainty;
	float gpsToGlonassTimeBiasMilliseconds;
	float gpsToGlonassTimeBiasMillisecondsUncertainty;
	uint16_t gpsWeek;
	uint32_t gpsMilliseconds;
	uint32_t gpsTimeBiasMs;
	uint32_t gpsClockTimeUncertaintyMs;
	uint8_t gpsClockSource;
	uint8_t glonassClockSource;
	uint8_t glonassYear;
	uint16_t glonassDay;
	uint32_t glonassMilliseconds;
	float glonassTimeBias;
	float glonassClockTimeUncertainty;
	float clockFrequencyBias;
	float clockFrequencyUncertainty;
	uint8_t frequencySource;
	enum cereal_QcomGnss_MeasurementSource source;
	cereal_QcomGnss_DrMeasurementReport_SV_list sv;
};

static const size_t cereal_QcomGnss_DrMeasurementReport_word_count = 10;

static const size_t cereal_QcomGnss_DrMeasurementReport_pointer_count = 1;

static const size_t cereal_QcomGnss_DrMeasurementReport_struct_bytes_count = 88;

struct cereal_QcomGnss_DrMeasurementReport_SV {
	uint8_t svId;
	int8_t glonassFrequencyIndex;
	enum cereal_QcomGnss_SVObservationState observationState;
	uint8_t observations;
	uint8_t goodObservations;
	uint8_t filterStages;
	uint8_t predetectInterval;
	uint8_t cycleSlipCount;
	uint16_t postdetections;
	cereal_QcomGnss_MeasurementStatus_ptr measurementStatus;
	uint16_t carrierNoise;
	uint16_t rfLoss;
	int16_t latency;
	float filteredMeasurementFraction;
	uint32_t filteredMeasurementIntegral;
	float filteredTimeUncertainty;
	float filteredSpeed;
	float filteredSpeedUncertainty;
	float unfilteredMeasurementFraction;
	uint32_t unfilteredMeasurementIntegral;
	float unfilteredTimeUncertainty;
	float unfilteredSpeed;
	float unfilteredSpeedUncertainty;
	uint32_t multipathEstimate;
	float azimuth;
	float elevation;
	float dopplerAcceleration;
	float fineSpeed;
	float fineSpeedUncertainty;
	double carrierPhase;
	uint32_t fCount;
	uint16_t parityErrorCount;
	unsigned goodParity : 1;
};

static const size_t cereal_QcomGnss_DrMeasurementReport_SV_word_count = 12;

static const size_t cereal_QcomGnss_DrMeasurementReport_SV_pointer_count = 1;

static const size_t cereal_QcomGnss_DrMeasurementReport_SV_struct_bytes_count = 104;

struct cereal_QcomGnss_DrSvPolyReport {
	uint16_t svId;
	int8_t frequencyIndex;
	unsigned hasPosition : 1;
	unsigned hasIono : 1;
	unsigned hasTropo : 1;
	unsigned hasElevation : 1;
	unsigned polyFromXtra : 1;
	unsigned hasSbasIono : 1;
	uint16_t iode;
	double t0;
	capn_list64 xyz0;
	capn_list64 xyzN;
	capn_list32 other;
	float positionUncertainty;
	float ionoDelay;
	float ionoDot;
	float sbasIonoDelay;
	float sbasIonoDot;
	float tropoDelay;
	float elevation;
	float elevationDot;
	float elevationUncertainty;
	capn_list64 velocityCoeff;
};

static const size_t cereal_QcomGnss_DrSvPolyReport_word_count = 7;

static const size_t cereal_QcomGnss_DrSvPolyReport_pointer_count = 4;

static const size_t cereal_QcomGnss_DrSvPolyReport_struct_bytes_count = 88;

struct cereal_LidarPts {
	capn_list16 r;
	capn_list16 theta;
	capn_list8 reflect;
	uint64_t idx;
	capn_data pkt;
};

static const size_t cereal_LidarPts_word_count = 1;

static const size_t cereal_LidarPts_pointer_count = 4;

static const size_t cereal_LidarPts_struct_bytes_count = 40;

struct cereal_ProcLog {
	cereal_ProcLog_CPUTimes_list cpuTimes;
	cereal_ProcLog_Mem_ptr mem;
	cereal_ProcLog_Process_list procs;
};

static const size_t cereal_ProcLog_word_count = 0;

static const size_t cereal_ProcLog_pointer_count = 3;

static const size_t cereal_ProcLog_struct_bytes_count = 24;

struct cereal_ProcLog_Process {
	int32_t pid;
	capn_text name;
	uint8_t state;
	int32_t ppid;
	float cpuUser;
	float cpuSystem;
	float cpuChildrenUser;
	float cpuChildrenSystem;
	int64_t priority;
	int32_t nice;
	int32_t numThreads;
	double startTime;
	uint64_t memVms;
	uint64_t memRss;
	int32_t processor;
	capn_ptr cmdline;
	capn_text exe;
};

static const size_t cereal_ProcLog_Process_word_count = 9;

static const size_t cereal_ProcLog_Process_pointer_count = 3;

static const size_t cereal_ProcLog_Process_struct_bytes_count = 96;

struct cereal_ProcLog_CPUTimes {
	int64_t cpuNum;
	float user;
	float nice;
	float system;
	float idle;
	float iowait;
	float irq;
	float softirq;
};

static const size_t cereal_ProcLog_CPUTimes_word_count = 5;

static const size_t cereal_ProcLog_CPUTimes_pointer_count = 0;

static const size_t cereal_ProcLog_CPUTimes_struct_bytes_count = 40;

struct cereal_ProcLog_Mem {
	uint64_t total;
	uint64_t free;
	uint64_t available;
	uint64_t buffers;
	uint64_t cached;
	uint64_t active;
	uint64_t inactive;
	uint64_t shared;
};

static const size_t cereal_ProcLog_Mem_word_count = 8;

static const size_t cereal_ProcLog_Mem_pointer_count = 0;

static const size_t cereal_ProcLog_Mem_struct_bytes_count = 64;
enum cereal_UbloxGnss_which {
	cereal_UbloxGnss_measurementReport = 0,
	cereal_UbloxGnss_ephemeris = 1,
	cereal_UbloxGnss_ionoData = 2,
	cereal_UbloxGnss_hwStatus = 3
};

struct cereal_UbloxGnss {
	enum cereal_UbloxGnss_which which;
	union {
		cereal_UbloxGnss_MeasurementReport_ptr measurementReport;
		cereal_UbloxGnss_Ephemeris_ptr ephemeris;
		cereal_UbloxGnss_IonoData_ptr ionoData;
		cereal_UbloxGnss_HwStatus_ptr hwStatus;
	};
};

static const size_t cereal_UbloxGnss_word_count = 1;

static const size_t cereal_UbloxGnss_pointer_count = 1;

static const size_t cereal_UbloxGnss_struct_bytes_count = 16;

struct cereal_UbloxGnss_MeasurementReport {
	double rcvTow;
	uint16_t gpsWeek;
	uint16_t leapSeconds;
	cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr receiverStatus;
	uint8_t numMeas;
	cereal_UbloxGnss_MeasurementReport_Measurement_list measurements;
};

static const size_t cereal_UbloxGnss_MeasurementReport_word_count = 2;

static const size_t cereal_UbloxGnss_MeasurementReport_pointer_count = 2;

static const size_t cereal_UbloxGnss_MeasurementReport_struct_bytes_count = 32;

struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus {
	unsigned leapSecValid : 1;
	unsigned clkReset : 1;
};

static const size_t cereal_UbloxGnss_MeasurementReport_ReceiverStatus_word_count = 1;

static const size_t cereal_UbloxGnss_MeasurementReport_ReceiverStatus_pointer_count = 0;

static const size_t cereal_UbloxGnss_MeasurementReport_ReceiverStatus_struct_bytes_count = 8;

struct cereal_UbloxGnss_MeasurementReport_Measurement {
	uint8_t svId;
	cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr trackingStatus;
	double pseudorange;
	double carrierCycles;
	float doppler;
	uint8_t gnssId;
	uint8_t glonassFrequencyIndex;
	uint16_t locktime;
	uint8_t cno;
	float pseudorangeStdev;
	float carrierPhaseStdev;
	float dopplerStdev;
	uint8_t sigId;
};

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_word_count = 5;

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_pointer_count = 1;

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_struct_bytes_count = 48;

struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus {
	unsigned pseudorangeValid : 1;
	unsigned carrierPhaseValid : 1;
	unsigned halfCycleValid : 1;
	unsigned halfCycleSubtracted : 1;
};

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_word_count = 1;

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_pointer_count = 0;

static const size_t cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_struct_bytes_count = 8;

struct cereal_UbloxGnss_Ephemeris {
	uint16_t svId;
	uint16_t year;
	uint16_t month;
	uint16_t day;
	uint16_t hour;
	uint16_t minute;
	float second;
	double af0;
	double af1;
	double af2;
	double iode;
	double crs;
	double deltaN;
	double m0;
	double cuc;
	double ecc;
	double cus;
	double a;
	double toe;
	double cic;
	double omega0;
	double cis;
	double i0;
	double crc;
	double omega;
	double omegaDot;
	double iDot;
	double codesL2;
	double gpsWeek;
	double l2;
	double svAcc;
	double svHealth;
	double tgd;
	double iodc;
	double transmissionTime;
	double fitInterval;
	double toc;
	unsigned ionoCoeffsValid : 1;
	capn_list64 ionoAlpha;
	capn_list64 ionoBeta;
};

static const size_t cereal_UbloxGnss_Ephemeris_word_count = 33;

static const size_t cereal_UbloxGnss_Ephemeris_pointer_count = 2;

static const size_t cereal_UbloxGnss_Ephemeris_struct_bytes_count = 280;

struct cereal_UbloxGnss_IonoData {
	uint32_t svHealth;
	double tow;
	double gpsWeek;
	capn_list64 ionoAlpha;
	capn_list64 ionoBeta;
	unsigned healthValid : 1;
	unsigned ionoCoeffsValid : 1;
};

static const size_t cereal_UbloxGnss_IonoData_word_count = 3;

static const size_t cereal_UbloxGnss_IonoData_pointer_count = 2;

static const size_t cereal_UbloxGnss_IonoData_struct_bytes_count = 40;

struct cereal_UbloxGnss_HwStatus {
	uint16_t noisePerMS;
	uint16_t agcCnt;
	enum cereal_UbloxGnss_HwStatus_AntennaSupervisorState aStatus;
	enum cereal_UbloxGnss_HwStatus_AntennaPowerStatus aPower;
	uint8_t jamInd;
};

static const size_t cereal_UbloxGnss_HwStatus_word_count = 2;

static const size_t cereal_UbloxGnss_HwStatus_pointer_count = 0;

static const size_t cereal_UbloxGnss_HwStatus_struct_bytes_count = 16;

struct cereal_Clocks {
	uint64_t bootTimeNanos;
	uint64_t monotonicNanos;
	uint64_t monotonicRawNanos;
	uint64_t wallTimeNanos;
	uint64_t modemUptimeMillis;
};

static const size_t cereal_Clocks_word_count = 5;

static const size_t cereal_Clocks_pointer_count = 0;

static const size_t cereal_Clocks_struct_bytes_count = 40;

struct cereal_LiveMpcData {
	capn_list32 x;
	capn_list32 y;
	capn_list32 psi;
	capn_list32 delta;
	uint32_t qpIterations;
	uint64_t calculationTime;
	double cost;
};

static const size_t cereal_LiveMpcData_word_count = 3;

static const size_t cereal_LiveMpcData_pointer_count = 4;

static const size_t cereal_LiveMpcData_struct_bytes_count = 56;

struct cereal_LiveLongitudinalMpcData {
	capn_list32 xEgo;
	capn_list32 vEgo;
	capn_list32 aEgo;
	capn_list32 xLead;
	capn_list32 vLead;
	capn_list32 aLead;
	float aLeadTau;
	uint32_t qpIterations;
	uint32_t mpcId;
	uint64_t calculationTime;
	double cost;
};

static const size_t cereal_LiveLongitudinalMpcData_word_count = 4;

static const size_t cereal_LiveLongitudinalMpcData_pointer_count = 6;

static const size_t cereal_LiveLongitudinalMpcData_struct_bytes_count = 80;

struct cereal_ECEFPointDEPRECATED {
	float x;
	float y;
	float z;
};

static const size_t cereal_ECEFPointDEPRECATED_word_count = 2;

static const size_t cereal_ECEFPointDEPRECATED_pointer_count = 0;

static const size_t cereal_ECEFPointDEPRECATED_struct_bytes_count = 16;

struct cereal_ECEFPoint {
	double x;
	double y;
	double z;
};

static const size_t cereal_ECEFPoint_word_count = 3;

static const size_t cereal_ECEFPoint_pointer_count = 0;

static const size_t cereal_ECEFPoint_struct_bytes_count = 24;

struct cereal_GPSPlannerPoints {
	cereal_ECEFPointDEPRECATED_ptr curPosDEPRECATED;
	cereal_ECEFPointDEPRECATED_list pointsDEPRECATED;
	cereal_ECEFPoint_ptr curPos;
	cereal_ECEFPoint_list points;
	unsigned valid : 1;
	capn_text trackName;
	float speedLimit;
	float accelTarget;
};

static const size_t cereal_GPSPlannerPoints_word_count = 2;

static const size_t cereal_GPSPlannerPoints_pointer_count = 5;

static const size_t cereal_GPSPlannerPoints_struct_bytes_count = 56;

struct cereal_GPSPlannerPlan {
	unsigned valid : 1;
	capn_list32 poly;
	capn_text trackName;
	float speed;
	float acceleration;
	cereal_ECEFPointDEPRECATED_list pointsDEPRECATED;
	cereal_ECEFPoint_list points;
	float xLookahead;
};

static const size_t cereal_GPSPlannerPlan_word_count = 2;

static const size_t cereal_GPSPlannerPlan_pointer_count = 4;

static const size_t cereal_GPSPlannerPlan_struct_bytes_count = 48;

struct cereal_TrafficEvent {
	enum cereal_TrafficEvent_Type type;
	float distance;
	enum cereal_TrafficEvent_Action action;
	unsigned resuming : 1;
};

static const size_t cereal_TrafficEvent_word_count = 2;

static const size_t cereal_TrafficEvent_pointer_count = 0;

static const size_t cereal_TrafficEvent_struct_bytes_count = 16;

struct cereal_OrbslamCorrection {
	uint64_t correctionMonoTime;
	capn_list64 prePositionECEF;
	capn_list64 postPositionECEF;
	capn_list32 prePoseQuatECEF;
	capn_list32 postPoseQuatECEF;
	uint32_t numInliers;
};

static const size_t cereal_OrbslamCorrection_word_count = 2;

static const size_t cereal_OrbslamCorrection_pointer_count = 4;

static const size_t cereal_OrbslamCorrection_struct_bytes_count = 48;

struct cereal_OrbObservation {
	uint64_t observationMonoTime;
	capn_list32 normalizedCoordinates;
	capn_list64 locationECEF;
	uint32_t matchDistance;
};

static const size_t cereal_OrbObservation_word_count = 2;

static const size_t cereal_OrbObservation_pointer_count = 2;

static const size_t cereal_OrbObservation_struct_bytes_count = 32;

struct cereal_UiNavigationEvent {
	enum cereal_UiNavigationEvent_Type type;
	enum cereal_UiNavigationEvent_Status status;
	float distanceTo;
	cereal_ECEFPointDEPRECATED_ptr endRoadPointDEPRECATED;
	cereal_ECEFPoint_ptr endRoadPoint;
};

static const size_t cereal_UiNavigationEvent_word_count = 1;

static const size_t cereal_UiNavigationEvent_pointer_count = 2;

static const size_t cereal_UiNavigationEvent_struct_bytes_count = 24;

struct cereal_UiLayoutState {
	enum cereal_UiLayoutState_App activeApp;
	unsigned sidebarCollapsed : 1;
	unsigned mapEnabled : 1;
};

static const size_t cereal_UiLayoutState_word_count = 1;

static const size_t cereal_UiLayoutState_pointer_count = 0;

static const size_t cereal_UiLayoutState_struct_bytes_count = 8;

struct cereal_Joystick {
	capn_list32 axes;
	capn_list1 buttons;
};

static const size_t cereal_Joystick_word_count = 0;

static const size_t cereal_Joystick_pointer_count = 2;

static const size_t cereal_Joystick_struct_bytes_count = 16;

struct cereal_OrbOdometry {
	uint64_t startMonoTime;
	uint64_t endMonoTime;
	capn_list64 f;
	double err;
	int32_t inliers;
	capn_list16 matches;
};

static const size_t cereal_OrbOdometry_word_count = 4;

static const size_t cereal_OrbOdometry_pointer_count = 2;

static const size_t cereal_OrbOdometry_struct_bytes_count = 48;

struct cereal_OrbFeatures {
	uint64_t timestampEof;
	capn_list32 xs;
	capn_list32 ys;
	capn_data descriptors;
	capn_list8 octaves;
	uint64_t timestampLastEof;
	capn_list16 matches;
};

static const size_t cereal_OrbFeatures_word_count = 2;

static const size_t cereal_OrbFeatures_pointer_count = 5;

static const size_t cereal_OrbFeatures_struct_bytes_count = 56;

struct cereal_OrbFeaturesSummary {
	uint64_t timestampEof;
	uint64_t timestampLastEof;
	uint16_t featureCount;
	uint16_t matchCount;
	uint64_t computeNs;
};

static const size_t cereal_OrbFeaturesSummary_word_count = 4;

static const size_t cereal_OrbFeaturesSummary_pointer_count = 0;

static const size_t cereal_OrbFeaturesSummary_struct_bytes_count = 32;

struct cereal_OrbKeyFrame {
	uint64_t id;
	cereal_ECEFPoint_ptr pos;
	cereal_ECEFPoint_list dpos;
	capn_data descriptors;
};

static const size_t cereal_OrbKeyFrame_word_count = 1;

static const size_t cereal_OrbKeyFrame_pointer_count = 3;

static const size_t cereal_OrbKeyFrame_struct_bytes_count = 32;

struct cereal_DriverState {
	uint32_t frameId;
	capn_list32 descriptorDEPRECATED;
	float stdDEPRECATED;
	capn_list32 faceOrientation;
	capn_list32 facePosition;
	float faceProb;
	float leftEyeProb;
	float rightEyeProb;
	float leftBlinkProb;
	float rightBlinkProb;
	float irPwrDEPRECATED;
	capn_list32 faceOrientationStd;
	capn_list32 facePositionStd;
};

static const size_t cereal_DriverState_word_count = 4;

static const size_t cereal_DriverState_pointer_count = 5;

static const size_t cereal_DriverState_struct_bytes_count = 72;

struct cereal_DMonitoringState {
	cereal_CarEvent_list events;
	unsigned faceDetected : 1;
	unsigned isDistracted : 1;
	float awarenessStatus;
	unsigned isRHD : 1;
	unsigned rhdChecked : 1;
	float posePitchOffset;
	uint32_t posePitchValidCount;
	float poseYawOffset;
	uint32_t poseYawValidCount;
	float stepChange;
	float awarenessActive;
	float awarenessPassive;
	unsigned isLowStd : 1;
	uint32_t hiStdCount;
};

static const size_t cereal_DMonitoringState_word_count = 5;

static const size_t cereal_DMonitoringState_pointer_count = 1;

static const size_t cereal_DMonitoringState_struct_bytes_count = 48;

struct cereal_Boot {
	uint64_t wallTimeNanos;
	capn_data lastKmsg;
	capn_data lastPmsg;
};

static const size_t cereal_Boot_word_count = 1;

static const size_t cereal_Boot_pointer_count = 2;

static const size_t cereal_Boot_struct_bytes_count = 24;

struct cereal_LiveParametersData {
	unsigned valid : 1;
	float gyroBias;
	float angleOffset;
	float angleOffsetAverage;
	float stiffnessFactor;
	float steerRatio;
	unsigned sensorValid : 1;
	float yawRate;
	float posenetSpeed;
	unsigned posenetValid : 1;
};

static const size_t cereal_LiveParametersData_word_count = 4;

static const size_t cereal_LiveParametersData_pointer_count = 0;

static const size_t cereal_LiveParametersData_struct_bytes_count = 32;

struct cereal_LiveMapData {
	unsigned speedLimitValid : 1;
	float speedLimit;
	unsigned speedAdvisoryValid : 1;
	float speedAdvisory;
	unsigned speedLimitAheadValid : 1;
	float speedLimitAhead;
	float speedLimitAheadDistance;
	unsigned curvatureValid : 1;
	float curvature;
	uint64_t wayId;
	capn_list32 roadX;
	capn_list32 roadY;
	cereal_GpsLocationData_ptr lastGps;
	capn_list32 roadCurvatureX;
	capn_list32 roadCurvature;
	float distToTurn;
	unsigned mapValid : 1;
};

static const size_t cereal_LiveMapData_word_count = 5;

static const size_t cereal_LiveMapData_pointer_count = 5;

static const size_t cereal_LiveMapData_struct_bytes_count = 80;

struct cereal_CameraOdometry {
	uint32_t frameId;
	uint64_t timestampEof;
	capn_list32 trans;
	capn_list32 rot;
	capn_list32 transStd;
	capn_list32 rotStd;
};

static const size_t cereal_CameraOdometry_word_count = 2;

static const size_t cereal_CameraOdometry_pointer_count = 4;

static const size_t cereal_CameraOdometry_struct_bytes_count = 48;

struct cereal_KalmanOdometry {
	capn_list32 trans;
	capn_list32 rot;
	capn_list32 transStd;
	capn_list32 rotStd;
};

static const size_t cereal_KalmanOdometry_word_count = 0;

static const size_t cereal_KalmanOdometry_pointer_count = 4;

static const size_t cereal_KalmanOdometry_struct_bytes_count = 32;
enum cereal_Event_which {
	cereal_Event_initData = 0,
	cereal_Event_frame = 1,
	cereal_Event_gpsNMEA = 2,
	cereal_Event_sensorEventDEPRECATED = 3,
	cereal_Event_can = 4,
	cereal_Event_thermal = 5,
	cereal_Event_controlsState = 6,
	cereal_Event_liveEventDEPRECATED = 7,
	cereal_Event_model = 8,
	cereal_Event_features = 9,
	cereal_Event_sensorEvents = 10,
	cereal_Event_health = 11,
	cereal_Event_radarState = 12,
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
	cereal_Event_qcomGnss = 30,
	cereal_Event_lidarPts = 31,
	cereal_Event_procLog = 32,
	cereal_Event_ubloxGnss = 33,
	cereal_Event_clocks = 34,
	cereal_Event_liveMpc = 35,
	cereal_Event_liveLongitudinalMpc = 36,
	cereal_Event_navStatus = 37,
	cereal_Event_ubloxRaw = 38,
	cereal_Event_gpsPlannerPoints = 39,
	cereal_Event_gpsPlannerPlan = 40,
	cereal_Event_applanixRaw = 41,
	cereal_Event_trafficEvents = 42,
	cereal_Event_liveLocationTiming = 43,
	cereal_Event_orbslamCorrectionDEPRECATED = 44,
	cereal_Event_liveLocationCorrected = 45,
	cereal_Event_orbObservation = 46,
	cereal_Event_gpsLocationExternal = 47,
	cereal_Event_location = 48,
	cereal_Event_uiNavigationEvent = 49,
	cereal_Event_liveLocationKalman = 50,
	cereal_Event_testJoystick = 51,
	cereal_Event_orbOdometry = 52,
	cereal_Event_orbFeatures = 53,
	cereal_Event_applanixLocation = 54,
	cereal_Event_orbKeyFrame = 55,
	cereal_Event_uiLayoutState = 56,
	cereal_Event_orbFeaturesSummary = 57,
	cereal_Event_driverState = 58,
	cereal_Event_boot = 59,
	cereal_Event_liveParameters = 60,
	cereal_Event_liveMapData = 61,
	cereal_Event_cameraOdometry = 62,
	cereal_Event_pathPlan = 63,
	cereal_Event_kalmanOdometry = 64,
	cereal_Event_thumbnail = 65,
	cereal_Event_carEvents = 66,
	cereal_Event_carParams = 67,
	cereal_Event_frontFrame = 68,
	cereal_Event_dMonitoringState = 69
};

struct cereal_Event {
	uint64_t logMonoTime;
	unsigned valid : 1;
	enum cereal_Event_which which;
	union {
		cereal_InitData_ptr initData;
		cereal_FrameData_ptr frame;
		cereal_GPSNMEAData_ptr gpsNMEA;
		cereal_SensorEventData_ptr sensorEventDEPRECATED;
		cereal_CanData_list can;
		cereal_ThermalData_ptr thermal;
		cereal_ControlsState_ptr controlsState;
		cereal_LiveEventData_list liveEventDEPRECATED;
		cereal_ModelData_ptr model;
		cereal_CalibrationFeatures_ptr features;
		cereal_SensorEventData_list sensorEvents;
		cereal_HealthData_ptr health;
		cereal_RadarState_ptr radarState;
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
		cereal_LidarPts_ptr lidarPts;
		cereal_ProcLog_ptr procLog;
		cereal_UbloxGnss_ptr ubloxGnss;
		cereal_Clocks_ptr clocks;
		cereal_LiveMpcData_ptr liveMpc;
		cereal_LiveLongitudinalMpcData_ptr liveLongitudinalMpc;
		cereal_NavStatus_ptr navStatus;
		capn_data ubloxRaw;
		cereal_GPSPlannerPoints_ptr gpsPlannerPoints;
		cereal_GPSPlannerPlan_ptr gpsPlannerPlan;
		capn_data applanixRaw;
		cereal_TrafficEvent_list trafficEvents;
		cereal_LiveLocationData_ptr liveLocationTiming;
		cereal_OrbslamCorrection_ptr orbslamCorrectionDEPRECATED;
		cereal_LiveLocationData_ptr liveLocationCorrected;
		cereal_OrbObservation_list orbObservation;
		cereal_GpsLocationData_ptr gpsLocationExternal;
		cereal_LiveLocationData_ptr location;
		cereal_UiNavigationEvent_ptr uiNavigationEvent;
		cereal_LiveLocationData_ptr liveLocationKalman;
		cereal_Joystick_ptr testJoystick;
		cereal_OrbOdometry_ptr orbOdometry;
		cereal_OrbFeatures_ptr orbFeatures;
		cereal_LiveLocationData_ptr applanixLocation;
		cereal_OrbKeyFrame_ptr orbKeyFrame;
		cereal_UiLayoutState_ptr uiLayoutState;
		cereal_OrbFeaturesSummary_ptr orbFeaturesSummary;
		cereal_DriverState_ptr driverState;
		cereal_Boot_ptr boot;
		cereal_LiveParametersData_ptr liveParameters;
		cereal_LiveMapData_ptr liveMapData;
		cereal_CameraOdometry_ptr cameraOdometry;
		cereal_PathPlan_ptr pathPlan;
		cereal_KalmanOdometry_ptr kalmanOdometry;
		cereal_Thumbnail_ptr thumbnail;
		cereal_CarEvent_list carEvents;
		cereal_CarParams_ptr carParams;
		cereal_FrameData_ptr frontFrame;
		cereal_DMonitoringState_ptr dMonitoringState;
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
cereal_InitData_IosBuildInfo_ptr cereal_new_InitData_IosBuildInfo(struct capn_segment*);
cereal_InitData_PandaInfo_ptr cereal_new_InitData_PandaInfo(struct capn_segment*);
cereal_FrameData_ptr cereal_new_FrameData(struct capn_segment*);
cereal_FrameData_AndroidCaptureResult_ptr cereal_new_FrameData_AndroidCaptureResult(struct capn_segment*);
cereal_Thumbnail_ptr cereal_new_Thumbnail(struct capn_segment*);
cereal_GPSNMEAData_ptr cereal_new_GPSNMEAData(struct capn_segment*);
cereal_SensorEventData_ptr cereal_new_SensorEventData(struct capn_segment*);
cereal_SensorEventData_SensorVec_ptr cereal_new_SensorEventData_SensorVec(struct capn_segment*);
cereal_GpsLocationData_ptr cereal_new_GpsLocationData(struct capn_segment*);
cereal_CanData_ptr cereal_new_CanData(struct capn_segment*);
cereal_ThermalData_ptr cereal_new_ThermalData(struct capn_segment*);
cereal_HealthData_ptr cereal_new_HealthData(struct capn_segment*);
cereal_LiveUI_ptr cereal_new_LiveUI(struct capn_segment*);
cereal_RadarState_ptr cereal_new_RadarState(struct capn_segment*);
cereal_RadarState_LeadData_ptr cereal_new_RadarState_LeadData(struct capn_segment*);
cereal_LiveCalibrationData_ptr cereal_new_LiveCalibrationData(struct capn_segment*);
cereal_LiveTracks_ptr cereal_new_LiveTracks(struct capn_segment*);
cereal_ControlsState_ptr cereal_new_ControlsState(struct capn_segment*);
cereal_ControlsState_LateralINDIState_ptr cereal_new_ControlsState_LateralINDIState(struct capn_segment*);
cereal_ControlsState_LateralPIDState_ptr cereal_new_ControlsState_LateralPIDState(struct capn_segment*);
cereal_ControlsState_LateralLQRState_ptr cereal_new_ControlsState_LateralLQRState(struct capn_segment*);
cereal_LiveEventData_ptr cereal_new_LiveEventData(struct capn_segment*);
cereal_ModelData_ptr cereal_new_ModelData(struct capn_segment*);
cereal_ModelData_PathData_ptr cereal_new_ModelData_PathData(struct capn_segment*);
cereal_ModelData_LeadData_ptr cereal_new_ModelData_LeadData(struct capn_segment*);
cereal_ModelData_ModelSettings_ptr cereal_new_ModelData_ModelSettings(struct capn_segment*);
cereal_ModelData_MetaData_ptr cereal_new_ModelData_MetaData(struct capn_segment*);
cereal_ModelData_LongitudinalData_ptr cereal_new_ModelData_LongitudinalData(struct capn_segment*);
cereal_CalibrationFeatures_ptr cereal_new_CalibrationFeatures(struct capn_segment*);
cereal_EncodeIndex_ptr cereal_new_EncodeIndex(struct capn_segment*);
cereal_AndroidLogEntry_ptr cereal_new_AndroidLogEntry(struct capn_segment*);
cereal_LogRotate_ptr cereal_new_LogRotate(struct capn_segment*);
cereal_Plan_ptr cereal_new_Plan(struct capn_segment*);
cereal_Plan_GpsTrajectory_ptr cereal_new_Plan_GpsTrajectory(struct capn_segment*);
cereal_PathPlan_ptr cereal_new_PathPlan(struct capn_segment*);
cereal_LiveLocationData_ptr cereal_new_LiveLocationData(struct capn_segment*);
cereal_LiveLocationData_Accuracy_ptr cereal_new_LiveLocationData_Accuracy(struct capn_segment*);
cereal_EthernetPacket_ptr cereal_new_EthernetPacket(struct capn_segment*);
cereal_NavUpdate_ptr cereal_new_NavUpdate(struct capn_segment*);
cereal_NavUpdate_LatLng_ptr cereal_new_NavUpdate_LatLng(struct capn_segment*);
cereal_NavUpdate_Segment_ptr cereal_new_NavUpdate_Segment(struct capn_segment*);
cereal_NavStatus_ptr cereal_new_NavStatus(struct capn_segment*);
cereal_NavStatus_Address_ptr cereal_new_NavStatus_Address(struct capn_segment*);
cereal_CellInfo_ptr cereal_new_CellInfo(struct capn_segment*);
cereal_WifiScan_ptr cereal_new_WifiScan(struct capn_segment*);
cereal_AndroidGnss_ptr cereal_new_AndroidGnss(struct capn_segment*);
cereal_AndroidGnss_Measurements_ptr cereal_new_AndroidGnss_Measurements(struct capn_segment*);
cereal_AndroidGnss_Measurements_Clock_ptr cereal_new_AndroidGnss_Measurements_Clock(struct capn_segment*);
cereal_AndroidGnss_Measurements_Measurement_ptr cereal_new_AndroidGnss_Measurements_Measurement(struct capn_segment*);
cereal_AndroidGnss_NavigationMessage_ptr cereal_new_AndroidGnss_NavigationMessage(struct capn_segment*);
cereal_QcomGnss_ptr cereal_new_QcomGnss(struct capn_segment*);
cereal_QcomGnss_MeasurementStatus_ptr cereal_new_QcomGnss_MeasurementStatus(struct capn_segment*);
cereal_QcomGnss_MeasurementReport_ptr cereal_new_QcomGnss_MeasurementReport(struct capn_segment*);
cereal_QcomGnss_MeasurementReport_SV_ptr cereal_new_QcomGnss_MeasurementReport_SV(struct capn_segment*);
cereal_QcomGnss_ClockReport_ptr cereal_new_QcomGnss_ClockReport(struct capn_segment*);
cereal_QcomGnss_DrMeasurementReport_ptr cereal_new_QcomGnss_DrMeasurementReport(struct capn_segment*);
cereal_QcomGnss_DrMeasurementReport_SV_ptr cereal_new_QcomGnss_DrMeasurementReport_SV(struct capn_segment*);
cereal_QcomGnss_DrSvPolyReport_ptr cereal_new_QcomGnss_DrSvPolyReport(struct capn_segment*);
cereal_LidarPts_ptr cereal_new_LidarPts(struct capn_segment*);
cereal_ProcLog_ptr cereal_new_ProcLog(struct capn_segment*);
cereal_ProcLog_Process_ptr cereal_new_ProcLog_Process(struct capn_segment*);
cereal_ProcLog_CPUTimes_ptr cereal_new_ProcLog_CPUTimes(struct capn_segment*);
cereal_ProcLog_Mem_ptr cereal_new_ProcLog_Mem(struct capn_segment*);
cereal_UbloxGnss_ptr cereal_new_UbloxGnss(struct capn_segment*);
cereal_UbloxGnss_MeasurementReport_ptr cereal_new_UbloxGnss_MeasurementReport(struct capn_segment*);
cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr cereal_new_UbloxGnss_MeasurementReport_ReceiverStatus(struct capn_segment*);
cereal_UbloxGnss_MeasurementReport_Measurement_ptr cereal_new_UbloxGnss_MeasurementReport_Measurement(struct capn_segment*);
cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr cereal_new_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct capn_segment*);
cereal_UbloxGnss_Ephemeris_ptr cereal_new_UbloxGnss_Ephemeris(struct capn_segment*);
cereal_UbloxGnss_IonoData_ptr cereal_new_UbloxGnss_IonoData(struct capn_segment*);
cereal_UbloxGnss_HwStatus_ptr cereal_new_UbloxGnss_HwStatus(struct capn_segment*);
cereal_Clocks_ptr cereal_new_Clocks(struct capn_segment*);
cereal_LiveMpcData_ptr cereal_new_LiveMpcData(struct capn_segment*);
cereal_LiveLongitudinalMpcData_ptr cereal_new_LiveLongitudinalMpcData(struct capn_segment*);
cereal_ECEFPointDEPRECATED_ptr cereal_new_ECEFPointDEPRECATED(struct capn_segment*);
cereal_ECEFPoint_ptr cereal_new_ECEFPoint(struct capn_segment*);
cereal_GPSPlannerPoints_ptr cereal_new_GPSPlannerPoints(struct capn_segment*);
cereal_GPSPlannerPlan_ptr cereal_new_GPSPlannerPlan(struct capn_segment*);
cereal_TrafficEvent_ptr cereal_new_TrafficEvent(struct capn_segment*);
cereal_OrbslamCorrection_ptr cereal_new_OrbslamCorrection(struct capn_segment*);
cereal_OrbObservation_ptr cereal_new_OrbObservation(struct capn_segment*);
cereal_UiNavigationEvent_ptr cereal_new_UiNavigationEvent(struct capn_segment*);
cereal_UiLayoutState_ptr cereal_new_UiLayoutState(struct capn_segment*);
cereal_Joystick_ptr cereal_new_Joystick(struct capn_segment*);
cereal_OrbOdometry_ptr cereal_new_OrbOdometry(struct capn_segment*);
cereal_OrbFeatures_ptr cereal_new_OrbFeatures(struct capn_segment*);
cereal_OrbFeaturesSummary_ptr cereal_new_OrbFeaturesSummary(struct capn_segment*);
cereal_OrbKeyFrame_ptr cereal_new_OrbKeyFrame(struct capn_segment*);
cereal_DriverState_ptr cereal_new_DriverState(struct capn_segment*);
cereal_DMonitoringState_ptr cereal_new_DMonitoringState(struct capn_segment*);
cereal_Boot_ptr cereal_new_Boot(struct capn_segment*);
cereal_LiveParametersData_ptr cereal_new_LiveParametersData(struct capn_segment*);
cereal_LiveMapData_ptr cereal_new_LiveMapData(struct capn_segment*);
cereal_CameraOdometry_ptr cereal_new_CameraOdometry(struct capn_segment*);
cereal_KalmanOdometry_ptr cereal_new_KalmanOdometry(struct capn_segment*);
cereal_Event_ptr cereal_new_Event(struct capn_segment*);

cereal_Map_list cereal_new_Map_list(struct capn_segment*, int len);
cereal_Map_Entry_list cereal_new_Map_Entry_list(struct capn_segment*, int len);
cereal_InitData_list cereal_new_InitData_list(struct capn_segment*, int len);
cereal_InitData_AndroidBuildInfo_list cereal_new_InitData_AndroidBuildInfo_list(struct capn_segment*, int len);
cereal_InitData_AndroidSensor_list cereal_new_InitData_AndroidSensor_list(struct capn_segment*, int len);
cereal_InitData_ChffrAndroidExtra_list cereal_new_InitData_ChffrAndroidExtra_list(struct capn_segment*, int len);
cereal_InitData_IosBuildInfo_list cereal_new_InitData_IosBuildInfo_list(struct capn_segment*, int len);
cereal_InitData_PandaInfo_list cereal_new_InitData_PandaInfo_list(struct capn_segment*, int len);
cereal_FrameData_list cereal_new_FrameData_list(struct capn_segment*, int len);
cereal_FrameData_AndroidCaptureResult_list cereal_new_FrameData_AndroidCaptureResult_list(struct capn_segment*, int len);
cereal_Thumbnail_list cereal_new_Thumbnail_list(struct capn_segment*, int len);
cereal_GPSNMEAData_list cereal_new_GPSNMEAData_list(struct capn_segment*, int len);
cereal_SensorEventData_list cereal_new_SensorEventData_list(struct capn_segment*, int len);
cereal_SensorEventData_SensorVec_list cereal_new_SensorEventData_SensorVec_list(struct capn_segment*, int len);
cereal_GpsLocationData_list cereal_new_GpsLocationData_list(struct capn_segment*, int len);
cereal_CanData_list cereal_new_CanData_list(struct capn_segment*, int len);
cereal_ThermalData_list cereal_new_ThermalData_list(struct capn_segment*, int len);
cereal_HealthData_list cereal_new_HealthData_list(struct capn_segment*, int len);
cereal_LiveUI_list cereal_new_LiveUI_list(struct capn_segment*, int len);
cereal_RadarState_list cereal_new_RadarState_list(struct capn_segment*, int len);
cereal_RadarState_LeadData_list cereal_new_RadarState_LeadData_list(struct capn_segment*, int len);
cereal_LiveCalibrationData_list cereal_new_LiveCalibrationData_list(struct capn_segment*, int len);
cereal_LiveTracks_list cereal_new_LiveTracks_list(struct capn_segment*, int len);
cereal_ControlsState_list cereal_new_ControlsState_list(struct capn_segment*, int len);
cereal_ControlsState_LateralINDIState_list cereal_new_ControlsState_LateralINDIState_list(struct capn_segment*, int len);
cereal_ControlsState_LateralPIDState_list cereal_new_ControlsState_LateralPIDState_list(struct capn_segment*, int len);
cereal_ControlsState_LateralLQRState_list cereal_new_ControlsState_LateralLQRState_list(struct capn_segment*, int len);
cereal_LiveEventData_list cereal_new_LiveEventData_list(struct capn_segment*, int len);
cereal_ModelData_list cereal_new_ModelData_list(struct capn_segment*, int len);
cereal_ModelData_PathData_list cereal_new_ModelData_PathData_list(struct capn_segment*, int len);
cereal_ModelData_LeadData_list cereal_new_ModelData_LeadData_list(struct capn_segment*, int len);
cereal_ModelData_ModelSettings_list cereal_new_ModelData_ModelSettings_list(struct capn_segment*, int len);
cereal_ModelData_MetaData_list cereal_new_ModelData_MetaData_list(struct capn_segment*, int len);
cereal_ModelData_LongitudinalData_list cereal_new_ModelData_LongitudinalData_list(struct capn_segment*, int len);
cereal_CalibrationFeatures_list cereal_new_CalibrationFeatures_list(struct capn_segment*, int len);
cereal_EncodeIndex_list cereal_new_EncodeIndex_list(struct capn_segment*, int len);
cereal_AndroidLogEntry_list cereal_new_AndroidLogEntry_list(struct capn_segment*, int len);
cereal_LogRotate_list cereal_new_LogRotate_list(struct capn_segment*, int len);
cereal_Plan_list cereal_new_Plan_list(struct capn_segment*, int len);
cereal_Plan_GpsTrajectory_list cereal_new_Plan_GpsTrajectory_list(struct capn_segment*, int len);
cereal_PathPlan_list cereal_new_PathPlan_list(struct capn_segment*, int len);
cereal_LiveLocationData_list cereal_new_LiveLocationData_list(struct capn_segment*, int len);
cereal_LiveLocationData_Accuracy_list cereal_new_LiveLocationData_Accuracy_list(struct capn_segment*, int len);
cereal_EthernetPacket_list cereal_new_EthernetPacket_list(struct capn_segment*, int len);
cereal_NavUpdate_list cereal_new_NavUpdate_list(struct capn_segment*, int len);
cereal_NavUpdate_LatLng_list cereal_new_NavUpdate_LatLng_list(struct capn_segment*, int len);
cereal_NavUpdate_Segment_list cereal_new_NavUpdate_Segment_list(struct capn_segment*, int len);
cereal_NavStatus_list cereal_new_NavStatus_list(struct capn_segment*, int len);
cereal_NavStatus_Address_list cereal_new_NavStatus_Address_list(struct capn_segment*, int len);
cereal_CellInfo_list cereal_new_CellInfo_list(struct capn_segment*, int len);
cereal_WifiScan_list cereal_new_WifiScan_list(struct capn_segment*, int len);
cereal_AndroidGnss_list cereal_new_AndroidGnss_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_list cereal_new_AndroidGnss_Measurements_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_Clock_list cereal_new_AndroidGnss_Measurements_Clock_list(struct capn_segment*, int len);
cereal_AndroidGnss_Measurements_Measurement_list cereal_new_AndroidGnss_Measurements_Measurement_list(struct capn_segment*, int len);
cereal_AndroidGnss_NavigationMessage_list cereal_new_AndroidGnss_NavigationMessage_list(struct capn_segment*, int len);
cereal_QcomGnss_list cereal_new_QcomGnss_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementStatus_list cereal_new_QcomGnss_MeasurementStatus_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementReport_list cereal_new_QcomGnss_MeasurementReport_list(struct capn_segment*, int len);
cereal_QcomGnss_MeasurementReport_SV_list cereal_new_QcomGnss_MeasurementReport_SV_list(struct capn_segment*, int len);
cereal_QcomGnss_ClockReport_list cereal_new_QcomGnss_ClockReport_list(struct capn_segment*, int len);
cereal_QcomGnss_DrMeasurementReport_list cereal_new_QcomGnss_DrMeasurementReport_list(struct capn_segment*, int len);
cereal_QcomGnss_DrMeasurementReport_SV_list cereal_new_QcomGnss_DrMeasurementReport_SV_list(struct capn_segment*, int len);
cereal_QcomGnss_DrSvPolyReport_list cereal_new_QcomGnss_DrSvPolyReport_list(struct capn_segment*, int len);
cereal_LidarPts_list cereal_new_LidarPts_list(struct capn_segment*, int len);
cereal_ProcLog_list cereal_new_ProcLog_list(struct capn_segment*, int len);
cereal_ProcLog_Process_list cereal_new_ProcLog_Process_list(struct capn_segment*, int len);
cereal_ProcLog_CPUTimes_list cereal_new_ProcLog_CPUTimes_list(struct capn_segment*, int len);
cereal_ProcLog_Mem_list cereal_new_ProcLog_Mem_list(struct capn_segment*, int len);
cereal_UbloxGnss_list cereal_new_UbloxGnss_list(struct capn_segment*, int len);
cereal_UbloxGnss_MeasurementReport_list cereal_new_UbloxGnss_MeasurementReport_list(struct capn_segment*, int len);
cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list cereal_new_UbloxGnss_MeasurementReport_ReceiverStatus_list(struct capn_segment*, int len);
cereal_UbloxGnss_MeasurementReport_Measurement_list cereal_new_UbloxGnss_MeasurementReport_Measurement_list(struct capn_segment*, int len);
cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list cereal_new_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list(struct capn_segment*, int len);
cereal_UbloxGnss_Ephemeris_list cereal_new_UbloxGnss_Ephemeris_list(struct capn_segment*, int len);
cereal_UbloxGnss_IonoData_list cereal_new_UbloxGnss_IonoData_list(struct capn_segment*, int len);
cereal_UbloxGnss_HwStatus_list cereal_new_UbloxGnss_HwStatus_list(struct capn_segment*, int len);
cereal_Clocks_list cereal_new_Clocks_list(struct capn_segment*, int len);
cereal_LiveMpcData_list cereal_new_LiveMpcData_list(struct capn_segment*, int len);
cereal_LiveLongitudinalMpcData_list cereal_new_LiveLongitudinalMpcData_list(struct capn_segment*, int len);
cereal_ECEFPointDEPRECATED_list cereal_new_ECEFPointDEPRECATED_list(struct capn_segment*, int len);
cereal_ECEFPoint_list cereal_new_ECEFPoint_list(struct capn_segment*, int len);
cereal_GPSPlannerPoints_list cereal_new_GPSPlannerPoints_list(struct capn_segment*, int len);
cereal_GPSPlannerPlan_list cereal_new_GPSPlannerPlan_list(struct capn_segment*, int len);
cereal_TrafficEvent_list cereal_new_TrafficEvent_list(struct capn_segment*, int len);
cereal_OrbslamCorrection_list cereal_new_OrbslamCorrection_list(struct capn_segment*, int len);
cereal_OrbObservation_list cereal_new_OrbObservation_list(struct capn_segment*, int len);
cereal_UiNavigationEvent_list cereal_new_UiNavigationEvent_list(struct capn_segment*, int len);
cereal_UiLayoutState_list cereal_new_UiLayoutState_list(struct capn_segment*, int len);
cereal_Joystick_list cereal_new_Joystick_list(struct capn_segment*, int len);
cereal_OrbOdometry_list cereal_new_OrbOdometry_list(struct capn_segment*, int len);
cereal_OrbFeatures_list cereal_new_OrbFeatures_list(struct capn_segment*, int len);
cereal_OrbFeaturesSummary_list cereal_new_OrbFeaturesSummary_list(struct capn_segment*, int len);
cereal_OrbKeyFrame_list cereal_new_OrbKeyFrame_list(struct capn_segment*, int len);
cereal_DriverState_list cereal_new_DriverState_list(struct capn_segment*, int len);
cereal_DMonitoringState_list cereal_new_DMonitoringState_list(struct capn_segment*, int len);
cereal_Boot_list cereal_new_Boot_list(struct capn_segment*, int len);
cereal_LiveParametersData_list cereal_new_LiveParametersData_list(struct capn_segment*, int len);
cereal_LiveMapData_list cereal_new_LiveMapData_list(struct capn_segment*, int len);
cereal_CameraOdometry_list cereal_new_CameraOdometry_list(struct capn_segment*, int len);
cereal_KalmanOdometry_list cereal_new_KalmanOdometry_list(struct capn_segment*, int len);
cereal_Event_list cereal_new_Event_list(struct capn_segment*, int len);

void cereal_read_Map(struct cereal_Map*, cereal_Map_ptr);
void cereal_read_Map_Entry(struct cereal_Map_Entry*, cereal_Map_Entry_ptr);
void cereal_read_InitData(struct cereal_InitData*, cereal_InitData_ptr);
void cereal_read_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_ptr);
void cereal_read_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_ptr);
void cereal_read_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_ptr);
void cereal_read_InitData_IosBuildInfo(struct cereal_InitData_IosBuildInfo*, cereal_InitData_IosBuildInfo_ptr);
void cereal_read_InitData_PandaInfo(struct cereal_InitData_PandaInfo*, cereal_InitData_PandaInfo_ptr);
void cereal_read_FrameData(struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_read_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_ptr);
void cereal_read_Thumbnail(struct cereal_Thumbnail*, cereal_Thumbnail_ptr);
void cereal_read_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_read_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_read_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_read_GpsLocationData(struct cereal_GpsLocationData*, cereal_GpsLocationData_ptr);
void cereal_read_CanData(struct cereal_CanData*, cereal_CanData_ptr);
void cereal_read_ThermalData(struct cereal_ThermalData*, cereal_ThermalData_ptr);
void cereal_read_HealthData(struct cereal_HealthData*, cereal_HealthData_ptr);
void cereal_read_LiveUI(struct cereal_LiveUI*, cereal_LiveUI_ptr);
void cereal_read_RadarState(struct cereal_RadarState*, cereal_RadarState_ptr);
void cereal_read_RadarState_LeadData(struct cereal_RadarState_LeadData*, cereal_RadarState_LeadData_ptr);
void cereal_read_LiveCalibrationData(struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_ptr);
void cereal_read_LiveTracks(struct cereal_LiveTracks*, cereal_LiveTracks_ptr);
void cereal_read_ControlsState(struct cereal_ControlsState*, cereal_ControlsState_ptr);
void cereal_read_ControlsState_LateralINDIState(struct cereal_ControlsState_LateralINDIState*, cereal_ControlsState_LateralINDIState_ptr);
void cereal_read_ControlsState_LateralPIDState(struct cereal_ControlsState_LateralPIDState*, cereal_ControlsState_LateralPIDState_ptr);
void cereal_read_ControlsState_LateralLQRState(struct cereal_ControlsState_LateralLQRState*, cereal_ControlsState_LateralLQRState_ptr);
void cereal_read_LiveEventData(struct cereal_LiveEventData*, cereal_LiveEventData_ptr);
void cereal_read_ModelData(struct cereal_ModelData*, cereal_ModelData_ptr);
void cereal_read_ModelData_PathData(struct cereal_ModelData_PathData*, cereal_ModelData_PathData_ptr);
void cereal_read_ModelData_LeadData(struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_ptr);
void cereal_read_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_ptr);
void cereal_read_ModelData_MetaData(struct cereal_ModelData_MetaData*, cereal_ModelData_MetaData_ptr);
void cereal_read_ModelData_LongitudinalData(struct cereal_ModelData_LongitudinalData*, cereal_ModelData_LongitudinalData_ptr);
void cereal_read_CalibrationFeatures(struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_ptr);
void cereal_read_EncodeIndex(struct cereal_EncodeIndex*, cereal_EncodeIndex_ptr);
void cereal_read_AndroidLogEntry(struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_ptr);
void cereal_read_LogRotate(struct cereal_LogRotate*, cereal_LogRotate_ptr);
void cereal_read_Plan(struct cereal_Plan*, cereal_Plan_ptr);
void cereal_read_Plan_GpsTrajectory(struct cereal_Plan_GpsTrajectory*, cereal_Plan_GpsTrajectory_ptr);
void cereal_read_PathPlan(struct cereal_PathPlan*, cereal_PathPlan_ptr);
void cereal_read_LiveLocationData(struct cereal_LiveLocationData*, cereal_LiveLocationData_ptr);
void cereal_read_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_ptr);
void cereal_read_EthernetPacket(struct cereal_EthernetPacket*, cereal_EthernetPacket_ptr);
void cereal_read_NavUpdate(struct cereal_NavUpdate*, cereal_NavUpdate_ptr);
void cereal_read_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_ptr);
void cereal_read_NavUpdate_Segment(struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_ptr);
void cereal_read_NavStatus(struct cereal_NavStatus*, cereal_NavStatus_ptr);
void cereal_read_NavStatus_Address(struct cereal_NavStatus_Address*, cereal_NavStatus_Address_ptr);
void cereal_read_CellInfo(struct cereal_CellInfo*, cereal_CellInfo_ptr);
void cereal_read_WifiScan(struct cereal_WifiScan*, cereal_WifiScan_ptr);
void cereal_read_AndroidGnss(struct cereal_AndroidGnss*, cereal_AndroidGnss_ptr);
void cereal_read_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_ptr);
void cereal_read_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_ptr);
void cereal_read_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_ptr);
void cereal_read_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_ptr);
void cereal_read_QcomGnss(struct cereal_QcomGnss*, cereal_QcomGnss_ptr);
void cereal_read_QcomGnss_MeasurementStatus(struct cereal_QcomGnss_MeasurementStatus*, cereal_QcomGnss_MeasurementStatus_ptr);
void cereal_read_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_ptr);
void cereal_read_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_ptr);
void cereal_read_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_ptr);
void cereal_read_QcomGnss_DrMeasurementReport(struct cereal_QcomGnss_DrMeasurementReport*, cereal_QcomGnss_DrMeasurementReport_ptr);
void cereal_read_QcomGnss_DrMeasurementReport_SV(struct cereal_QcomGnss_DrMeasurementReport_SV*, cereal_QcomGnss_DrMeasurementReport_SV_ptr);
void cereal_read_QcomGnss_DrSvPolyReport(struct cereal_QcomGnss_DrSvPolyReport*, cereal_QcomGnss_DrSvPolyReport_ptr);
void cereal_read_LidarPts(struct cereal_LidarPts*, cereal_LidarPts_ptr);
void cereal_read_ProcLog(struct cereal_ProcLog*, cereal_ProcLog_ptr);
void cereal_read_ProcLog_Process(struct cereal_ProcLog_Process*, cereal_ProcLog_Process_ptr);
void cereal_read_ProcLog_CPUTimes(struct cereal_ProcLog_CPUTimes*, cereal_ProcLog_CPUTimes_ptr);
void cereal_read_ProcLog_Mem(struct cereal_ProcLog_Mem*, cereal_ProcLog_Mem_ptr);
void cereal_read_UbloxGnss(struct cereal_UbloxGnss*, cereal_UbloxGnss_ptr);
void cereal_read_UbloxGnss_MeasurementReport(struct cereal_UbloxGnss_MeasurementReport*, cereal_UbloxGnss_MeasurementReport_ptr);
void cereal_read_UbloxGnss_MeasurementReport_ReceiverStatus(struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus*, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr);
void cereal_read_UbloxGnss_MeasurementReport_Measurement(struct cereal_UbloxGnss_MeasurementReport_Measurement*, cereal_UbloxGnss_MeasurementReport_Measurement_ptr);
void cereal_read_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus*, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr);
void cereal_read_UbloxGnss_Ephemeris(struct cereal_UbloxGnss_Ephemeris*, cereal_UbloxGnss_Ephemeris_ptr);
void cereal_read_UbloxGnss_IonoData(struct cereal_UbloxGnss_IonoData*, cereal_UbloxGnss_IonoData_ptr);
void cereal_read_UbloxGnss_HwStatus(struct cereal_UbloxGnss_HwStatus*, cereal_UbloxGnss_HwStatus_ptr);
void cereal_read_Clocks(struct cereal_Clocks*, cereal_Clocks_ptr);
void cereal_read_LiveMpcData(struct cereal_LiveMpcData*, cereal_LiveMpcData_ptr);
void cereal_read_LiveLongitudinalMpcData(struct cereal_LiveLongitudinalMpcData*, cereal_LiveLongitudinalMpcData_ptr);
void cereal_read_ECEFPointDEPRECATED(struct cereal_ECEFPointDEPRECATED*, cereal_ECEFPointDEPRECATED_ptr);
void cereal_read_ECEFPoint(struct cereal_ECEFPoint*, cereal_ECEFPoint_ptr);
void cereal_read_GPSPlannerPoints(struct cereal_GPSPlannerPoints*, cereal_GPSPlannerPoints_ptr);
void cereal_read_GPSPlannerPlan(struct cereal_GPSPlannerPlan*, cereal_GPSPlannerPlan_ptr);
void cereal_read_TrafficEvent(struct cereal_TrafficEvent*, cereal_TrafficEvent_ptr);
void cereal_read_OrbslamCorrection(struct cereal_OrbslamCorrection*, cereal_OrbslamCorrection_ptr);
void cereal_read_OrbObservation(struct cereal_OrbObservation*, cereal_OrbObservation_ptr);
void cereal_read_UiNavigationEvent(struct cereal_UiNavigationEvent*, cereal_UiNavigationEvent_ptr);
void cereal_read_UiLayoutState(struct cereal_UiLayoutState*, cereal_UiLayoutState_ptr);
void cereal_read_Joystick(struct cereal_Joystick*, cereal_Joystick_ptr);
void cereal_read_OrbOdometry(struct cereal_OrbOdometry*, cereal_OrbOdometry_ptr);
void cereal_read_OrbFeatures(struct cereal_OrbFeatures*, cereal_OrbFeatures_ptr);
void cereal_read_OrbFeaturesSummary(struct cereal_OrbFeaturesSummary*, cereal_OrbFeaturesSummary_ptr);
void cereal_read_OrbKeyFrame(struct cereal_OrbKeyFrame*, cereal_OrbKeyFrame_ptr);
void cereal_read_DriverState(struct cereal_DriverState*, cereal_DriverState_ptr);
void cereal_read_DMonitoringState(struct cereal_DMonitoringState*, cereal_DMonitoringState_ptr);
void cereal_read_Boot(struct cereal_Boot*, cereal_Boot_ptr);
void cereal_read_LiveParametersData(struct cereal_LiveParametersData*, cereal_LiveParametersData_ptr);
void cereal_read_LiveMapData(struct cereal_LiveMapData*, cereal_LiveMapData_ptr);
void cereal_read_CameraOdometry(struct cereal_CameraOdometry*, cereal_CameraOdometry_ptr);
void cereal_read_KalmanOdometry(struct cereal_KalmanOdometry*, cereal_KalmanOdometry_ptr);
void cereal_read_Event(struct cereal_Event*, cereal_Event_ptr);

void cereal_write_Map(const struct cereal_Map*, cereal_Map_ptr);
void cereal_write_Map_Entry(const struct cereal_Map_Entry*, cereal_Map_Entry_ptr);
void cereal_write_InitData(const struct cereal_InitData*, cereal_InitData_ptr);
void cereal_write_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_ptr);
void cereal_write_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_ptr);
void cereal_write_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_ptr);
void cereal_write_InitData_IosBuildInfo(const struct cereal_InitData_IosBuildInfo*, cereal_InitData_IosBuildInfo_ptr);
void cereal_write_InitData_PandaInfo(const struct cereal_InitData_PandaInfo*, cereal_InitData_PandaInfo_ptr);
void cereal_write_FrameData(const struct cereal_FrameData*, cereal_FrameData_ptr);
void cereal_write_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_ptr);
void cereal_write_Thumbnail(const struct cereal_Thumbnail*, cereal_Thumbnail_ptr);
void cereal_write_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_ptr);
void cereal_write_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_ptr);
void cereal_write_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_ptr);
void cereal_write_GpsLocationData(const struct cereal_GpsLocationData*, cereal_GpsLocationData_ptr);
void cereal_write_CanData(const struct cereal_CanData*, cereal_CanData_ptr);
void cereal_write_ThermalData(const struct cereal_ThermalData*, cereal_ThermalData_ptr);
void cereal_write_HealthData(const struct cereal_HealthData*, cereal_HealthData_ptr);
void cereal_write_LiveUI(const struct cereal_LiveUI*, cereal_LiveUI_ptr);
void cereal_write_RadarState(const struct cereal_RadarState*, cereal_RadarState_ptr);
void cereal_write_RadarState_LeadData(const struct cereal_RadarState_LeadData*, cereal_RadarState_LeadData_ptr);
void cereal_write_LiveCalibrationData(const struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_ptr);
void cereal_write_LiveTracks(const struct cereal_LiveTracks*, cereal_LiveTracks_ptr);
void cereal_write_ControlsState(const struct cereal_ControlsState*, cereal_ControlsState_ptr);
void cereal_write_ControlsState_LateralINDIState(const struct cereal_ControlsState_LateralINDIState*, cereal_ControlsState_LateralINDIState_ptr);
void cereal_write_ControlsState_LateralPIDState(const struct cereal_ControlsState_LateralPIDState*, cereal_ControlsState_LateralPIDState_ptr);
void cereal_write_ControlsState_LateralLQRState(const struct cereal_ControlsState_LateralLQRState*, cereal_ControlsState_LateralLQRState_ptr);
void cereal_write_LiveEventData(const struct cereal_LiveEventData*, cereal_LiveEventData_ptr);
void cereal_write_ModelData(const struct cereal_ModelData*, cereal_ModelData_ptr);
void cereal_write_ModelData_PathData(const struct cereal_ModelData_PathData*, cereal_ModelData_PathData_ptr);
void cereal_write_ModelData_LeadData(const struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_ptr);
void cereal_write_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_ptr);
void cereal_write_ModelData_MetaData(const struct cereal_ModelData_MetaData*, cereal_ModelData_MetaData_ptr);
void cereal_write_ModelData_LongitudinalData(const struct cereal_ModelData_LongitudinalData*, cereal_ModelData_LongitudinalData_ptr);
void cereal_write_CalibrationFeatures(const struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_ptr);
void cereal_write_EncodeIndex(const struct cereal_EncodeIndex*, cereal_EncodeIndex_ptr);
void cereal_write_AndroidLogEntry(const struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_ptr);
void cereal_write_LogRotate(const struct cereal_LogRotate*, cereal_LogRotate_ptr);
void cereal_write_Plan(const struct cereal_Plan*, cereal_Plan_ptr);
void cereal_write_Plan_GpsTrajectory(const struct cereal_Plan_GpsTrajectory*, cereal_Plan_GpsTrajectory_ptr);
void cereal_write_PathPlan(const struct cereal_PathPlan*, cereal_PathPlan_ptr);
void cereal_write_LiveLocationData(const struct cereal_LiveLocationData*, cereal_LiveLocationData_ptr);
void cereal_write_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_ptr);
void cereal_write_EthernetPacket(const struct cereal_EthernetPacket*, cereal_EthernetPacket_ptr);
void cereal_write_NavUpdate(const struct cereal_NavUpdate*, cereal_NavUpdate_ptr);
void cereal_write_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_ptr);
void cereal_write_NavUpdate_Segment(const struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_ptr);
void cereal_write_NavStatus(const struct cereal_NavStatus*, cereal_NavStatus_ptr);
void cereal_write_NavStatus_Address(const struct cereal_NavStatus_Address*, cereal_NavStatus_Address_ptr);
void cereal_write_CellInfo(const struct cereal_CellInfo*, cereal_CellInfo_ptr);
void cereal_write_WifiScan(const struct cereal_WifiScan*, cereal_WifiScan_ptr);
void cereal_write_AndroidGnss(const struct cereal_AndroidGnss*, cereal_AndroidGnss_ptr);
void cereal_write_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_ptr);
void cereal_write_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_ptr);
void cereal_write_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_ptr);
void cereal_write_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_ptr);
void cereal_write_QcomGnss(const struct cereal_QcomGnss*, cereal_QcomGnss_ptr);
void cereal_write_QcomGnss_MeasurementStatus(const struct cereal_QcomGnss_MeasurementStatus*, cereal_QcomGnss_MeasurementStatus_ptr);
void cereal_write_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_ptr);
void cereal_write_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_ptr);
void cereal_write_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_ptr);
void cereal_write_QcomGnss_DrMeasurementReport(const struct cereal_QcomGnss_DrMeasurementReport*, cereal_QcomGnss_DrMeasurementReport_ptr);
void cereal_write_QcomGnss_DrMeasurementReport_SV(const struct cereal_QcomGnss_DrMeasurementReport_SV*, cereal_QcomGnss_DrMeasurementReport_SV_ptr);
void cereal_write_QcomGnss_DrSvPolyReport(const struct cereal_QcomGnss_DrSvPolyReport*, cereal_QcomGnss_DrSvPolyReport_ptr);
void cereal_write_LidarPts(const struct cereal_LidarPts*, cereal_LidarPts_ptr);
void cereal_write_ProcLog(const struct cereal_ProcLog*, cereal_ProcLog_ptr);
void cereal_write_ProcLog_Process(const struct cereal_ProcLog_Process*, cereal_ProcLog_Process_ptr);
void cereal_write_ProcLog_CPUTimes(const struct cereal_ProcLog_CPUTimes*, cereal_ProcLog_CPUTimes_ptr);
void cereal_write_ProcLog_Mem(const struct cereal_ProcLog_Mem*, cereal_ProcLog_Mem_ptr);
void cereal_write_UbloxGnss(const struct cereal_UbloxGnss*, cereal_UbloxGnss_ptr);
void cereal_write_UbloxGnss_MeasurementReport(const struct cereal_UbloxGnss_MeasurementReport*, cereal_UbloxGnss_MeasurementReport_ptr);
void cereal_write_UbloxGnss_MeasurementReport_ReceiverStatus(const struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus*, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_ptr);
void cereal_write_UbloxGnss_MeasurementReport_Measurement(const struct cereal_UbloxGnss_MeasurementReport_Measurement*, cereal_UbloxGnss_MeasurementReport_Measurement_ptr);
void cereal_write_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(const struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus*, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_ptr);
void cereal_write_UbloxGnss_Ephemeris(const struct cereal_UbloxGnss_Ephemeris*, cereal_UbloxGnss_Ephemeris_ptr);
void cereal_write_UbloxGnss_IonoData(const struct cereal_UbloxGnss_IonoData*, cereal_UbloxGnss_IonoData_ptr);
void cereal_write_UbloxGnss_HwStatus(const struct cereal_UbloxGnss_HwStatus*, cereal_UbloxGnss_HwStatus_ptr);
void cereal_write_Clocks(const struct cereal_Clocks*, cereal_Clocks_ptr);
void cereal_write_LiveMpcData(const struct cereal_LiveMpcData*, cereal_LiveMpcData_ptr);
void cereal_write_LiveLongitudinalMpcData(const struct cereal_LiveLongitudinalMpcData*, cereal_LiveLongitudinalMpcData_ptr);
void cereal_write_ECEFPointDEPRECATED(const struct cereal_ECEFPointDEPRECATED*, cereal_ECEFPointDEPRECATED_ptr);
void cereal_write_ECEFPoint(const struct cereal_ECEFPoint*, cereal_ECEFPoint_ptr);
void cereal_write_GPSPlannerPoints(const struct cereal_GPSPlannerPoints*, cereal_GPSPlannerPoints_ptr);
void cereal_write_GPSPlannerPlan(const struct cereal_GPSPlannerPlan*, cereal_GPSPlannerPlan_ptr);
void cereal_write_TrafficEvent(const struct cereal_TrafficEvent*, cereal_TrafficEvent_ptr);
void cereal_write_OrbslamCorrection(const struct cereal_OrbslamCorrection*, cereal_OrbslamCorrection_ptr);
void cereal_write_OrbObservation(const struct cereal_OrbObservation*, cereal_OrbObservation_ptr);
void cereal_write_UiNavigationEvent(const struct cereal_UiNavigationEvent*, cereal_UiNavigationEvent_ptr);
void cereal_write_UiLayoutState(const struct cereal_UiLayoutState*, cereal_UiLayoutState_ptr);
void cereal_write_Joystick(const struct cereal_Joystick*, cereal_Joystick_ptr);
void cereal_write_OrbOdometry(const struct cereal_OrbOdometry*, cereal_OrbOdometry_ptr);
void cereal_write_OrbFeatures(const struct cereal_OrbFeatures*, cereal_OrbFeatures_ptr);
void cereal_write_OrbFeaturesSummary(const struct cereal_OrbFeaturesSummary*, cereal_OrbFeaturesSummary_ptr);
void cereal_write_OrbKeyFrame(const struct cereal_OrbKeyFrame*, cereal_OrbKeyFrame_ptr);
void cereal_write_DriverState(const struct cereal_DriverState*, cereal_DriverState_ptr);
void cereal_write_DMonitoringState(const struct cereal_DMonitoringState*, cereal_DMonitoringState_ptr);
void cereal_write_Boot(const struct cereal_Boot*, cereal_Boot_ptr);
void cereal_write_LiveParametersData(const struct cereal_LiveParametersData*, cereal_LiveParametersData_ptr);
void cereal_write_LiveMapData(const struct cereal_LiveMapData*, cereal_LiveMapData_ptr);
void cereal_write_CameraOdometry(const struct cereal_CameraOdometry*, cereal_CameraOdometry_ptr);
void cereal_write_KalmanOdometry(const struct cereal_KalmanOdometry*, cereal_KalmanOdometry_ptr);
void cereal_write_Event(const struct cereal_Event*, cereal_Event_ptr);

void cereal_get_Map(struct cereal_Map*, cereal_Map_list, int i);
void cereal_get_Map_Entry(struct cereal_Map_Entry*, cereal_Map_Entry_list, int i);
void cereal_get_InitData(struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_get_InitData_AndroidBuildInfo(struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_list, int i);
void cereal_get_InitData_AndroidSensor(struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_list, int i);
void cereal_get_InitData_ChffrAndroidExtra(struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_list, int i);
void cereal_get_InitData_IosBuildInfo(struct cereal_InitData_IosBuildInfo*, cereal_InitData_IosBuildInfo_list, int i);
void cereal_get_InitData_PandaInfo(struct cereal_InitData_PandaInfo*, cereal_InitData_PandaInfo_list, int i);
void cereal_get_FrameData(struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_get_FrameData_AndroidCaptureResult(struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_list, int i);
void cereal_get_Thumbnail(struct cereal_Thumbnail*, cereal_Thumbnail_list, int i);
void cereal_get_GPSNMEAData(struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_get_SensorEventData(struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_get_SensorEventData_SensorVec(struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_get_GpsLocationData(struct cereal_GpsLocationData*, cereal_GpsLocationData_list, int i);
void cereal_get_CanData(struct cereal_CanData*, cereal_CanData_list, int i);
void cereal_get_ThermalData(struct cereal_ThermalData*, cereal_ThermalData_list, int i);
void cereal_get_HealthData(struct cereal_HealthData*, cereal_HealthData_list, int i);
void cereal_get_LiveUI(struct cereal_LiveUI*, cereal_LiveUI_list, int i);
void cereal_get_RadarState(struct cereal_RadarState*, cereal_RadarState_list, int i);
void cereal_get_RadarState_LeadData(struct cereal_RadarState_LeadData*, cereal_RadarState_LeadData_list, int i);
void cereal_get_LiveCalibrationData(struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_list, int i);
void cereal_get_LiveTracks(struct cereal_LiveTracks*, cereal_LiveTracks_list, int i);
void cereal_get_ControlsState(struct cereal_ControlsState*, cereal_ControlsState_list, int i);
void cereal_get_ControlsState_LateralINDIState(struct cereal_ControlsState_LateralINDIState*, cereal_ControlsState_LateralINDIState_list, int i);
void cereal_get_ControlsState_LateralPIDState(struct cereal_ControlsState_LateralPIDState*, cereal_ControlsState_LateralPIDState_list, int i);
void cereal_get_ControlsState_LateralLQRState(struct cereal_ControlsState_LateralLQRState*, cereal_ControlsState_LateralLQRState_list, int i);
void cereal_get_LiveEventData(struct cereal_LiveEventData*, cereal_LiveEventData_list, int i);
void cereal_get_ModelData(struct cereal_ModelData*, cereal_ModelData_list, int i);
void cereal_get_ModelData_PathData(struct cereal_ModelData_PathData*, cereal_ModelData_PathData_list, int i);
void cereal_get_ModelData_LeadData(struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_list, int i);
void cereal_get_ModelData_ModelSettings(struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_list, int i);
void cereal_get_ModelData_MetaData(struct cereal_ModelData_MetaData*, cereal_ModelData_MetaData_list, int i);
void cereal_get_ModelData_LongitudinalData(struct cereal_ModelData_LongitudinalData*, cereal_ModelData_LongitudinalData_list, int i);
void cereal_get_CalibrationFeatures(struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_list, int i);
void cereal_get_EncodeIndex(struct cereal_EncodeIndex*, cereal_EncodeIndex_list, int i);
void cereal_get_AndroidLogEntry(struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_list, int i);
void cereal_get_LogRotate(struct cereal_LogRotate*, cereal_LogRotate_list, int i);
void cereal_get_Plan(struct cereal_Plan*, cereal_Plan_list, int i);
void cereal_get_Plan_GpsTrajectory(struct cereal_Plan_GpsTrajectory*, cereal_Plan_GpsTrajectory_list, int i);
void cereal_get_PathPlan(struct cereal_PathPlan*, cereal_PathPlan_list, int i);
void cereal_get_LiveLocationData(struct cereal_LiveLocationData*, cereal_LiveLocationData_list, int i);
void cereal_get_LiveLocationData_Accuracy(struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_list, int i);
void cereal_get_EthernetPacket(struct cereal_EthernetPacket*, cereal_EthernetPacket_list, int i);
void cereal_get_NavUpdate(struct cereal_NavUpdate*, cereal_NavUpdate_list, int i);
void cereal_get_NavUpdate_LatLng(struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_list, int i);
void cereal_get_NavUpdate_Segment(struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_list, int i);
void cereal_get_NavStatus(struct cereal_NavStatus*, cereal_NavStatus_list, int i);
void cereal_get_NavStatus_Address(struct cereal_NavStatus_Address*, cereal_NavStatus_Address_list, int i);
void cereal_get_CellInfo(struct cereal_CellInfo*, cereal_CellInfo_list, int i);
void cereal_get_WifiScan(struct cereal_WifiScan*, cereal_WifiScan_list, int i);
void cereal_get_AndroidGnss(struct cereal_AndroidGnss*, cereal_AndroidGnss_list, int i);
void cereal_get_AndroidGnss_Measurements(struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_list, int i);
void cereal_get_AndroidGnss_Measurements_Clock(struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_list, int i);
void cereal_get_AndroidGnss_Measurements_Measurement(struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_list, int i);
void cereal_get_AndroidGnss_NavigationMessage(struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_list, int i);
void cereal_get_QcomGnss(struct cereal_QcomGnss*, cereal_QcomGnss_list, int i);
void cereal_get_QcomGnss_MeasurementStatus(struct cereal_QcomGnss_MeasurementStatus*, cereal_QcomGnss_MeasurementStatus_list, int i);
void cereal_get_QcomGnss_MeasurementReport(struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_list, int i);
void cereal_get_QcomGnss_MeasurementReport_SV(struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_list, int i);
void cereal_get_QcomGnss_ClockReport(struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_list, int i);
void cereal_get_QcomGnss_DrMeasurementReport(struct cereal_QcomGnss_DrMeasurementReport*, cereal_QcomGnss_DrMeasurementReport_list, int i);
void cereal_get_QcomGnss_DrMeasurementReport_SV(struct cereal_QcomGnss_DrMeasurementReport_SV*, cereal_QcomGnss_DrMeasurementReport_SV_list, int i);
void cereal_get_QcomGnss_DrSvPolyReport(struct cereal_QcomGnss_DrSvPolyReport*, cereal_QcomGnss_DrSvPolyReport_list, int i);
void cereal_get_LidarPts(struct cereal_LidarPts*, cereal_LidarPts_list, int i);
void cereal_get_ProcLog(struct cereal_ProcLog*, cereal_ProcLog_list, int i);
void cereal_get_ProcLog_Process(struct cereal_ProcLog_Process*, cereal_ProcLog_Process_list, int i);
void cereal_get_ProcLog_CPUTimes(struct cereal_ProcLog_CPUTimes*, cereal_ProcLog_CPUTimes_list, int i);
void cereal_get_ProcLog_Mem(struct cereal_ProcLog_Mem*, cereal_ProcLog_Mem_list, int i);
void cereal_get_UbloxGnss(struct cereal_UbloxGnss*, cereal_UbloxGnss_list, int i);
void cereal_get_UbloxGnss_MeasurementReport(struct cereal_UbloxGnss_MeasurementReport*, cereal_UbloxGnss_MeasurementReport_list, int i);
void cereal_get_UbloxGnss_MeasurementReport_ReceiverStatus(struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus*, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list, int i);
void cereal_get_UbloxGnss_MeasurementReport_Measurement(struct cereal_UbloxGnss_MeasurementReport_Measurement*, cereal_UbloxGnss_MeasurementReport_Measurement_list, int i);
void cereal_get_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus*, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list, int i);
void cereal_get_UbloxGnss_Ephemeris(struct cereal_UbloxGnss_Ephemeris*, cereal_UbloxGnss_Ephemeris_list, int i);
void cereal_get_UbloxGnss_IonoData(struct cereal_UbloxGnss_IonoData*, cereal_UbloxGnss_IonoData_list, int i);
void cereal_get_UbloxGnss_HwStatus(struct cereal_UbloxGnss_HwStatus*, cereal_UbloxGnss_HwStatus_list, int i);
void cereal_get_Clocks(struct cereal_Clocks*, cereal_Clocks_list, int i);
void cereal_get_LiveMpcData(struct cereal_LiveMpcData*, cereal_LiveMpcData_list, int i);
void cereal_get_LiveLongitudinalMpcData(struct cereal_LiveLongitudinalMpcData*, cereal_LiveLongitudinalMpcData_list, int i);
void cereal_get_ECEFPointDEPRECATED(struct cereal_ECEFPointDEPRECATED*, cereal_ECEFPointDEPRECATED_list, int i);
void cereal_get_ECEFPoint(struct cereal_ECEFPoint*, cereal_ECEFPoint_list, int i);
void cereal_get_GPSPlannerPoints(struct cereal_GPSPlannerPoints*, cereal_GPSPlannerPoints_list, int i);
void cereal_get_GPSPlannerPlan(struct cereal_GPSPlannerPlan*, cereal_GPSPlannerPlan_list, int i);
void cereal_get_TrafficEvent(struct cereal_TrafficEvent*, cereal_TrafficEvent_list, int i);
void cereal_get_OrbslamCorrection(struct cereal_OrbslamCorrection*, cereal_OrbslamCorrection_list, int i);
void cereal_get_OrbObservation(struct cereal_OrbObservation*, cereal_OrbObservation_list, int i);
void cereal_get_UiNavigationEvent(struct cereal_UiNavigationEvent*, cereal_UiNavigationEvent_list, int i);
void cereal_get_UiLayoutState(struct cereal_UiLayoutState*, cereal_UiLayoutState_list, int i);
void cereal_get_Joystick(struct cereal_Joystick*, cereal_Joystick_list, int i);
void cereal_get_OrbOdometry(struct cereal_OrbOdometry*, cereal_OrbOdometry_list, int i);
void cereal_get_OrbFeatures(struct cereal_OrbFeatures*, cereal_OrbFeatures_list, int i);
void cereal_get_OrbFeaturesSummary(struct cereal_OrbFeaturesSummary*, cereal_OrbFeaturesSummary_list, int i);
void cereal_get_OrbKeyFrame(struct cereal_OrbKeyFrame*, cereal_OrbKeyFrame_list, int i);
void cereal_get_DriverState(struct cereal_DriverState*, cereal_DriverState_list, int i);
void cereal_get_DMonitoringState(struct cereal_DMonitoringState*, cereal_DMonitoringState_list, int i);
void cereal_get_Boot(struct cereal_Boot*, cereal_Boot_list, int i);
void cereal_get_LiveParametersData(struct cereal_LiveParametersData*, cereal_LiveParametersData_list, int i);
void cereal_get_LiveMapData(struct cereal_LiveMapData*, cereal_LiveMapData_list, int i);
void cereal_get_CameraOdometry(struct cereal_CameraOdometry*, cereal_CameraOdometry_list, int i);
void cereal_get_KalmanOdometry(struct cereal_KalmanOdometry*, cereal_KalmanOdometry_list, int i);
void cereal_get_Event(struct cereal_Event*, cereal_Event_list, int i);

void cereal_set_Map(const struct cereal_Map*, cereal_Map_list, int i);
void cereal_set_Map_Entry(const struct cereal_Map_Entry*, cereal_Map_Entry_list, int i);
void cereal_set_InitData(const struct cereal_InitData*, cereal_InitData_list, int i);
void cereal_set_InitData_AndroidBuildInfo(const struct cereal_InitData_AndroidBuildInfo*, cereal_InitData_AndroidBuildInfo_list, int i);
void cereal_set_InitData_AndroidSensor(const struct cereal_InitData_AndroidSensor*, cereal_InitData_AndroidSensor_list, int i);
void cereal_set_InitData_ChffrAndroidExtra(const struct cereal_InitData_ChffrAndroidExtra*, cereal_InitData_ChffrAndroidExtra_list, int i);
void cereal_set_InitData_IosBuildInfo(const struct cereal_InitData_IosBuildInfo*, cereal_InitData_IosBuildInfo_list, int i);
void cereal_set_InitData_PandaInfo(const struct cereal_InitData_PandaInfo*, cereal_InitData_PandaInfo_list, int i);
void cereal_set_FrameData(const struct cereal_FrameData*, cereal_FrameData_list, int i);
void cereal_set_FrameData_AndroidCaptureResult(const struct cereal_FrameData_AndroidCaptureResult*, cereal_FrameData_AndroidCaptureResult_list, int i);
void cereal_set_Thumbnail(const struct cereal_Thumbnail*, cereal_Thumbnail_list, int i);
void cereal_set_GPSNMEAData(const struct cereal_GPSNMEAData*, cereal_GPSNMEAData_list, int i);
void cereal_set_SensorEventData(const struct cereal_SensorEventData*, cereal_SensorEventData_list, int i);
void cereal_set_SensorEventData_SensorVec(const struct cereal_SensorEventData_SensorVec*, cereal_SensorEventData_SensorVec_list, int i);
void cereal_set_GpsLocationData(const struct cereal_GpsLocationData*, cereal_GpsLocationData_list, int i);
void cereal_set_CanData(const struct cereal_CanData*, cereal_CanData_list, int i);
void cereal_set_ThermalData(const struct cereal_ThermalData*, cereal_ThermalData_list, int i);
void cereal_set_HealthData(const struct cereal_HealthData*, cereal_HealthData_list, int i);
void cereal_set_LiveUI(const struct cereal_LiveUI*, cereal_LiveUI_list, int i);
void cereal_set_RadarState(const struct cereal_RadarState*, cereal_RadarState_list, int i);
void cereal_set_RadarState_LeadData(const struct cereal_RadarState_LeadData*, cereal_RadarState_LeadData_list, int i);
void cereal_set_LiveCalibrationData(const struct cereal_LiveCalibrationData*, cereal_LiveCalibrationData_list, int i);
void cereal_set_LiveTracks(const struct cereal_LiveTracks*, cereal_LiveTracks_list, int i);
void cereal_set_ControlsState(const struct cereal_ControlsState*, cereal_ControlsState_list, int i);
void cereal_set_ControlsState_LateralINDIState(const struct cereal_ControlsState_LateralINDIState*, cereal_ControlsState_LateralINDIState_list, int i);
void cereal_set_ControlsState_LateralPIDState(const struct cereal_ControlsState_LateralPIDState*, cereal_ControlsState_LateralPIDState_list, int i);
void cereal_set_ControlsState_LateralLQRState(const struct cereal_ControlsState_LateralLQRState*, cereal_ControlsState_LateralLQRState_list, int i);
void cereal_set_LiveEventData(const struct cereal_LiveEventData*, cereal_LiveEventData_list, int i);
void cereal_set_ModelData(const struct cereal_ModelData*, cereal_ModelData_list, int i);
void cereal_set_ModelData_PathData(const struct cereal_ModelData_PathData*, cereal_ModelData_PathData_list, int i);
void cereal_set_ModelData_LeadData(const struct cereal_ModelData_LeadData*, cereal_ModelData_LeadData_list, int i);
void cereal_set_ModelData_ModelSettings(const struct cereal_ModelData_ModelSettings*, cereal_ModelData_ModelSettings_list, int i);
void cereal_set_ModelData_MetaData(const struct cereal_ModelData_MetaData*, cereal_ModelData_MetaData_list, int i);
void cereal_set_ModelData_LongitudinalData(const struct cereal_ModelData_LongitudinalData*, cereal_ModelData_LongitudinalData_list, int i);
void cereal_set_CalibrationFeatures(const struct cereal_CalibrationFeatures*, cereal_CalibrationFeatures_list, int i);
void cereal_set_EncodeIndex(const struct cereal_EncodeIndex*, cereal_EncodeIndex_list, int i);
void cereal_set_AndroidLogEntry(const struct cereal_AndroidLogEntry*, cereal_AndroidLogEntry_list, int i);
void cereal_set_LogRotate(const struct cereal_LogRotate*, cereal_LogRotate_list, int i);
void cereal_set_Plan(const struct cereal_Plan*, cereal_Plan_list, int i);
void cereal_set_Plan_GpsTrajectory(const struct cereal_Plan_GpsTrajectory*, cereal_Plan_GpsTrajectory_list, int i);
void cereal_set_PathPlan(const struct cereal_PathPlan*, cereal_PathPlan_list, int i);
void cereal_set_LiveLocationData(const struct cereal_LiveLocationData*, cereal_LiveLocationData_list, int i);
void cereal_set_LiveLocationData_Accuracy(const struct cereal_LiveLocationData_Accuracy*, cereal_LiveLocationData_Accuracy_list, int i);
void cereal_set_EthernetPacket(const struct cereal_EthernetPacket*, cereal_EthernetPacket_list, int i);
void cereal_set_NavUpdate(const struct cereal_NavUpdate*, cereal_NavUpdate_list, int i);
void cereal_set_NavUpdate_LatLng(const struct cereal_NavUpdate_LatLng*, cereal_NavUpdate_LatLng_list, int i);
void cereal_set_NavUpdate_Segment(const struct cereal_NavUpdate_Segment*, cereal_NavUpdate_Segment_list, int i);
void cereal_set_NavStatus(const struct cereal_NavStatus*, cereal_NavStatus_list, int i);
void cereal_set_NavStatus_Address(const struct cereal_NavStatus_Address*, cereal_NavStatus_Address_list, int i);
void cereal_set_CellInfo(const struct cereal_CellInfo*, cereal_CellInfo_list, int i);
void cereal_set_WifiScan(const struct cereal_WifiScan*, cereal_WifiScan_list, int i);
void cereal_set_AndroidGnss(const struct cereal_AndroidGnss*, cereal_AndroidGnss_list, int i);
void cereal_set_AndroidGnss_Measurements(const struct cereal_AndroidGnss_Measurements*, cereal_AndroidGnss_Measurements_list, int i);
void cereal_set_AndroidGnss_Measurements_Clock(const struct cereal_AndroidGnss_Measurements_Clock*, cereal_AndroidGnss_Measurements_Clock_list, int i);
void cereal_set_AndroidGnss_Measurements_Measurement(const struct cereal_AndroidGnss_Measurements_Measurement*, cereal_AndroidGnss_Measurements_Measurement_list, int i);
void cereal_set_AndroidGnss_NavigationMessage(const struct cereal_AndroidGnss_NavigationMessage*, cereal_AndroidGnss_NavigationMessage_list, int i);
void cereal_set_QcomGnss(const struct cereal_QcomGnss*, cereal_QcomGnss_list, int i);
void cereal_set_QcomGnss_MeasurementStatus(const struct cereal_QcomGnss_MeasurementStatus*, cereal_QcomGnss_MeasurementStatus_list, int i);
void cereal_set_QcomGnss_MeasurementReport(const struct cereal_QcomGnss_MeasurementReport*, cereal_QcomGnss_MeasurementReport_list, int i);
void cereal_set_QcomGnss_MeasurementReport_SV(const struct cereal_QcomGnss_MeasurementReport_SV*, cereal_QcomGnss_MeasurementReport_SV_list, int i);
void cereal_set_QcomGnss_ClockReport(const struct cereal_QcomGnss_ClockReport*, cereal_QcomGnss_ClockReport_list, int i);
void cereal_set_QcomGnss_DrMeasurementReport(const struct cereal_QcomGnss_DrMeasurementReport*, cereal_QcomGnss_DrMeasurementReport_list, int i);
void cereal_set_QcomGnss_DrMeasurementReport_SV(const struct cereal_QcomGnss_DrMeasurementReport_SV*, cereal_QcomGnss_DrMeasurementReport_SV_list, int i);
void cereal_set_QcomGnss_DrSvPolyReport(const struct cereal_QcomGnss_DrSvPolyReport*, cereal_QcomGnss_DrSvPolyReport_list, int i);
void cereal_set_LidarPts(const struct cereal_LidarPts*, cereal_LidarPts_list, int i);
void cereal_set_ProcLog(const struct cereal_ProcLog*, cereal_ProcLog_list, int i);
void cereal_set_ProcLog_Process(const struct cereal_ProcLog_Process*, cereal_ProcLog_Process_list, int i);
void cereal_set_ProcLog_CPUTimes(const struct cereal_ProcLog_CPUTimes*, cereal_ProcLog_CPUTimes_list, int i);
void cereal_set_ProcLog_Mem(const struct cereal_ProcLog_Mem*, cereal_ProcLog_Mem_list, int i);
void cereal_set_UbloxGnss(const struct cereal_UbloxGnss*, cereal_UbloxGnss_list, int i);
void cereal_set_UbloxGnss_MeasurementReport(const struct cereal_UbloxGnss_MeasurementReport*, cereal_UbloxGnss_MeasurementReport_list, int i);
void cereal_set_UbloxGnss_MeasurementReport_ReceiverStatus(const struct cereal_UbloxGnss_MeasurementReport_ReceiverStatus*, cereal_UbloxGnss_MeasurementReport_ReceiverStatus_list, int i);
void cereal_set_UbloxGnss_MeasurementReport_Measurement(const struct cereal_UbloxGnss_MeasurementReport_Measurement*, cereal_UbloxGnss_MeasurementReport_Measurement_list, int i);
void cereal_set_UbloxGnss_MeasurementReport_Measurement_TrackingStatus(const struct cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus*, cereal_UbloxGnss_MeasurementReport_Measurement_TrackingStatus_list, int i);
void cereal_set_UbloxGnss_Ephemeris(const struct cereal_UbloxGnss_Ephemeris*, cereal_UbloxGnss_Ephemeris_list, int i);
void cereal_set_UbloxGnss_IonoData(const struct cereal_UbloxGnss_IonoData*, cereal_UbloxGnss_IonoData_list, int i);
void cereal_set_UbloxGnss_HwStatus(const struct cereal_UbloxGnss_HwStatus*, cereal_UbloxGnss_HwStatus_list, int i);
void cereal_set_Clocks(const struct cereal_Clocks*, cereal_Clocks_list, int i);
void cereal_set_LiveMpcData(const struct cereal_LiveMpcData*, cereal_LiveMpcData_list, int i);
void cereal_set_LiveLongitudinalMpcData(const struct cereal_LiveLongitudinalMpcData*, cereal_LiveLongitudinalMpcData_list, int i);
void cereal_set_ECEFPointDEPRECATED(const struct cereal_ECEFPointDEPRECATED*, cereal_ECEFPointDEPRECATED_list, int i);
void cereal_set_ECEFPoint(const struct cereal_ECEFPoint*, cereal_ECEFPoint_list, int i);
void cereal_set_GPSPlannerPoints(const struct cereal_GPSPlannerPoints*, cereal_GPSPlannerPoints_list, int i);
void cereal_set_GPSPlannerPlan(const struct cereal_GPSPlannerPlan*, cereal_GPSPlannerPlan_list, int i);
void cereal_set_TrafficEvent(const struct cereal_TrafficEvent*, cereal_TrafficEvent_list, int i);
void cereal_set_OrbslamCorrection(const struct cereal_OrbslamCorrection*, cereal_OrbslamCorrection_list, int i);
void cereal_set_OrbObservation(const struct cereal_OrbObservation*, cereal_OrbObservation_list, int i);
void cereal_set_UiNavigationEvent(const struct cereal_UiNavigationEvent*, cereal_UiNavigationEvent_list, int i);
void cereal_set_UiLayoutState(const struct cereal_UiLayoutState*, cereal_UiLayoutState_list, int i);
void cereal_set_Joystick(const struct cereal_Joystick*, cereal_Joystick_list, int i);
void cereal_set_OrbOdometry(const struct cereal_OrbOdometry*, cereal_OrbOdometry_list, int i);
void cereal_set_OrbFeatures(const struct cereal_OrbFeatures*, cereal_OrbFeatures_list, int i);
void cereal_set_OrbFeaturesSummary(const struct cereal_OrbFeaturesSummary*, cereal_OrbFeaturesSummary_list, int i);
void cereal_set_OrbKeyFrame(const struct cereal_OrbKeyFrame*, cereal_OrbKeyFrame_list, int i);
void cereal_set_DriverState(const struct cereal_DriverState*, cereal_DriverState_list, int i);
void cereal_set_DMonitoringState(const struct cereal_DMonitoringState*, cereal_DMonitoringState_list, int i);
void cereal_set_Boot(const struct cereal_Boot*, cereal_Boot_list, int i);
void cereal_set_LiveParametersData(const struct cereal_LiveParametersData*, cereal_LiveParametersData_list, int i);
void cereal_set_LiveMapData(const struct cereal_LiveMapData*, cereal_LiveMapData_list, int i);
void cereal_set_CameraOdometry(const struct cereal_CameraOdometry*, cereal_CameraOdometry_list, int i);
void cereal_set_KalmanOdometry(const struct cereal_KalmanOdometry*, cereal_KalmanOdometry_list, int i);
void cereal_set_Event(const struct cereal_Event*, cereal_Event_list, int i);

#ifdef __cplusplus
}
#endif
#endif
