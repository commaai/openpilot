/*==============================================================================
@file
    HAP_power.h

@brief
    Header file of DSP power APIs.

Copyright (c) 2015,2019 Qualcomm Technologies, Inc.
All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/

#ifndef _HAP_POWER_H
#define _HAP_POWER_H

#include "AEEStdErr.h"
//#include <string.h>
//#include <stdlib.h>
#define boolean char
#define FALSE 0
#define TRUE 1
#define uint64 unsigned long long
#define uint32 unsigned int
#define NULL 0

#ifdef __cplusplus
extern "C" {
#endif

//Add a weak reference so shared objects do not throw link error
#pragma weak HAP_power_destroy_client

/**
* Possible error codes returned
*/
typedef enum {
	HAP_POWER_ERR_UNKNOWN           = -1,
	HAP_POWER_ERR_INVALID_PARAM     = -2,
	HAP_POWER_ERR_UNSUPPORTED_API   = -3
} HAP_power_error_codes;

/** Payload for HAP_power_set_mips_bw */
typedef struct {
	boolean set_mips;						/**< Set to TRUE to request MIPS */
	unsigned int mipsPerThread;				/**< mips requested per thread, to establish a minimal clock frequency per HW thread */
	unsigned int mipsTotal;					/**< Total mips requested, to establish total number of MIPS required across all HW threads */
	boolean set_bus_bw;						/**< Set to TRUE to request bus_bw */
	uint64 bwBytePerSec;					/**< Max bus BW requested (bytes per second) */
	unsigned short busbwUsagePercentage;	/**< Percentage of time during which bwBytesPerSec BW is required from the bus (0..100) */
	boolean set_latency;					/**< Set to TRUE to set latency */
	int latency;							/**< maximum hardware wakeup latency in microseconds.  The higher the value,
											*	the deeper state of sleep that can be entered but the longer it may take
											*	to awaken. Only values > 0 are supported (1 microsecond is the smallest valid value) */
} HAP_power_mips_bw_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
 /** Clock frequency match type*/
typedef enum {
	HAP_FREQ_AT_LEAST,				/**< Matches at least the specified frequency. */
	HAP_FREQ_AT_MOST,				/**< Matches at most the specified frequency. */
	HAP_FREQ_CLOSEST,				/**< Closest match to the specified frequency. */
	HAP_FREQ_EXACT,					/**< Exact match with the specified frequency. */
	HAP_FREQ_MAX_COUNT				/**< Maximum count. */
} HAP_freq_match_type;
/**
 * @} // HAP_power_enums
 */

/** Configuration for bus bandwidth */
typedef struct {
	boolean set_bus_bw;						/**< Set to TRUE to request bus_bw */
	uint64 bwBytePerSec;					/**< Max bus BW requested (bytes per second) */
	unsigned short busbwUsagePercentage;	/**< Percentage of time during which bwBytesPerSec BW is required from the bus (0..100) */
} HAP_power_bus_bw;

/**
* @brief Payload for vapps power request
* vapps core is used for Video post processing
*/
typedef struct {
	boolean set_clk;						/**< Set to TRUE to request clock frequency */
	unsigned int clkFreqHz;					/**< Clock frequency in Hz */
	HAP_freq_match_type freqMatch;			/**< Clock frequency match */
	HAP_power_bus_bw dma_ext;				/**< DMA external bus bandwidth */
	HAP_power_bus_bw hcp_ext;				/**< HCP external bus bandwidth */
	HAP_power_bus_bw dma_int;				/**< DMA internal bus bandwidth */
	HAP_power_bus_bw hcp_int;				/**< HCP internal bus bandwidth */
} HAP_power_vapss_payload;

/**
* @brief Payload for vapps_v2 power request
* Supported in targets which have split VAPPS core(DMA and HCP) form Hana onwards
*/
typedef struct {
	boolean set_dma_clk;					/**< Set to TRUE to reqeust DMA clock frequency */
	boolean set_hcp_clk;					/**< Set to TRUE to reqeust HCP clock frequency */
	unsigned int dmaClkFreqHz;				/**< DMA Clock frequency in Hz */
	unsigned int hcpClkFreqHz;				/**< HCP Clock frequency in Hz */
	HAP_freq_match_type freqMatch;			/**< Clock frequency match type */
	HAP_power_bus_bw dma_ext;				/**< DMA external bus bandwidth */
	HAP_power_bus_bw hcp_ext;				/**< HCP external bus bandwidth */
	HAP_power_bus_bw dma_int;				/**< DMA internal bus bandwidth */
	HAP_power_bus_bw hcp_int;				/**< HCP internal bus bandwidth */
} HAP_power_vapss_payload_v2;

/** Payload for HAP_power_set_HVX */
typedef struct {
	boolean power_up;						/**< Set to TRUE to turn on HVX, and FALSE to turn off. */
} HAP_power_hvx_payload;

/**
* Payload for HAP_power_set_HMX
* Supported from Lahaina onwards*/
typedef struct {
	boolean power_up;						/**< Set to TRUE to turn on HMX, and FALSE to turn off. */
} HAP_power_hmx_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
/** Payload for HAP power client classes */
typedef enum {
	HAP_POWER_UNKNOWN_CLIENT_CLASS			= 0x00,		/**< Unknown client class */
	HAP_POWER_AUDIO_CLIENT_CLASS			= 0x01,		/**< Audio client class */
	HAP_POWER_VOICE_CLIENT_CLASS			= 0x02,		/**< Voice client class */
	HAP_POWER_COMPUTE_CLIENT_CLASS			= 0x04,		/**< Compute client class */
	HAP_POWER_STREAMING_1HVX_CLIENT_CLASS	= 0x08,		/**< Camera streaming with 1 HVX client class */
	HAP_POWER_STREAMING_2HVX_CLIENT_CLASS = 0x10,		/**< Camera streaming with 2 HVX client class */
} HAP_power_app_type_payload;
/**
 * @} // HAP_power_enums
 */

/** Payload for HAP_power_set_linelock */
typedef struct {
	void* startAddress;					/**< Start address of the memory region to be locked. */
	uint32 size;						/**< Size (bytes) of the memory region to be locked. Set size
										*	to 0 to unlock memory. */
	uint32 throttleBlockSize;			/**< Block size for throttling, in bytes;
										* 0 for no throttling.  The region to be locked will be divided into
										* blocks of this size for throttling purposes.
										* Use for locking larger cache blocks.
										* Applicable only when enabling line locking.Only ONE throttled linelock call is supported at this time.
										* You can linelock additional regions (without throttling) using HAP_power_set_linelock_nothrottle*/
	uint32 throttlePauseUs;				/**< Pause to be applied between locking each block, in microseconds. Applicable only when enabling line locking*/
} HAP_power_linelock_payload;

/** Payload for HAP_power_set_linelock_nothrottle */
typedef struct {
	void* startAddress;							/**< Start address of the memory region to be locked. */
	uint32 size;								/**< Size (bytes) of the memory region to be locked. Set size to 0
												* to unlock memory */
} HAP_power_linelock_nothrottle_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
/** Option for dcvs payload */
typedef enum {
	HAP_DCVS_ADJUST_UP_DOWN =   0x1,		/**< increase and decrease core/bus clock speed. */
	HAP_DCVS_ADJUST_ONLY_UP =   0x2,		/**< restricts DCVS from lowering the clock speed below the requested value . */
} HAP_power_dcvs_payload_option;
/**
 * @} // HAP_power_enums
 */

/** Payload for HAP_power_set_DCVS */
typedef struct {
	boolean dcvs_enable;								/**< Set to TRUE to participate in DCVS, and FALSE otherwise. */
	HAP_power_dcvs_payload_option dcvs_option;			/**< Set to one of
														*		HAP_DCVS_ADJUST_UP_DOWN  - Allows for DCVS to adjust up and down.
														*		HAP_DCVS_ADJUST_ONLY_UP  - Allows for DCVS to adjust up only. */
} HAP_power_dcvs_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
/** Voltage corners for HAP DCVS V2 interface */
typedef enum {
	HAP_DCVS_VCORNER_DISABLE,
	HAP_DCVS_VCORNER_SVS2,
	HAP_DCVS_VCORNER_SVS,
	HAP_DCVS_VCORNER_SVS_PLUS,
	HAP_DCVS_VCORNER_NOM,
	HAP_DCVS_VCORNER_NOM_PLUS,
	HAP_DCVS_VCORNER_TURBO,
	HAP_DCVS_VCORNER_TURBO_PLUS,
	HAP_DCVS_VCORNER_MAX = 255,
} HAP_dcvs_voltage_corner_t;
/**
 * @} // HAP_power_enums
 */

#define HAP_DCVS_VCORNER_SVSPLUS HAP_DCVS_VCORNER_SVS_PLUS
#define HAP_DCVS_VCORNER_NOMPLUS HAP_DCVS_VCORNER_NOM_PLUS

/** DCVS parameters for HAP_power_dcvs_v2_payload */
typedef struct {
	HAP_dcvs_voltage_corner_t target_corner;	/**< target voltage corner */
	HAP_dcvs_voltage_corner_t min_corner;		/**< minimum voltage corner */
	HAP_dcvs_voltage_corner_t max_corner;		/**< maximum voltage corner */
	uint32 param1;								/**< reserved */
	uint32 param2;								/**< reserved */
	uint32 param3;								/**< reserved */
} HAP_dcvs_params_t;

/** Core clock parameters for HAP_power_dcvs_v3_payload */
typedef struct {
	HAP_dcvs_voltage_corner_t target_corner;	/**< target voltage corner */
	HAP_dcvs_voltage_corner_t min_corner;		/**< minimum voltage corner */
	HAP_dcvs_voltage_corner_t max_corner;		/**< maximum voltage corner */
	uint32 param1;								/**< reserved */
	uint32 param2;								/**< reserved */
	uint32 param3;								/**< reserved */
} HAP_core_params_t;

/** Bus clock parameters for HAP_power_dcvs_v3_payload */
typedef struct {
	HAP_dcvs_voltage_corner_t target_corner;	/**< target voltage corner */
	HAP_dcvs_voltage_corner_t min_corner;		/**< minimum voltage corner */
	HAP_dcvs_voltage_corner_t max_corner;		/**< maximum voltage corner */
	uint32 param1;								/**< reserved */
	uint32 param2;								/**< reserved */
	uint32 param3;								/**< reserved */
} HAP_bus_params_t;

/** DCVS v3 parameters for HAP_power_dcvs_v3_payload */
typedef struct {
	uint32 param1;					/**< reserved */
	uint32 param2;					/**< reserved */
	uint32 param3;					/**< reserved */
	uint32 param4;					/**< reserved */
	uint32 param5;					/**< reserved */
	uint32 param6;					/**< reserved */
} HAP_dcvs_v3_params_t;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
/** option for dcvs_v2 payload */
typedef enum {
	HAP_DCVS_V2_ADJUST_UP_DOWN =   0x1,					/**< Allows for DCVS to adjust up and down. */
	HAP_DCVS_V2_ADJUST_ONLY_UP =   0x2,					/**< Allows for DCVS to adjust up only. */
	HAP_DCVS_V2_POWER_SAVER_MODE = 0x4,					/**< HAP_DCVS_POWER_SAVER_MODE				-	Higher thresholds for power efficiency. */
	HAP_DCVS_V2_POWER_SAVER_AGGRESSIVE_MODE = 0x8,		/**< HAP_DCVS_POWER_SAVER_AGGRESSIVE_MODE	-	Higher thresholds for power efficiency with faster ramp down. */
	HAP_DCVS_V2_PERFORMANCE_MODE = 0x10,				/**< HAP_DCVS_PERFORMANCE_MODE				-	Lower thresholds for maximum performance */
	HAP_DCVS_V2_DUTY_CYCLE_MODE = 0x20,					/**< HAP_DCVS_DUTY_CYCLE_MODE				-	only for HVX based clients.
														*												For streaming class clients:
														*													> detects periodicity based on HVX usage
														*													> lowers clocks in the no HVX activity region of each period.
														*												For compute class clients:
														*													> Lowers clocks on no HVX activity detects and brings clocks up on detecting HVX activity again.
														*													> Latency involved in bringing up the clock with be at max 1 to 2 ms. */



} HAP_power_dcvs_v2_payload_option;
/**
 * @} // HAP_power_enums
 */
/** Payload for HAP_power_set_DCVS_v2 */
typedef struct {
	boolean dcvs_enable;								/**< Set to TRUE to participate in DCVS, and FALSE otherwise */
	HAP_power_dcvs_v2_payload_option dcvs_option;		/**< Set to one of HAP_power_dcvs_v2_payload_option */
	boolean set_latency;								/**< TRUE to set latency parameter, otherwise FALSE */
	uint32 latency;										/**< sleep latency */
	boolean set_dcvs_params;							/**< TRUE to set DCVS params, otherwise FALSE */
	HAP_dcvs_params_t dcvs_params;						/**< DCVS parameters */
} HAP_power_dcvs_v2_payload;

/** Payload for HAP_power_set_DCVS_v3 */
typedef struct {
	boolean set_dcvs_enable;							/**< TRUE to consider DCVS enable/disable and option parameters, otherwise FALSE */
	boolean dcvs_enable;								/**< Set to TRUE to participate in DCVS, and FALSE otherwise. */
	HAP_power_dcvs_v2_payload_option dcvs_option;		/**< Set to one of HAP_power_dcvs_v2_payload_option */
	boolean set_latency;								/**< TRUE to consider latency parameter, otherwise FALSE */
	uint32 latency;										/**< sleep latency */
	boolean set_core_params;							/**< TRUE to consider core clock params, otherwise FALSE */
	HAP_core_params_t core_params;						/**< Core clock parameters */
	boolean set_bus_params;								/**< TRUE to consider bus clock params, otherwise FALSE */
	HAP_bus_params_t bus_params;						/**< Bus clock parameters */
	boolean set_dcvs_v3_params;							/**< TRUE to consider DCVS v3 params, otherwise FALSE */
	HAP_dcvs_v3_params_t dcvs_v3_params;				/**< DCVS v3 parameters */
	boolean set_sleep_disable;							/**< TRUE to consider sleep disable/enable parameter, otherwise FALSE */
	boolean sleep_disable;								/**< TRUE to disable sleep/LPM modes, FALSE to enable */
} HAP_power_dcvs_v3_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
 /** Type for dcvs update request */
typedef enum {
	HAP_POWER_UPDATE_DCVS = 1,
	HAP_POWER_UPDATE_SLEEP_LATENCY,
	HAP_POWER_UPDATE_DCVS_PARAMS,
} HAP_power_update_type_t;
/**
 * @} // HAP_power_enums
 */
/** Payload for DCVS update */
typedef struct {
	boolean dcvs_enable;							/**< TRUE for DCVS enable and FALSE for DCVS disable */
	HAP_power_dcvs_v2_payload_option dcvs_option;	/**< Requested DCVS policy in case DCVS enable is TRUE */
} HAP_power_update_dcvs_t;

/** Payload for latency update */
typedef struct {
	boolean set_latency;							/**< TRUE if sleep latency request has to be considered */
	unsigned int latency;							/**< Sleep latency request in micro seconds */
} HAP_power_update_latency_t;

/** Payload for DCVS params update */
typedef struct {
    boolean set_dcvs_params;						/**< Flag to mark DCVS params structure validity, TRUE for valid DCVS
													*params request and FALSE otherwise */
    HAP_dcvs_params_t dcvs_params;					/**< Intended DCVS params if set_dcvs_params is set to TRUE */
} HAP_power_update_dcvs_params_t;

/** Payload for HAP_power_set_DCVS_v2 */
typedef struct {
	HAP_power_update_type_t update_param;			/**< Type for which param to update */
	union {
		HAP_power_update_dcvs_t dcvs_payload;
		HAP_power_update_latency_t latency_payload;
		HAP_power_update_dcvs_params_t dcvs_params_payload;
	};												/**< Update payload for DCVS, latency or DCVS params */
} HAP_power_dcvs_v2_update_payload;

/** Payload for HAP_power_set_streamer */
typedef struct {
	boolean set_streamer0_clk;				/**< Set streamer 0 clock */
	boolean set_streamer1_clk;				/**< Set streamer 1 clock */
	unsigned int streamer0_clkFreqHz;		/**< Streamer 0 clock frequency */
	unsigned int streamer1_clkFreqHz;		/**< Streamer 1 clock frequency */
	HAP_freq_match_type freqMatch;			/**< Clock frequency match */
	uint32 param1;							/**< Reserved for future streamer parameters */
	uint32 param2;							/**< Reserved for future streamer parameters */
	uint32 param3;							/**< Reserved for future streamer parameters */
} HAP_power_streamer_payload;

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
 /** Identifies the HAP power request type */
typedef enum {
	HAP_power_set_mips_bw = 1,				/**< Requests for MIPS. Provides
											* fine-grained control to set MIPS values.
											* Payload is set to HAP_power_payload */
	HAP_power_set_HVX,						/**< Requests to enable / disable HVX
											* Payload is set to HAP_power_hvx_payload */
	HAP_power_set_apptype,					/**< Sets the app_type
											* Payload is set to HAP_power_app_type_payload */
	HAP_power_set_linelock,					/**< Sets the throttled L2 cache line locking parameters.
											* Only one throttled call is supported at this time. Additional
											* un-throttled line-locks can be performed using HAP_power_set_linelock_nothrottle
											* Payload is set to HAP_power_linelock_payload */
	HAP_power_set_DCVS,						/**< Requests to participate / stop participating in DCVS */
	HAP_power_set_linelock_nothrottle,		/**< Sets the L2 cache line locking parameters (non-throttled).
											* Payload is set to HAP_power_linelock_nothrottle_payload */
	HAP_power_set_DCVS_v2,					/**< Requests to participate / stop participating in DCVS_v2 */
	HAP_power_set_vapss,					/**< Sets the VAPSS core clock and DDR/IPNOC bandwidth
											* Payload is set to HAP_power_vapss_payload */
	HAP_power_set_vapss_v2,					/**< Sets the VAPSS core DMA/HCP clocks and DDR/IPNOC bandwidths
											* Payload is set to HAP_power_vapss_payload_v2 */
	HAP_power_set_dcvs_v2_update,			/**< Updates DCVS params
											* Payload is set to HAP_power_dcvs_v2_update_payload */
	HAP_power_set_streamer,					/**< Sets the streamer core clocks
											* Payload is set to HAP_power_streamer_payload */
	HAP_power_set_DCVS_v3,					/**< Updates DCVS params
											* Payload is set to HAP_power_dcvs_v3_payload */
	HAP_power_set_HMX,						/**< Requests to enable / disable HMX
											* Payload is set to HAP_power_hmx_payload */
} HAP_Power_request_type;
/**
 * @} // HAP_power_enums
 */

/** Data type to change power values on the DSP */
typedef struct {
	HAP_Power_request_type type;									/**< Identifies the request type */
	union{
		HAP_power_mips_bw_payload mips_bw;							/**< Requests for performance level */
		HAP_power_vapss_payload vapss;								/**< Sets the VAPSS core clock and DDR/IPNOC bandwidth  */
		HAP_power_vapss_payload_v2 vapss_v2;						/**< Sets the VAPSS core clock and DDR/IPNOC bandwidth  */
		HAP_power_streamer_payload streamer;						/**< Sets the streamer core clocks */
		HAP_power_hvx_payload hvx;									/**< Requests to enable / disable HVX */
		HAP_power_app_type_payload apptype;							/**< Sets the app_type */
		HAP_power_linelock_payload linelock;						/**< Sets the throttled L2 cache linelock parameters. Only one
																	* throttled linelock is permitted at this time. Additional
																	* un-throttled linelocks can be performed using linelock_nothrottle */
		HAP_power_dcvs_payload dcvs;								/**< Updates DCVS params */
		HAP_power_dcvs_v2_payload dcvs_v2;							/**< Updates DCVS_v2 params */
		HAP_power_dcvs_v2_update_payload dcvs_v2_update;			/**< Updates DCVS_v2_update params */
		HAP_power_linelock_nothrottle_payload linelock_nothrottle;	/**< Sets the un-throttled L2 cache linelock parameters */
		HAP_power_dcvs_v3_payload dcvs_v3;							/**< Updates DCVS_v3 params */
		HAP_power_hmx_payload hmx;									/**< Requests to turn on / off HMX */
	};
} HAP_power_request_t;

/** @defgroup HAP_power_functions HAP POWER functions
 *  @{
 */
/**
* Method to set power values from the DSP
* @param[in] context	-	To identify the power client
* @param[in] request	-	Request params.
* @retval 0 on success, AEE_EMMPMREGISTER on MMPM client register request failure, -1 on unknown error
*/
int HAP_power_set(void* context, HAP_power_request_t* request);
/**
 * @} // HAP_power_functions
 */

/** @defgroup HAP_power_enums HAP POWER enums
 *  @{
 */
 /** Identifies the HAP power response type */
typedef enum {
	HAP_power_get_max_mips = 1,				/**< Returns the max mips supported (max_mips) */
	HAP_power_get_max_bus_bw,				/**< Returns the max bus bandwidth supported (max_bus_bw) */
	HAP_power_get_client_class,				/**< Returns the client class (client_class) */
	HAP_power_get_clk_Freq,					/**< Returns the core clock frequency (clkFreqHz) */
	HAP_power_get_aggregateAVSMpps,			/**< Returns the aggregate Mpps used by audio and voice (clkFreqHz) */
	HAP_power_get_dcvsEnabled,				/**< Returns the dcvs status (enabled / disabled) */
	HAP_power_get_vapss_core_clk_Freq,		/**< Returns the VAPSS core clock frequency (clkFreqHz) */
	HAP_power_get_dma_core_clk_Freq,		/**< Returns the DMA core clock frequency (clkFreqHz) */
	HAP_power_get_hcp_core_clk_Freq,		/**< Returns the HCP core clock frequency (clkFreqHz) */
	HAP_power_get_streamer0_core_clk_Freq,	/**< Returns the streamer 0 core clock frequency (clkFreqHz) */
	HAP_power_get_streamer1_core_clk_Freq,	/**< Returns the streamer 1 core clock frequency (clkFreqHz) */
} HAP_Power_response_type;
/**
 * @} // HAP_power_enums
 */

/** Data type to retrieve power values from the DSP */
typedef struct {
	HAP_Power_response_type type;			/**< Identifies the type to retrieve. */
	union{
		unsigned int max_mips;				/**< Max mips supported */
		uint64 max_bus_bw;					/**< Max bus bw supported */
		unsigned int client_class;			/**< Current client class */
		unsigned int clkFreqHz;				/**< Current core CPU frequency */
		unsigned int aggregateAVSMpps;		/**< Aggregate AVS Mpps used by audio and voice */
		boolean dcvsEnabled;				/**< Indicates if dcvs is enabled / disabled. */
	};
} HAP_power_response_t;

/** @defgroup HAP_power_functions HAP POWER functions
 *  @{
 */

/**
* Method to retrieve power values from the DSP
* @param[in] context	-	Ignored
* @param[out] response	-	Response.
*/
int HAP_power_get(void* context, HAP_power_response_t* response);

/**
* Method to initialize dcvs v3 structure in request param. It enables
*		flags and resets params for all fields in dcvs v3. So, this
*		can also be used to remove applied dcvs v3 params and restore
*		defaults.
* @param[in] request	-	Pointer to request params.
*/
/*static inline void HAP_power_set_dcvs_v3_init(HAP_power_request_t* request) {
	memset(request, 0, sizeof(HAP_power_request_t) );
	request->type = HAP_power_set_DCVS_v3;
	request->dcvs_v3.set_dcvs_enable = TRUE;
	request->dcvs_v3.dcvs_enable = TRUE;
	request->dcvs_v3.dcvs_option = HAP_DCVS_V2_POWER_SAVER_MODE;
	request->dcvs_v3.set_latency = TRUE;
	request->dcvs_v3.latency = 65535;
	request->dcvs_v3.set_core_params = TRUE;
	request->dcvs_v3.set_bus_params = TRUE;
	request->dcvs_v3.set_dcvs_v3_params = TRUE;
	request->dcvs_v3.set_sleep_disable = TRUE;
	return;
}*/

/**
* Method to enable/disable dcvs and set particular dcvs policy.
* @param[in] context		-	User context.
* @param[in] dcvs_enable	-	TRUE to enable dcvs, FALSE to disable dcvs.
* @param[in] dcvs_option	-	To set particular dcvs policy. In case of dcvs disable
*                           request, this param will be ignored.
* @returns	-	0 on success
*/
/*static inline int HAP_power_set_dcvs_option(void* context, boolean dcvs_enable,
		HAP_power_dcvs_v2_payload_option dcvs_option) {
	HAP_power_request_t request;
	memset(&request, 0, sizeof(HAP_power_request_t) );
	request.type = HAP_power_set_DCVS_v3;
	request.dcvs_v3.set_dcvs_enable = TRUE;
	request.dcvs_v3.dcvs_enable = dcvs_enable;
	if(dcvs_enable)
		request.dcvs_v3.dcvs_option = dcvs_option;
	return HAP_power_set(context, &request);
}*/

/**
* Method to set/reset sleep latency.
* @param[in] context	-	User context.
* @param[in] latency	-	Sleep latency value in microseconds, should be > 1.
*						Use 65535 max value to reset it to default.
* @returns	-	0 on success
*/
/*static inline int HAP_power_set_sleep_latency(void* context, uint32 latency) {
	HAP_power_request_t request;
	memset(&request, 0, sizeof(HAP_power_request_t) );
	request.type = HAP_power_set_DCVS_v3;
	request.dcvs_v3.set_latency = TRUE;
	request.dcvs_v3.latency = latency;
	return HAP_power_set(context, &request);
}*/

/**
* Method to set/reset DSP core clock voltage corners.
* @param[in] context		-	User context.
* @param[in] target_corner	-	Target voltage corner.
* @param[in] min_corner		-	Minimum voltage corner.
* @param[in] max_corner		-	Maximum voltage corner.
* @returns	-	0 on success
*/
/*static inline int HAP_power_set_core_corner(void* context, uint32 target_corner,
		uint32 min_corner, uint32 max_corner) {
	HAP_power_request_t request;
	memset(&request, 0, sizeof(HAP_power_request_t) );
	request.type = HAP_power_set_DCVS_v3;
	request.dcvs_v3.set_core_params = TRUE;
	request.dcvs_v3.core_params.min_corner = (HAP_dcvs_voltage_corner_t) (min_corner);
	request.dcvs_v3.core_params.max_corner = (HAP_dcvs_voltage_corner_t) (max_corner);
	request.dcvs_v3.core_params.target_corner = (HAP_dcvs_voltage_corner_t) (target_corner);
	return HAP_power_set(context, &request);
}*/

/**
* Method to set/reset bus clock voltage corners.
* @param[in] context		-	User context.
* @param[in] target_corner	-	Target voltage corner.
* @param[in] min_corner		-	Minimum voltage corner.
* @param[in] max_corner		-	Maximum voltage corner.
* @returns	-	0 on success
*/
/*static inline int HAP_power_set_bus_corner(void* context, uint32 target_corner,
		uint32 min_corner, uint32 max_corner) {
    HAP_power_request_t request;
	memset(&request, 0, sizeof(HAP_power_request_t) );
	request.type = HAP_power_set_DCVS_v3;
	request.dcvs_v3.set_bus_params = TRUE;
	request.dcvs_v3.bus_params.min_corner = (HAP_dcvs_voltage_corner_t) (min_corner);
	request.dcvs_v3.bus_params.max_corner = (HAP_dcvs_voltage_corner_t) (max_corner);
	request.dcvs_v3.bus_params.target_corner = (HAP_dcvs_voltage_corner_t) (target_corner);
	return HAP_power_set(context, &request);
}*/

/**
* Method to disable/enable all low power modes.
* @param[in] context		-	User context.
* @param[in] sleep_disable	-	TRUE to disable all low power modes.
*							FALSE to re-enable all low power modes.
* @returns	-	0 on success
*/
/*static inline int HAP_power_set_sleep_mode(void* context, boolean sleep_disable) {
	HAP_power_request_t request;
	memset(&request, 0, sizeof(HAP_power_request_t) );
	request.type = HAP_power_set_DCVS_v3;
	request.dcvs_v3.set_sleep_disable = TRUE;
	request.dcvs_v3.sleep_disable = sleep_disable;
	return HAP_power_set(context, &request);
}*/


/**
* This API is deprecated and might generate undesired results.
* Please use the HAP_power_get() and HAP_power_set() APIs instead.
* Requests a performance level by percentage for clock speed
* and bus speed.  Passing 0 for any parameter results in no
* request being issued for that particular attribute.
* @param[in] clock		-	percentage of target's maximum clock speed
* @param[in] bus		-	percentage of target's maximum bus speed
* @param[in] latency	-	maximum hardware wake up latency in microseconds.  The
*						higher the value the deeper state of sleep
*						that can be entered but the longer it may
*						take to awaken.
* @retval 0 on success
* @par Comments	:	Performance metrics vary from target to target so the
*					intent of this API is to allow callers to set a relative
*					performance level to achieve the desired balance between
*					performance and power saving.
*/
int HAP_power_request(int clock, int bus, int latency);

/**
* This API is deprecated and might generate undesired results.
* Please use the HAP_power_get() and HAP_power_set() APIs instead.
* Requests a performance level by absolute values.  Passing 0
* for any parameter results in no request being issued for that
* particular attribute.
* @param[in] clock		-	speed in MHz
* @param[in] bus		-	bus speed in MHz
* @param[in] latency	-	maximum hardware wakeup latency in microseconds.  The
*						higher the value the deeper state of
*						sleep that can be entered but the
*						longer it may take to awaken.
* @retval 0 on success
* @par Comments	:	This API allows callers who are aware of their target
*					specific capabilities to set them explicitly.
*/
int HAP_power_request_abs(int clock, int bus, int latency);

/**
* This API is deprecated and might generate undesired results.
* Please use the HAP_power_get() and HAP_power_set() APIs instead.
* queries the target for its clock and bus speed capabilities
* @param[out] clock_max	-	maximum clock speed supported in MHz
* @param[out] bus_max	-	maximum bus speed supported in MHz
* @retval 0 on success
*/
int HAP_power_get_max_speed(int* clock_max, int* bus_max);

/**
* This API is deprecated and might generate undesired results.
* Please use the HAP_power_get() and HAP_power_set() APIs instead.
* Upvote for HVX power
* @retval 0 on success
*/
int HVX_power_request(void);

/**
* This API is deprecated and might generate undesired results.
* Please use the HAP_power_get() and HAP_power_set() APIs instead.
* Downvote for HVX power
* @retval 0 on success
*/
int HVX_power_release(void);

/**
* Method to destroy clients created through HAP_power_set
* @param[in] context	-	To uniquely identify the client
* @retval 0 on success, AEE_ENOSUCHCLIENT on Invalid context, -1 on unknown error
* @brief DO NOT call this API directly, use HAP_power_destroy instead.
*/
int HAP_power_destroy_client(void *context);

/**
* @param[in] client	-	To uniquely identify the client context.
* @retval 0 on success, AEE_EUNSUPPORTEDAPI if the API is not supported on the DSP image, AEE_ENOSUCHCLIENT on Invalid context, -1 on unknown error
* @brief Method to destroy clients created through HAP_power_set, wrapper to HAP_power_destroy_client API
*/
static inline int HAP_power_destroy(void *client){
	if(0 != HAP_power_destroy_client)
		return HAP_power_destroy_client(client);
	return AEE_EUNSUPPORTEDAPI;
}

/**
* Method to create user client context
* @retval context for client
*/
//static inline void* HAP_utils_create_context(void) {
	/*
	 * Allocate 1 byte of memory for a unique context identifier
	 * Clients can also allocate memory and use it as unique context identifier
	 */
//	return malloc(1);
//}

/**
* Method to destroy user client context
* @param context of client
*/
/*static inline void HAP_utils_destroy_context(void* context) {
	free(context);
}*/

/**
 * @} // HAP_power_functions
 */
#ifdef __cplusplus
}
#endif
#endif //_HAP_POWER_H

