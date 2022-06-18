//
// Copyright (c) 2015-2016 DashLogic, Inc.
// All Rights Reserved.
//
// http://www.dashlogic.com
// sales@dashlogic.com
//
// Redistribution and use in source and binary forms, with or without
// modification, including use for commercial purposes, are permitted
// provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// 4. Redistributions of any form whatsoever must retain the following
//    acknowledgment: 'This product includes software developed by
//    "DashLogic, Inc." (http://www.dashlogic.com/).'
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


//
// Formatting:
//  Indents:	Use tabs only (1 tab per indent)
//	Tab Size:	4 spaces
//
// File Revision:
// $Rev: 5216 $
// $Date: 2016-03-15 09:32:34 -0600 (Tue, 15 Mar 2016) $
//

#pragma once

#ifdef PANDAJ2534DLL_EXPORTS
#define PANDAJ2534DLL_API extern "C" __declspec(dllexport)
#else
#define PANDAJ2534DLL_API
//__declspec(dllimport)
#endif

//
// Platform-specific Defines:
//
// PTAPI: Define this yourself if you want a specific calling
//        convention or other modifiers on the Pass-Thru API
//        functions. Typically, on Windows, PTAPI will be defined
//        as WINAPI, which enables the __stdcall convention.
//
#define PTAPI			__stdcall	//WINAPI

//
// J2534-1 v04.04 ProtocolID Values
//
#define J1850VPW						0x01
#define J1850PWM						0x02
#define ISO9141							0x03
#define ISO14230						0x04
#define CAN								0x05
#define ISO15765						0x06
#define SCI_A_ENGINE					0x07
#define SCI_A_TRANS						0x08
#define SCI_B_ENGINE					0x09
#define SCI_B_TRANS						0x0A


//
// J2534-2 ProtocolID Values
//
#define J1850VPW_PS						0x00008000
#define J1850PWM_PS						0x00008001
#define ISO9141_PS						0x00008002
#define ISO14230_PS						0x00008003
#define CAN_PS							0x00008004
#define ISO15765_PS						0x00008005
#define J2610_PS						0x00008006
#define SW_ISO15765_PS					0x00008007
#define SW_CAN_PS						0x00008008
#define GM_UART_PS						0x00008009
#define CAN_CH1							0x00009000
#define CAN_CH2							(CAN_CH1 + 1)
#define CAN_CH128						(CAN_CH1 + 127)
#define J1850VPW_CH1					0x00009080
#define J1850VPW_CH2					(J1850VPW_CH1 + 1)
#define J1850VPW_CH128					(J1850VPW_CH1 + 127)
#define J1850PWM_CH1					0x00009160
#define J1850PWM_CH2					(J1850PWM_CH1 + 1)
#define J1850PWM_CH128					(J1850PWM_CH1 + 127)
#define ISO9141_CH1						0x00009240
#define ISO9141_CH2						(ISO9141_CH1 + 1)
#define ISO9141_CH128					(ISO9141_CH1 + 127)
#define ISO14230_CH1					0x00009320
#define ISO14230_CH2					(ISO14230_CH1 + 1)
#define ISO14230_CH128					(ISO14230_CH1 + 127)
#define ISO15765_CH1					0x00009400
#define ISO15765_CH2					(ISO15765_CH1 + 1)
#define ISO15765_CH128					(ISO15765_CH1 + 127)
#define SW_CAN_CAN_CH1					0x00009480
#define SW_CAN_CAN_CH2					(SW_CAN_CAN_CH1 + 1)
#define SW_CAN_CAN_CH128				(SW_CAN_CAN_CH1 + 127)
#define SW_CAN_ISO15765_CH1				0x00009560
#define SW_CAN_ISO15765_CH2				(SW_CAN_ISO15765_CH1 + 1)
#define SW_CAN_ISO15765_CH128			(SW_CAN_ISO15765_CH1 + 127)
#define J2610_CH1						0x00009640
#define J2610_CH2						(J2610_CH1 + 1)
#define J2610_CH128						(J2610_CH1 + 127)
#define ANALOG_IN_CH1					0x0000C000
#define ANALOG_IN_CH2					0x0000C001
#define ANALOG_IN_CH32					0x0000C01F


//
// J2534-1 v04.04 Error Values
//
#define STATUS_NOERROR					0x00	// Function call successful.
#define ERR_NOT_SUPPORTED				0x01	// Device cannot support requested functionality mandated in J2534. Device is not fully SAE J2534 compliant.
#define ERR_INVALID_CHANNEL_ID			0x02	// Invalid ChannelID value.
#define ERR_INVALID_PROTOCOL_ID			0x03	// Invalid or unsupported ProtocolID, or there is a resource conflict (i.e. trying to connect to multiple mutually exclusive protocols such as J1850PWM and J1850VPW, or CAN and SCI, etc.).
#define ERR_NULL_PARAMETER				0x04	// NULL pointer supplied where a valid pointer is required.
#define ERR_INVALID_IOCTL_VALUE			0x05	// Invalid value for Ioctl parameter.
#define ERR_INVALID_FLAGS				0x06	// Invalid flag values.
#define ERR_FAILED						0x07	// Undefined error, use PassThruGetLastError() for text description.
#define ERR_DEVICE_NOT_CONNECTED		0x08	// Unable to communicate with device.
#define ERR_TIMEOUT						0x09	// Read or write timeout:
												// PassThruReadMsgs() - No message available to read or could not read the specified number of messages. The actual number of messages read is placed in <NumMsgs>.
												// PassThruWriteMsgs() - Device could not write the specified number of messages. The actual number of messages sent on the vehicle network is placed in <NumMsgs>.
#define ERR_INVALID_MSG					0x0A	// Invalid message structure pointed to by pMsg.
#define ERR_INVALID_TIME_INTERVAL		0x0B	// Invalid TimeInterval value.
#define ERR_EXCEEDED_LIMIT				0x0C	// Exceeded maximum number of message IDs or allocated space.
#define ERR_INVALID_MSG_ID				0x0D	// Invalid MsgID value.
#define ERR_DEVICE_IN_USE				0x0E	// Device is currently open.
#define ERR_INVALID_IOCTL_ID			0x0F	// Invalid IoctlID value.
#define ERR_BUFFER_EMPTY				0x10	// Protocol message buffer empty, no messages available to read.
#define ERR_BUFFER_FULL					0x11	// Protocol message buffer full. All the messages specified may not have been transmitted.
#define ERR_BUFFER_OVERFLOW				0x12	// Indicates a buffer overflow occurred and messages were lost.
#define ERR_PIN_INVALID					0x13	// Invalid pin number, pin number already in use, or voltage already applied to a different pin.
#define ERR_CHANNEL_IN_USE				0x14	// Channel number is currently connected.
#define ERR_MSG_PROTOCOL_ID				0x15	// Protocol type in the message does not match the protocol associated with the Channel ID
#define ERR_INVALID_FILTER_ID			0x16	// Invalid Filter ID value.
#define ERR_NO_FLOW_CONTROL				0x17	// No flow control filter set or matched (for ProtocolID ISO15765 only).
#define ERR_NOT_UNIQUE					0x18	// A CAN ID in pPatternMsg or pFlowControlMsg matches either ID in an existing FLOW_CONTROL_FILTER
#define ERR_INVALID_BAUDRATE			0x19	// The desired baud rate cannot be achieved within the tolerance specified in SAE J2534-1 Section 6.5
#define ERR_INVALID_DEVICE_ID			0x1A	// Device ID invalid.


//
// J2534-1 v04.04 Connect Flags
//
#define CAN_29BIT_ID						0x0100
#define ISO9141_NO_CHECKSUM					0x0200
#define CAN_ID_BOTH							0x0800
#define ISO9141_K_LINE_ONLY					0x1000


//
// J2534-1 v04.04 Filter Type Values
//
#define PASS_FILTER							0x00000001
#define BLOCK_FILTER						0x00000002
#define FLOW_CONTROL_FILTER					0x00000003


//
// J2534-1 v04.04 Programming Voltage Pin Numbers
//
#define AUXILIARY_OUTPUT_PIN				0
#define SAE_J1962_CONNECTOR_PIN_6			6
#define SAE_J1962_CONNECTOR_PIN_9			9
#define SAE_J1962_CONNECTOR_PIN_11			11
#define SAE_J1962_CONNECTOR_PIN_12			12
#define SAE_J1962_CONNECTOR_PIN_13			13
#define SAE_J1962_CONNECTOR_PIN_14			14
#define SAE_J1962_CONNECTOR_PIN_15			15        // Short to ground only


//
// J2534-1 v04.04 Programming Voltage Values
//
#define SHORT_TO_GROUND						0xFFFFFFFE
#define VOLTAGE_OFF							0xFFFFFFFF


//
// J2534-1 v04.04 API Version Values
//
#define J2534_APIVER_FEBRUARY_2002		"02.02"
#define J2534_APIVER_NOVEMBER_2004		"04.04"


//
// J2534-1 v04.04 IOCTL ID Values
//
#define GET_CONFIG								0x01	// pInput = SCONFIG_LIST, pOutput = NULL
#define SET_CONFIG								0x02	// pInput = SCONFIG_LIST, pOutput = NULL
#define READ_VBATT								0x03	// pInput = NULL, pOutput = unsigned long
#define FIVE_BAUD_INIT							0x04	// pInput = SBYTE_ARRAY, pOutput = SBYTE_ARRAY
#define FAST_INIT								0x05	// pInput = PASSTHRU_MSG, pOutput = PASSTHRU_MSG
#define CLEAR_TX_BUFFER							0x07	// pInput = NULL, pOutput = NULL
#define CLEAR_RX_BUFFER							0x08	// pInput = NULL, pOutput = NULL
#define CLEAR_PERIODIC_MSGS						0x09	// pInput = NULL, pOutput = NULL
#define CLEAR_MSG_FILTERS						0x0A	// pInput = NULL, pOutput = NULL
#define CLEAR_FUNCT_MSG_LOOKUP_TABLE			0x0B	// pInput = NULL, pOutput = NULL
#define ADD_TO_FUNCT_MSG_LOOKUP_TABLE			0x0C	// pInput = SBYTE_ARRAY, pOutput = NULL
#define DELETE_FROM_FUNCT_MSG_LOOKUP_TABLE		0x0D	// pInput = SBYTE_ARRAY, pOutput = NULL
#define READ_PROG_VOLTAGE						0x0E	// pInput = NULL, pOutput = unsigned long


//
// J2534-2 IOCTL ID Values
//
#define SW_CAN_HS								0x00008000	// pInput = NULL, pOutput = NULL
#define SW_CAN_NS								0x00008001	// pInput = NULL, pOutput = NULL
#define SET_POLL_RESPONSE						0x00008002	// pInput = SBYTE_ARRAY, pOutput = NULL
#define BECOME_MASTER							0x00008003	// pInput = unsigned char, pOutput = NULL


//
// J2534-1 v04.04 Configuration Parameter Values
// Default value is enclosed in square brackets "[" and "]"
//
#define DATA_RATE				0x01	// 5-500000
#define LOOPBACK				0x03	// 0 (OFF), 1 (ON) [0]
#define NODE_ADDRESS			0x04	// J1850PWM: 0x00-0xFF
#define NETWORK_LINE			0x05	// J1850PWM: 0 (BUS_NORMAL), 1 (BUS_PLUS), 2 (BUS_MINUS) [0]
#define P1_MIN					0x06	// ISO9141 or ISO14230: Not used by interface
#define P1_MAX					0x07	// ISO9141 or ISO14230: 0x1-0xFFFF (.5 ms per bit) [40 (20ms)]
#define P2_MIN					0x08	// ISO9141 or ISO14230: Not used by interface
#define P2_MAX					0x09	// ISO9141 or ISO14230: Not used by interface
#define P3_MIN					0x0A	// ISO9141 or ISO14230: 0x0-0xFFFF (.5 ms per bit) [110 (55ms)]
#define P3_MAX					0x0B	// ISO9141 or ISO14230: Not used by interface
#define P4_MIN					0x0C	// ISO9141 or ISO14230: 0x0-0xFFFF (.5 ms per bit) [10 (5ms)]
#define P4_MAX					0x0D	// ISO9141 or ISO14230: Not used by interface
#define W0						0x19	// ISO9141: 0x0-0xFFFF (1 ms per bit) [300]
#define W1						0x0E	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [300]
#define W2						0x0F	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [20]
#define W3						0x10	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [20]
#define W4						0x11	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [50]
#define W5						0x12	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [300]
#define TIDLE					0x13	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [300]
#define TINIL					0x14	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [25]
#define TWUP					0x15	// ISO9141 or ISO14230: 0x0-0xFFFF (1 ms per bit) [50]
#define PARITY					0x16	// ISO9141 or ISO14230: 0 (NO_PARITY), 1 (ODD_PARITY), 2 (EVEN_PARITY) [0]
#define BIT_SAMPLE_POINT		0x17	// CAN: 0-100 (1% per bit) [80]
#define SYNC_JUMP_WIDTH			0x18	// CAN: 0-100 (1% per bit) [15]
#define T1_MAX					0x1A	// SCI: 0x0-0xFFFF (1 ms per bit) [20]
#define T2_MAX					0x1B	// SCI: 0x0-0xFFFF (1 ms per bit) [100]
#define T3_MAX					0x24	// SCI: 0x0-0xFFFF (1 ms per bit) [50]
#define T4_MAX					0x1C	// SCI: 0x0-0xFFFF (1 ms per bit) [20]
#define T5_MAX					0x1D	// SCI: 0x0-0xFFFF (1 ms per bit) [100]
#define ISO15765_BS				0x1E	// ISO15765: 0x0-0xFF [0]
#define ISO15765_STMIN			0x1F	// ISO15765: 0x0-0xFF [0]
#define ISO15765_BS_TX			0x22	// ISO15765: 0x0-0xFF,0xFFFF [0xFFFF]
#define ISO15765_STMIN_TX		0x23	// ISO15765: 0x0-0xFF,0xFFFF [0xFFFF]
#define DATA_BITS				0x20	// ISO9141 or ISO14230: 0 (8 data bits), 1 (7 data bits) [0]
#define FIVE_BAUD_MOD			0x21	// ISO9141 or ISO14230: 0 (ISO 9141-2/14230-4), 1 (Inv KB2), 2 (Inv Addr), 3 (ISO 9141) [0]
#define ISO15765_WFT_MAX		0x25	// ISO15765: 0x0-0xFF [0]


//
// J2534-2 Configuration Parameter Values
// Default value is enclosed in square brackets "[" and "]"
//
#define CAN_MIXED_FORMAT				0x00008000	// See #defines below. [0]
#define J1962_PINS						0x00008001	// 0xPPSS PP: 0x00-0x10 SS: 0x00-0x10 PP!=SS, except 0x0000. Exclude pins 4, 5, and 16. [0]
#define SW_CAN_HS_DATA_RATE				0x00008010	// SWCAN: 5-500000 [83333]
#define SW_CAN_SPEEDCHANGE_ENABLE		0x00008011	// SWCAN: 0 (DISABLE_SPDCHANGE), 1 (ENABLE_SPDCHANGE) [0]
#define SW_CAN_RES_SWITCH				0x00008012	// SWCAN: 0 (DISCONNECT_RESISTOR), 1 (CONNECT_RESISTOR), 2 (AUTO_ RESISTOR) [0]
#define ACTIVE_CHANNELS					0x00008020	// ANALOG: 0-0xFFFFFFFF
#define SAMPLE_RATE						0x00008021	// ANALOG: 0-0xFFFFFFFF [0] (high bit changes meaning from samples/sec to seconds/sample)
#define SAMPLES_PER_READING				0x00008022	// ANALOG: 1-0xFFFFFFFF [1]
#define READINGS_PER_MSG				0x00008023	// ANALOG: 1-0x00000408 (1 - 1032) [1]
#define AVERAGING_METHOD				0x00008024	// ANALOG: 0-0xFFFFFFFF [0]
#define SAMPLE_RESOLUTION				0x00008025	// ANALOG READ-ONLY: 0x1-0x20 (1 - 32)
#define INPUT_RANGE_LOW					0x00008026	// ANALOG READ-ONLY: 0x80000000-0x7FFFFFFF (-2147483648-2147483647)
#define INPUT_RANGE_HIGH				0x00008027	// ANALOG READ-ONLY: 0x80000000-0x7FFFFFFF (-2147483648-2147483647)


//
// J2534-2 Mixed-Mode/Format CAN Definitions
//
#define CAN_MIXED_FORMAT_OFF			0	// Messages will be treated as ISO 15765 ONLY.
#define CAN_MIXED_FORMAT_ON				1	// Messages will be treated as either ISO 15765 or an unformatted CAN frame.
#define CAN_MIXED_FORMAT_ALL_FRAMES		2	// Messages will be treated as ISO 15765, an unformatted CAN frame, or both.


//
// J2534-2 Analog Channel Averaging Method Definitions
//
#define SIMPLE_AVERAGE			0x00000000        // Simple arithmetic mean
#define MAX_LIMIT_AVERAGE		0x00000001        // Choose the biggest value
#define MIN_LIMIT_AVERAGE		0x00000002        // Choose the lowest value
#define MEDIAN_AVERAGE			0x00000003        // Choose arithmetic median


//
// J2534-1 v04.04 RxStatus Definitions
//
#define TX_MSG_TYPE					0x0001
#define START_OF_MESSAGE			0x0002
#define RX_BREAK					0x0004
#define TX_INDICATION				0x0008
#define ISO15765_PADDING_ERROR		0x0010
#define ISO15765_ADDR_TYPE			0x0080
//#define CAN_29BIT_ID				0x0100		// Defined above


//
// J2534-2 RxStatus Definitions
//
#define SW_CAN_HV_RX				0x00010000	// SWCAN Channels Only
#define SW_CAN_HS_RX				0x00020000	// SWCAN Channels Only
#define SW_CAN_NS_RX				0x00040000	// SWCAN Channels Only
#define OVERFLOW_					0x00010000	// Analog Input Channels Only


//
// J2534-1 v04.04 TxFlags Definitions
//
#define ISO15765_FRAME_PAD			0x0040
//#define ISO15765_ADDR_TYPE		0x0080		// Defined above
//#define CAN_29BIT_ID				0x0100		// Defined above
#define WAIT_P3_MIN_ONLY			0x0200
#define SCI_MODE					0x400000
#define SCI_TX_VOLTAGE				0x800000


//
// J2534-2 TxFlags Definitions
//
#define SW_CAN_HV_TX				0x00000400


//
// J2534-1 v04.04 Structure Definitions
//
typedef struct
{
	unsigned long	Parameter;	// Name of parameter
	unsigned long	Value;		// Value of the parameter
} SCONFIG;


typedef struct
{
	unsigned long	NumOfParams;	// Number of SCONFIG elements
	SCONFIG*		ConfigPtr;		// Array of SCONFIG
} SCONFIG_LIST;


typedef struct
{
	unsigned long	NumOfBytes;		// Number of bytes in the array
	unsigned char*	BytePtr;		// Array of bytes
} SBYTE_ARRAY;


typedef struct
{
	unsigned long	ProtocolID;
	unsigned long	RxStatus;
	unsigned long	TxFlags;
	unsigned long	Timestamp;
	unsigned long	DataSize;
	unsigned long	ExtraDataIndex;
	unsigned char	Data[4128];
} PASSTHRU_MSG;

//
// J2534-1 v04.04 Function Prototypes
//
PANDAJ2534DLL_API long PTAPI	PassThruOpen(void *pName, unsigned long *pDeviceID);
PANDAJ2534DLL_API long PTAPI	PassThruClose(unsigned long DeviceID);
PANDAJ2534DLL_API long PTAPI	PassThruConnect(unsigned long DeviceID, unsigned long ProtocolID, unsigned long Flags, unsigned long BaudRate, unsigned long *pChannelID);
PANDAJ2534DLL_API long PTAPI	PassThruDisconnect(unsigned long ChannelID);
PANDAJ2534DLL_API long PTAPI	PassThruReadMsgs(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
PANDAJ2534DLL_API long PTAPI	PassThruWriteMsgs(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
PANDAJ2534DLL_API long PTAPI	PassThruStartPeriodicMsg(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pMsgID, unsigned long TimeInterval);
PANDAJ2534DLL_API long PTAPI	PassThruStopPeriodicMsg(unsigned long ChannelID, unsigned long MsgID);
PANDAJ2534DLL_API long PTAPI	PassThruStartMsgFilter(unsigned long ChannelID, unsigned long FilterType, PASSTHRU_MSG *pMaskMsg, PASSTHRU_MSG *pPatternMsg, PASSTHRU_MSG *pFlowControlMsg, unsigned long *pFilterID);
PANDAJ2534DLL_API long PTAPI	PassThruStopMsgFilter(unsigned long ChannelID, unsigned long FilterID);
PANDAJ2534DLL_API long PTAPI	PassThruSetProgrammingVoltage(unsigned long DeviceID, unsigned long PinNumber, unsigned long Voltage);
PANDAJ2534DLL_API long PTAPI	PassThruReadVersion(unsigned long DeviceID, char *pFirmwareVersion, char *pDllVersion, char *pApiVersion);
PANDAJ2534DLL_API long PTAPI	PassThruGetLastError(char *pErrorDescription);
PANDAJ2534DLL_API long PTAPI	PassThruIoctl(unsigned long ChannelID, unsigned long IoctlID, void *pInput, void *pOutput);


//
// J2534-1 v04.04 Function Typedefs
// These function typedefs allow simpler use of the J2534 API by
// allowing you to do things like this:
// PTCONNECT	pPassThruConnectFunc = GetProcAddress(hModule, "PassThruConnect");
// if (pPassThruConnectFunc == NULL)
//     return FALSE;
// pPassThruConnectFunc(DeviceID, CAN, CAN_29BIT_ID, 500000, &ChannelID);
//
typedef long	(PTAPI *PTOPEN)(void *pName, unsigned long *pDeviceID);
typedef long	(PTAPI *PTCLOSE)(unsigned long DeviceID);
typedef long	(PTAPI *PTCONNECT)(unsigned long DeviceID, unsigned long ProtocolID, unsigned long Flags, unsigned long BaudRate, unsigned long *pChannelID);
typedef long	(PTAPI *PTDISCONNECT)(unsigned long ChannelID);
typedef long	(PTAPI *PTREADMSGS)(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
typedef long	(PTAPI *PTWRITEMSGS)(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
typedef long	(PTAPI *PTSTARTPERIODICMSG)(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pMsgID, unsigned long TimeInterval);
typedef long	(PTAPI *PTSTOPPERIODICMSG)(unsigned long ChannelID, unsigned long MsgID);
typedef long	(PTAPI *PTSTARTMSGFILTER)(unsigned long ChannelID, unsigned long FilterType, PASSTHRU_MSG *pMaskMsg, PASSTHRU_MSG *pPatternMsg, PASSTHRU_MSG *pFlowControlMsg, unsigned long *pFilterID);
typedef long	(PTAPI *PTSTOPMSGFILTER)(unsigned long ChannelID, unsigned long FilterID);
typedef long	(PTAPI *PTSETPROGRAMMINGVOLTAGE)(unsigned long DeviceID, unsigned long PinNumber, unsigned long Voltage);
typedef long	(PTAPI *PTREADVERSION)(unsigned long DeviceID, char *pFirmwareVersion, char *pDllVersion, char *pApiVersion);
typedef long	(PTAPI *PTGETLASTERROR)(char *pErrorDescription);
typedef long	(PTAPI *PTIOCTL)(unsigned long ChannelID, unsigned long IoctlID, void *pInput, void *pOutput);
