// Loader4.cpp
// (c) 2005 National Control Systems, Inc.
// Portions (c) 2004 Drew Technologies, Inc.
// Dynamic J2534 v04.04 dll loader for VB

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to:
// the Free Software Foundation, Inc.
// 51 Franklin Street, Fifth Floor
// Boston, MA  02110-1301, USA

// National Control Systems, Inc.
// 10737 Hamburg Rd
// Hamburg, MI 48139
// 810-231-2901

// Drew Technologies, Inc.
// 7012  E.M -36, Suite 3B
// Whitmore Lake, MI 48189
// 810-231-3171

#define STRICT
#include "stdafx.h"
#include <windows.h>
#include "Loader4.h"

PTOPEN LocalOpen;
PTCLOSE LocalClose;
PTCONNECT LocalConnect;
PTDISCONNECT LocalDisconnect;
PTREADMSGS LocalReadMsgs;
PTWRITEMSGS LocalWriteMsgs;
PTSTARTPERIODICMSG LocalStartPeriodicMsg;
PTSTOPPERIODICMSG LocalStopPeriodicMsg;
PTSTARTMSGFILTER LocalStartMsgFilter;
PTSTOPMSGFILTER LocalStopMsgFilter;
PTSETPROGRAMMINGVOLTAGE LocalSetProgrammingVoltage;
PTREADVERSION LocalReadVersion;
PTGETLASTERROR LocalGetLastError;
PTIOCTL LocalIoctl;

HINSTANCE hDLL = NULL;
//BOOL bIsCorrectVersion = FALSE;

BOOL WINAPI DllMain(HINSTANCE hInstA, DWORD dwReason, LPVOID lpvReserved)
{
	switch (dwReason) {
	case DLL_PROCESS_ATTACH:
		// The DLL is being mapped into the process's address space

	case DLL_THREAD_ATTACH:
		// A thread is being created
	    break;

	case DLL_THREAD_DETACH:
		// A thread is exiting cleanly
	    break;

	case DLL_PROCESS_DETACH:
		// The DLL is being unmapped from the process's address space
	    break;
	}

	return TRUE;
}


long WINAPI LoadJ2534Dll(char *sLib)
{
	long lFuncList = 0;

	if (hDLL != NULL) UnloadJ2534Dll();
	hDLL = LoadLibraryA (sLib);
	if (hDLL == NULL) return ERR_NO_DLL;

	LocalOpen = (PTOPEN)(GetProcAddress(hDLL, "PassThruOpen"));
	if (LocalOpen == NULL) lFuncList = lFuncList | ERR_NO_PTOPEN;

	LocalClose = (PTCLOSE)(GetProcAddress(hDLL, "PassThruClose"));
	if (LocalClose == NULL) lFuncList = lFuncList | ERR_NO_PTCLOSE;

	LocalConnect = (PTCONNECT)(GetProcAddress(hDLL,"PassThruConnect"));
	if (LocalConnect == NULL) lFuncList = lFuncList | ERR_NO_PTCONNECT;

	LocalDisconnect = (PTDISCONNECT)(GetProcAddress(hDLL,"PassThruDisconnect"));
	if (LocalDisconnect == NULL) lFuncList = lFuncList | ERR_NO_PTDISCONNECT;

	LocalReadMsgs = (PTREADMSGS)(GetProcAddress(hDLL,"PassThruReadMsgs"));
	if (LocalReadMsgs == NULL) lFuncList = lFuncList | ERR_NO_PTREADMSGS;

	LocalWriteMsgs = (PTWRITEMSGS)(GetProcAddress(hDLL,"PassThruWriteMsgs"));
	if (LocalWriteMsgs == NULL) lFuncList = lFuncList | ERR_NO_PTWRITEMSGS;

	LocalStartPeriodicMsg = (PTSTARTPERIODICMSG)(GetProcAddress(hDLL,"PassThruStartPeriodicMsg"));
	if (LocalStartPeriodicMsg == NULL) lFuncList = lFuncList | ERR_NO_PTSTARTPERIODICMSG;

	LocalStopPeriodicMsg = (PTSTOPPERIODICMSG)(GetProcAddress(hDLL,"PassThruStopPeriodicMsg"));
	if (LocalStopPeriodicMsg == NULL) lFuncList = lFuncList | ERR_NO_PTSTOPPERIODICMSG;

	LocalStartMsgFilter = (PTSTARTMSGFILTER)(GetProcAddress(hDLL,"PassThruStartMsgFilter"));
	if (LocalStartPeriodicMsg == NULL) lFuncList = lFuncList | ERR_NO_PTSTARTMSGFILTER;

	LocalStopMsgFilter = (PTSTOPMSGFILTER)(GetProcAddress(hDLL,"PassThruStopMsgFilter"));
	if (LocalStopMsgFilter == NULL) lFuncList = lFuncList | ERR_NO_PTSTOPMSGFILTER;

	LocalSetProgrammingVoltage = (PTSETPROGRAMMINGVOLTAGE)(GetProcAddress(hDLL,"PassThruSetProgrammingVoltage"));
	if (LocalSetProgrammingVoltage == NULL) lFuncList = lFuncList | ERR_NO_PTSETPROGRAMMINGVOLTAGE;

	LocalReadVersion = (PTREADVERSION)(GetProcAddress(hDLL,"PassThruReadVersion"));
	if (LocalReadVersion == NULL) lFuncList = lFuncList | ERR_NO_PTREADVERSION;

	LocalGetLastError = (PTGETLASTERROR)(GetProcAddress(hDLL,"PassThruGetLastError"));
	if (LocalGetLastError == NULL) lFuncList = lFuncList | ERR_NO_PTGETLASTERROR;

	LocalIoctl = (PTIOCTL)(GetProcAddress(hDLL,"PassThruIoctl"));
	if (LocalIoctl == NULL) lFuncList = lFuncList | ERR_NO_PTIOCTL;

	if (lFuncList == ERR_NO_FUNCTIONS) return ERR_WRONG_DLL_VER;

	return lFuncList;
}

long WINAPI UnloadJ2534Dll()
{
	if (FreeLibrary(hDLL))
	{
		hDLL = NULL;
		LocalOpen = NULL;
		LocalClose = NULL;
		LocalConnect = NULL;
		LocalDisconnect = NULL;
		LocalReadMsgs = NULL;
		LocalWriteMsgs = NULL;
		LocalStartPeriodicMsg = NULL;
		LocalStopPeriodicMsg = NULL;
		LocalStartMsgFilter = NULL;
		LocalStopMsgFilter = NULL;
		LocalSetProgrammingVoltage = NULL;
		LocalReadVersion = NULL;
		LocalGetLastError = NULL;
		LocalIoctl = NULL;
		return 0;
	}
	return ERR_NO_DLL;
}

long WINAPI PassThruOpen(void *pName, unsigned long *pDeviceID)
{
	if (LocalOpen == NULL) return ERR_FUNC_MISSING;
	return LocalOpen(pName, pDeviceID);
}

long WINAPI PassThruClose(unsigned long DeviceID)
{
	if (LocalOpen == NULL) return ERR_FUNC_MISSING;
	return LocalClose(DeviceID);
}

long WINAPI PassThruConnect(unsigned long DeviceID, unsigned long ProtocolID, unsigned long Flags, unsigned long Baudrate, unsigned long *pChannelID)
{
	if (LocalConnect == NULL) return ERR_FUNC_MISSING;
	return LocalConnect(DeviceID, ProtocolID, Flags, Baudrate, pChannelID);
}

long WINAPI PassThruDisconnect(unsigned long ChannelID)
{
	if (LocalDisconnect == NULL) return ERR_FUNC_MISSING;
	return LocalDisconnect(ChannelID);
}

long WINAPI PassThruReadMsgs(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout)
{
	if (LocalReadMsgs == NULL) return ERR_FUNC_MISSING;
	return LocalReadMsgs(ChannelID, pMsg, pNumMsgs, Timeout);
}

long WINAPI PassThruWriteMsgs(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout)
{
	if (LocalWriteMsgs == NULL) return ERR_FUNC_MISSING;
	return LocalWriteMsgs(ChannelID, pMsg, pNumMsgs, Timeout);
}

long WINAPI PassThruStartPeriodicMsg(unsigned long ChannelID, PASSTHRU_MSG *pMsg, unsigned long *pMsgID, unsigned long TimeInterval)
{
	if (LocalStartPeriodicMsg == NULL) return ERR_FUNC_MISSING;
	return LocalStartPeriodicMsg(ChannelID, pMsg, pMsgID, TimeInterval);
}

long WINAPI PassThruStopPeriodicMsg(unsigned long ChannelID, unsigned long MsgID)
{
	if (LocalStopPeriodicMsg == NULL) return ERR_FUNC_MISSING;
	return LocalStopPeriodicMsg(ChannelID, MsgID);
}

long WINAPI PassThruStartMsgFilter(unsigned long ChannelID, unsigned long FilterType,
	PASSTHRU_MSG *pMaskMsg, PASSTHRU_MSG *pPatternMsg, PASSTHRU_MSG *pFlowControlMsg, unsigned long *pFilterID)
{
	if (LocalStartMsgFilter == NULL) return ERR_FUNC_MISSING;
	return LocalStartMsgFilter(ChannelID, FilterType, pMaskMsg, pPatternMsg, pFlowControlMsg, pFilterID);
}

long WINAPI PassThruStopMsgFilter(unsigned long ChannelID, unsigned long FilterID)
{
	if (LocalStopMsgFilter == NULL) return ERR_FUNC_MISSING;
	return LocalStopMsgFilter(ChannelID, FilterID);
}

long WINAPI PassThruSetProgrammingVoltage(unsigned long DeviceID, unsigned long PinNumber, unsigned long Voltage)
{
	if (LocalSetProgrammingVoltage == NULL) return ERR_FUNC_MISSING;
	return LocalSetProgrammingVoltage(DeviceID, PinNumber, Voltage);
}

long WINAPI PassThruReadVersion(unsigned long DeviceID, char *pFirmwareVersion, char *pDllVersion, char *pApiVersion)
{
	if (LocalReadVersion == NULL) return ERR_FUNC_MISSING;
	return LocalReadVersion(DeviceID, pFirmwareVersion, pDllVersion, pApiVersion);
}

long WINAPI PassThruGetLastError(char *pErrorDescription)
{
	if (LocalGetLastError == NULL) return ERR_FUNC_MISSING;
	return LocalGetLastError(pErrorDescription);
}

long WINAPI PassThruIoctl(unsigned long ChannelID, unsigned long IoctlID, void *pInput, void *pOutput)
{
	if (LocalIoctl == NULL) return ERR_FUNC_MISSING;
	return LocalIoctl(ChannelID, IoctlID, pInput, pOutput);
}